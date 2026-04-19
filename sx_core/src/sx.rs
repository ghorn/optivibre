use std::cell::{Cell, RefCell};
use std::collections::hash_map::{DefaultHasher, Entry};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::ccs::CCS;
use crate::error::{Result, SxError};
use crate::expr::{SXExpr, SXExprMatrix};
use crate::function::{
    DependencyProfile, FunctionId, SXFunction, dependency_profile, forward_batch_helper,
    forward_helper, function_by_id, function_name, reverse_output_batch_helper,
    reverse_scalar_helper,
};
use crate::{Index, SXMatrix};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SX(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SXContext {
    id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Abs,
    Sign,
    Floor,
    Ceil,
    Round,
    Trunc,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Atan2,
    Hypot,
    Mod,
    Copysign,
}

#[derive(Clone, Debug)]
pub(crate) enum NodeKind {
    Constant(f64),
    Symbol {
        serial: usize,
        name: Arc<str>,
    },
    Unary {
        op: UnaryOp,
        arg: SX,
    },
    Binary {
        op: BinaryOp,
        lhs: SX,
        rhs: SX,
    },
    Call {
        function: FunctionId,
        inputs: CallInputs,
        output_slot: Index,
        output_offset: Index,
    },
}

#[derive(Clone, Debug)]
struct Node {
    kind: NodeKind,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum NodeKey {
    Constant(u64),
    Unary {
        op: UnaryOp,
        arg: SX,
    },
    Binary {
        op: BinaryOp,
        lhs: SX,
        rhs: SX,
    },
    Call {
        function: FunctionId,
        inputs: CallInputs,
        output_slot: Index,
        output_offset: Index,
    },
}

struct NodeArena {
    context_id: u32,
    nodes: Vec<Node>,
    keyed: HashMap<NodeKey, SX>,
    next_symbol_serial: usize,
}

struct ContextArena {
    arena: Mutex<NodeArena>,
}

#[derive(Default)]
struct ContextRegistry {
    next_context_id: u32,
    arenas: HashMap<u32, Arc<ContextArena>>,
}

const ROOT_CONTEXT_ID: u32 = 1;

static CONTEXT_REGISTRY: OnceLock<Mutex<ContextRegistry>> = OnceLock::new();
thread_local! {
    static CURRENT_CONTEXT_ID: Cell<u32> = const { Cell::new(ROOT_CONTEXT_ID) };
    static CONTEXT_ARENA_CACHE: RefCell<HashMap<u32, Arc<ContextArena>>> = RefCell::new(HashMap::new());
}

#[derive(Clone, Debug, PartialEq)]
pub enum NodeView {
    Constant(f64),
    Symbol {
        name: String,
        serial: usize,
    },
    Unary {
        op: UnaryOp,
        arg: SX,
    },
    Binary {
        op: BinaryOp,
        lhs: SX,
        rhs: SX,
    },
    Call {
        function_id: FunctionId,
        function_name: String,
        inputs: Vec<SXMatrix>,
        output_slot: Index,
        output_offset: Index,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct CallInputs {
    values: Arc<[SXMatrix]>,
    hash: u64,
    site_id: CallSiteId,
}

impl CallInputs {
    pub(crate) fn new(inputs: Vec<SXMatrix>) -> Self {
        let values: Arc<[SXMatrix]> = inputs.into();
        let mut hasher = DefaultHasher::new();
        values.hash(&mut hasher);
        let hash = hasher.finish();
        let site_id = CallSiteId {
            ptr: values.as_ptr() as usize,
            len: values.len(),
        };
        Self {
            values,
            hash,
            site_id,
        }
    }

    pub(crate) fn from_slice(inputs: &[SXMatrix]) -> Self {
        Self::new(inputs.to_vec())
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, SXMatrix> {
        self.values.iter()
    }

    #[allow(dead_code)]
    pub(crate) fn as_slice(&self) -> &[SXMatrix] {
        &self.values
    }

    pub(crate) fn to_vec(&self) -> Vec<SXMatrix> {
        self.values.iter().cloned().collect()
    }

    pub(crate) fn site_id(&self) -> CallSiteId {
        self.site_id
    }
}

impl PartialEq for CallInputs {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl Eq for CallInputs {}

impl Hash for CallInputs {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CallSiteId {
    ptr: usize,
    len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ForwardCallCacheKey {
    function: FunctionId,
    site_id: CallSiteId,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ForwardCallMemoKey {
    site_key: ForwardCallCacheKey,
    seed_inputs: CallInputs,
}

#[derive(Default)]
struct AdLocalCaches {
    dependency_profiles: HashMap<FunctionId, Arc<DependencyProfile>>,
    forward_helpers: HashMap<FunctionId, Arc<SXFunction>>,
    forward_batch_helpers: HashMap<(FunctionId, usize), Arc<SXFunction>>,
    reverse_scalar_helpers: HashMap<(FunctionId, Index, Index), Arc<SXFunction>>,
    reverse_output_batch_helpers: HashMap<(FunctionId, Index, Index, usize), Arc<SXFunction>>,
    adexpr_slot_scratch: HashMap<FunctionId, Vec<Vec<AdExpr>>>,
    adexpr_derivative_scratch: HashMap<(FunctionId, usize), Vec<Vec<AdExpr>>>,
}

impl AdLocalCaches {
    fn dependency_profile(&mut self, function_id: FunctionId) -> Arc<DependencyProfile> {
        if let Some(profile) = self.dependency_profiles.get(&function_id) {
            return Arc::clone(profile);
        }
        let profile = dependency_profile(function_id);
        self.dependency_profiles
            .insert(function_id, Arc::clone(&profile));
        profile
    }

    fn forward_helper(&mut self, function_id: FunctionId) -> Result<Arc<SXFunction>> {
        if let Some(helper) = self.forward_helpers.get(&function_id) {
            return Ok(Arc::clone(helper));
        }
        let helper = forward_helper(function_id)?;
        self.forward_helpers
            .insert(function_id, Arc::clone(&helper));
        Ok(helper)
    }

    fn forward_batch_helper(
        &mut self,
        function_id: FunctionId,
        directions: usize,
    ) -> Result<Arc<SXFunction>> {
        let key = (function_id, directions);
        if let Some(helper) = self.forward_batch_helpers.get(&key) {
            return Ok(Arc::clone(helper));
        }
        let helper = forward_batch_helper(function_id, directions)?;
        self.forward_batch_helpers.insert(key, Arc::clone(&helper));
        Ok(helper)
    }

    fn reverse_scalar_helper(
        &mut self,
        function_id: FunctionId,
        output_slot: Index,
        output_offset: Index,
    ) -> Result<Arc<SXFunction>> {
        let key = (function_id, output_slot, output_offset);
        if let Some(helper) = self.reverse_scalar_helpers.get(&key) {
            return Ok(Arc::clone(helper));
        }
        let helper = reverse_scalar_helper(function_id, output_slot, output_offset)?;
        self.reverse_scalar_helpers.insert(key, Arc::clone(&helper));
        Ok(helper)
    }

    fn reverse_output_batch_helper(
        &mut self,
        function_id: FunctionId,
        output_slot: Index,
        output_offset: Index,
        directions: usize,
    ) -> Result<Arc<SXFunction>> {
        let key = (function_id, output_slot, output_offset, directions);
        if let Some(helper) = self.reverse_output_batch_helpers.get(&key) {
            return Ok(Arc::clone(helper));
        }
        let helper =
            reverse_output_batch_helper(function_id, output_slot, output_offset, directions)?;
        self.reverse_output_batch_helpers
            .insert(key, Arc::clone(&helper));
        Ok(helper)
    }

    fn take_adexpr_slot_scratch(
        &mut self,
        function_id: FunctionId,
        template: &[AdExpr],
    ) -> Vec<AdExpr> {
        if let Some(buffer) = self
            .adexpr_slot_scratch
            .get_mut(&function_id)
            .and_then(|buffers| buffers.pop())
        {
            let mut buffer = buffer;
            buffer.clear();
            buffer.extend_from_slice(template);
            buffer
        } else {
            template.to_vec()
        }
    }

    fn recycle_adexpr_slot_scratch(&mut self, function_id: FunctionId, buffer: Vec<AdExpr>) {
        self.adexpr_slot_scratch
            .entry(function_id)
            .or_default()
            .push(buffer);
    }

    fn take_adexpr_derivative_scratch(
        &mut self,
        function_id: FunctionId,
        directions: usize,
        len: usize,
        zero: AdExpr,
    ) -> Vec<AdExpr> {
        let key = (function_id, directions);
        if let Some(buffer) = self
            .adexpr_derivative_scratch
            .get_mut(&key)
            .and_then(|buffers| buffers.pop())
        {
            let mut buffer = buffer;
            if buffer.len() != len {
                buffer.resize(len, zero);
            } else {
                buffer.fill(zero);
            }
            buffer
        } else {
            vec![zero; len]
        }
    }

    fn recycle_adexpr_derivative_scratch(
        &mut self,
        function_id: FunctionId,
        directions: usize,
        buffer: Vec<AdExpr>,
    ) {
        self.adexpr_derivative_scratch
            .entry((function_id, directions))
            .or_default()
            .push(buffer);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct JacobianStructureKey {
    outputs: Vec<SX>,
    vars: Vec<SX>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProgramKey {
    outputs: Vec<SX>,
}

#[derive(Clone, Debug)]
struct ProgramCallInput {
    matrix: SXMatrix,
    slots: Vec<usize>,
    relevant_offsets: Vec<usize>,
}

#[derive(Clone, Debug)]
enum ProgramInstruction {
    Unary {
        result_slot: usize,
        op: UnaryOp,
        arg_slot: usize,
        arg_expr: SX,
    },
    Binary {
        result_slot: usize,
        op: BinaryOp,
        lhs_slot: usize,
        rhs_slot: usize,
        lhs_expr: SX,
        rhs_expr: SX,
    },
    Call {
        result_slot: usize,
        function: FunctionId,
        site_key: ForwardCallCacheKey,
        output_slot: Index,
        output_offset: Index,
        inputs: Vec<ProgramCallInput>,
    },
}

#[derive(Clone, Debug)]
struct SxProgram {
    slot_exprs: Vec<SX>,
    slot_by_node: HashMap<SX, usize>,
    output_slots: Vec<usize>,
    instructions: Vec<ProgramInstruction>,
}

#[derive(Clone, Debug)]
struct FunctionProgramPlan {
    program: Arc<SxProgram>,
    input_slots: Vec<Vec<Option<usize>>>,
    output_ccs: Vec<CCS>,
    output_nnz: Vec<usize>,
}

#[derive(Clone, Debug)]
struct FunctionExprProgramPlan {
    program: Arc<SxProgram>,
    slot_exprs: Vec<SXExpr>,
    input_slots: Vec<Vec<Option<usize>>>,
    output_ccs: Vec<CCS>,
    output_nnz: Vec<usize>,
}

#[derive(Clone, Debug)]
struct FunctionAdExprArenaPlan {
    program: Arc<SxProgram>,
    slot_exprs: Vec<AdExpr>,
    input_slots: Vec<Vec<Option<usize>>>,
    #[allow(dead_code)]
    output_ccs: Vec<CCS>,
    output_nnz: Vec<usize>,
}

fn ensure_function_adexpr_arena_plan(
    function_id: FunctionId,
    arena: &mut AdExprArena,
    lower_memo: &mut HashMap<SX, AdExpr>,
    function_plan_cache: &mut HashMap<FunctionId, FunctionAdExprArenaPlan>,
) -> Result<()> {
    if let Entry::Vacant(entry) = function_plan_cache.entry(function_id) {
        let plan = function_program_plan(function_id)?;
        let arena_plan = FunctionAdExprArenaPlan {
            program: Arc::clone(&plan.program),
            slot_exprs: arena.lower_sx_collection_with_memo(&plan.program.slot_exprs, lower_memo),
            input_slots: plan.input_slots.clone(),
            output_ccs: plan.output_ccs.clone(),
            output_nnz: plan.output_nnz.clone(),
        };
        entry.insert(arena_plan);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct AdExpr(u32);

#[derive(Clone, Debug)]
enum AdExprKind {
    Constant(f64),
    Symbol {
        name: Arc<str>,
    },
    Unary {
        op: UnaryOp,
        arg: AdExpr,
    },
    Binary {
        op: BinaryOp,
        lhs: AdExpr,
        rhs: AdExpr,
    },
    Call {
        function_id: FunctionId,
        inputs: Vec<AdExprMatrix>,
        output_slot: Index,
        output_offset: Index,
    },
}

#[derive(Clone, Debug)]
struct AdExprNode {
    source: Option<SX>,
    kind: AdExprKind,
}

#[derive(Clone, Debug)]
struct AdExprMatrix {
    ccs: CCS,
    nonzeros: Vec<AdExpr>,
}

#[allow(dead_code)]
#[derive(Clone)]
struct AdExprSparseSeedMatrix {
    entries: Vec<(usize, AdExpr)>,
}

struct AdExprArena {
    context_id: u32,
    nodes: Vec<AdExprNode>,
    constant_cache: HashMap<u64, AdExpr>,
}

const HESSIAN_REIFY_SIMPLIFY_DEPTH: usize = 1;

#[allow(dead_code)]
impl AdExprMatrix {
    fn new(ccs: CCS, nonzeros: Vec<AdExpr>) -> Result<Self> {
        if ccs.nnz() != nonzeros.len() {
            return Err(SxError::Shape(format!(
                "CCS nnz {} does not match value nnz {}",
                ccs.nnz(),
                nonzeros.len()
            )));
        }
        Ok(Self { ccs, nonzeros })
    }

    fn nz(&self, offset: Index) -> AdExpr {
        self.nonzeros[offset]
    }
}

impl AdExprArena {
    fn new(context_id: u32) -> Self {
        Self {
            context_id,
            nodes: Vec::new(),
            constant_cache: HashMap::new(),
        }
    }

    fn push(&mut self, source: Option<SX>, kind: AdExprKind) -> AdExpr {
        let id = AdExpr(self.nodes.len() as u32);
        self.nodes.push(AdExprNode { source, kind });
        id
    }

    fn kind(&self, expr: AdExpr) -> &AdExprKind {
        &self.nodes[expr.0 as usize].kind
    }

    fn source(&self, expr: AdExpr) -> Option<SX> {
        self.nodes[expr.0 as usize].source
    }

    fn constant(&mut self, value: f64) -> AdExpr {
        let key = value.to_bits();
        if let Some(&existing) = self.constant_cache.get(&key) {
            return existing;
        }
        let expr = self.push(None, AdExprKind::Constant(value));
        self.constant_cache.insert(key, expr);
        expr
    }

    fn zero(&mut self) -> AdExpr {
        self.constant(0.0)
    }

    fn one(&mut self) -> AdExpr {
        self.constant(1.0)
    }

    fn neg_one(&mut self) -> AdExpr {
        self.constant(-1.0)
    }

    fn constant_value(&self, expr: AdExpr) -> Option<f64> {
        match self.kind(expr) {
            AdExprKind::Constant(value) => Some(*value),
            AdExprKind::Symbol { .. }
            | AdExprKind::Unary { .. }
            | AdExprKind::Binary { .. }
            | AdExprKind::Call { .. } => None,
        }
    }

    fn is_zero(&self, expr: AdExpr) -> bool {
        matches!(self.kind(expr), AdExprKind::Constant(value) if *value == 0.0)
    }

    fn is_one(&self, expr: AdExpr) -> bool {
        matches!(self.kind(expr), AdExprKind::Constant(value) if *value == 1.0)
    }

    fn canonical_pair(lhs: AdExpr, rhs: AdExpr) -> (AdExpr, AdExpr) {
        if lhs <= rhs { (lhs, rhs) } else { (rhs, lhs) }
    }

    fn lower_sx_expr_with_memo(&mut self, expr: SX, memo: &mut HashMap<SX, AdExpr>) -> AdExpr {
        if let Some(&existing) = memo.get(&expr) {
            return existing;
        }
        let lowered = match expr.inspect() {
            NodeView::Constant(value) => self.push(Some(expr), AdExprKind::Constant(value)),
            NodeView::Symbol { name, serial: _ } => self.push(
                Some(expr),
                AdExprKind::Symbol {
                    name: Arc::<str>::from(name),
                },
            ),
            NodeView::Unary { op, arg } => {
                let lowered_arg = self.lower_sx_expr_with_memo(arg, memo);
                self.push(
                    Some(expr),
                    AdExprKind::Unary {
                        op,
                        arg: lowered_arg,
                    },
                )
            }
            NodeView::Binary { op, lhs, rhs } => {
                let lowered_lhs = self.lower_sx_expr_with_memo(lhs, memo);
                let lowered_rhs = self.lower_sx_expr_with_memo(rhs, memo);
                self.push(
                    Some(expr),
                    AdExprKind::Binary {
                        op,
                        lhs: lowered_lhs,
                        rhs: lowered_rhs,
                    },
                )
            }
            NodeView::Call {
                function_id,
                inputs,
                output_slot,
                output_offset,
                ..
            } => {
                let lowered_inputs = inputs
                    .into_iter()
                    .map(|input| self.lower_sx_matrix_with_memo(&input, memo))
                    .collect();
                self.push(
                    Some(expr),
                    AdExprKind::Call {
                        function_id,
                        inputs: lowered_inputs,
                        output_slot,
                        output_offset,
                    },
                )
            }
        };
        memo.insert(expr, lowered);
        lowered
    }

    fn lower_sx_matrix_with_memo(
        &mut self,
        matrix: &SXMatrix,
        memo: &mut HashMap<SX, AdExpr>,
    ) -> AdExprMatrix {
        AdExprMatrix {
            ccs: matrix.ccs().clone(),
            nonzeros: matrix
                .nonzeros()
                .iter()
                .copied()
                .map(|expr| self.lower_sx_expr_with_memo(expr, memo))
                .collect(),
        }
    }

    fn lower_sx_collection_with_memo(
        &mut self,
        values: &[SX],
        memo: &mut HashMap<SX, AdExpr>,
    ) -> Vec<AdExpr> {
        values
            .iter()
            .copied()
            .map(|value| self.lower_sx_expr_with_memo(value, memo))
            .collect()
    }

    fn to_sx_with_simplify_depth(
        &self,
        expr: AdExpr,
        memo: &mut HashMap<AdExpr, SX>,
        simplify_depth: Option<usize>,
    ) -> Result<SX> {
        if let Some(&existing) = memo.get(&expr) {
            return Ok(existing);
        }

        let mut stack = vec![(expr, false)];
        while let Some((current, expanded)) = stack.pop() {
            if memo.contains_key(&current) {
                continue;
            }
            if let Some(source) = self.source(current) {
                memo.insert(current, source);
                continue;
            }

            let kind = self.kind(current).clone();
            if !expanded {
                stack.push((current, true));
                match &kind {
                    AdExprKind::Constant(_) | AdExprKind::Symbol { .. } => {}
                    AdExprKind::Unary { arg, .. } => {
                        if !memo.contains_key(arg) {
                            stack.push((*arg, false));
                        }
                    }
                    AdExprKind::Binary { lhs, rhs, .. } => {
                        if !memo.contains_key(rhs) {
                            stack.push((*rhs, false));
                        }
                        if !memo.contains_key(lhs) {
                            stack.push((*lhs, false));
                        }
                    }
                    AdExprKind::Call { inputs, .. } => {
                        for input in inputs.iter().rev() {
                            for input_expr in input.nonzeros.iter().rev().copied() {
                                if !memo.contains_key(&input_expr) {
                                    stack.push((input_expr, false));
                                }
                            }
                        }
                    }
                }
                continue;
            }

            let lowered = match kind {
                AdExprKind::Constant(value) => {
                    with_sx_context_id(self.context_id, || SX::from(value))
                }
                AdExprKind::Symbol { name } => {
                    with_sx_context_id(self.context_id, || SX::sym(name.as_ref().to_string()))
                }
                AdExprKind::Unary { op, arg } => apply_adexpr_sx_unary_with_budget(
                    self.context_id,
                    op,
                    *memo
                        .get(&arg)
                        .expect("postorder reification should visit unary arg first"),
                    simplify_depth,
                ),
                AdExprKind::Binary { op, lhs, rhs } => apply_adexpr_sx_binary_with_budget(
                    self.context_id,
                    op,
                    *memo
                        .get(&lhs)
                        .expect("postorder reification should visit binary lhs first"),
                    *memo
                        .get(&rhs)
                        .expect("postorder reification should visit binary rhs first"),
                    simplify_depth,
                ),
                AdExprKind::Call {
                    function_id,
                    inputs,
                    output_slot,
                    output_offset,
                } => {
                    let function = function_by_id(function_id).ok_or_else(|| {
                        SxError::Graph(format!("unknown function id {}", function_id))
                    })?;
                    let inputs = inputs
                        .iter()
                        .map(|input| {
                            self.to_sx_matrix_with_simplify_depth(input, memo, simplify_depth)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let outputs = function.call(&inputs)?;
                    outputs[output_slot].nz(output_offset)
                }
            };
            memo.insert(current, lowered);
        }

        memo.get(&expr)
            .copied()
            .ok_or_else(|| SxError::Graph("failed to reify AdExpr to SX".to_string()))
    }

    fn to_sx_matrix_with_simplify_depth(
        &self,
        matrix: &AdExprMatrix,
        memo: &mut HashMap<AdExpr, SX>,
        simplify_depth: Option<usize>,
    ) -> Result<SXMatrix> {
        SXMatrix::new(
            matrix.ccs.clone(),
            matrix
                .nonzeros
                .iter()
                .copied()
                .map(|expr| self.to_sx_with_simplify_depth(expr, memo, simplify_depth))
                .collect::<Result<Vec<_>>>()?,
        )
    }

    fn unary(&mut self, op: UnaryOp, arg: AdExpr) -> AdExpr {
        if let Some(value) = self.constant_value(arg) {
            return self.constant(op.apply_constant(value));
        }
        match op {
            UnaryOp::Abs => {
                if self.is_zero(arg) {
                    return self.zero();
                }
                if matches!(
                    self.kind(arg),
                    AdExprKind::Unary {
                        op: UnaryOp::Abs,
                        ..
                    }
                ) {
                    return arg;
                }
            }
            UnaryOp::Sign => {
                if self.is_zero(arg) {
                    return self.zero();
                }
            }
            UnaryOp::Sqrt => {
                if self.is_zero(arg) {
                    return self.zero();
                }
                if self.is_one(arg) {
                    return self.one();
                }
            }
            UnaryOp::Exp | UnaryOp::Cos | UnaryOp::Cosh => {
                if self.is_zero(arg) {
                    return self.one();
                }
            }
            UnaryOp::Log => {
                if self.is_one(arg) {
                    return self.zero();
                }
            }
            UnaryOp::Floor
            | UnaryOp::Ceil
            | UnaryOp::Round
            | UnaryOp::Trunc
            | UnaryOp::Sin
            | UnaryOp::Tan
            | UnaryOp::Asin
            | UnaryOp::Atan
            | UnaryOp::Sinh
            | UnaryOp::Tanh
            | UnaryOp::Asinh => {
                if self.is_zero(arg) {
                    return self.zero();
                }
            }
            UnaryOp::Acos | UnaryOp::Acosh | UnaryOp::Atanh => {}
        }
        self.push(None, AdExprKind::Unary { op, arg })
    }

    fn binary_ad(&mut self, op: BinaryOp, lhs: AdExpr, rhs: AdExpr) -> AdExpr {
        if let (Some(a), Some(b)) = (self.constant_value(lhs), self.constant_value(rhs)) {
            return self.constant(op.apply_constant(a, b));
        }
        match op {
            BinaryOp::Add => {
                if self.is_zero(lhs) {
                    return rhs;
                }
                if self.is_zero(rhs) {
                    return lhs;
                }
                let (lhs, rhs) = Self::canonical_pair(lhs, rhs);
                self.push(None, AdExprKind::Binary { op, lhs, rhs })
            }
            BinaryOp::Sub => {
                if self.is_zero(rhs) {
                    lhs
                } else if self.is_zero(lhs) {
                    let neg_one = self.neg_one();
                    self.binary_ad(BinaryOp::Mul, neg_one, rhs)
                } else {
                    self.push(None, AdExprKind::Binary { op, lhs, rhs })
                }
            }
            BinaryOp::Mul => {
                if self.is_zero(lhs) || self.is_zero(rhs) {
                    self.zero()
                } else if self.is_one(lhs) {
                    rhs
                } else if self.is_one(rhs) {
                    lhs
                } else {
                    let (lhs, rhs) = Self::canonical_pair(lhs, rhs);
                    self.push(None, AdExprKind::Binary { op, lhs, rhs })
                }
            }
            BinaryOp::Div => {
                if self.is_zero(lhs) {
                    self.zero()
                } else if self.is_one(rhs) {
                    lhs
                } else {
                    self.push(None, AdExprKind::Binary { op, lhs, rhs })
                }
            }
            BinaryOp::Pow => {
                if self.is_zero(rhs) {
                    self.one()
                } else if self.is_one(rhs) {
                    lhs
                } else if self.is_one(lhs) {
                    self.one()
                } else {
                    self.push(None, AdExprKind::Binary { op, lhs, rhs })
                }
            }
            BinaryOp::Atan2 | BinaryOp::Hypot | BinaryOp::Mod | BinaryOp::Copysign => {
                if op.is_commutative() {
                    let (lhs, rhs) = Self::canonical_pair(lhs, rhs);
                    self.push(None, AdExprKind::Binary { op, lhs, rhs })
                } else {
                    self.push(None, AdExprKind::Binary { op, lhs, rhs })
                }
            }
        }
    }

    fn neg(&mut self, expr: AdExpr) -> AdExpr {
        let neg_one = self.neg_one();
        self.binary_ad(BinaryOp::Mul, neg_one, expr)
    }

    #[allow(dead_code)]
    fn weighted_sum2(
        &mut self,
        coeff_a: AdExpr,
        value_a: AdExpr,
        coeff_b: AdExpr,
        value_b: AdExpr,
    ) -> AdExpr {
        let left = if self.is_zero(coeff_a) || self.is_zero(value_a) {
            self.zero()
        } else {
            self.binary_ad(BinaryOp::Mul, coeff_a, value_a)
        };
        let right = if self.is_zero(coeff_b) || self.is_zero(value_b) {
            self.zero()
        } else {
            self.binary_ad(BinaryOp::Mul, coeff_b, value_b)
        };
        self.binary_ad(BinaryOp::Add, left, right)
    }

    fn unary_derivative(&mut self, op: UnaryOp, arg: AdExpr) -> AdExpr {
        match op {
            UnaryOp::Abs => self.unary(UnaryOp::Sign, arg),
            UnaryOp::Sign | UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Trunc => {
                self.zero()
            }
            UnaryOp::Sqrt => {
                let half = self.constant(0.5);
                let sqrt = self.unary(UnaryOp::Sqrt, arg);
                self.binary_ad(BinaryOp::Div, half, sqrt)
            }
            UnaryOp::Exp => self.unary(UnaryOp::Exp, arg),
            UnaryOp::Log => {
                let one = self.one();
                self.binary_ad(BinaryOp::Div, one, arg)
            }
            UnaryOp::Sin => self.unary(UnaryOp::Cos, arg),
            UnaryOp::Cos => {
                let sin = self.unary(UnaryOp::Sin, arg);
                self.neg(sin)
            }
            UnaryOp::Tan => {
                let cos = self.unary(UnaryOp::Cos, arg);
                let cos_sq = self.binary_ad(BinaryOp::Mul, cos, cos);
                let one = self.one();
                self.binary_ad(BinaryOp::Div, one, cos_sq)
            }
            UnaryOp::Asin => {
                let one = self.one();
                let arg_sq = self.binary_ad(BinaryOp::Mul, arg, arg);
                let radicand = self.binary_ad(BinaryOp::Sub, one, arg_sq);
                let sqrt = self.unary(UnaryOp::Sqrt, radicand);
                let one = self.one();
                self.binary_ad(BinaryOp::Div, one, sqrt)
            }
            UnaryOp::Acos => {
                let one = self.one();
                let arg_sq = self.binary_ad(BinaryOp::Mul, arg, arg);
                let radicand = self.binary_ad(BinaryOp::Sub, one, arg_sq);
                let sqrt = self.unary(UnaryOp::Sqrt, radicand);
                let neg_one = self.neg_one();
                self.binary_ad(BinaryOp::Div, neg_one, sqrt)
            }
            UnaryOp::Atan => {
                let one = self.one();
                let arg_sq = self.binary_ad(BinaryOp::Mul, arg, arg);
                let denom = self.binary_ad(BinaryOp::Add, one, arg_sq);
                let one = self.one();
                self.binary_ad(BinaryOp::Div, one, denom)
            }
            UnaryOp::Sinh => self.unary(UnaryOp::Cosh, arg),
            UnaryOp::Cosh => self.unary(UnaryOp::Sinh, arg),
            UnaryOp::Tanh => {
                let cosh = self.unary(UnaryOp::Cosh, arg);
                let cosh_sq = self.binary_ad(BinaryOp::Mul, cosh, cosh);
                let one = self.one();
                self.binary_ad(BinaryOp::Div, one, cosh_sq)
            }
            UnaryOp::Asinh => {
                let one = self.one();
                let arg_sq = self.binary_ad(BinaryOp::Mul, arg, arg);
                let radicand = self.binary_ad(BinaryOp::Add, arg_sq, one);
                let denom = self.unary(UnaryOp::Sqrt, radicand);
                let one = self.one();
                self.binary_ad(BinaryOp::Div, one, denom)
            }
            UnaryOp::Acosh => {
                let one = self.one();
                let left_arg = self.binary_ad(BinaryOp::Sub, arg, one);
                let left = self.unary(UnaryOp::Sqrt, left_arg);
                let one = self.one();
                let right_arg = self.binary_ad(BinaryOp::Add, arg, one);
                let right = self.unary(UnaryOp::Sqrt, right_arg);
                let denom = self.binary_ad(BinaryOp::Mul, left, right);
                let one = self.one();
                self.binary_ad(BinaryOp::Div, one, denom)
            }
            UnaryOp::Atanh => {
                let one = self.one();
                let arg_sq = self.binary_ad(BinaryOp::Mul, arg, arg);
                let denom = self.binary_ad(BinaryOp::Sub, one, arg_sq);
                let one = self.one();
                self.binary_ad(BinaryOp::Div, one, denom)
            }
        }
    }

    fn binary_partials(&mut self, op: BinaryOp, lhs: AdExpr, rhs: AdExpr) -> (AdExpr, AdExpr) {
        match op {
            BinaryOp::Add => (self.one(), self.one()),
            BinaryOp::Sub => (self.one(), self.neg_one()),
            BinaryOp::Mul => (rhs, lhs),
            BinaryOp::Div => {
                let one = self.one();
                let lhs_partial = self.binary_ad(BinaryOp::Div, one, rhs);
                let neg_one = self.neg_one();
                let numer = self.binary_ad(BinaryOp::Mul, neg_one, lhs);
                let rhs_sq = self.binary_ad(BinaryOp::Mul, rhs, rhs);
                let rhs_partial = self.binary_ad(BinaryOp::Div, numer, rhs_sq);
                (lhs_partial, rhs_partial)
            }
            BinaryOp::Pow => {
                let pow = self.binary_ad(BinaryOp::Pow, lhs, rhs);
                let one = self.one();
                let rhs_minus_one = self.binary_ad(BinaryOp::Sub, rhs, one);
                let pow_term = self.binary_ad(BinaryOp::Pow, lhs, rhs_minus_one);
                let lhs_partial = self.binary_ad(BinaryOp::Mul, rhs, pow_term);
                let log_lhs = self.unary(UnaryOp::Log, lhs);
                let rhs_partial = self.binary_ad(BinaryOp::Mul, pow, log_lhs);
                (lhs_partial, rhs_partial)
            }
            BinaryOp::Atan2 => {
                let lhs_sq = self.binary_ad(BinaryOp::Mul, lhs, lhs);
                let rhs_sq = self.binary_ad(BinaryOp::Mul, rhs, rhs);
                let denom = self.binary_ad(BinaryOp::Add, lhs_sq, rhs_sq);
                let neg_one = self.neg_one();
                let neg_lhs = self.binary_ad(BinaryOp::Mul, neg_one, lhs);
                (
                    self.binary_ad(BinaryOp::Div, rhs, denom),
                    self.binary_ad(BinaryOp::Div, neg_lhs, denom),
                )
            }
            BinaryOp::Hypot => {
                let hypot = self.binary_ad(BinaryOp::Hypot, lhs, rhs);
                (
                    self.binary_ad(BinaryOp::Div, lhs, hypot),
                    self.binary_ad(BinaryOp::Div, rhs, hypot),
                )
            }
            BinaryOp::Mod => {
                let div = self.binary_ad(BinaryOp::Div, lhs, rhs);
                let trunc = self.unary(UnaryOp::Trunc, div);
                let neg_one = self.neg_one();
                (self.one(), self.binary_ad(BinaryOp::Mul, neg_one, trunc))
            }
            BinaryOp::Copysign => {
                let rhs_sign = self.unary(UnaryOp::Sign, rhs);
                let rhs_abs = self.unary(UnaryOp::Abs, rhs_sign);
                let one = self.one();
                let one_minus_abs = self.binary_ad(BinaryOp::Sub, one, rhs_abs);
                let sign_term = self.binary_ad(BinaryOp::Add, rhs_sign, one_minus_abs);
                let lhs_sign = self.unary(UnaryOp::Sign, lhs);
                (
                    self.binary_ad(BinaryOp::Mul, lhs_sign, sign_term),
                    self.zero(),
                )
            }
        }
    }

    fn unary_second_directional(
        &mut self,
        op: UnaryOp,
        arg: AdExpr,
        arg_tangent: AdExpr,
    ) -> AdExpr {
        if self.is_zero(arg_tangent) {
            return self.zero();
        }
        match op {
            UnaryOp::Abs
            | UnaryOp::Sign
            | UnaryOp::Floor
            | UnaryOp::Ceil
            | UnaryOp::Round
            | UnaryOp::Trunc => self.zero(),
            UnaryOp::Sqrt => {
                let neg_quarter = self.constant(-0.25);
                let sqrt_arg = self.unary(UnaryOp::Sqrt, arg);
                let denom = self.binary_ad(BinaryOp::Mul, arg, sqrt_arg);
                let scaled = self.binary_ad(BinaryOp::Mul, neg_quarter, arg_tangent);
                self.binary_ad(BinaryOp::Div, scaled, denom)
            }
            UnaryOp::Exp => {
                let exp_arg = self.unary(UnaryOp::Exp, arg);
                self.binary_ad(BinaryOp::Mul, exp_arg, arg_tangent)
            }
            UnaryOp::Log => {
                let neg_one = self.neg_one();
                let denom = self.binary_ad(BinaryOp::Mul, arg, arg);
                let numer = self.binary_ad(BinaryOp::Mul, neg_one, arg_tangent);
                self.binary_ad(BinaryOp::Div, numer, denom)
            }
            UnaryOp::Sin => {
                let sin = self.unary(UnaryOp::Sin, arg);
                let neg_sin = self.neg(sin);
                self.binary_ad(BinaryOp::Mul, neg_sin, arg_tangent)
            }
            UnaryOp::Cos => {
                let cos = self.unary(UnaryOp::Cos, arg);
                let neg_cos = self.neg(cos);
                self.binary_ad(BinaryOp::Mul, neg_cos, arg_tangent)
            }
            UnaryOp::Tan => {
                let two = self.constant(2.0);
                let tan_arg = self.unary(UnaryOp::Tan, arg);
                let cos_arg = self.unary(UnaryOp::Cos, arg);
                let cos_sq = self.binary_ad(BinaryOp::Mul, cos_arg, cos_arg);
                let factor = self.binary_ad(BinaryOp::Mul, two, tan_arg);
                let factor = self.binary_ad(BinaryOp::Div, factor, cos_sq);
                self.binary_ad(BinaryOp::Mul, factor, arg_tangent)
            }
            UnaryOp::Asin => {
                let one = self.one();
                let arg_sq = self.binary_ad(BinaryOp::Mul, arg, arg);
                let radicand = self.binary_ad(BinaryOp::Sub, one, arg_sq);
                let sqrt = self.unary(UnaryOp::Sqrt, radicand);
                let denom = self.binary_ad(BinaryOp::Mul, radicand, sqrt);
                let numer = self.binary_ad(BinaryOp::Mul, arg, arg_tangent);
                self.binary_ad(BinaryOp::Div, numer, denom)
            }
            UnaryOp::Acos => {
                let asin_second = self.unary_second_directional(UnaryOp::Asin, arg, arg_tangent);
                self.neg(asin_second)
            }
            UnaryOp::Atan => {
                let neg_two = self.constant(-2.0);
                let one = self.one();
                let arg_sq = self.binary_ad(BinaryOp::Mul, arg, arg);
                let denom_base = self.binary_ad(BinaryOp::Add, one, arg_sq);
                let denom = self.binary_ad(BinaryOp::Mul, denom_base, denom_base);
                let numer = self.binary_ad(BinaryOp::Mul, arg, arg_tangent);
                let numer = self.binary_ad(BinaryOp::Mul, neg_two, numer);
                self.binary_ad(BinaryOp::Div, numer, denom)
            }
            UnaryOp::Sinh => {
                let sinh = self.unary(UnaryOp::Sinh, arg);
                self.binary_ad(BinaryOp::Mul, sinh, arg_tangent)
            }
            UnaryOp::Cosh => {
                let cosh = self.unary(UnaryOp::Cosh, arg);
                self.binary_ad(BinaryOp::Mul, cosh, arg_tangent)
            }
            UnaryOp::Tanh => {
                let neg_two = self.constant(-2.0);
                let tanh_arg = self.unary(UnaryOp::Tanh, arg);
                let cosh_arg = self.unary(UnaryOp::Cosh, arg);
                let cosh_sq = self.binary_ad(BinaryOp::Mul, cosh_arg, cosh_arg);
                let factor = self.binary_ad(BinaryOp::Mul, neg_two, tanh_arg);
                let factor = self.binary_ad(BinaryOp::Div, factor, cosh_sq);
                self.binary_ad(BinaryOp::Mul, factor, arg_tangent)
            }
            UnaryOp::Asinh => {
                let one = self.one();
                let arg_sq = self.binary_ad(BinaryOp::Mul, arg, arg);
                let radicand = self.binary_ad(BinaryOp::Add, arg_sq, one);
                let sqrt = self.unary(UnaryOp::Sqrt, radicand);
                let denom = self.binary_ad(BinaryOp::Mul, radicand, sqrt);
                let mul = self.binary_ad(BinaryOp::Mul, arg, arg_tangent);
                let numer = self.neg(mul);
                self.binary_ad(BinaryOp::Div, numer, denom)
            }
            UnaryOp::Acosh => {
                let one = self.one();
                let arg_sq = self.binary_ad(BinaryOp::Mul, arg, arg);
                let denom_base = self.binary_ad(BinaryOp::Sub, arg_sq, one);
                let sqrt = self.unary(UnaryOp::Sqrt, denom_base);
                let denom = self.binary_ad(BinaryOp::Mul, denom_base, sqrt);
                let mul = self.binary_ad(BinaryOp::Mul, arg, arg_tangent);
                let numer = self.neg(mul);
                self.binary_ad(BinaryOp::Div, numer, denom)
            }
            UnaryOp::Atanh => {
                let two = self.constant(2.0);
                let one = self.one();
                let arg_sq = self.binary_ad(BinaryOp::Mul, arg, arg);
                let denom_base = self.binary_ad(BinaryOp::Sub, one, arg_sq);
                let denom = self.binary_ad(BinaryOp::Mul, denom_base, denom_base);
                let numer = self.binary_ad(BinaryOp::Mul, arg, arg_tangent);
                let numer = self.binary_ad(BinaryOp::Mul, two, numer);
                self.binary_ad(BinaryOp::Div, numer, denom)
            }
        }
    }

    #[allow(dead_code)]
    fn binary_partial_directionals(
        &mut self,
        op: BinaryOp,
        lhs: AdExpr,
        rhs: AdExpr,
        lhs_tangent: AdExpr,
        rhs_tangent: AdExpr,
    ) -> (AdExpr, AdExpr) {
        match op {
            BinaryOp::Add | BinaryOp::Sub => {
                let zero = self.zero();
                (zero, zero)
            }
            BinaryOp::Mul => (rhs_tangent, lhs_tangent),
            BinaryOp::Div => {
                let rhs_sq = self.binary_ad(BinaryOp::Mul, rhs, rhs);
                let neg_rhs_tangent = self.neg(rhs_tangent);
                let lhs_dir = self.binary_ad(BinaryOp::Div, neg_rhs_tangent, rhs_sq);
                let lhs_over_rhs_sq = self.binary_ad(BinaryOp::Div, lhs_tangent, rhs_sq);
                let left = self.neg(lhs_over_rhs_sq);
                let two = self.constant(2.0);
                let rhs_cu = self.binary_ad(BinaryOp::Mul, rhs_sq, rhs);
                let numer = self.binary_ad(BinaryOp::Mul, lhs, rhs_tangent);
                let numer = self.binary_ad(BinaryOp::Mul, two, numer);
                let right = self.binary_ad(BinaryOp::Div, numer, rhs_cu);
                (lhs_dir, self.binary_ad(BinaryOp::Add, left, right))
            }
            BinaryOp::Pow => {
                let pow = self.binary_ad(BinaryOp::Pow, lhs, rhs);
                let log_lhs = self.unary(UnaryOp::Log, lhs);
                let one = self.one();
                let lhs_inv = self.binary_ad(BinaryOp::Div, one, lhs);
                let rhs_lhs = self.binary_ad(BinaryOp::Mul, rhs, lhs_tangent);
                let rhs_lhs = self.binary_ad(BinaryOp::Mul, rhs_lhs, lhs_inv);
                let rhs_log = self.binary_ad(BinaryOp::Mul, rhs_tangent, log_lhs);
                let pow_factor = self.binary_ad(BinaryOp::Add, rhs_log, rhs_lhs);
                let pow_t = self.binary_ad(BinaryOp::Mul, pow, pow_factor);

                let rhs_lhs_inv = self.binary_ad(BinaryOp::Mul, rhs, lhs_inv);
                let lhs_term1 = self.binary_ad(BinaryOp::Mul, pow_t, rhs_lhs_inv);
                let rhs_tangent_lhs_inv = self.binary_ad(BinaryOp::Mul, rhs_tangent, lhs_inv);
                let lhs_term2 = self.binary_ad(BinaryOp::Mul, pow, rhs_tangent_lhs_inv);
                let lhs_inv_sq = self.binary_ad(BinaryOp::Mul, lhs_inv, lhs_inv);
                let rhs_lhs_tangent = self.binary_ad(BinaryOp::Mul, rhs, lhs_tangent);
                let rhs_lhs_tangent_inv_sq =
                    self.binary_ad(BinaryOp::Mul, rhs_lhs_tangent, lhs_inv_sq);
                let lhs_term3_inner = self.binary_ad(BinaryOp::Mul, pow, rhs_lhs_tangent_inv_sq);
                let lhs_terms = self.binary_ad(BinaryOp::Add, lhs_term1, lhs_term2);
                let neg_lhs_term3_inner = self.neg(lhs_term3_inner);
                let lhs_dir = self.binary_ad(BinaryOp::Add, lhs_terms, neg_lhs_term3_inner);

                let rhs_term1 = self.binary_ad(BinaryOp::Mul, pow_t, log_lhs);
                let one_over_lhs = self.binary_ad(BinaryOp::Div, one, lhs);
                let lhs_tangent_lhs_inv = self.binary_ad(BinaryOp::Mul, lhs_tangent, one_over_lhs);
                let rhs_term2 = self.binary_ad(BinaryOp::Mul, pow, lhs_tangent_lhs_inv);
                let rhs_dir = self.binary_ad(BinaryOp::Add, rhs_term1, rhs_term2);
                (lhs_dir, rhs_dir)
            }
            BinaryOp::Atan2 => {
                let two = self.constant(2.0);
                let lhs_sq = self.binary_ad(BinaryOp::Mul, lhs, lhs);
                let rhs_sq = self.binary_ad(BinaryOp::Mul, rhs, rhs);
                let denom = self.binary_ad(BinaryOp::Add, lhs_sq, rhs_sq);
                let denom_sq = self.binary_ad(BinaryOp::Mul, denom, denom);
                let two_lhs_lhs_tangent = self.binary_ad(BinaryOp::Mul, lhs, lhs_tangent);
                let two_lhs_lhs_tangent = self.binary_ad(BinaryOp::Mul, two, two_lhs_lhs_tangent);
                let two_rhs_rhs_tangent = self.binary_ad(BinaryOp::Mul, rhs, rhs_tangent);
                let two_rhs_rhs_tangent = self.binary_ad(BinaryOp::Mul, two, two_rhs_rhs_tangent);
                let denom_t =
                    self.binary_ad(BinaryOp::Add, two_lhs_lhs_tangent, two_rhs_rhs_tangent);
                let left = self.binary_ad(BinaryOp::Div, rhs_tangent, denom);
                let rhs_denom_t = self.binary_ad(BinaryOp::Mul, rhs, denom_t);
                let right = self.binary_ad(BinaryOp::Div, rhs_denom_t, denom_sq);
                let lhs_dir = self.binary_ad(BinaryOp::Sub, left, right);
                let lhs_tangent_over_denom = self.binary_ad(BinaryOp::Div, lhs_tangent, denom);
                let left = self.neg(lhs_tangent_over_denom);
                let lhs_denom_t = self.binary_ad(BinaryOp::Mul, lhs, denom_t);
                let right = self.binary_ad(BinaryOp::Div, lhs_denom_t, denom_sq);
                let rhs_dir = self.binary_ad(BinaryOp::Add, left, right);
                (lhs_dir, rhs_dir)
            }
            BinaryOp::Hypot => {
                let hypot = self.binary_ad(BinaryOp::Hypot, lhs, rhs);
                let lhs_lhs_tangent = self.binary_ad(BinaryOp::Mul, lhs, lhs_tangent);
                let rhs_rhs_tangent = self.binary_ad(BinaryOp::Mul, rhs, rhs_tangent);
                let numer_t = self.binary_ad(BinaryOp::Add, lhs_lhs_tangent, rhs_rhs_tangent);
                let hypot_t = self.binary_ad(BinaryOp::Div, numer_t, hypot);
                let hypot_sq = self.binary_ad(BinaryOp::Mul, hypot, hypot);
                let left = self.binary_ad(BinaryOp::Div, lhs_tangent, hypot);
                let lhs_hypot_t = self.binary_ad(BinaryOp::Mul, lhs, hypot_t);
                let right = self.binary_ad(BinaryOp::Div, lhs_hypot_t, hypot_sq);
                let lhs_dir = self.binary_ad(BinaryOp::Sub, left, right);
                let left = self.binary_ad(BinaryOp::Div, rhs_tangent, hypot);
                let rhs_hypot_t = self.binary_ad(BinaryOp::Mul, rhs, hypot_t);
                let right = self.binary_ad(BinaryOp::Div, rhs_hypot_t, hypot_sq);
                let rhs_dir = self.binary_ad(BinaryOp::Sub, left, right);
                (lhs_dir, rhs_dir)
            }
            BinaryOp::Mod | BinaryOp::Copysign => {
                let zero = self.zero();
                (zero, zero)
            }
        }
    }

    #[allow(dead_code)]
    fn binary_second_partial_coefficients(
        &mut self,
        op: BinaryOp,
        lhs: AdExpr,
        rhs: AdExpr,
    ) -> (AdExpr, AdExpr, AdExpr, AdExpr) {
        match op {
            BinaryOp::Add | BinaryOp::Sub => {
                let zero = self.zero();
                (zero, zero, zero, zero)
            }
            BinaryOp::Mul => {
                let zero = self.zero();
                let one = self.one();
                (zero, one, one, zero)
            }
            BinaryOp::Div => {
                let zero = self.zero();
                let rhs_sq = self.binary_ad(BinaryOp::Mul, rhs, rhs);
                let minus_inv_rhs_sq = {
                    let neg_one = self.neg_one();
                    self.binary_ad(BinaryOp::Div, neg_one, rhs_sq)
                };
                let two = self.constant(2.0);
                let rhs_cu = self.binary_ad(BinaryOp::Mul, rhs_sq, rhs);
                let numer = self.binary_ad(BinaryOp::Mul, two, lhs);
                let rhs_from_rhs = self.binary_ad(BinaryOp::Div, numer, rhs_cu);
                (zero, minus_inv_rhs_sq, minus_inv_rhs_sq, rhs_from_rhs)
            }
            BinaryOp::Pow => {
                let pow = self.binary_ad(BinaryOp::Pow, lhs, rhs);
                let one = self.one();
                let lhs_inv = self.binary_ad(BinaryOp::Div, one, lhs);
                let lhs_inv_sq = self.binary_ad(BinaryOp::Mul, lhs_inv, lhs_inv);
                let rhs_minus_one = self.binary_ad(BinaryOp::Sub, rhs, one);
                let rhs_times_rhs_minus_one = self.binary_ad(BinaryOp::Mul, rhs, rhs_minus_one);
                let f_xx = {
                    let scaled = self.binary_ad(BinaryOp::Mul, pow, rhs_times_rhs_minus_one);
                    self.binary_ad(BinaryOp::Mul, scaled, lhs_inv_sq)
                };
                let log_lhs = self.unary(UnaryOp::Log, lhs);
                let rhs_log = self.binary_ad(BinaryOp::Mul, rhs, log_lhs);
                let one_plus_rhs_log = self.binary_ad(BinaryOp::Add, one, rhs_log);
                let common = {
                    let scaled = self.binary_ad(BinaryOp::Mul, pow, lhs_inv);
                    self.binary_ad(BinaryOp::Mul, scaled, one_plus_rhs_log)
                };
                let log_sq = self.binary_ad(BinaryOp::Mul, log_lhs, log_lhs);
                let f_yy = self.binary_ad(BinaryOp::Mul, pow, log_sq);
                (f_xx, common, common, f_yy)
            }
            BinaryOp::Atan2 => {
                let lhs_sq = self.binary_ad(BinaryOp::Mul, lhs, lhs);
                let rhs_sq = self.binary_ad(BinaryOp::Mul, rhs, rhs);
                let denom = self.binary_ad(BinaryOp::Add, lhs_sq, rhs_sq);
                let denom_sq = self.binary_ad(BinaryOp::Mul, denom, denom);
                let two = self.constant(2.0);
                let two_lhs_rhs = {
                    let prod = self.binary_ad(BinaryOp::Mul, lhs, rhs);
                    let scaled = self.binary_ad(BinaryOp::Mul, two, prod);
                    self.binary_ad(BinaryOp::Div, scaled, denom_sq)
                };
                let neg_two_lhs_rhs = self.neg(two_lhs_rhs);
                let lhs_sq_minus_rhs_sq = self.binary_ad(BinaryOp::Sub, lhs_sq, rhs_sq);
                let mixed = self.binary_ad(BinaryOp::Div, lhs_sq_minus_rhs_sq, denom_sq);
                (neg_two_lhs_rhs, mixed, mixed, self.neg(neg_two_lhs_rhs))
            }
            BinaryOp::Hypot => {
                let hypot = self.binary_ad(BinaryOp::Hypot, lhs, rhs);
                let hypot_sq = self.binary_ad(BinaryOp::Mul, hypot, hypot);
                let hypot_cu = self.binary_ad(BinaryOp::Mul, hypot_sq, hypot);
                let lhs_sq = self.binary_ad(BinaryOp::Mul, lhs, lhs);
                let rhs_sq = self.binary_ad(BinaryOp::Mul, rhs, rhs);
                let rhs_from_lhs = {
                    let prod = self.binary_ad(BinaryOp::Mul, lhs, rhs);
                    let numer = self.neg(prod);
                    self.binary_ad(BinaryOp::Div, numer, hypot_cu)
                };
                let lhs_from_rhs = rhs_from_lhs;
                let lhs_from_lhs = self.binary_ad(BinaryOp::Div, rhs_sq, hypot_cu);
                let rhs_from_rhs = self.binary_ad(BinaryOp::Div, lhs_sq, hypot_cu);
                (lhs_from_lhs, lhs_from_rhs, rhs_from_lhs, rhs_from_rhs)
            }
            BinaryOp::Mod | BinaryOp::Copysign => {
                let zero = self.zero();
                (zero, zero, zero, zero)
            }
        }
    }
}

#[allow(dead_code)]
fn apply_adexpr_sx_unary(op: UnaryOp, arg: SX) -> SX {
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

fn extract_linear_term_shallow(arena: &NodeArena, expr: SX) -> (f64, SX) {
    if let Some((coeff, factor)) = mul_constant_factor_with_arena(arena, expr) {
        (coeff, factor)
    } else {
        (1.0, expr)
    }
}

fn combine_like_terms_shallow(
    arena: &mut NodeArena,
    lhs: SX,
    rhs: SX,
    rhs_sign: f64,
) -> Option<SX> {
    let (lhs_coeff, lhs_term) = extract_linear_term_shallow(arena, lhs);
    let (rhs_coeff, rhs_term) = extract_linear_term_shallow(arena, rhs);
    if lhs_term != rhs_term {
        return None;
    }

    let coeff = lhs_coeff + rhs_sign * rhs_coeff;
    if coeff == 0.0 {
        return Some(append_constant_in_arena(arena, 0.0));
    }
    if coeff == 1.0 {
        return Some(lhs_term);
    }
    if coeff == -1.0 {
        return Some(neg_in_arena(arena, lhs_term));
    }

    let coeff_expr = append_constant_in_arena(arena, coeff);
    Some(binary_in_arena(arena, BinaryOp::Mul, coeff_expr, lhs_term))
}

fn simplify_mul_shallow(arena: &mut NodeArena, lhs: SX, rhs: SX) -> Option<SX> {
    if let Some((lhs_scale, lhs_factor)) = mul_constant_factor_with_arena(arena, lhs)
        && let Some((rhs_numerator, rhs_divisor)) = div_constant_numerator_with_arena(arena, rhs)
        && lhs_factor == rhs_divisor
    {
        return Some(append_constant_in_arena(arena, lhs_scale * rhs_numerator));
    }
    if let Some((rhs_scale, rhs_factor)) = mul_constant_factor_with_arena(arena, rhs)
        && let Some((lhs_numerator, lhs_divisor)) = div_constant_numerator_with_arena(arena, lhs)
        && rhs_factor == lhs_divisor
    {
        return Some(append_constant_in_arena(arena, rhs_scale * lhs_numerator));
    }
    if let Some((numerator, divisor)) = div_constant_numerator_with_arena(arena, rhs)
        && divisor == lhs
    {
        return Some(append_constant_in_arena(arena, numerator));
    }
    if let Some((numerator, divisor)) = div_constant_numerator_with_arena(arena, lhs)
        && divisor == rhs
    {
        return Some(append_constant_in_arena(arena, numerator));
    }
    if let Some(lhs_value) = constant_value_with_arena(arena, lhs)
        && let Some((numerator, denominator)) = div_constant_factor_with_arena(arena, rhs)
        && denominator != 0.0
    {
        let scaled = lhs_value / denominator;
        if scaled == 0.0 {
            return Some(append_constant_in_arena(arena, 0.0));
        }
        if scaled == 1.0 {
            return Some(numerator);
        }
        if scaled == -1.0 {
            return Some(neg_in_arena(arena, numerator));
        }
        let scaled = append_constant_in_arena(arena, scaled);
        return Some(binary_in_arena(arena, BinaryOp::Mul, scaled, numerator));
    }
    if let Some(rhs_value) = constant_value_with_arena(arena, rhs)
        && let Some((numerator, denominator)) = div_constant_factor_with_arena(arena, lhs)
        && denominator != 0.0
    {
        let scaled = rhs_value / denominator;
        if scaled == 0.0 {
            return Some(append_constant_in_arena(arena, 0.0));
        }
        if scaled == 1.0 {
            return Some(numerator);
        }
        if scaled == -1.0 {
            return Some(neg_in_arena(arena, numerator));
        }
        let scaled = append_constant_in_arena(arena, scaled);
        return Some(binary_in_arena(arena, BinaryOp::Mul, scaled, numerator));
    }
    if let Some(lhs_value) = constant_value_with_arena(arena, lhs)
        && let Some((rhs_value, factor)) = mul_constant_factor_with_arena(arena, rhs)
    {
        let scaled = append_constant_in_arena(arena, lhs_value * rhs_value);
        return Some(binary_in_arena(arena, BinaryOp::Mul, scaled, factor));
    }
    if let Some(rhs_value) = constant_value_with_arena(arena, rhs)
        && let Some((lhs_value, factor)) = mul_constant_factor_with_arena(arena, lhs)
    {
        let scaled = append_constant_in_arena(arena, lhs_value * rhs_value);
        return Some(binary_in_arena(arena, BinaryOp::Mul, scaled, factor));
    }
    None
}

fn simplify_div_shallow(arena: &mut NodeArena, lhs: SX, rhs: SX) -> Option<SX> {
    if lhs == rhs {
        return Some(append_constant_in_arena(arena, 1.0));
    }
    if let Some((lhs_value, lhs_factor)) = mul_constant_factor_with_arena(arena, lhs)
        && let Some((rhs_value, rhs_factor)) = mul_constant_factor_with_arena(arena, rhs)
        && rhs_value != 0.0
        && lhs_factor == rhs_factor
    {
        return Some(append_constant_in_arena(arena, lhs_value / rhs_value));
    }
    match arena.node_kind_ref(lhs).clone() {
        NodeKind::Binary {
            op: BinaryOp::Mul,
            lhs: mul_lhs,
            rhs: mul_rhs,
        } => {
            if mul_lhs == rhs {
                return Some(mul_rhs);
            }
            if mul_rhs == rhs {
                return Some(mul_lhs);
            }
        }
        NodeKind::Binary {
            op: BinaryOp::Div,
            lhs: div_lhs,
            rhs: div_rhs,
        } => {
            if div_lhs == rhs {
                let one = append_constant_in_arena(arena, 1.0);
                return Some(append_binary_in_arena(arena, BinaryOp::Div, one, div_rhs));
            }
        }
        NodeKind::Constant(_)
        | NodeKind::Symbol { .. }
        | NodeKind::Unary { .. }
        | NodeKind::Binary { .. }
        | NodeKind::Call { .. } => {}
    }

    if let Some(lhs_value) = constant_value_with_arena(arena, lhs)
        && let Some((rhs_value, factor)) = mul_constant_factor_with_arena(arena, rhs)
        && rhs_value != 0.0
    {
        let scaled = append_constant_in_arena(arena, lhs_value / rhs_value);
        return Some(binary_in_arena(arena, BinaryOp::Div, scaled, factor));
    }
    if let Some(rhs_value) = constant_value_with_arena(arena, rhs)
        && rhs_value != 0.0
        && let Some((lhs_value, factor)) = mul_constant_factor_with_arena(arena, lhs)
    {
        let scaled = lhs_value / rhs_value;
        if scaled == 0.0 {
            return Some(append_constant_in_arena(arena, 0.0));
        }
        if scaled == 1.0 {
            return Some(factor);
        }
        if scaled == -1.0 {
            return Some(neg_in_arena(arena, factor));
        }
        let scaled = append_constant_in_arena(arena, scaled);
        return Some(binary_in_arena(arena, BinaryOp::Mul, scaled, factor));
    }
    if let Some((rhs_value, factor)) = mul_constant_factor_with_arena(arena, rhs)
        && rhs_value != 0.0
        && factor == lhs
    {
        return Some(append_constant_in_arena(arena, 1.0 / rhs_value));
    }
    if let Some(rhs_value) = constant_value_with_arena(arena, rhs)
        && let Some((numerator, denominator)) = div_constant_factor_with_arena(arena, lhs)
    {
        let scaled_denominator = denominator * rhs_value;
        if scaled_denominator == 1.0 {
            return Some(numerator);
        }
        let scaled_denominator = append_constant_in_arena(arena, scaled_denominator);
        return Some(binary_in_arena(
            arena,
            BinaryOp::Div,
            numerator,
            scaled_denominator,
        ));
    }

    None
}

fn binary_in_arena_with_simplify_depth(
    arena: &mut NodeArena,
    op: BinaryOp,
    lhs: SX,
    rhs: SX,
    _simplify_depth: Option<usize>,
) -> SX {
    use NodeKind as N;

    let lhs_kind = arena.node_kind_ref(lhs).clone();
    let rhs_kind = arena.node_kind_ref(rhs).clone();

    if let (N::Constant(a), N::Constant(b)) = (&lhs_kind, &rhs_kind) {
        return append_constant_in_arena(arena, op.apply_constant(*a, *b));
    }

    match op {
        BinaryOp::Add => {
            if is_zero_kind(&lhs_kind) {
                return rhs;
            }
            if is_zero_kind(&rhs_kind) {
                return lhs;
            }
            if let Some(combined) = combine_like_terms_shallow(arena, lhs, rhs, 1.0) {
                return combined;
            }
        }
        BinaryOp::Sub => {
            if is_zero_kind(&lhs_kind) {
                return neg_in_arena(arena, rhs);
            }
            if is_zero_kind(&rhs_kind) {
                return lhs;
            }
            if lhs == rhs {
                return append_constant_in_arena(arena, 0.0);
            }
            if let Some(combined) = combine_like_terms_shallow(arena, lhs, rhs, -1.0) {
                return combined;
            }
        }
        BinaryOp::Mul => {
            if is_zero_kind(&lhs_kind) || is_zero_kind(&rhs_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
            if is_neg_one_kind(&lhs_kind) {
                return neg_in_arena(arena, rhs);
            }
            if is_neg_one_kind(&rhs_kind) {
                return neg_in_arena(arena, lhs);
            }
            if is_one_kind(&lhs_kind) {
                return rhs;
            }
            if is_one_kind(&rhs_kind) {
                return lhs;
            }
            if let Some(simplified) = simplify_mul_shallow(arena, lhs, rhs) {
                return simplified;
            }
        }
        BinaryOp::Div => {
            if is_zero_kind(&lhs_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
            if is_one_kind(&rhs_kind) {
                return lhs;
            }
            if let Some(simplified) = simplify_div_shallow(arena, lhs, rhs) {
                return simplified;
            }
        }
        BinaryOp::Pow => {
            if is_zero_kind(&rhs_kind) {
                return append_constant_in_arena(arena, 1.0);
            }
            if is_one_kind(&rhs_kind) {
                return lhs;
            }
            if let Some(exponent) = constant_value_with_arena(arena, rhs) {
                if exponent == 2.0 {
                    return append_binary_in_arena(arena, BinaryOp::Mul, lhs, lhs);
                }
                if exponent == 0.5 {
                    return unary_in_arena(arena, UnaryOp::Sqrt, lhs);
                }
                if is_zero_kind(&lhs_kind) && exponent > 0.0 {
                    return append_constant_in_arena(arena, 0.0);
                }
            }
            if is_one_kind(&lhs_kind) {
                return append_constant_in_arena(arena, 1.0);
            }
        }
        BinaryOp::Atan2 => {
            if is_zero_kind(&lhs_kind) && is_one_kind(&rhs_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
        }
        BinaryOp::Hypot => {
            if is_zero_kind(&lhs_kind) {
                return unary_in_arena(arena, UnaryOp::Abs, rhs);
            }
            if is_zero_kind(&rhs_kind) {
                return unary_in_arena(arena, UnaryOp::Abs, lhs);
            }
        }
        BinaryOp::Mod => {
            if is_zero_kind(&lhs_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
        }
        BinaryOp::Copysign => {
            if is_zero_kind(&lhs_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
            if is_zero_kind(&rhs_kind) {
                return unary_in_arena(arena, UnaryOp::Abs, lhs);
            }
        }
    }

    append_binary_in_arena(arena, op, lhs, rhs)
}

fn apply_adexpr_sx_unary_with_budget(
    context_id: u32,
    op: UnaryOp,
    arg: SX,
    _simplify_depth: Option<usize>,
) -> SX {
    with_arena_for_context(context_id, |arena| unary_in_arena(arena, op, arg))
}

#[allow(dead_code)]
fn apply_adexpr_sx_binary(op: BinaryOp, lhs: SX, rhs: SX) -> SX {
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

fn apply_adexpr_sx_binary_with_budget(
    context_id: u32,
    op: BinaryOp,
    lhs: SX,
    rhs: SX,
    simplify_depth: Option<usize>,
) -> SX {
    with_arena_for_context(context_id, |arena| {
        binary_in_arena_with_simplify_depth(arena, op, lhs, rhs, simplify_depth)
    })
}

#[derive(Default)]
struct ProgramBuilder {
    slot_exprs: Vec<SX>,
    slot_by_node: HashMap<SX, usize>,
    instructions: Vec<ProgramInstruction>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ForwardBatchCallCacheKey {
    site_key: ForwardCallCacheKey,
    direction_count: usize,
    direction_mask: u64,
    seed_inputs: CallInputs,
}

#[derive(Clone, Debug)]
pub(crate) struct JacobianStructure {
    pub(crate) ccs: CCS,
    pub(crate) forward_color_groups: Vec<Vec<Index>>,
    pub(crate) reverse_color_groups: Vec<Vec<Index>>,
    pub(crate) positions_by_row: Vec<Vec<(Index, Index)>>,
}

#[derive(Clone, Debug)]
struct DependencyMask {
    words: Vec<u64>,
}

impl DependencyMask {
    fn new(bit_count: usize) -> Self {
        Self {
            words: vec![0; bit_count.div_ceil(64)],
        }
    }

    fn singleton(bit_count: usize, index: usize) -> Self {
        let mut mask = Self::new(bit_count);
        mask.set(index);
        mask
    }

    fn set(&mut self, index: usize) {
        let word = index / 64;
        let bit = index % 64;
        self.words[word] |= 1_u64 << bit;
    }

    fn union_assign(&mut self, other: &Self) {
        for (dst, src) in self.words.iter_mut().zip(other.words.iter()) {
            *dst |= *src;
        }
    }

    fn iter_ones(&self) -> impl Iterator<Item = Index> + '_ {
        self.words
            .iter()
            .copied()
            .enumerate()
            .flat_map(|(word_index, mut word)| {
                let mut bits = Vec::new();
                while word != 0 {
                    let bit = word.trailing_zeros() as usize;
                    bits.push(word_index * 64 + bit);
                    word &= word - 1;
                }
                bits.into_iter()
            })
    }
}

fn iter_direction_bits(mask: u64) -> impl Iterator<Item = usize> {
    let mut bits = Vec::new();
    let mut word = mask;
    while word != 0 {
        let bit = word.trailing_zeros() as usize;
        bits.push(bit);
        word &= word - 1;
    }
    bits.into_iter()
}

static JACOBIAN_STRUCTURE_CACHE: OnceLock<Mutex<HashMap<JacobianStructureKey, JacobianStructure>>> =
    OnceLock::new();
static PROGRAM_CACHE: OnceLock<Mutex<HashMap<ProgramKey, Arc<SxProgram>>>> = OnceLock::new();
static FUNCTION_PROGRAM_PLAN_CACHE: OnceLock<Mutex<HashMap<FunctionId, Arc<FunctionProgramPlan>>>> =
    OnceLock::new();
static FUNCTION_EXPR_PROGRAM_PLAN_CACHE: OnceLock<
    Mutex<HashMap<FunctionId, Arc<FunctionExprProgramPlan>>>,
> = OnceLock::new();

impl ContextRegistry {
    fn with_root() -> Self {
        let mut arenas = HashMap::new();
        arenas.insert(
            ROOT_CONTEXT_ID,
            Arc::new(ContextArena {
                arena: Mutex::new(NodeArena::new(ROOT_CONTEXT_ID)),
            }),
        );
        Self {
            next_context_id: ROOT_CONTEXT_ID + 1,
            arenas,
        }
    }
}

impl UnaryOp {
    pub fn name(self) -> &'static str {
        match self {
            Self::Abs => "abs",
            Self::Sign => "sign",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Round => "round",
            Self::Trunc => "trunc",
            Self::Sqrt => "sqrt",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Tanh => "tanh",
            Self::Asinh => "asinh",
            Self::Acosh => "acosh",
            Self::Atanh => "atanh",
        }
    }

    pub(crate) fn apply_constant(self, arg: f64) -> f64 {
        match self {
            Self::Abs => arg.abs(),
            Self::Sign => {
                if arg > 0.0 {
                    1.0
                } else if arg < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::Floor => arg.floor(),
            Self::Ceil => arg.ceil(),
            Self::Round => arg.round(),
            Self::Trunc => arg.trunc(),
            Self::Sqrt => arg.sqrt(),
            Self::Exp => arg.exp(),
            Self::Log => arg.ln(),
            Self::Sin => arg.sin(),
            Self::Cos => arg.cos(),
            Self::Tan => arg.tan(),
            Self::Asin => arg.asin(),
            Self::Acos => arg.acos(),
            Self::Atan => arg.atan(),
            Self::Sinh => arg.sinh(),
            Self::Cosh => arg.cosh(),
            Self::Tanh => arg.tanh(),
            Self::Asinh => arg.asinh(),
            Self::Acosh => arg.acosh(),
            Self::Atanh => arg.atanh(),
        }
    }
}

impl BinaryOp {
    pub fn symbol(self) -> Option<&'static str> {
        match self {
            Self::Add => Some("+"),
            Self::Sub => Some("-"),
            Self::Mul => Some("*"),
            Self::Div => Some("/"),
            Self::Mod => Some("%"),
            Self::Pow | Self::Atan2 | Self::Hypot | Self::Copysign => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Pow => "pow",
            Self::Atan2 => "atan2",
            Self::Hypot => "hypot",
            Self::Mod => "mod",
            Self::Copysign => "copysign",
        }
    }

    pub(crate) fn apply_constant(self, lhs: f64, rhs: f64) -> f64 {
        match self {
            Self::Add => lhs + rhs,
            Self::Sub => lhs - rhs,
            Self::Mul => lhs * rhs,
            Self::Div => lhs / rhs,
            Self::Pow => lhs.powf(rhs),
            Self::Atan2 => lhs.atan2(rhs),
            Self::Hypot => lhs.hypot(rhs),
            Self::Mod => lhs % rhs,
            Self::Copysign => lhs.copysign(rhs),
        }
    }

    pub(crate) fn is_commutative(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Hypot)
    }
}

fn with_arena<R>(f: impl FnOnce(&mut NodeArena) -> R) -> R {
    let arena = lookup_context_arena(current_context_id());
    let mut guard = lock_arena(&arena.arena);
    f(&mut guard)
}

fn lock_arena(mutex: &Mutex<NodeArena>) -> MutexGuard<'_, NodeArena> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn lock_context_registry() -> MutexGuard<'static, ContextRegistry> {
    match CONTEXT_REGISTRY
        .get_or_init(|| Mutex::new(ContextRegistry::with_root()))
        .lock()
    {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn current_context_id() -> u32 {
    CURRENT_CONTEXT_ID.with(Cell::get)
}

pub(crate) fn with_sx_context_id<R>(context_id: u32, f: impl FnOnce() -> R) -> R {
    CURRENT_CONTEXT_ID.with(|current| {
        let previous = current.replace(context_id);
        let result = f();
        current.set(previous);
        result
    })
}

fn lookup_context_arena(context_id: u32) -> Arc<ContextArena> {
    CONTEXT_ARENA_CACHE.with(|cache| {
        if let Some(arena) = cache.borrow().get(&context_id).cloned() {
            return arena;
        }
        let arena = lock_context_registry()
            .arenas
            .get(&context_id)
            .cloned()
            .unwrap_or_else(|| panic!("unknown SX context {context_id}"));
        cache.borrow_mut().insert(context_id, arena.clone());
        arena
    })
}

fn with_arena_for_context<R>(context_id: u32, f: impl FnOnce(&mut NodeArena) -> R) -> R {
    let arena = lookup_context_arena(context_id);
    let mut guard = lock_arena(&arena.arena);
    f(&mut guard)
}

fn with_arena_ref_for_context<R>(context_id: u32, f: impl FnOnce(&NodeArena) -> R) -> R {
    let arena = lookup_context_arena(context_id);
    let guard = lock_arena(&arena.arena);
    f(&guard)
}

fn with_arena_for_sx<R>(sx: SX, f: impl FnOnce(&mut NodeArena) -> R) -> R {
    with_arena_for_context(sx.context_id(), f)
}

fn with_arena_ref_for_sx<R>(sx: SX, f: impl FnOnce(&NodeArena) -> R) -> R {
    with_arena_ref_for_context(sx.context_id(), f)
}

fn ensure_same_context(lhs: SX, rhs: SX) -> u32 {
    let lhs_context = lhs.context_id();
    let rhs_context = rhs.context_id();
    assert!(
        lhs_context == rhs_context,
        "mixed SX contexts are not supported: lhs context {} vs rhs context {}",
        lhs_context,
        rhs_context
    );
    lhs_context
}

fn ensure_slice_context(label: &str, values: &[SX]) -> Option<u32> {
    let mut context = None;
    for &value in values {
        let value_context = value.context_id();
        if let Some(existing) = context {
            assert!(
                existing == value_context,
                "mixed SX contexts in {}: {} vs {}",
                label,
                existing,
                value_context
            );
        } else {
            context = Some(value_context);
        }
    }
    context
}

fn resolve_context_for_slices(slices: &[(&str, &[SX])]) -> u32 {
    let mut context = None;
    for (label, values) in slices {
        if let Some(slice_context) = ensure_slice_context(label, values) {
            if let Some(existing) = context {
                assert!(
                    existing == slice_context,
                    "mixed SX contexts across inputs: {} vs {}",
                    existing,
                    slice_context
                );
            } else {
                context = Some(slice_context);
            }
        }
    }
    context.unwrap_or_else(current_context_id)
}

fn lock_jacobian_structure_cache()
-> MutexGuard<'static, HashMap<JacobianStructureKey, JacobianStructure>> {
    match JACOBIAN_STRUCTURE_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
    {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn lock_program_cache() -> MutexGuard<'static, HashMap<ProgramKey, Arc<SxProgram>>> {
    match PROGRAM_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
    {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn lock_function_program_plan_cache()
-> MutexGuard<'static, HashMap<FunctionId, Arc<FunctionProgramPlan>>> {
    match FUNCTION_PROGRAM_PLAN_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
    {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn lock_function_expr_program_plan_cache()
-> MutexGuard<'static, HashMap<FunctionId, Arc<FunctionExprProgramPlan>>> {
    match FUNCTION_EXPR_PROGRAM_PLAN_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
    {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn lower_sx_collection_to_expr(values: &[SX]) -> Vec<SXExpr> {
    let mut memo = HashMap::new();
    values
        .iter()
        .copied()
        .map(|value| SXExpr::from_sx_with_memo(value, &mut memo))
        .collect()
}

impl ProgramBuilder {
    fn alloc_slot(&mut self, expr: SX) -> usize {
        if let Some(&slot) = self.slot_by_node.get(&expr) {
            return slot;
        }
        let slot = self.slot_exprs.len();
        self.slot_exprs.push(expr);
        self.slot_by_node.insert(expr, slot);
        slot
    }

    fn ensure_slot(&mut self, expr: SX) -> usize {
        if let Some(&slot) = self.slot_by_node.get(&expr) {
            return slot;
        }

        match node_kind(expr) {
            NodeKind::Constant(_) | NodeKind::Symbol { .. } => self.alloc_slot(expr),
            NodeKind::Unary { op, arg } => {
                let arg_slot = self.ensure_slot(arg);
                let result_slot = self.alloc_slot(expr);
                self.instructions.push(ProgramInstruction::Unary {
                    result_slot,
                    op,
                    arg_slot,
                    arg_expr: arg,
                });
                result_slot
            }
            NodeKind::Binary { op, lhs, rhs } => {
                let lhs_slot = self.ensure_slot(lhs);
                let rhs_slot = self.ensure_slot(rhs);
                let result_slot = self.alloc_slot(expr);
                self.instructions.push(ProgramInstruction::Binary {
                    result_slot,
                    op,
                    lhs_slot,
                    rhs_slot,
                    lhs_expr: lhs,
                    rhs_expr: rhs,
                });
                result_slot
            }
            NodeKind::Call {
                function,
                inputs,
                output_slot,
                output_offset,
            } => {
                let profile = dependency_profile(function);
                let call_inputs = inputs
                    .iter()
                    .enumerate()
                    .map(|(slot, input)| ProgramCallInput {
                        matrix: input.clone(),
                        slots: input
                            .nonzeros()
                            .iter()
                            .copied()
                            .map(|value| self.ensure_slot(value))
                            .collect(),
                        relevant_offsets: input
                            .nonzeros()
                            .iter()
                            .enumerate()
                            .filter_map(|(offset, _)| {
                                profile
                                    .output_depends_on(output_slot, output_offset, slot, offset)
                                    .then_some(offset)
                            })
                            .collect(),
                    })
                    .collect::<Vec<_>>();
                let result_slot = self.alloc_slot(expr);
                self.instructions.push(ProgramInstruction::Call {
                    result_slot,
                    function,
                    site_key: ForwardCallCacheKey {
                        function,
                        site_id: inputs.site_id(),
                    },
                    output_slot,
                    output_offset,
                    inputs: call_inputs,
                });
                result_slot
            }
        }
    }
}

fn program_for_outputs(outputs: &[SX]) -> Arc<SxProgram> {
    let key = ProgramKey {
        outputs: outputs.to_vec(),
    };
    if let Some(existing) = lock_program_cache().get(&key).cloned() {
        return existing;
    }

    let mut builder = ProgramBuilder::default();
    let output_slots = outputs
        .iter()
        .copied()
        .map(|output| builder.ensure_slot(output))
        .collect::<Vec<_>>();
    let program = Arc::new(SxProgram {
        slot_exprs: builder.slot_exprs,
        slot_by_node: builder.slot_by_node,
        output_slots,
        instructions: builder.instructions,
    });
    lock_program_cache().insert(key, Arc::clone(&program));
    program
}

fn function_program_plan(function_id: FunctionId) -> Result<Arc<FunctionProgramPlan>> {
    if let Some(existing) = lock_function_program_plan_cache()
        .get(&function_id)
        .cloned()
    {
        return Ok(existing);
    }

    let function = function_by_id(function_id)
        .ok_or_else(|| SxError::Graph(format!("unknown function id {function_id}")))?;
    let flat_outputs = function
        .outputs()
        .iter()
        .flat_map(|output| output.matrix().nonzeros().iter().copied())
        .collect::<Vec<_>>();
    let program = program_for_outputs(&flat_outputs);
    let input_slots = function
        .inputs()
        .iter()
        .map(|input| {
            input
                .matrix()
                .nonzeros()
                .iter()
                .map(|&formal| program.slot_by_node.get(&formal).copied())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let output_ccs = function
        .outputs()
        .iter()
        .map(|output| output.matrix().ccs().clone())
        .collect::<Vec<_>>();
    let output_nnz = function
        .outputs()
        .iter()
        .map(|output| output.matrix().nnz())
        .collect::<Vec<_>>();
    let plan = Arc::new(FunctionProgramPlan {
        program,
        input_slots,
        output_ccs,
        output_nnz,
    });
    lock_function_program_plan_cache().insert(function_id, Arc::clone(&plan));
    Ok(plan)
}

fn function_expr_program_plan(function_id: FunctionId) -> Result<Arc<FunctionExprProgramPlan>> {
    if let Some(existing) = lock_function_expr_program_plan_cache()
        .get(&function_id)
        .cloned()
    {
        return Ok(existing);
    }

    let plan = function_program_plan(function_id)?;
    let expr_plan = Arc::new(FunctionExprProgramPlan {
        program: Arc::clone(&plan.program),
        slot_exprs: lower_sx_collection_to_expr(&plan.program.slot_exprs),
        input_slots: plan.input_slots.clone(),
        output_ccs: plan.output_ccs.clone(),
        output_nnz: plan.output_nnz.clone(),
    });
    lock_function_expr_program_plan_cache().insert(function_id, Arc::clone(&expr_plan));
    Ok(expr_plan)
}

fn execute_program_symbolically_expr(
    program: &SxProgram,
    mut slot_values: Vec<SXExpr>,
) -> Result<Vec<SXExpr>> {
    for instruction in &program.instructions {
        match instruction {
            ProgramInstruction::Unary {
                result_slot,
                op,
                arg_slot,
                ..
            } => {
                slot_values[*result_slot] = SXExpr::unary(*op, slot_values[*arg_slot].clone());
            }
            ProgramInstruction::Binary {
                result_slot,
                op,
                lhs_slot,
                rhs_slot,
                ..
            } => {
                slot_values[*result_slot] = SXExpr::binary_ad(
                    *op,
                    slot_values[*lhs_slot].clone(),
                    slot_values[*rhs_slot].clone(),
                );
            }
            ProgramInstruction::Call {
                result_slot,
                function,
                output_slot,
                output_offset,
                inputs,
                ..
            } => {
                let call_inputs = inputs
                    .iter()
                    .map(|input| {
                        let nonzeros = input
                            .slots
                            .iter()
                            .map(|&slot| slot_values[slot].clone())
                            .collect::<Vec<_>>();
                        SXExprMatrix::new(input.matrix.ccs().clone(), nonzeros)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let call_outputs =
                    execute_function_symbolically_expr_by_id(*function, &call_inputs)?;
                slot_values[*result_slot] = call_outputs[*output_slot].nz(*output_offset);
            }
        }
    }

    Ok(program
        .output_slots
        .iter()
        .map(|&slot| slot_values[slot].clone())
        .collect::<Vec<_>>())
}

#[allow(dead_code)]
fn collect_adexpr_output_matrices(
    flat_values: &[AdExpr],
    output_ccs: &[CCS],
    output_nnz: &[usize],
) -> Result<Vec<AdExprMatrix>> {
    let mut offset = 0;
    let mut outputs = Vec::with_capacity(output_ccs.len());
    for (ccs, nnz) in output_ccs.iter().zip(output_nnz.iter().copied()) {
        let next_offset = offset + nnz;
        outputs.push(AdExprMatrix::new(
            ccs.clone(),
            flat_values[offset..next_offset].to_vec(),
        )?);
        offset = next_offset;
    }
    Ok(outputs)
}

fn collect_adexpr_output_vectors(flat_values: &[AdExpr], output_nnz: &[usize]) -> Vec<Vec<AdExpr>> {
    let mut offset = 0;
    let mut outputs = Vec::with_capacity(output_nnz.len());
    for nnz in output_nnz.iter().copied() {
        let next_offset = offset + nnz;
        outputs.push(flat_values[offset..next_offset].to_vec());
        offset = next_offset;
    }
    outputs
}

fn execute_program_symbolically_adexpr_slots(
    program: &SxProgram,
    mut slot_values: Vec<AdExpr>,
    arena: &mut AdExprArena,
    lower_memo: &mut HashMap<SX, AdExpr>,
    local_caches: &mut AdLocalCaches,
    function_plan_cache: &mut HashMap<FunctionId, FunctionAdExprArenaPlan>,
) -> Result<Vec<AdExpr>> {
    for instruction in &program.instructions {
        match instruction {
            ProgramInstruction::Unary {
                result_slot,
                op,
                arg_slot,
                ..
            } => {
                slot_values[*result_slot] = arena.unary(*op, slot_values[*arg_slot]);
            }
            ProgramInstruction::Binary {
                result_slot,
                op,
                lhs_slot,
                rhs_slot,
                ..
            } => {
                slot_values[*result_slot] =
                    arena.binary_ad(*op, slot_values[*lhs_slot], slot_values[*rhs_slot]);
            }
            ProgramInstruction::Call {
                result_slot,
                function,
                output_slot,
                output_offset,
                inputs,
                ..
            } => {
                let call_inputs = inputs
                    .iter()
                    .map(|input| {
                        input
                            .slots
                            .iter()
                            .map(|&slot| slot_values[slot])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let call_outputs = execute_function_symbolically_adexpr_flat_by_id(
                    *function,
                    &call_inputs,
                    arena,
                    lower_memo,
                    local_caches,
                    function_plan_cache,
                )?;
                slot_values[*result_slot] = call_outputs[*output_slot][*output_offset];
            }
        }
    }

    Ok(slot_values)
}

fn execute_function_symbolically_expr_by_id(
    function_id: FunctionId,
    inputs: &[SXExprMatrix],
) -> Result<Vec<SXExprMatrix>> {
    let function = function_by_id(function_id)
        .ok_or_else(|| SxError::Graph(format!("unknown function id {function_id}")))?;
    if inputs.len() != function.inputs().len() {
        return Err(SxError::Shape(format!(
            "function {} expected {} input slots, got {}",
            function.name(),
            function.inputs().len(),
            inputs.len()
        )));
    }
    for (slot, (actual, formal)) in inputs.iter().zip(function.inputs()).enumerate() {
        if actual.ccs() != formal.matrix().ccs() {
            return Err(SxError::Shape(format!(
                "call input slot {slot} for {} must match declared CCS",
                function.name()
            )));
        }
    }

    let plan = function_expr_program_plan(function_id)?;
    let mut slot_values = plan.slot_exprs.clone();
    for ((actual, _formal), input_slots) in inputs
        .iter()
        .zip(function.inputs())
        .zip(plan.input_slots.iter())
    {
        for (actual_nz, slot) in actual.nonzeros().iter().cloned().zip(input_slots.iter()) {
            if let Some(slot) = slot {
                slot_values[*slot] = actual_nz;
            }
        }
    }

    let flat_values = execute_program_symbolically_expr(&plan.program, slot_values)?;
    let mut offset = 0;
    let mut outputs = Vec::with_capacity(plan.output_ccs.len());
    for (ccs, nnz) in plan.output_ccs.iter().zip(plan.output_nnz.iter().copied()) {
        let next_offset = offset + nnz;
        outputs.push(SXExprMatrix::new(
            ccs.clone(),
            flat_values[offset..next_offset].to_vec(),
        )?);
        offset = next_offset;
    }
    Ok(outputs)
}

#[allow(dead_code)]
fn execute_function_symbolically_adexpr_by_id(
    function_id: FunctionId,
    inputs: &[AdExprMatrix],
    arena: &mut AdExprArena,
    lower_memo: &mut HashMap<SX, AdExpr>,
    function_plan_cache: &mut HashMap<FunctionId, FunctionAdExprArenaPlan>,
) -> Result<Vec<AdExprMatrix>> {
    let mut local_caches = AdLocalCaches::default();
    let function = function_by_id(function_id)
        .ok_or_else(|| SxError::Graph(format!("unknown function id {function_id}")))?;
    if inputs.len() != function.inputs().len() {
        return Err(SxError::Shape(format!(
            "function {} expected {} input slots, got {}",
            function.name(),
            function.inputs().len(),
            inputs.len()
        )));
    }
    for (slot, (actual, formal)) in inputs.iter().zip(function.inputs()).enumerate() {
        if actual.ccs != *formal.matrix().ccs() {
            return Err(SxError::Shape(format!(
                "call input slot {slot} for {} must match declared CCS",
                function.name()
            )));
        }
    }

    ensure_function_adexpr_arena_plan(function_id, arena, lower_memo, function_plan_cache)?;
    let plan = function_plan_cache
        .get(&function_id)
        .expect("function plan inserted");
    let program = Arc::clone(&plan.program);
    let mut slot_values = local_caches.take_adexpr_slot_scratch(function_id, &plan.slot_exprs);
    let input_slots = plan.input_slots.clone();
    let output_ccs = plan.output_ccs.clone();
    let output_nnz = plan.output_nnz.clone();
    for (actual, input_slots) in inputs.iter().zip(input_slots.iter()) {
        for (actual_nz, slot) in actual.nonzeros.iter().copied().zip(input_slots.iter()) {
            if let Some(slot) = slot {
                slot_values[*slot] = actual_nz;
            }
        }
    }

    let slot_values = execute_program_symbolically_adexpr_slots(
        &program,
        slot_values,
        arena,
        lower_memo,
        &mut local_caches,
        function_plan_cache,
    )?;
    let flat_values = program
        .output_slots
        .iter()
        .map(|&slot| slot_values[slot])
        .collect::<Vec<_>>();
    let outputs = collect_adexpr_output_matrices(&flat_values, &output_ccs, &output_nnz)?;
    local_caches.recycle_adexpr_slot_scratch(function_id, slot_values);
    Ok(outputs)
}

fn execute_function_symbolically_adexpr_flat_by_id(
    function_id: FunctionId,
    inputs: &[Vec<AdExpr>],
    arena: &mut AdExprArena,
    lower_memo: &mut HashMap<SX, AdExpr>,
    local_caches: &mut AdLocalCaches,
    function_plan_cache: &mut HashMap<FunctionId, FunctionAdExprArenaPlan>,
) -> Result<Vec<Vec<AdExpr>>> {
    let function = function_by_id(function_id)
        .ok_or_else(|| SxError::Graph(format!("unknown function id {function_id}")))?;
    if inputs.len() != function.inputs().len() {
        return Err(SxError::Shape(format!(
            "function {} expected {} input slots, got {}",
            function.name(),
            function.inputs().len(),
            inputs.len()
        )));
    }

    ensure_function_adexpr_arena_plan(function_id, arena, lower_memo, function_plan_cache)?;
    let plan = function_plan_cache
        .get(&function_id)
        .expect("function plan inserted");
    let program = Arc::clone(&plan.program);
    let mut slot_values = local_caches.take_adexpr_slot_scratch(function_id, &plan.slot_exprs);
    let input_slots = plan.input_slots.clone();
    let output_nnz = plan.output_nnz.clone();
    for (slot, (actual, input_slots)) in inputs.iter().zip(input_slots.iter()).enumerate() {
        if actual.len() != input_slots.len() {
            return Err(SxError::Shape(format!(
                "function {} input slot {slot} expected {} nonzeros, got {}",
                function.name(),
                input_slots.len(),
                actual.len()
            )));
        }
        for (actual_nz, slot) in actual.iter().copied().zip(input_slots.iter()) {
            if let Some(slot) = slot {
                slot_values[*slot] = actual_nz;
            }
        }
    }

    let slot_values = execute_program_symbolically_adexpr_slots(
        &program,
        slot_values,
        arena,
        lower_memo,
        local_caches,
        function_plan_cache,
    )?;
    let flat_values = program
        .output_slots
        .iter()
        .map(|&slot| slot_values[slot])
        .collect::<Vec<_>>();
    local_caches.recycle_adexpr_slot_scratch(function_id, slot_values);
    Ok(collect_adexpr_output_vectors(&flat_values, &output_nnz))
}

#[expect(
    clippy::too_many_arguments,
    reason = "forward batch AD keeps hot-path scratch buffers and caches explicit"
)]
fn execute_program_forward_batch_adexpr_with_slots(
    program: &SxProgram,
    slot_values: &[AdExpr],
    mut derivative_slots: Vec<AdExpr>,
    mut active_masks: Option<Vec<u64>>,
    direction_count: usize,
    arena: &mut AdExprArena,
    lower_memo: &mut HashMap<SX, AdExpr>,
    local_caches: &mut AdLocalCaches,
    function_plan_cache: &mut HashMap<FunctionId, FunctionAdExprArenaPlan>,
) -> Result<Vec<AdExpr>> {
    let slot_base = |slot: usize| slot * direction_count;
    let zero = arena.zero();

    for instruction in &program.instructions {
        match instruction {
            ProgramInstruction::Unary {
                result_slot,
                op,
                arg_slot,
                ..
            } => {
                let result_base = slot_base(*result_slot);
                let arg_base = slot_base(*arg_slot);
                let derivative = arena.unary_derivative(*op, slot_values[*arg_slot]);
                if let Some(masks) = active_masks.as_mut() {
                    let arg_mask = masks[*arg_slot];
                    masks[*result_slot] = arg_mask;
                    if arg_mask == 0 {
                        continue;
                    }
                    for direction in iter_direction_bits(arg_mask) {
                        derivative_slots[result_base + direction] = arena.binary_ad(
                            BinaryOp::Mul,
                            derivative_slots[arg_base + direction],
                            derivative,
                        );
                    }
                } else {
                    for direction in 0..direction_count {
                        derivative_slots[result_base + direction] = arena.binary_ad(
                            BinaryOp::Mul,
                            derivative_slots[arg_base + direction],
                            derivative,
                        );
                    }
                }
            }
            ProgramInstruction::Binary {
                result_slot,
                op,
                lhs_slot,
                rhs_slot,
                ..
            } => {
                let result_base = slot_base(*result_slot);
                let lhs_base = slot_base(*lhs_slot);
                let rhs_base = slot_base(*rhs_slot);
                let (d_lhs, d_rhs) =
                    arena.binary_partials(*op, slot_values[*lhs_slot], slot_values[*rhs_slot]);
                if let Some(masks) = active_masks.as_mut() {
                    let lhs_mask = masks[*lhs_slot];
                    let rhs_mask = masks[*rhs_slot];
                    let result_mask = lhs_mask | rhs_mask;
                    masks[*result_slot] = result_mask;
                    if result_mask == 0 {
                        continue;
                    }
                    for direction in iter_direction_bits(result_mask) {
                        let left = arena.binary_ad(
                            BinaryOp::Mul,
                            derivative_slots[lhs_base + direction],
                            d_lhs,
                        );
                        let right = arena.binary_ad(
                            BinaryOp::Mul,
                            derivative_slots[rhs_base + direction],
                            d_rhs,
                        );
                        derivative_slots[result_base + direction] =
                            arena.binary_ad(BinaryOp::Add, left, right);
                    }
                } else {
                    for direction in 0..direction_count {
                        let left = arena.binary_ad(
                            BinaryOp::Mul,
                            derivative_slots[lhs_base + direction],
                            d_lhs,
                        );
                        let right = arena.binary_ad(
                            BinaryOp::Mul,
                            derivative_slots[rhs_base + direction],
                            d_rhs,
                        );
                        derivative_slots[result_base + direction] =
                            arena.binary_ad(BinaryOp::Add, left, right);
                    }
                }
            }
            ProgramInstruction::Call {
                result_slot,
                function,
                output_slot,
                output_offset,
                inputs,
                ..
            } => {
                let result_base = slot_base(*result_slot);
                let active_directions = if let Some(masks) = active_masks.as_mut() {
                    let mut direction_mask = 0_u64;
                    for input in inputs {
                        for &offset in &input.relevant_offsets {
                            direction_mask |= masks[input.slots[offset]];
                        }
                    }
                    masks[*result_slot] = direction_mask;
                    if direction_mask == 0 {
                        Vec::new()
                    } else {
                        iter_direction_bits(direction_mask).collect::<Vec<_>>()
                    }
                } else {
                    (0..direction_count)
                        .filter(|&direction| {
                            inputs.iter().any(|input| {
                                input.relevant_offsets.iter().any(|&offset| {
                                    !arena.is_zero(
                                        derivative_slots
                                            [slot_base(input.slots[offset]) + direction],
                                    )
                                })
                            })
                        })
                        .collect::<Vec<_>>()
                };

                if active_directions.is_empty() {
                    for direction in 0..direction_count {
                        derivative_slots[result_base + direction] = zero;
                    }
                    continue;
                }

                let call_inputs = inputs
                    .iter()
                    .map(|input| {
                        input
                            .slots
                            .iter()
                            .map(|&slot| slot_values[slot])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let directional_outputs = execute_function_forward_batch_adexpr_from_call_by_id(
                    *function,
                    &call_inputs,
                    inputs,
                    &active_directions,
                    direction_count,
                    &derivative_slots,
                    arena,
                    lower_memo,
                    local_caches,
                    function_plan_cache,
                )?;
                for direction in 0..direction_count {
                    derivative_slots[result_base + direction] = zero;
                }
                for (local_direction, &direction) in active_directions.iter().enumerate() {
                    derivative_slots[result_base + direction] =
                        directional_outputs[local_direction][*output_slot][*output_offset];
                }
            }
        }
    }

    Ok(derivative_slots)
}

#[allow(dead_code)]
fn execute_function_forward_batch_adexpr_by_id(
    function_id: FunctionId,
    inputs: &[AdExprMatrix],
    direction_seed_inputs: &[Vec<AdExprMatrix>],
    arena: &mut AdExprArena,
    lower_memo: &mut HashMap<SX, AdExpr>,
    function_plan_cache: &mut HashMap<FunctionId, FunctionAdExprArenaPlan>,
) -> Result<Vec<Vec<AdExprMatrix>>> {
    let mut local_caches = AdLocalCaches::default();
    let function = function_by_id(function_id)
        .ok_or_else(|| SxError::Graph(format!("unknown function id {function_id}")))?;
    if inputs.len() != function.inputs().len() {
        return Err(SxError::Shape(format!(
            "function {} expected {} input slots, got {}",
            function.name(),
            function.inputs().len(),
            inputs.len()
        )));
    }
    for (slot, (actual, formal)) in inputs.iter().zip(function.inputs()).enumerate() {
        if actual.ccs != *formal.matrix().ccs() {
            return Err(SxError::Shape(format!(
                "call input slot {slot} for {} must match declared CCS",
                function.name()
            )));
        }
    }
    for direction_inputs in direction_seed_inputs {
        if direction_inputs.len() != inputs.len() {
            return Err(SxError::Shape(format!(
                "function {} expected {} seed input slots per direction, got {}",
                function.name(),
                inputs.len(),
                direction_inputs.len()
            )));
        }
        for (slot, (actual, formal)) in direction_inputs.iter().zip(function.inputs()).enumerate() {
            if actual.ccs != *formal.matrix().ccs() {
                return Err(SxError::Shape(format!(
                    "seed input slot {slot} for {} must match declared CCS",
                    function.name()
                )));
            }
        }
    }

    ensure_function_adexpr_arena_plan(function_id, arena, lower_memo, function_plan_cache)?;
    let plan = function_plan_cache
        .get(&function_id)
        .expect("function plan inserted");
    let program = Arc::clone(&plan.program);
    let input_slots = plan.input_slots.clone();
    let output_ccs = plan.output_ccs.clone();
    let output_nnz = plan.output_nnz.clone();

    let mut slot_values = local_caches.take_adexpr_slot_scratch(function_id, &plan.slot_exprs);
    for (actual, input_slots) in inputs.iter().zip(input_slots.iter()) {
        for (actual_nz, slot) in actual.nonzeros.iter().copied().zip(input_slots.iter()) {
            if let Some(slot) = slot {
                slot_values[*slot] = actual_nz;
            }
        }
    }
    let slot_values = execute_program_symbolically_adexpr_slots(
        &program,
        slot_values,
        arena,
        lower_memo,
        &mut local_caches,
        function_plan_cache,
    )?;

    let direction_count = direction_seed_inputs.len();
    if direction_count == 0 {
        return Ok(Vec::new());
    }
    let slot_count = program.slot_exprs.len();
    let slot_base = |slot: usize| slot * direction_count;
    let zero = arena.zero();
    let mut derivative_slots = local_caches.take_adexpr_derivative_scratch(
        function_id,
        direction_count,
        slot_count * direction_count,
        zero,
    );
    let mut active_masks = (direction_count <= 64).then(|| vec![0_u64; slot_count]);
    for (direction, direction_inputs) in direction_seed_inputs.iter().enumerate() {
        for (actual, input_slots) in direction_inputs.iter().zip(input_slots.iter()) {
            for (actual_nz, slot) in actual.nonzeros.iter().copied().zip(input_slots.iter()) {
                if let Some(slot) = slot {
                    derivative_slots[slot_base(*slot) + direction] = actual_nz;
                    if let Some(masks) = active_masks.as_mut()
                        && !arena.is_zero(actual_nz)
                    {
                        masks[*slot] |= 1_u64 << direction;
                    }
                }
            }
        }
    }

    let derivative_slots = execute_program_forward_batch_adexpr_with_slots(
        &program,
        &slot_values,
        derivative_slots,
        active_masks,
        direction_count,
        arena,
        lower_memo,
        &mut local_caches,
        function_plan_cache,
    )?;

    let flat_directionals = (0..direction_count)
        .map(|direction| {
            program
                .output_slots
                .iter()
                .map(|&slot| derivative_slots[slot_base(slot) + direction])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    local_caches.recycle_adexpr_slot_scratch(function_id, slot_values);
    local_caches.recycle_adexpr_derivative_scratch(function_id, direction_count, derivative_slots);

    flat_directionals
        .iter()
        .map(|flat| collect_adexpr_output_matrices(flat, &output_ccs, &output_nnz))
        .collect()
}

#[allow(dead_code)]
fn execute_function_forward_batch_adexpr_flat_by_id(
    function_id: FunctionId,
    inputs: &[Vec<AdExpr>],
    direction_seed_inputs: &[Vec<Vec<AdExpr>>],
    arena: &mut AdExprArena,
    lower_memo: &mut HashMap<SX, AdExpr>,
    local_caches: &mut AdLocalCaches,
    function_plan_cache: &mut HashMap<FunctionId, FunctionAdExprArenaPlan>,
) -> Result<Vec<Vec<Vec<AdExpr>>>> {
    let function = function_by_id(function_id)
        .ok_or_else(|| SxError::Graph(format!("unknown function id {function_id}")))?;
    if inputs.len() != function.inputs().len() {
        return Err(SxError::Shape(format!(
            "function {} expected {} input slots, got {}",
            function.name(),
            function.inputs().len(),
            inputs.len()
        )));
    }

    ensure_function_adexpr_arena_plan(function_id, arena, lower_memo, function_plan_cache)?;
    let plan = function_plan_cache
        .get(&function_id)
        .expect("function plan inserted");
    let program = Arc::clone(&plan.program);
    let input_slots = plan.input_slots.clone();
    let output_nnz = plan.output_nnz.clone();

    let mut slot_values = local_caches.take_adexpr_slot_scratch(function_id, &plan.slot_exprs);
    for (slot, (actual, input_slots)) in inputs.iter().zip(input_slots.iter()).enumerate() {
        if actual.len() != input_slots.len() {
            return Err(SxError::Shape(format!(
                "function {} input slot {slot} expected {} nonzeros, got {}",
                function.name(),
                input_slots.len(),
                actual.len()
            )));
        }
        for (actual_nz, slot) in actual.iter().copied().zip(input_slots.iter()) {
            if let Some(slot) = slot {
                slot_values[*slot] = actual_nz;
            }
        }
    }
    let slot_values = execute_program_symbolically_adexpr_slots(
        &program,
        slot_values,
        arena,
        lower_memo,
        local_caches,
        function_plan_cache,
    )?;

    let direction_count = direction_seed_inputs.len();
    if direction_count == 0 {
        return Ok(Vec::new());
    }
    let slot_count = program.slot_exprs.len();
    let slot_base = |slot: usize| slot * direction_count;
    let zero = arena.zero();
    let mut derivative_slots = local_caches.take_adexpr_derivative_scratch(
        function_id,
        direction_count,
        slot_count * direction_count,
        zero,
    );
    let mut active_masks = (direction_count <= 64).then(|| vec![0_u64; slot_count]);
    for (direction, direction_inputs) in direction_seed_inputs.iter().enumerate() {
        if direction_inputs.len() != input_slots.len() {
            return Err(SxError::Shape(format!(
                "function {} expected {} seed input slots per direction, got {}",
                function.name(),
                input_slots.len(),
                direction_inputs.len()
            )));
        }
        for (slot, (actual, input_slots)) in
            direction_inputs.iter().zip(input_slots.iter()).enumerate()
        {
            if actual.len() != input_slots.len() {
                return Err(SxError::Shape(format!(
                    "function {} seed input slot {slot} expected {} nonzeros, got {}",
                    function.name(),
                    input_slots.len(),
                    actual.len()
                )));
            }
            for (actual_nz, slot) in actual.iter().copied().zip(input_slots.iter()) {
                if let Some(slot) = slot {
                    derivative_slots[slot_base(*slot) + direction] = actual_nz;
                    if let Some(masks) = active_masks.as_mut()
                        && !arena.is_zero(actual_nz)
                    {
                        masks[*slot] |= 1_u64 << direction;
                    }
                }
            }
        }
    }

    let derivative_slots = execute_program_forward_batch_adexpr_with_slots(
        &program,
        &slot_values,
        derivative_slots,
        active_masks,
        direction_count,
        arena,
        lower_memo,
        local_caches,
        function_plan_cache,
    )?;

    let flat_directionals = (0..direction_count)
        .map(|direction| {
            program
                .output_slots
                .iter()
                .map(|&slot| derivative_slots[slot_base(slot) + direction])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    local_caches.recycle_adexpr_slot_scratch(function_id, slot_values);
    local_caches.recycle_adexpr_derivative_scratch(function_id, direction_count, derivative_slots);

    Ok(flat_directionals
        .iter()
        .map(|flat| collect_adexpr_output_vectors(flat, &output_nnz))
        .collect())
}

#[expect(
    clippy::too_many_arguments,
    reason = "callee forward propagation needs explicit caller and callee tangent metadata"
)]
fn execute_function_forward_batch_adexpr_from_call_by_id(
    function_id: FunctionId,
    inputs: &[Vec<AdExpr>],
    call_inputs: &[ProgramCallInput],
    selected_directions: &[usize],
    caller_direction_count: usize,
    caller_tangent_slots: &[AdExpr],
    arena: &mut AdExprArena,
    lower_memo: &mut HashMap<SX, AdExpr>,
    local_caches: &mut AdLocalCaches,
    function_plan_cache: &mut HashMap<FunctionId, FunctionAdExprArenaPlan>,
) -> Result<Vec<Vec<Vec<AdExpr>>>> {
    let function = function_by_id(function_id)
        .ok_or_else(|| SxError::Graph(format!("unknown function id {function_id}")))?;
    if inputs.len() != function.inputs().len() || call_inputs.len() != function.inputs().len() {
        return Err(SxError::Shape(format!(
            "function {} expected {} input slots, got {} values and {} call specs",
            function.name(),
            function.inputs().len(),
            inputs.len(),
            call_inputs.len()
        )));
    }

    ensure_function_adexpr_arena_plan(function_id, arena, lower_memo, function_plan_cache)?;
    let plan = function_plan_cache
        .get(&function_id)
        .expect("function plan inserted");
    let program = Arc::clone(&plan.program);
    let input_slots = plan.input_slots.clone();
    let output_nnz = plan.output_nnz.clone();

    let mut slot_values = local_caches.take_adexpr_slot_scratch(function_id, &plan.slot_exprs);
    for (slot, (actual, input_slots)) in inputs.iter().zip(input_slots.iter()).enumerate() {
        if actual.len() != input_slots.len() {
            return Err(SxError::Shape(format!(
                "function {} input slot {slot} expected {} nonzeros, got {}",
                function.name(),
                input_slots.len(),
                actual.len()
            )));
        }
        for (actual_nz, slot) in actual.iter().copied().zip(input_slots.iter()) {
            if let Some(slot) = slot {
                slot_values[*slot] = actual_nz;
            }
        }
    }
    let slot_values = execute_program_symbolically_adexpr_slots(
        &program,
        slot_values,
        arena,
        lower_memo,
        local_caches,
        function_plan_cache,
    )?;

    let direction_count = selected_directions.len();
    if direction_count == 0 {
        local_caches.recycle_adexpr_slot_scratch(function_id, slot_values);
        return Ok(Vec::new());
    }

    let slot_count = program.slot_exprs.len();
    let callee_slot_base = |slot: usize| slot * direction_count;
    let caller_slot_base = |slot: usize| slot * caller_direction_count;
    let zero = arena.zero();
    let mut derivative_slots = local_caches.take_adexpr_derivative_scratch(
        function_id,
        direction_count,
        slot_count * direction_count,
        zero,
    );
    let mut active_masks = (direction_count <= 64).then(|| vec![0_u64; slot_count]);
    for ((input, call_input), input_slots) in inputs
        .iter()
        .zip(call_inputs.iter())
        .zip(input_slots.iter())
    {
        if input.len() != call_input.slots.len() {
            return Err(SxError::Shape(format!(
                "function {} call input slot expected {} nonzeros, got {}",
                function.name(),
                call_input.slots.len(),
                input.len()
            )));
        }
        for &offset in &call_input.relevant_offsets {
            if let Some(slot) = input_slots[offset] {
                for (local_direction, &direction) in selected_directions.iter().enumerate() {
                    let tangent = caller_tangent_slots
                        [caller_slot_base(call_input.slots[offset]) + direction];
                    derivative_slots[callee_slot_base(slot) + local_direction] = tangent;
                    if let Some(masks) = active_masks.as_mut()
                        && !arena.is_zero(tangent)
                    {
                        masks[slot] |= 1_u64 << local_direction;
                    }
                }
            }
        }
    }

    let derivative_slots = execute_program_forward_batch_adexpr_with_slots(
        &program,
        &slot_values,
        derivative_slots,
        active_masks,
        direction_count,
        arena,
        lower_memo,
        local_caches,
        function_plan_cache,
    )?;

    let flat_directionals = (0..direction_count)
        .map(|direction| {
            program
                .output_slots
                .iter()
                .map(|&slot| derivative_slots[callee_slot_base(slot) + direction])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    local_caches.recycle_adexpr_slot_scratch(function_id, slot_values);
    local_caches.recycle_adexpr_derivative_scratch(function_id, direction_count, derivative_slots);

    Ok(flat_directionals
        .iter()
        .map(|flat| collect_adexpr_output_vectors(flat, &output_nnz))
        .collect())
}

#[allow(dead_code)]
fn execute_function_symbolically_by_id(
    function_id: FunctionId,
    inputs: &[SXMatrix],
) -> Result<Vec<SXMatrix>> {
    let expr_inputs = inputs
        .iter()
        .map(SXExprMatrix::from_sx_matrix)
        .collect::<Vec<_>>();
    execute_function_symbolically_expr_by_id(function_id, &expr_inputs)?
        .into_iter()
        .map(|output| output.to_sx_matrix())
        .collect()
}

impl NodeArena {
    fn new(context_id: u32) -> Self {
        Self {
            context_id,
            nodes: Vec::new(),
            keyed: HashMap::new(),
            next_symbol_serial: 0,
        }
    }

    fn node(&self, sx: SX) -> &Node {
        &self.nodes[sx.node_id() as usize]
    }

    fn node_kind_ref(&self, sx: SX) -> &NodeKind {
        &self.node(sx).kind
    }

    fn fresh_symbol(&mut self, name: impl Into<String>) -> SX {
        let serial = self.next_symbol_serial;
        self.next_symbol_serial += 1;
        let id = SX::from_parts(self.context_id, self.nodes.len() as u32);
        self.nodes.push(Node {
            kind: NodeKind::Symbol {
                serial,
                name: Arc::<str>::from(name.into()),
            },
        });
        id
    }

    fn intern_node(&mut self, kind: NodeKind) -> SX {
        let key = match &kind {
            NodeKind::Constant(value) => NodeKey::Constant(value.to_bits()),
            NodeKind::Symbol { .. } => {
                unreachable!("symbol nodes must be created via fresh_symbol")
            }
            NodeKind::Unary { op, arg } => NodeKey::Unary { op: *op, arg: *arg },
            NodeKind::Binary { op, lhs, rhs } => NodeKey::Binary {
                op: *op,
                lhs: *lhs,
                rhs: *rhs,
            },
            NodeKind::Call {
                function,
                inputs,
                output_slot,
                output_offset,
            } => NodeKey::Call {
                function: *function,
                inputs: inputs.clone(),
                output_slot: *output_slot,
                output_offset: *output_offset,
            },
        };
        if let Some(&existing) = self.keyed.get(&key) {
            return existing;
        }
        let id = SX::from_parts(self.context_id, self.nodes.len() as u32);
        self.nodes.push(Node { kind });
        self.keyed.insert(key, id);
        id
    }
}

pub(crate) fn node_kind(sx: SX) -> NodeKind {
    with_arena_ref_for_sx(sx, |arena| arena.node_kind_ref(sx).clone())
}

fn is_zero_kind(kind: &NodeKind) -> bool {
    matches!(kind, NodeKind::Constant(value) if *value == 0.0)
}

fn is_one_kind(kind: &NodeKind) -> bool {
    matches!(kind, NodeKind::Constant(value) if *value == 1.0)
}

fn is_neg_one_kind(kind: &NodeKind) -> bool {
    matches!(kind, NodeKind::Constant(value) if *value == -1.0)
}

fn constant_value_with_arena(arena: &NodeArena, sx: SX) -> Option<f64> {
    match arena.node_kind_ref(sx) {
        NodeKind::Constant(value) => Some(*value),
        NodeKind::Symbol { .. }
        | NodeKind::Unary { .. }
        | NodeKind::Binary { .. }
        | NodeKind::Call { .. } => None,
    }
}

fn mul_constant_factor_with_arena(arena: &NodeArena, sx: SX) -> Option<(f64, SX)> {
    match arena.node_kind_ref(sx) {
        NodeKind::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
        } => {
            if let Some(value) = constant_value_with_arena(arena, *lhs) {
                Some((value, *rhs))
            } else {
                constant_value_with_arena(arena, *rhs).map(|value| (value, *lhs))
            }
        }
        NodeKind::Constant(_value) => None,
        NodeKind::Symbol { .. }
        | NodeKind::Unary { .. }
        | NodeKind::Binary { .. }
        | NodeKind::Call { .. } => None,
    }
}

fn div_constant_factor_with_arena(arena: &NodeArena, sx: SX) -> Option<(SX, f64)> {
    match arena.node_kind_ref(sx) {
        NodeKind::Binary {
            op: BinaryOp::Div,
            lhs,
            rhs,
        } => constant_value_with_arena(arena, *rhs).map(|value| (*lhs, value)),
        NodeKind::Constant(_)
        | NodeKind::Symbol { .. }
        | NodeKind::Unary { .. }
        | NodeKind::Binary { .. }
        | NodeKind::Call { .. } => None,
    }
}

fn div_constant_numerator_with_arena(arena: &NodeArena, sx: SX) -> Option<(f64, SX)> {
    match arena.node_kind_ref(sx) {
        NodeKind::Binary {
            op: BinaryOp::Div,
            lhs,
            rhs,
        } => constant_value_with_arena(arena, *lhs).map(|value| (value, *rhs)),
        NodeKind::Constant(_)
        | NodeKind::Symbol { .. }
        | NodeKind::Unary { .. }
        | NodeKind::Binary { .. }
        | NodeKind::Call { .. } => None,
    }
}

fn append_constant_in_arena(arena: &mut NodeArena, value: f64) -> SX {
    arena.intern_node(NodeKind::Constant(value))
}

fn append_binary_in_arena(arena: &mut NodeArena, op: BinaryOp, lhs: SX, rhs: SX) -> SX {
    let (lhs, rhs) = if op.is_commutative() {
        canonical_pair(lhs, rhs)
    } else {
        (lhs, rhs)
    };
    arena.intern_node(NodeKind::Binary { op, lhs, rhs })
}

fn neg_in_arena(arena: &mut NodeArena, expr: SX) -> SX {
    if let Some(value) = constant_value_with_arena(arena, expr) {
        return append_constant_in_arena(arena, -value);
    }
    if let Some((value, factor)) = mul_constant_factor_with_arena(arena, expr)
        && value == -1.0
    {
        return factor;
    }
    let minus_one = append_constant_in_arena(arena, -1.0);
    append_binary_in_arena(arena, BinaryOp::Mul, minus_one, expr)
}

fn unary_in_arena(arena: &mut NodeArena, op: UnaryOp, arg: SX) -> SX {
    let arg_kind = arena.node_kind_ref(arg).clone();
    if let Some(value) = constant_value_with_arena(arena, arg) {
        return append_constant_in_arena(arena, op.apply_constant(value));
    }

    match op {
        UnaryOp::Abs => {
            if is_zero_kind(&arg_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
            if matches!(
                arg_kind,
                NodeKind::Unary {
                    op: UnaryOp::Abs,
                    ..
                }
            ) {
                return arg;
            }
        }
        UnaryOp::Sign => {
            if is_zero_kind(&arg_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
            if matches!(
                arg_kind,
                NodeKind::Unary {
                    op: UnaryOp::Sign,
                    ..
                }
            ) {
                return arg;
            }
        }
        UnaryOp::Floor
        | UnaryOp::Ceil
        | UnaryOp::Round
        | UnaryOp::Trunc
        | UnaryOp::Sin
        | UnaryOp::Tan
        | UnaryOp::Asin
        | UnaryOp::Atan
        | UnaryOp::Sinh
        | UnaryOp::Tanh
        | UnaryOp::Asinh => {
            if is_zero_kind(&arg_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
        }
        UnaryOp::Sqrt => {
            if is_zero_kind(&arg_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
            if is_one_kind(&arg_kind) {
                return append_constant_in_arena(arena, 1.0);
            }
        }
        UnaryOp::Exp | UnaryOp::Cos | UnaryOp::Cosh => {
            if is_zero_kind(&arg_kind) {
                return append_constant_in_arena(arena, 1.0);
            }
        }
        UnaryOp::Log => {
            if is_one_kind(&arg_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
        }
        UnaryOp::Acos | UnaryOp::Acosh | UnaryOp::Atanh => {}
    }

    arena.intern_node(NodeKind::Unary { op, arg })
}

fn binary_in_arena(arena: &mut NodeArena, op: BinaryOp, lhs: SX, rhs: SX) -> SX {
    use NodeKind as N;

    let lhs_kind = arena.node_kind_ref(lhs).clone();
    let rhs_kind = arena.node_kind_ref(rhs).clone();

    if let (N::Constant(a), N::Constant(b)) = (&lhs_kind, &rhs_kind) {
        return append_constant_in_arena(arena, op.apply_constant(*a, *b));
    }

    match op {
        BinaryOp::Add => {
            if is_zero_kind(&lhs_kind) {
                return rhs;
            }
            if is_zero_kind(&rhs_kind) {
                return lhs;
            }
            if let Some(combined) = combine_like_terms_shallow(arena, lhs, rhs, 1.0) {
                return combined;
            }
        }
        BinaryOp::Sub => {
            if is_zero_kind(&lhs_kind) {
                return neg_in_arena(arena, rhs);
            }
            if is_zero_kind(&rhs_kind) {
                return lhs;
            }
            if lhs == rhs {
                return append_constant_in_arena(arena, 0.0);
            }
            if let Some(combined) = combine_like_terms_shallow(arena, lhs, rhs, -1.0) {
                return combined;
            }
        }
        BinaryOp::Mul => {
            if is_zero_kind(&lhs_kind) || is_zero_kind(&rhs_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
            if is_neg_one_kind(&lhs_kind) {
                return neg_in_arena(arena, rhs);
            }
            if is_neg_one_kind(&rhs_kind) {
                return neg_in_arena(arena, lhs);
            }
            if is_one_kind(&lhs_kind) {
                return rhs;
            }
            if is_one_kind(&rhs_kind) {
                return lhs;
            }
            if let Some(simplified) = simplify_mul_shallow(arena, lhs, rhs) {
                return simplified;
            }
        }
        BinaryOp::Div => {
            if is_zero_kind(&lhs_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
            if is_one_kind(&rhs_kind) {
                return lhs;
            }
            if let Some(simplified) = simplify_div_shallow(arena, lhs, rhs) {
                return simplified;
            }
        }
        BinaryOp::Pow => {
            if is_zero_kind(&rhs_kind) {
                return append_constant_in_arena(arena, 1.0);
            }
            if is_one_kind(&rhs_kind) {
                return lhs;
            }
            if let Some(exponent) = constant_value_with_arena(arena, rhs) {
                if exponent == 2.0 {
                    return binary_in_arena(arena, BinaryOp::Mul, lhs, lhs);
                }
                if exponent == 0.5 {
                    return unary_in_arena(arena, UnaryOp::Sqrt, lhs);
                }
                if is_zero_kind(&lhs_kind) && exponent > 0.0 {
                    return append_constant_in_arena(arena, 0.0);
                }
            }
            if is_one_kind(&lhs_kind) {
                return append_constant_in_arena(arena, 1.0);
            }
        }
        BinaryOp::Atan2 => {
            if is_zero_kind(&lhs_kind) && is_one_kind(&rhs_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
        }
        BinaryOp::Hypot => {
            if is_zero_kind(&lhs_kind) {
                return unary_in_arena(arena, UnaryOp::Abs, rhs);
            }
            if is_zero_kind(&rhs_kind) {
                return unary_in_arena(arena, UnaryOp::Abs, lhs);
            }
        }
        BinaryOp::Mod => {
            if is_zero_kind(&lhs_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
        }
        BinaryOp::Copysign => {
            if is_zero_kind(&lhs_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
            if is_zero_kind(&rhs_kind) {
                return unary_in_arena(arena, UnaryOp::Abs, lhs);
            }
        }
    }

    append_binary_in_arena(arena, op, lhs, rhs)
}

#[allow(dead_code)]
fn neg_ad_in_arena(arena: &mut NodeArena, expr: SX) -> SX {
    let minus_one = append_constant_in_arena(arena, -1.0);
    binary_ad_in_arena(arena, BinaryOp::Mul, minus_one, expr)
}

#[allow(dead_code)]
fn binary_ad_in_arena(arena: &mut NodeArena, op: BinaryOp, lhs: SX, rhs: SX) -> SX {
    use NodeKind as N;

    let lhs_kind = arena.node_kind_ref(lhs).clone();
    let rhs_kind = arena.node_kind_ref(rhs).clone();

    if let (N::Constant(a), N::Constant(b)) = (&lhs_kind, &rhs_kind) {
        return append_constant_in_arena(arena, op.apply_constant(*a, *b));
    }

    match op {
        BinaryOp::Add => {
            if is_zero_kind(&lhs_kind) {
                return rhs;
            }
            if is_zero_kind(&rhs_kind) {
                return lhs;
            }
            let (lhs, rhs) = canonical_pair(lhs, rhs);
            return append_binary_in_arena(arena, op, lhs, rhs);
        }
        BinaryOp::Sub => {
            if is_zero_kind(&rhs_kind) {
                return lhs;
            }
            if is_zero_kind(&lhs_kind) {
                return neg_ad_in_arena(arena, rhs);
            }
        }
        BinaryOp::Mul => {
            if is_zero_kind(&lhs_kind) || is_zero_kind(&rhs_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
            if is_one_kind(&lhs_kind) {
                return rhs;
            }
            if is_one_kind(&rhs_kind) {
                return lhs;
            }
            let (lhs, rhs) = canonical_pair(lhs, rhs);
            return append_binary_in_arena(arena, op, lhs, rhs);
        }
        BinaryOp::Div => {
            if is_zero_kind(&lhs_kind) {
                return append_constant_in_arena(arena, 0.0);
            }
            if is_one_kind(&rhs_kind) {
                return lhs;
            }
        }
        BinaryOp::Pow => {
            if is_zero_kind(&rhs_kind) {
                return append_constant_in_arena(arena, 1.0);
            }
            if is_one_kind(&rhs_kind) {
                return lhs;
            }
            if is_one_kind(&lhs_kind) {
                return append_constant_in_arena(arena, 1.0);
            }
        }
        BinaryOp::Atan2 | BinaryOp::Hypot | BinaryOp::Mod | BinaryOp::Copysign => {
            if op.is_commutative() {
                let (lhs, rhs) = canonical_pair(lhs, rhs);
                return append_binary_in_arena(arena, op, lhs, rhs);
            }
        }
    }

    append_binary_in_arena(arena, op, lhs, rhs)
}

fn format_expression(expr: SX, arena: &NodeArena, memo: &mut HashMap<SX, String>) -> String {
    if let Some(existing) = memo.get(&expr) {
        return existing.clone();
    }
    let formatted = match arena.node(expr).kind.clone() {
        NodeKind::Constant(v) => {
            if v.fract() == 0.0 {
                format!("{v:.1}")
            } else {
                format!("{v}")
            }
        }
        NodeKind::Symbol { name, .. } => name.to_string(),
        NodeKind::Unary { op, arg } => {
            format!("{}({})", op.name(), format_expression(arg, arena, memo))
        }
        NodeKind::Binary { op, lhs, rhs } => {
            if op == BinaryOp::Mul {
                if matches!(arena.node(lhs).kind, NodeKind::Constant(value) if value == -1.0) {
                    let rendered = format_expression(rhs, arena, memo);
                    let negated = format!("(-{rendered})");
                    memo.insert(expr, negated.clone());
                    return negated;
                }
                if matches!(arena.node(rhs).kind, NodeKind::Constant(value) if value == -1.0) {
                    let rendered = format_expression(lhs, arena, memo);
                    let negated = format!("(-{rendered})");
                    memo.insert(expr, negated.clone());
                    return negated;
                }
            }
            let lhs_rendered = format_expression(lhs, arena, memo);
            let rhs_rendered = format_expression(rhs, arena, memo);
            if let Some(symbol) = op.symbol() {
                format!("({lhs_rendered} {symbol} {rhs_rendered})")
            } else {
                format!("{}({lhs_rendered}, {rhs_rendered})", op.name())
            }
        }
        NodeKind::Call {
            function,
            output_slot,
            output_offset,
            ..
        } => {
            let name = function_name(function).unwrap_or_else(|| format!("fn_{function}"));
            format!("{name}(..)[{output_slot}:{output_offset}]")
        }
    };
    memo.insert(expr, formatted.clone());
    formatted
}

pub(crate) fn call_output(
    function: FunctionId,
    inputs: Vec<SXMatrix>,
    output_slot: Index,
    output_offset: Index,
) -> SX {
    call_output_with_inputs(
        function,
        CallInputs::new(inputs),
        output_slot,
        output_offset,
    )
}

pub(crate) fn call_output_with_inputs(
    function: FunctionId,
    inputs: CallInputs,
    output_slot: Index,
    output_offset: Index,
) -> SX {
    let context_id = function_by_id(function)
        .unwrap_or_else(|| panic!("unknown function id {function}"))
        .context_id();
    with_arena_for_context(context_id, |arena| {
        arena.intern_node(NodeKind::Call {
            function,
            inputs,
            output_slot,
            output_offset,
        })
    })
}

fn canonical_pair(lhs: SX, rhs: SX) -> (SX, SX) {
    if lhs <= rhs { (lhs, rhs) } else { (rhs, lhs) }
}

fn unary(op: UnaryOp, arg: SX) -> SX {
    with_arena_for_sx(arg, |arena| unary_in_arena(arena, op, arg))
}

fn binary(op: BinaryOp, lhs: SX, rhs: SX) -> SX {
    let context_id = ensure_same_context(lhs, rhs);
    with_arena_for_context(context_id, |arena| binary_in_arena(arena, op, lhs, rhs))
}

fn unary_derivative(op: UnaryOp, arg: SX) -> SX {
    with_arena_for_sx(arg, |arena| unary_derivative_in_arena(arena, op, arg))
}

fn unary_derivative_in_arena(arena: &mut NodeArena, op: UnaryOp, arg: SX) -> SX {
    match op {
        UnaryOp::Abs => unary_in_arena(arena, UnaryOp::Sign, arg),
        UnaryOp::Sign | UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Trunc => {
            append_constant_in_arena(arena, 0.0)
        }
        UnaryOp::Sqrt => {
            let half = append_constant_in_arena(arena, 0.5);
            let sqrt = unary_in_arena(arena, UnaryOp::Sqrt, arg);
            binary_in_arena(arena, BinaryOp::Div, half, sqrt)
        }
        UnaryOp::Exp => unary_in_arena(arena, UnaryOp::Exp, arg),
        UnaryOp::Log => {
            let one = append_constant_in_arena(arena, 1.0);
            binary_in_arena(arena, BinaryOp::Div, one, arg)
        }
        UnaryOp::Sin => unary_in_arena(arena, UnaryOp::Cos, arg),
        UnaryOp::Cos => {
            let sin = unary_in_arena(arena, UnaryOp::Sin, arg);
            let neg_one = append_constant_in_arena(arena, -1.0);
            binary_in_arena(arena, BinaryOp::Mul, neg_one, sin)
        }
        UnaryOp::Tan => {
            let cos = unary_in_arena(arena, UnaryOp::Cos, arg);
            let cos_sq = binary_in_arena(arena, BinaryOp::Mul, cos, cos);
            let one = append_constant_in_arena(arena, 1.0);
            binary_in_arena(arena, BinaryOp::Div, one, cos_sq)
        }
        UnaryOp::Asin => {
            let arg_sq = binary_in_arena(arena, BinaryOp::Mul, arg, arg);
            let one = append_constant_in_arena(arena, 1.0);
            let radicand = binary_in_arena(arena, BinaryOp::Sub, one, arg_sq);
            let denom = unary_in_arena(arena, UnaryOp::Sqrt, radicand);
            let one = append_constant_in_arena(arena, 1.0);
            binary_in_arena(arena, BinaryOp::Div, one, denom)
        }
        UnaryOp::Acos => {
            let arg_sq = binary_in_arena(arena, BinaryOp::Mul, arg, arg);
            let one = append_constant_in_arena(arena, 1.0);
            let radicand = binary_in_arena(arena, BinaryOp::Sub, one, arg_sq);
            let denom = unary_in_arena(arena, UnaryOp::Sqrt, radicand);
            let neg_one = append_constant_in_arena(arena, -1.0);
            binary_in_arena(arena, BinaryOp::Div, neg_one, denom)
        }
        UnaryOp::Atan => {
            let arg_sq = binary_in_arena(arena, BinaryOp::Mul, arg, arg);
            let one = append_constant_in_arena(arena, 1.0);
            let denom = binary_in_arena(arena, BinaryOp::Add, one, arg_sq);
            let one = append_constant_in_arena(arena, 1.0);
            binary_in_arena(arena, BinaryOp::Div, one, denom)
        }
        UnaryOp::Sinh => unary_in_arena(arena, UnaryOp::Cosh, arg),
        UnaryOp::Cosh => unary_in_arena(arena, UnaryOp::Sinh, arg),
        UnaryOp::Tanh => {
            let cosh = unary_in_arena(arena, UnaryOp::Cosh, arg);
            let cosh_sq = binary_in_arena(arena, BinaryOp::Mul, cosh, cosh);
            let one = append_constant_in_arena(arena, 1.0);
            binary_in_arena(arena, BinaryOp::Div, one, cosh_sq)
        }
        UnaryOp::Asinh => {
            let arg_sq = binary_in_arena(arena, BinaryOp::Mul, arg, arg);
            let one = append_constant_in_arena(arena, 1.0);
            let radicand = binary_in_arena(arena, BinaryOp::Add, arg_sq, one);
            let denom = unary_in_arena(arena, UnaryOp::Sqrt, radicand);
            let one = append_constant_in_arena(arena, 1.0);
            binary_in_arena(arena, BinaryOp::Div, one, denom)
        }
        UnaryOp::Acosh => {
            let one = append_constant_in_arena(arena, 1.0);
            let arg_minus_one = binary_in_arena(arena, BinaryOp::Sub, arg, one);
            let left = unary_in_arena(arena, UnaryOp::Sqrt, arg_minus_one);
            let one = append_constant_in_arena(arena, 1.0);
            let arg_plus_one = binary_in_arena(arena, BinaryOp::Add, arg, one);
            let right = unary_in_arena(arena, UnaryOp::Sqrt, arg_plus_one);
            let denom = binary_in_arena(arena, BinaryOp::Mul, left, right);
            let one = append_constant_in_arena(arena, 1.0);
            binary_in_arena(arena, BinaryOp::Div, one, denom)
        }
        UnaryOp::Atanh => {
            let arg_sq = binary_in_arena(arena, BinaryOp::Mul, arg, arg);
            let one = append_constant_in_arena(arena, 1.0);
            let denom = binary_in_arena(arena, BinaryOp::Sub, one, arg_sq);
            let one = append_constant_in_arena(arena, 1.0);
            binary_in_arena(arena, BinaryOp::Div, one, denom)
        }
    }
}

fn binary_partials(op: BinaryOp, lhs: SX, rhs: SX) -> (SX, SX) {
    let context_id = ensure_same_context(lhs, rhs);
    with_arena_for_context(context_id, |arena| {
        binary_partials_in_arena(arena, op, lhs, rhs)
    })
}

fn binary_partials_in_arena(arena: &mut NodeArena, op: BinaryOp, lhs: SX, rhs: SX) -> (SX, SX) {
    match op {
        BinaryOp::Add => (
            append_constant_in_arena(arena, 1.0),
            append_constant_in_arena(arena, 1.0),
        ),
        BinaryOp::Sub => (
            append_constant_in_arena(arena, 1.0),
            append_constant_in_arena(arena, -1.0),
        ),
        BinaryOp::Mul => (rhs, lhs),
        BinaryOp::Div => {
            let one = append_constant_in_arena(arena, 1.0);
            let lhs_partial = binary_in_arena(arena, BinaryOp::Div, one, rhs);
            let rhs_sq = binary_in_arena(arena, BinaryOp::Mul, rhs, rhs);
            let neg_one = append_constant_in_arena(arena, -1.0);
            let neg_lhs = binary_in_arena(arena, BinaryOp::Mul, neg_one, lhs);
            let rhs_partial = binary_in_arena(arena, BinaryOp::Div, neg_lhs, rhs_sq);
            (lhs_partial, rhs_partial)
        }
        BinaryOp::Pow => {
            let pow = binary_in_arena(arena, BinaryOp::Pow, lhs, rhs);
            let one = append_constant_in_arena(arena, 1.0);
            let rhs_minus_one = binary_in_arena(arena, BinaryOp::Sub, rhs, one);
            let lhs_pow = binary_in_arena(arena, BinaryOp::Pow, lhs, rhs_minus_one);
            let lhs_partial = binary_in_arena(arena, BinaryOp::Mul, rhs, lhs_pow);
            let lhs_log = unary_in_arena(arena, UnaryOp::Log, lhs);
            let rhs_partial = binary_in_arena(arena, BinaryOp::Mul, pow, lhs_log);
            (lhs_partial, rhs_partial)
        }
        BinaryOp::Atan2 => {
            let lhs_sq = binary_in_arena(arena, BinaryOp::Mul, lhs, lhs);
            let rhs_sq = binary_in_arena(arena, BinaryOp::Mul, rhs, rhs);
            let denom = binary_in_arena(arena, BinaryOp::Add, lhs_sq, rhs_sq);
            let lhs_partial = binary_in_arena(arena, BinaryOp::Div, rhs, denom);
            let neg_one = append_constant_in_arena(arena, -1.0);
            let neg_lhs = binary_in_arena(arena, BinaryOp::Mul, neg_one, lhs);
            let rhs_partial = binary_in_arena(arena, BinaryOp::Div, neg_lhs, denom);
            (lhs_partial, rhs_partial)
        }
        BinaryOp::Hypot => {
            let hypot = binary_in_arena(arena, BinaryOp::Hypot, lhs, rhs);
            let lhs_partial = binary_in_arena(arena, BinaryOp::Div, lhs, hypot);
            let rhs_partial = binary_in_arena(arena, BinaryOp::Div, rhs, hypot);
            (lhs_partial, rhs_partial)
        }
        BinaryOp::Mod => {
            let lhs_over_rhs = binary_in_arena(arena, BinaryOp::Div, lhs, rhs);
            let trunc = unary_in_arena(arena, UnaryOp::Trunc, lhs_over_rhs);
            let neg_one = append_constant_in_arena(arena, -1.0);
            let rhs_partial = binary_in_arena(arena, BinaryOp::Mul, neg_one, trunc);
            let one = append_constant_in_arena(arena, 1.0);
            (one, rhs_partial)
        }
        BinaryOp::Copysign => {
            let rhs_sign = unary_in_arena(arena, UnaryOp::Sign, rhs);
            let rhs_sign_abs = unary_in_arena(arena, UnaryOp::Abs, rhs_sign);
            let one = append_constant_in_arena(arena, 1.0);
            let rhs_sign_term = binary_in_arena(arena, BinaryOp::Sub, one, rhs_sign_abs);
            let rhs_sign_full = binary_in_arena(arena, BinaryOp::Add, rhs_sign, rhs_sign_term);
            let lhs_sign = unary_in_arena(arena, UnaryOp::Sign, lhs);
            (
                binary_in_arena(arena, BinaryOp::Mul, lhs_sign, rhs_sign_full),
                append_constant_in_arena(arena, 0.0),
            )
        }
    }
}

pub(crate) fn greedy_color_disjoint(sets: &[Vec<Index>]) -> Vec<Vec<Index>> {
    let mut color_unions = Vec::<BTreeSet<Index>>::new();
    let mut color_members = Vec::<Vec<Index>>::new();
    for (member, set) in sets.iter().enumerate() {
        let color_idx = color_unions
            .iter()
            .position(|union| set.iter().all(|index| !union.contains(index)))
            .unwrap_or_else(|| {
                color_unions.push(BTreeSet::new());
                color_members.push(Vec::new());
                color_unions.len() - 1
            });
        for &index in set {
            color_unions[color_idx].insert(index);
        }
        color_members[color_idx].push(member);
    }
    color_members
}

fn derivative_dependency_mask(
    expr: SX,
    var_index_by_symbol: &HashMap<SX, usize>,
    memo: &mut HashMap<SX, DependencyMask>,
) -> Result<DependencyMask> {
    if let Some(existing) = memo.get(&expr) {
        return Ok(existing.clone());
    }

    let var_count = var_index_by_symbol.len();
    let mask = match node_kind(expr) {
        NodeKind::Constant(_) => DependencyMask::new(var_count),
        NodeKind::Symbol { .. } => match var_index_by_symbol.get(&expr).copied() {
            Some(index) => DependencyMask::singleton(var_count, index),
            None => DependencyMask::new(var_count),
        },
        NodeKind::Unary { op, arg } => {
            if unary_derivative(op, arg).is_zero() {
                DependencyMask::new(var_count)
            } else {
                derivative_dependency_mask(arg, var_index_by_symbol, memo)?
            }
        }
        NodeKind::Binary { op, lhs, rhs } => {
            let (d_lhs, d_rhs) = binary_partials(op, lhs, rhs);
            let mut mask = DependencyMask::new(var_count);
            if !d_lhs.is_zero() {
                mask.union_assign(&derivative_dependency_mask(lhs, var_index_by_symbol, memo)?);
            }
            if !d_rhs.is_zero() {
                mask.union_assign(&derivative_dependency_mask(rhs, var_index_by_symbol, memo)?);
            }
            mask
        }
        NodeKind::Call {
            function,
            inputs,
            output_slot,
            output_offset,
        } => {
            let helper = forward_helper(function)?;
            let helper_profile = dependency_profile(helper.id());
            let mut mask = DependencyMask::new(var_count);
            let input_count = inputs.iter().len();
            for (slot, input) in inputs.iter().enumerate() {
                let seed_slot = input_count + slot;
                for (offset, &value) in input.nonzeros().iter().enumerate() {
                    if helper_profile.output_depends_on(
                        output_slot,
                        output_offset,
                        seed_slot,
                        offset,
                    ) {
                        mask.union_assign(&derivative_dependency_mask(
                            value,
                            var_index_by_symbol,
                            memo,
                        )?);
                    }
                }
            }
            mask
        }
    };

    memo.insert(expr, mask.clone());
    Ok(mask)
}

pub(crate) fn jacobian_structure(outputs: &[SX], vars: &[SX]) -> Result<JacobianStructure> {
    let context_id = resolve_context_for_slices(&[("outputs", outputs), ("vars", vars)]);
    with_sx_context_id(context_id, || {
        let key = JacobianStructureKey {
            outputs: outputs.to_vec(),
            vars: vars.to_vec(),
        };
        if let Some(existing) = lock_jacobian_structure_cache().get(&key).cloned() {
            return Ok(existing);
        }

        let var_index_by_symbol = vars
            .iter()
            .copied()
            .enumerate()
            .map(|(index, symbol)| (symbol, index))
            .collect::<HashMap<_, _>>();
        let mut memo = HashMap::<SX, DependencyMask>::new();
        let mut rows_by_col = vec![Vec::<Index>::new(); vars.len()];
        for (row, &output) in outputs.iter().enumerate() {
            let mask = derivative_dependency_mask(output, &var_index_by_symbol, &mut memo)?;
            for col in mask.iter_ones() {
                rows_by_col[col].push(row);
            }
        }

        let mut col_ptrs = Vec::with_capacity(vars.len() + 1);
        let mut row_indices = Vec::new();
        col_ptrs.push(0);
        for rows in &rows_by_col {
            row_indices.extend(rows.iter().copied());
            col_ptrs.push(row_indices.len());
        }
        let ccs = CCS::new(outputs.len(), vars.len(), col_ptrs, row_indices)?;

        let mut positions_by_row = vec![Vec::<(Index, Index)>::new(); outputs.len()];
        for col in 0..vars.len() {
            for nz_index in ccs.col_ptrs()[col]..ccs.col_ptrs()[col + 1] {
                let row = ccs.row_indices()[nz_index];
                positions_by_row[row].push((col, nz_index));
            }
        }
        let row_columns = positions_by_row
            .iter()
            .map(|entries| entries.iter().map(|(col, _)| *col).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let structure = JacobianStructure {
            ccs,
            forward_color_groups: greedy_color_disjoint(&rows_by_col),
            reverse_color_groups: greedy_color_disjoint(&row_columns),
            positions_by_row,
        };
        lock_jacobian_structure_cache().insert(key, structure.clone());
        Ok(structure)
    })
}

fn execute_program_forward(program: &SxProgram, vars: &[SX], seeds: &[SX]) -> Result<Vec<SX>> {
    let mut derivative_slots = vec![SX::zero(); program.slot_exprs.len()];
    for (var, seed) in vars.iter().copied().zip(seeds.iter().copied()) {
        if let Some(&slot) = program.slot_by_node.get(&var) {
            derivative_slots[slot] = seed;
        }
    }

    let mut local_caches = AdLocalCaches::default();
    let mut call_memo = HashMap::<ForwardCallMemoKey, Vec<SXMatrix>>::new();
    for instruction in &program.instructions {
        match instruction {
            ProgramInstruction::Unary {
                result_slot,
                op,
                arg_slot,
                arg_expr,
            } => {
                with_arena(|arena| {
                    let derivative = unary_derivative_in_arena(arena, *op, *arg_expr);
                    derivative_slots[*result_slot] = binary_in_arena(
                        arena,
                        BinaryOp::Mul,
                        derivative_slots[*arg_slot],
                        derivative,
                    );
                });
            }
            ProgramInstruction::Binary {
                result_slot,
                op,
                lhs_slot,
                rhs_slot,
                lhs_expr,
                rhs_expr,
            } => {
                with_arena(|arena| {
                    derivative_slots[*result_slot] = match op {
                        BinaryOp::Add => binary_in_arena(
                            arena,
                            BinaryOp::Add,
                            derivative_slots[*lhs_slot],
                            derivative_slots[*rhs_slot],
                        ),
                        BinaryOp::Sub => binary_in_arena(
                            arena,
                            BinaryOp::Sub,
                            derivative_slots[*lhs_slot],
                            derivative_slots[*rhs_slot],
                        ),
                        BinaryOp::Mul => {
                            let left = binary_in_arena(
                                arena,
                                BinaryOp::Mul,
                                derivative_slots[*lhs_slot],
                                *rhs_expr,
                            );
                            let right = binary_in_arena(
                                arena,
                                BinaryOp::Mul,
                                *lhs_expr,
                                derivative_slots[*rhs_slot],
                            );
                            binary_in_arena(arena, BinaryOp::Add, left, right)
                        }
                        BinaryOp::Div => {
                            let left = binary_in_arena(
                                arena,
                                BinaryOp::Mul,
                                derivative_slots[*lhs_slot],
                                *rhs_expr,
                            );
                            let right = binary_in_arena(
                                arena,
                                BinaryOp::Mul,
                                *lhs_expr,
                                derivative_slots[*rhs_slot],
                            );
                            let numer = binary_in_arena(arena, BinaryOp::Sub, left, right);
                            let rhs_sq =
                                binary_in_arena(arena, BinaryOp::Mul, *rhs_expr, *rhs_expr);
                            binary_in_arena(arena, BinaryOp::Div, numer, rhs_sq)
                        }
                        BinaryOp::Pow
                        | BinaryOp::Atan2
                        | BinaryOp::Hypot
                        | BinaryOp::Mod
                        | BinaryOp::Copysign => {
                            let (d_lhs, d_rhs) =
                                binary_partials_in_arena(arena, *op, *lhs_expr, *rhs_expr);
                            let left = binary_in_arena(
                                arena,
                                BinaryOp::Mul,
                                derivative_slots[*lhs_slot],
                                d_lhs,
                            );
                            let right = binary_in_arena(
                                arena,
                                BinaryOp::Mul,
                                derivative_slots[*rhs_slot],
                                d_rhs,
                            );
                            binary_in_arena(arena, BinaryOp::Add, left, right)
                        }
                    };
                });
            }
            ProgramInstruction::Call {
                result_slot,
                function,
                site_key,
                output_slot,
                output_offset,
                inputs,
            } => {
                let has_relevant_seed = inputs.iter().any(|input| {
                    input
                        .relevant_offsets
                        .iter()
                        .any(|&offset| !derivative_slots[input.slots[offset]].is_zero())
                });

                derivative_slots[*result_slot] = if !has_relevant_seed {
                    SX::zero()
                } else {
                    let mut seed_inputs = Vec::with_capacity(inputs.len());
                    for input in inputs {
                        let mut seed_nonzeros = vec![SX::zero(); input.slots.len()];
                        for &offset in &input.relevant_offsets {
                            seed_nonzeros[offset] = derivative_slots[input.slots[offset]];
                        }
                        seed_inputs.push(SXMatrix::new(input.matrix.ccs().clone(), seed_nonzeros)?);
                    }
                    let memo_key = ForwardCallMemoKey {
                        site_key: site_key.clone(),
                        seed_inputs: CallInputs::new(seed_inputs.clone()),
                    };
                    if let Some(existing) = call_memo.get(&memo_key) {
                        existing[*output_slot].nz(*output_offset)
                    } else {
                        let helper = local_caches.forward_helper(*function)?;
                        let mut helper_inputs = inputs
                            .iter()
                            .map(|input| input.matrix.clone())
                            .collect::<Vec<_>>();
                        helper_inputs.extend(seed_inputs);
                        let helper_outputs = helper.call(&helper_inputs)?;
                        let selected = helper_outputs[*output_slot].nz(*output_offset);
                        call_memo.insert(memo_key, helper_outputs);
                        selected
                    }
                };
            }
        }
    }

    Ok(program
        .output_slots
        .iter()
        .map(|&slot| derivative_slots[slot])
        .collect())
}

fn execute_program_forward_batch(
    program: &SxProgram,
    vars: &[SX],
    seeds_by_direction: &[Vec<SX>],
) -> Result<Vec<Vec<SX>>> {
    let direction_count = seeds_by_direction.len();
    let slot_count = program.slot_exprs.len();
    let slot_base = |slot: usize| slot * direction_count;
    let mut derivative_slots = vec![SX::zero(); slot_count * direction_count];
    let mut active_masks = (direction_count <= 64).then(|| vec![0_u64; slot_count]);
    for (var_index, var) in vars.iter().copied().enumerate() {
        if let Some(&slot) = program.slot_by_node.get(&var) {
            let base = slot_base(slot);
            let mut slot_mask = 0_u64;
            for (direction, seeds) in seeds_by_direction.iter().enumerate() {
                let value = seeds[var_index];
                derivative_slots[base + direction] = value;
                if !value.is_zero() && direction < 64 {
                    slot_mask |= 1_u64 << direction;
                }
            }
            if let Some(masks) = active_masks.as_mut() {
                masks[slot] = slot_mask;
            }
        }
    }

    let derivative_slots = execute_program_forward_batch_with_slots(
        program,
        derivative_slots,
        active_masks,
        direction_count,
    )?;
    Ok(collect_program_output_directionals(
        program,
        &derivative_slots,
        direction_count,
    ))
}

fn execute_program_forward_basis_batch(
    program: &SxProgram,
    vars: &[SX],
    active_var_groups: &[Vec<Index>],
) -> Result<Vec<Vec<SX>>> {
    let direction_count = active_var_groups.len();
    let derivative_slots =
        execute_program_forward_basis_batch_slots(program, vars, active_var_groups)?;
    Ok(collect_program_output_directionals(
        program,
        &derivative_slots,
        direction_count,
    ))
}

fn execute_program_forward_batch_with_slots(
    program: &SxProgram,
    mut derivative_slots: Vec<SX>,
    mut active_masks: Option<Vec<u64>>,
    direction_count: usize,
) -> Result<Vec<SX>> {
    let slot_base = |slot: usize| slot * direction_count;

    let mut local_caches = AdLocalCaches::default();
    let mut call_memo = HashMap::<ForwardBatchCallCacheKey, Vec<SXMatrix>>::new();
    for instruction in &program.instructions {
        match instruction {
            ProgramInstruction::Unary {
                result_slot,
                op,
                arg_slot,
                arg_expr,
            } => {
                let result_base = slot_base(*result_slot);
                let arg_base = slot_base(*arg_slot);
                if let Some(masks) = active_masks.as_mut() {
                    let arg_mask = masks[*arg_slot];
                    masks[*result_slot] = arg_mask;
                    if arg_mask == 0 {
                        continue;
                    }
                    with_arena(|arena| {
                        let derivative = unary_derivative_in_arena(arena, *op, *arg_expr);
                        for direction in iter_direction_bits(arg_mask) {
                            derivative_slots[result_base + direction] = binary_in_arena(
                                arena,
                                BinaryOp::Mul,
                                derivative_slots[arg_base + direction],
                                derivative,
                            );
                        }
                    });
                } else {
                    with_arena(|arena| {
                        let derivative = unary_derivative_in_arena(arena, *op, *arg_expr);
                        for direction in 0..direction_count {
                            derivative_slots[result_base + direction] = binary_in_arena(
                                arena,
                                BinaryOp::Mul,
                                derivative_slots[arg_base + direction],
                                derivative,
                            );
                        }
                    });
                }
            }
            ProgramInstruction::Binary {
                result_slot,
                op,
                lhs_slot,
                rhs_slot,
                lhs_expr,
                rhs_expr,
            } => {
                let result_base = slot_base(*result_slot);
                let lhs_base = slot_base(*lhs_slot);
                let rhs_base = slot_base(*rhs_slot);
                if let Some(masks) = active_masks.as_mut() {
                    let lhs_mask = masks[*lhs_slot];
                    let rhs_mask = masks[*rhs_slot];
                    let result_mask = lhs_mask | rhs_mask;
                    masks[*result_slot] = result_mask;
                    if result_mask == 0 {
                        continue;
                    }
                    with_arena(|arena| match op {
                        BinaryOp::Add => {
                            for direction in iter_direction_bits(result_mask) {
                                derivative_slots[result_base + direction] = binary_in_arena(
                                    arena,
                                    BinaryOp::Add,
                                    derivative_slots[lhs_base + direction],
                                    derivative_slots[rhs_base + direction],
                                );
                            }
                        }
                        BinaryOp::Sub => {
                            for direction in iter_direction_bits(result_mask) {
                                derivative_slots[result_base + direction] = binary_in_arena(
                                    arena,
                                    BinaryOp::Sub,
                                    derivative_slots[lhs_base + direction],
                                    derivative_slots[rhs_base + direction],
                                );
                            }
                        }
                        BinaryOp::Mul => {
                            for direction in iter_direction_bits(result_mask) {
                                let left = binary_in_arena(
                                    arena,
                                    BinaryOp::Mul,
                                    derivative_slots[lhs_base + direction],
                                    *rhs_expr,
                                );
                                let right = binary_in_arena(
                                    arena,
                                    BinaryOp::Mul,
                                    *lhs_expr,
                                    derivative_slots[rhs_base + direction],
                                );
                                derivative_slots[result_base + direction] =
                                    binary_in_arena(arena, BinaryOp::Add, left, right);
                            }
                        }
                        BinaryOp::Div => {
                            let rhs_sq =
                                binary_in_arena(arena, BinaryOp::Mul, *rhs_expr, *rhs_expr);
                            for direction in iter_direction_bits(result_mask) {
                                let left = binary_in_arena(
                                    arena,
                                    BinaryOp::Mul,
                                    derivative_slots[lhs_base + direction],
                                    *rhs_expr,
                                );
                                let right = binary_in_arena(
                                    arena,
                                    BinaryOp::Mul,
                                    *lhs_expr,
                                    derivative_slots[rhs_base + direction],
                                );
                                let numer = binary_in_arena(arena, BinaryOp::Sub, left, right);
                                derivative_slots[result_base + direction] =
                                    binary_in_arena(arena, BinaryOp::Div, numer, rhs_sq);
                            }
                        }
                        BinaryOp::Pow
                        | BinaryOp::Atan2
                        | BinaryOp::Hypot
                        | BinaryOp::Mod
                        | BinaryOp::Copysign => {
                            let (d_lhs, d_rhs) =
                                binary_partials_in_arena(arena, *op, *lhs_expr, *rhs_expr);
                            for direction in iter_direction_bits(result_mask) {
                                let left = binary_in_arena(
                                    arena,
                                    BinaryOp::Mul,
                                    derivative_slots[lhs_base + direction],
                                    d_lhs,
                                );
                                let right = binary_in_arena(
                                    arena,
                                    BinaryOp::Mul,
                                    derivative_slots[rhs_base + direction],
                                    d_rhs,
                                );
                                derivative_slots[result_base + direction] =
                                    binary_in_arena(arena, BinaryOp::Add, left, right);
                            }
                        }
                    });
                } else {
                    with_arena(|arena| match op {
                        BinaryOp::Add => {
                            for direction in 0..direction_count {
                                derivative_slots[result_base + direction] = binary_in_arena(
                                    arena,
                                    BinaryOp::Add,
                                    derivative_slots[lhs_base + direction],
                                    derivative_slots[rhs_base + direction],
                                );
                            }
                        }
                        BinaryOp::Sub => {
                            for direction in 0..direction_count {
                                derivative_slots[result_base + direction] = binary_in_arena(
                                    arena,
                                    BinaryOp::Sub,
                                    derivative_slots[lhs_base + direction],
                                    derivative_slots[rhs_base + direction],
                                );
                            }
                        }
                        BinaryOp::Mul => {
                            for direction in 0..direction_count {
                                let left = binary_in_arena(
                                    arena,
                                    BinaryOp::Mul,
                                    derivative_slots[lhs_base + direction],
                                    *rhs_expr,
                                );
                                let right = binary_in_arena(
                                    arena,
                                    BinaryOp::Mul,
                                    *lhs_expr,
                                    derivative_slots[rhs_base + direction],
                                );
                                derivative_slots[result_base + direction] =
                                    binary_in_arena(arena, BinaryOp::Add, left, right);
                            }
                        }
                        BinaryOp::Div => {
                            let rhs_sq =
                                binary_in_arena(arena, BinaryOp::Mul, *rhs_expr, *rhs_expr);
                            for direction in 0..direction_count {
                                let left = binary_in_arena(
                                    arena,
                                    BinaryOp::Mul,
                                    derivative_slots[lhs_base + direction],
                                    *rhs_expr,
                                );
                                let right = binary_in_arena(
                                    arena,
                                    BinaryOp::Mul,
                                    *lhs_expr,
                                    derivative_slots[rhs_base + direction],
                                );
                                let numer = binary_in_arena(arena, BinaryOp::Sub, left, right);
                                derivative_slots[result_base + direction] =
                                    binary_in_arena(arena, BinaryOp::Div, numer, rhs_sq);
                            }
                        }
                        BinaryOp::Pow
                        | BinaryOp::Atan2
                        | BinaryOp::Hypot
                        | BinaryOp::Mod
                        | BinaryOp::Copysign => {
                            let (d_lhs, d_rhs) =
                                binary_partials_in_arena(arena, *op, *lhs_expr, *rhs_expr);
                            for direction in 0..direction_count {
                                let left = binary_in_arena(
                                    arena,
                                    BinaryOp::Mul,
                                    derivative_slots[lhs_base + direction],
                                    d_lhs,
                                );
                                let right = binary_in_arena(
                                    arena,
                                    BinaryOp::Mul,
                                    derivative_slots[rhs_base + direction],
                                    d_rhs,
                                );
                                derivative_slots[result_base + direction] =
                                    binary_in_arena(arena, BinaryOp::Add, left, right);
                            }
                        }
                    });
                }
            }
            ProgramInstruction::Call {
                result_slot,
                function,
                site_key,
                output_slot,
                output_offset,
                inputs,
            } => {
                let result_base = slot_base(*result_slot);
                let result_values = if let Some(masks) = active_masks.as_mut() {
                    let mut direction_mask = 0_u64;
                    for input in inputs {
                        for &offset in &input.relevant_offsets {
                            direction_mask |= masks[input.slots[offset]];
                        }
                    }
                    masks[*result_slot] = direction_mask;
                    if direction_mask == 0 {
                        vec![SX::zero(); direction_count]
                    } else {
                        let active_directions =
                            iter_direction_bits(direction_mask).collect::<Vec<_>>();
                        let active_count = active_directions.len();
                        let mut seed_inputs_by_direction = Vec::with_capacity(active_count);
                        for &direction in &active_directions {
                            let mut direction_seed_inputs = Vec::with_capacity(inputs.len());
                            for input in inputs {
                                let mut seed_nonzeros = vec![SX::zero(); input.slots.len()];
                                for &offset in &input.relevant_offsets {
                                    seed_nonzeros[offset] = derivative_slots
                                        [slot_base(input.slots[offset]) + direction];
                                }
                                direction_seed_inputs.push(SXMatrix::new(
                                    input.matrix.ccs().clone(),
                                    seed_nonzeros,
                                )?);
                            }
                            seed_inputs_by_direction.push(direction_seed_inputs);
                        }
                        let memo_seed_inputs = seed_inputs_by_direction
                            .iter()
                            .flat_map(|direction_seed_inputs| direction_seed_inputs.iter().cloned())
                            .collect::<Vec<_>>();
                        let key = ForwardBatchCallCacheKey {
                            site_key: site_key.clone(),
                            direction_count: active_count,
                            direction_mask,
                            seed_inputs: CallInputs::new(memo_seed_inputs),
                        };
                        if let Some(existing) = call_memo.get(&key) {
                            let mut selected = vec![SX::zero(); direction_count];
                            for (local_direction, &direction) in
                                active_directions.iter().enumerate()
                            {
                                selected[direction] = existing
                                    [*output_slot * active_count + local_direction]
                                    .nz(*output_offset);
                            }
                            selected
                        } else {
                            let helper =
                                local_caches.forward_batch_helper(*function, active_count)?;
                            let mut helper_inputs = inputs
                                .iter()
                                .map(|input| input.matrix.clone())
                                .collect::<Vec<_>>();
                            for direction_seed_inputs in &seed_inputs_by_direction {
                                helper_inputs.extend(direction_seed_inputs.iter().cloned());
                            }
                            let helper_outputs = helper.call(&helper_inputs)?;
                            let mut selected = vec![SX::zero(); direction_count];
                            for (local_direction, &direction) in
                                active_directions.iter().enumerate()
                            {
                                selected[direction] = helper_outputs
                                    [*output_slot * active_count + local_direction]
                                    .nz(*output_offset);
                            }
                            call_memo.insert(key, helper_outputs);
                            selected
                        }
                    }
                } else {
                    let has_relevant_seed = (0..direction_count).any(|direction| {
                        inputs.iter().any(|input| {
                            input.relevant_offsets.iter().any(|&offset| {
                                !derivative_slots[slot_base(input.slots[offset]) + direction]
                                    .is_zero()
                            })
                        })
                    });
                    if !has_relevant_seed {
                        vec![SX::zero(); direction_count]
                    } else {
                        let mut seed_inputs_by_direction = Vec::with_capacity(direction_count);
                        for direction in 0..direction_count {
                            let mut direction_seed_inputs = Vec::with_capacity(inputs.len());
                            for input in inputs {
                                let mut seed_nonzeros = vec![SX::zero(); input.slots.len()];
                                for &offset in &input.relevant_offsets {
                                    seed_nonzeros[offset] = derivative_slots
                                        [slot_base(input.slots[offset]) + direction];
                                }
                                direction_seed_inputs.push(SXMatrix::new(
                                    input.matrix.ccs().clone(),
                                    seed_nonzeros,
                                )?);
                            }
                            seed_inputs_by_direction.push(direction_seed_inputs);
                        }
                        let memo_seed_inputs = seed_inputs_by_direction
                            .iter()
                            .flat_map(|direction_seed_inputs| direction_seed_inputs.iter().cloned())
                            .collect::<Vec<_>>();
                        let key = ForwardBatchCallCacheKey {
                            site_key: site_key.clone(),
                            direction_count,
                            direction_mask: u64::MAX,
                            seed_inputs: CallInputs::new(memo_seed_inputs),
                        };
                        if let Some(existing) = call_memo.get(&key) {
                            (0..direction_count)
                                .map(|direction| {
                                    existing[*output_slot * direction_count + direction]
                                        .nz(*output_offset)
                                })
                                .collect()
                        } else {
                            let helper =
                                local_caches.forward_batch_helper(*function, direction_count)?;
                            let mut helper_inputs = inputs
                                .iter()
                                .map(|input| input.matrix.clone())
                                .collect::<Vec<_>>();
                            for direction_seed_inputs in &seed_inputs_by_direction {
                                helper_inputs.extend(direction_seed_inputs.iter().cloned());
                            }
                            let helper_outputs = helper.call(&helper_inputs)?;
                            let selected = (0..direction_count)
                                .map(|direction| {
                                    helper_outputs[*output_slot * direction_count + direction]
                                        .nz(*output_offset)
                                })
                                .collect::<Vec<_>>();
                            call_memo.insert(key, helper_outputs);
                            selected
                        }
                    }
                };
                derivative_slots[result_base..(direction_count + result_base)]
                    .copy_from_slice(&result_values[..direction_count]);
            }
        }
    }

    Ok(derivative_slots)
}

fn collect_program_output_directionals(
    program: &SxProgram,
    derivative_slots: &[SX],
    direction_count: usize,
) -> Vec<Vec<SX>> {
    let slot_base = |slot: usize| slot * direction_count;
    (0..direction_count)
        .map(|direction| {
            program
                .output_slots
                .iter()
                .map(|&slot| derivative_slots[slot_base(slot) + direction])
                .collect::<Vec<_>>()
        })
        .collect()
}

fn execute_program_forward_basis_batch_slots(
    program: &SxProgram,
    vars: &[SX],
    active_var_groups: &[Vec<Index>],
) -> Result<Vec<SX>> {
    let direction_count = active_var_groups.len();
    let slot_count = program.slot_exprs.len();
    let slot_base = |slot: usize| slot * direction_count;
    let mut derivative_slots = vec![SX::zero(); slot_count * direction_count];
    let mut active_masks = (direction_count <= 64).then(|| vec![0_u64; slot_count]);
    for (direction, active_vars) in active_var_groups.iter().enumerate() {
        for &var_index in active_vars {
            if var_index >= vars.len() {
                return Err(SxError::Shape(format!(
                    "forward basis variable index {} out of range for variable length {}",
                    var_index,
                    vars.len()
                )));
            }
            if let Some(&slot) = program.slot_by_node.get(&vars[var_index]) {
                derivative_slots[slot_base(slot) + direction] = SX::one();
                if let Some(masks) = active_masks.as_mut() {
                    masks[slot] |= 1_u64 << direction;
                }
            }
        }
    }

    execute_program_forward_batch_with_slots(
        program,
        derivative_slots,
        active_masks,
        direction_count,
    )
}

#[allow(dead_code)]
fn unary_second_directional_with_arena(
    arena: &mut NodeArena,
    op: UnaryOp,
    arg: SX,
    arg_tangent: SX,
) -> SX {
    let zero = append_constant_in_arena(arena, 0.0);
    if is_zero_kind(arena.node_kind_ref(arg_tangent)) {
        return zero;
    }
    match op {
        UnaryOp::Abs
        | UnaryOp::Sign
        | UnaryOp::Floor
        | UnaryOp::Ceil
        | UnaryOp::Round
        | UnaryOp::Trunc => zero,
        UnaryOp::Sqrt => {
            let neg_quarter = append_constant_in_arena(arena, -0.25);
            let sqrt_arg = unary_in_arena(arena, UnaryOp::Sqrt, arg);
            let denom = binary_ad_in_arena(arena, BinaryOp::Mul, arg, sqrt_arg);
            let scaled = binary_ad_in_arena(arena, BinaryOp::Mul, neg_quarter, arg_tangent);
            binary_ad_in_arena(arena, BinaryOp::Div, scaled, denom)
        }
        UnaryOp::Exp => {
            let exp_arg = unary_in_arena(arena, UnaryOp::Exp, arg);
            binary_ad_in_arena(arena, BinaryOp::Mul, exp_arg, arg_tangent)
        }
        UnaryOp::Log => {
            let neg_one = append_constant_in_arena(arena, -1.0);
            let numer = binary_ad_in_arena(arena, BinaryOp::Mul, neg_one, arg_tangent);
            let denom = binary_ad_in_arena(arena, BinaryOp::Mul, arg, arg);
            binary_ad_in_arena(arena, BinaryOp::Div, numer, denom)
        }
        UnaryOp::Sin => {
            let sin_arg = unary_in_arena(arena, UnaryOp::Sin, arg);
            let neg_sin = neg_ad_in_arena(arena, sin_arg);
            binary_ad_in_arena(arena, BinaryOp::Mul, neg_sin, arg_tangent)
        }
        UnaryOp::Cos => {
            let cos_arg = unary_in_arena(arena, UnaryOp::Cos, arg);
            let neg_cos = neg_ad_in_arena(arena, cos_arg);
            binary_ad_in_arena(arena, BinaryOp::Mul, neg_cos, arg_tangent)
        }
        UnaryOp::Tan => {
            let two = append_constant_in_arena(arena, 2.0);
            let tan_arg = unary_in_arena(arena, UnaryOp::Tan, arg);
            let cos_arg = unary_in_arena(arena, UnaryOp::Cos, arg);
            let cos_sq = binary_ad_in_arena(arena, BinaryOp::Mul, cos_arg, cos_arg);
            let factor = binary_ad_in_arena(arena, BinaryOp::Mul, two, tan_arg);
            let factor = binary_ad_in_arena(arena, BinaryOp::Div, factor, cos_sq);
            binary_ad_in_arena(arena, BinaryOp::Mul, factor, arg_tangent)
        }
        UnaryOp::Asin => {
            let one = append_constant_in_arena(arena, 1.0);
            let arg_sq = binary_ad_in_arena(arena, BinaryOp::Mul, arg, arg);
            let radicand = binary_ad_in_arena(arena, BinaryOp::Sub, one, arg_sq);
            let sqrt = unary_in_arena(arena, UnaryOp::Sqrt, radicand);
            let denom = binary_ad_in_arena(arena, BinaryOp::Mul, radicand, sqrt);
            let numer = binary_ad_in_arena(arena, BinaryOp::Mul, arg, arg_tangent);
            binary_ad_in_arena(arena, BinaryOp::Div, numer, denom)
        }
        UnaryOp::Acos => {
            let asin_second =
                unary_second_directional_with_arena(arena, UnaryOp::Asin, arg, arg_tangent);
            neg_ad_in_arena(arena, asin_second)
        }
        UnaryOp::Atan => {
            let neg_two = append_constant_in_arena(arena, -2.0);
            let arg_sq = binary_ad_in_arena(arena, BinaryOp::Mul, arg, arg);
            let one = append_constant_in_arena(arena, 1.0);
            let denom_base = binary_ad_in_arena(arena, BinaryOp::Add, one, arg_sq);
            let denom = binary_ad_in_arena(arena, BinaryOp::Mul, denom_base, denom_base);
            let numer = binary_ad_in_arena(arena, BinaryOp::Mul, arg, arg_tangent);
            let numer = binary_ad_in_arena(arena, BinaryOp::Mul, neg_two, numer);
            binary_ad_in_arena(arena, BinaryOp::Div, numer, denom)
        }
        UnaryOp::Sinh => {
            let sinh_arg = unary_in_arena(arena, UnaryOp::Sinh, arg);
            binary_ad_in_arena(arena, BinaryOp::Mul, sinh_arg, arg_tangent)
        }
        UnaryOp::Cosh => {
            let cosh_arg = unary_in_arena(arena, UnaryOp::Cosh, arg);
            binary_ad_in_arena(arena, BinaryOp::Mul, cosh_arg, arg_tangent)
        }
        UnaryOp::Tanh => {
            let neg_two = append_constant_in_arena(arena, -2.0);
            let tanh_arg = unary_in_arena(arena, UnaryOp::Tanh, arg);
            let cosh_arg = unary_in_arena(arena, UnaryOp::Cosh, arg);
            let cosh_sq = binary_ad_in_arena(arena, BinaryOp::Mul, cosh_arg, cosh_arg);
            let factor = binary_ad_in_arena(arena, BinaryOp::Mul, neg_two, tanh_arg);
            let factor = binary_ad_in_arena(arena, BinaryOp::Div, factor, cosh_sq);
            binary_ad_in_arena(arena, BinaryOp::Mul, factor, arg_tangent)
        }
        UnaryOp::Asinh => {
            let one = append_constant_in_arena(arena, 1.0);
            let arg_sq = binary_ad_in_arena(arena, BinaryOp::Mul, arg, arg);
            let radicand = binary_ad_in_arena(arena, BinaryOp::Add, arg_sq, one);
            let sqrt = unary_in_arena(arena, UnaryOp::Sqrt, radicand);
            let denom = binary_ad_in_arena(arena, BinaryOp::Mul, radicand, sqrt);
            let numer = binary_ad_in_arena(arena, BinaryOp::Mul, arg, arg_tangent);
            let neg_numer = neg_ad_in_arena(arena, numer);
            binary_ad_in_arena(arena, BinaryOp::Div, neg_numer, denom)
        }
        UnaryOp::Acosh => {
            let one = append_constant_in_arena(arena, 1.0);
            let arg_sq = binary_ad_in_arena(arena, BinaryOp::Mul, arg, arg);
            let denom_base = binary_ad_in_arena(arena, BinaryOp::Sub, arg_sq, one);
            let sqrt = unary_in_arena(arena, UnaryOp::Sqrt, denom_base);
            let denom = binary_ad_in_arena(arena, BinaryOp::Mul, denom_base, sqrt);
            let numer = binary_ad_in_arena(arena, BinaryOp::Mul, arg, arg_tangent);
            let neg_numer = neg_ad_in_arena(arena, numer);
            binary_ad_in_arena(arena, BinaryOp::Div, neg_numer, denom)
        }
        UnaryOp::Atanh => {
            let two = append_constant_in_arena(arena, 2.0);
            let one = append_constant_in_arena(arena, 1.0);
            let arg_sq = binary_ad_in_arena(arena, BinaryOp::Mul, arg, arg);
            let denom_base = binary_ad_in_arena(arena, BinaryOp::Sub, one, arg_sq);
            let denom = binary_ad_in_arena(arena, BinaryOp::Mul, denom_base, denom_base);
            let numer = binary_ad_in_arena(arena, BinaryOp::Mul, arg, arg_tangent);
            let numer = binary_ad_in_arena(arena, BinaryOp::Mul, two, numer);
            binary_ad_in_arena(arena, BinaryOp::Div, numer, denom)
        }
    }
}

#[allow(dead_code)]
fn binary_partial_directionals_with_arena(
    arena: &mut NodeArena,
    op: BinaryOp,
    lhs: SX,
    rhs: SX,
    lhs_tangent: SX,
    rhs_tangent: SX,
) -> (SX, SX) {
    let zero = append_constant_in_arena(arena, 0.0);
    match op {
        BinaryOp::Add | BinaryOp::Sub => (zero, zero),
        BinaryOp::Mul => (rhs_tangent, lhs_tangent),
        BinaryOp::Div => {
            let rhs_sq = binary_ad_in_arena(arena, BinaryOp::Mul, rhs, rhs);
            let lhs_dir = {
                let neg_rhs_t = neg_ad_in_arena(arena, rhs_tangent);
                binary_ad_in_arena(arena, BinaryOp::Div, neg_rhs_t, rhs_sq)
            };
            let rhs_dir = {
                let lhs_over_rhs_sq = binary_ad_in_arena(arena, BinaryOp::Div, lhs_tangent, rhs_sq);
                let left = neg_ad_in_arena(arena, lhs_over_rhs_sq);
                let two = append_constant_in_arena(arena, 2.0);
                let rhs_cu = binary_ad_in_arena(arena, BinaryOp::Mul, rhs_sq, rhs);
                let numer = binary_ad_in_arena(arena, BinaryOp::Mul, lhs, rhs_tangent);
                let numer = binary_ad_in_arena(arena, BinaryOp::Mul, two, numer);
                let right = binary_ad_in_arena(arena, BinaryOp::Div, numer, rhs_cu);
                binary_ad_in_arena(arena, BinaryOp::Add, left, right)
            };
            (lhs_dir, rhs_dir)
        }
        BinaryOp::Pow => {
            let pow = binary_ad_in_arena(arena, BinaryOp::Pow, lhs, rhs);
            let log_lhs = unary_in_arena(arena, UnaryOp::Log, lhs);
            let one = append_constant_in_arena(arena, 1.0);
            let lhs_inv = binary_ad_in_arena(arena, BinaryOp::Div, one, lhs);
            let rhs_lhs = binary_ad_in_arena(arena, BinaryOp::Mul, rhs, lhs_tangent);
            let rhs_lhs = binary_ad_in_arena(arena, BinaryOp::Mul, rhs_lhs, lhs_inv);
            let rhs_log = binary_ad_in_arena(arena, BinaryOp::Mul, rhs_tangent, log_lhs);
            let pow_factor = binary_ad_in_arena(arena, BinaryOp::Add, rhs_log, rhs_lhs);
            let pow_t = binary_ad_in_arena(arena, BinaryOp::Mul, pow, pow_factor);

            let rhs_lhs_inv = binary_ad_in_arena(arena, BinaryOp::Mul, rhs, lhs_inv);
            let lhs_term1 = binary_ad_in_arena(arena, BinaryOp::Mul, pow_t, rhs_lhs_inv);
            let rhs_tangent_lhs_inv =
                binary_ad_in_arena(arena, BinaryOp::Mul, rhs_tangent, lhs_inv);
            let lhs_term2 = binary_ad_in_arena(arena, BinaryOp::Mul, pow, rhs_tangent_lhs_inv);
            let lhs_inv_sq = binary_ad_in_arena(arena, BinaryOp::Mul, lhs_inv, lhs_inv);
            let rhs_lhs_tangent = binary_ad_in_arena(arena, BinaryOp::Mul, rhs, lhs_tangent);
            let rhs_lhs_tangent_inv_sq =
                binary_ad_in_arena(arena, BinaryOp::Mul, rhs_lhs_tangent, lhs_inv_sq);
            let lhs_term3_inner =
                binary_ad_in_arena(arena, BinaryOp::Mul, pow, rhs_lhs_tangent_inv_sq);
            let lhs_term3 = neg_ad_in_arena(arena, lhs_term3_inner);
            let lhs_terms = binary_ad_in_arena(arena, BinaryOp::Add, lhs_term1, lhs_term2);
            let lhs_dir = binary_ad_in_arena(arena, BinaryOp::Add, lhs_terms, lhs_term3);

            let rhs_term1 = binary_ad_in_arena(arena, BinaryOp::Mul, pow_t, log_lhs);
            let lhs_tangent_lhs_inv =
                binary_ad_in_arena(arena, BinaryOp::Mul, lhs_tangent, lhs_inv);
            let rhs_term2 = binary_ad_in_arena(arena, BinaryOp::Mul, pow, lhs_tangent_lhs_inv);
            let rhs_dir = binary_ad_in_arena(arena, BinaryOp::Add, rhs_term1, rhs_term2);
            (lhs_dir, rhs_dir)
        }
        BinaryOp::Atan2 => {
            let two = append_constant_in_arena(arena, 2.0);
            let lhs_sq = binary_ad_in_arena(arena, BinaryOp::Mul, lhs, lhs);
            let rhs_sq = binary_ad_in_arena(arena, BinaryOp::Mul, rhs, rhs);
            let denom = binary_ad_in_arena(arena, BinaryOp::Add, lhs_sq, rhs_sq);
            let denom_sq = binary_ad_in_arena(arena, BinaryOp::Mul, denom, denom);
            let two_lhs_lhs_tangent = binary_ad_in_arena(arena, BinaryOp::Mul, lhs, lhs_tangent);
            let two_lhs_lhs_tangent =
                binary_ad_in_arena(arena, BinaryOp::Mul, two, two_lhs_lhs_tangent);
            let two_rhs_rhs_tangent = binary_ad_in_arena(arena, BinaryOp::Mul, rhs, rhs_tangent);
            let two_rhs_rhs_tangent =
                binary_ad_in_arena(arena, BinaryOp::Mul, two, two_rhs_rhs_tangent);
            let denom_t = binary_ad_in_arena(
                arena,
                BinaryOp::Add,
                two_lhs_lhs_tangent,
                two_rhs_rhs_tangent,
            );
            let lhs_dir = {
                let left = binary_ad_in_arena(arena, BinaryOp::Div, rhs_tangent, denom);
                let rhs_denom_t = binary_ad_in_arena(arena, BinaryOp::Mul, rhs, denom_t);
                let right = binary_ad_in_arena(arena, BinaryOp::Div, rhs_denom_t, denom_sq);
                binary_ad_in_arena(arena, BinaryOp::Sub, left, right)
            };
            let rhs_dir = {
                let lhs_tangent_over_denom =
                    binary_ad_in_arena(arena, BinaryOp::Div, lhs_tangent, denom);
                let left = neg_ad_in_arena(arena, lhs_tangent_over_denom);
                let lhs_denom_t = binary_ad_in_arena(arena, BinaryOp::Mul, lhs, denom_t);
                let right = binary_ad_in_arena(arena, BinaryOp::Div, lhs_denom_t, denom_sq);
                binary_ad_in_arena(arena, BinaryOp::Add, left, right)
            };
            (lhs_dir, rhs_dir)
        }
        BinaryOp::Hypot => {
            let hypot = binary_ad_in_arena(arena, BinaryOp::Hypot, lhs, rhs);
            let lhs_lhs_tangent = binary_ad_in_arena(arena, BinaryOp::Mul, lhs, lhs_tangent);
            let rhs_rhs_tangent = binary_ad_in_arena(arena, BinaryOp::Mul, rhs, rhs_tangent);
            let numer_t =
                binary_ad_in_arena(arena, BinaryOp::Add, lhs_lhs_tangent, rhs_rhs_tangent);
            let hypot_t = binary_ad_in_arena(arena, BinaryOp::Div, numer_t, hypot);
            let hypot_sq = binary_ad_in_arena(arena, BinaryOp::Mul, hypot, hypot);
            let lhs_dir = {
                let left = binary_ad_in_arena(arena, BinaryOp::Div, lhs_tangent, hypot);
                let lhs_hypot_t = binary_ad_in_arena(arena, BinaryOp::Mul, lhs, hypot_t);
                let right = binary_ad_in_arena(arena, BinaryOp::Div, lhs_hypot_t, hypot_sq);
                binary_ad_in_arena(arena, BinaryOp::Sub, left, right)
            };
            let rhs_dir = {
                let left = binary_ad_in_arena(arena, BinaryOp::Div, rhs_tangent, hypot);
                let rhs_hypot_t = binary_ad_in_arena(arena, BinaryOp::Mul, rhs, hypot_t);
                let right = binary_ad_in_arena(arena, BinaryOp::Div, rhs_hypot_t, hypot_sq);
                binary_ad_in_arena(arena, BinaryOp::Sub, left, right)
            };
            (lhs_dir, rhs_dir)
        }
        BinaryOp::Mod | BinaryOp::Copysign => (zero, zero),
    }
}

fn execute_program_reverse(program: &SxProgram, vars: &[SX], seeds: &[SX]) -> Result<Vec<SX>> {
    let mut adjoint_slots = vec![SX::zero(); program.slot_exprs.len()];
    for (slot, seed) in program
        .output_slots
        .iter()
        .copied()
        .zip(seeds.iter().copied())
    {
        adjoint_slots[slot] += seed;
    }

    let mut local_caches = AdLocalCaches::default();
    let mut reverse_call_memo = HashMap::<usize, Vec<SXMatrix>>::new();
    for instruction in program.instructions.iter().rev() {
        match instruction {
            ProgramInstruction::Unary {
                result_slot,
                op,
                arg_slot,
                arg_expr,
            } => {
                let adj = adjoint_slots[*result_slot];
                if !adj.is_zero() {
                    with_arena(|arena| {
                        let derivative = unary_derivative_in_arena(arena, *op, *arg_expr);
                        let contribution = binary_in_arena(arena, BinaryOp::Mul, adj, derivative);
                        adjoint_slots[*arg_slot] = binary_in_arena(
                            arena,
                            BinaryOp::Add,
                            adjoint_slots[*arg_slot],
                            contribution,
                        );
                    });
                }
            }
            ProgramInstruction::Binary {
                result_slot,
                op,
                lhs_slot,
                rhs_slot,
                lhs_expr,
                rhs_expr,
            } => {
                let adj = adjoint_slots[*result_slot];
                if adj.is_zero() {
                    continue;
                }
                with_arena(|arena| match op {
                    BinaryOp::Add => {
                        adjoint_slots[*lhs_slot] =
                            binary_in_arena(arena, BinaryOp::Add, adjoint_slots[*lhs_slot], adj);
                        adjoint_slots[*rhs_slot] =
                            binary_in_arena(arena, BinaryOp::Add, adjoint_slots[*rhs_slot], adj);
                    }
                    BinaryOp::Sub => {
                        adjoint_slots[*lhs_slot] =
                            binary_in_arena(arena, BinaryOp::Add, adjoint_slots[*lhs_slot], adj);
                        adjoint_slots[*rhs_slot] =
                            binary_in_arena(arena, BinaryOp::Sub, adjoint_slots[*rhs_slot], adj);
                    }
                    BinaryOp::Mul => {
                        let lhs_contrib = binary_in_arena(arena, BinaryOp::Mul, adj, *rhs_expr);
                        let rhs_contrib = binary_in_arena(arena, BinaryOp::Mul, adj, *lhs_expr);
                        adjoint_slots[*lhs_slot] = binary_in_arena(
                            arena,
                            BinaryOp::Add,
                            adjoint_slots[*lhs_slot],
                            lhs_contrib,
                        );
                        adjoint_slots[*rhs_slot] = binary_in_arena(
                            arena,
                            BinaryOp::Add,
                            adjoint_slots[*rhs_slot],
                            rhs_contrib,
                        );
                    }
                    BinaryOp::Div => {
                        let lhs_contrib = binary_in_arena(arena, BinaryOp::Div, adj, *rhs_expr);
                        let neg_one = append_constant_in_arena(arena, -1.0);
                        let neg_adj = binary_in_arena(arena, BinaryOp::Mul, neg_one, adj);
                        let numer = binary_in_arena(arena, BinaryOp::Mul, neg_adj, *lhs_expr);
                        let rhs_sq = binary_in_arena(arena, BinaryOp::Mul, *rhs_expr, *rhs_expr);
                        let rhs_contrib = binary_in_arena(arena, BinaryOp::Div, numer, rhs_sq);
                        adjoint_slots[*lhs_slot] = binary_in_arena(
                            arena,
                            BinaryOp::Add,
                            adjoint_slots[*lhs_slot],
                            lhs_contrib,
                        );
                        adjoint_slots[*rhs_slot] = binary_in_arena(
                            arena,
                            BinaryOp::Add,
                            adjoint_slots[*rhs_slot],
                            rhs_contrib,
                        );
                    }
                    BinaryOp::Pow
                    | BinaryOp::Atan2
                    | BinaryOp::Hypot
                    | BinaryOp::Mod
                    | BinaryOp::Copysign => {
                        let (d_lhs, d_rhs) =
                            binary_partials_in_arena(arena, *op, *lhs_expr, *rhs_expr);
                        let lhs_contrib = binary_in_arena(arena, BinaryOp::Mul, adj, d_lhs);
                        let rhs_contrib = binary_in_arena(arena, BinaryOp::Mul, adj, d_rhs);
                        adjoint_slots[*lhs_slot] = binary_in_arena(
                            arena,
                            BinaryOp::Add,
                            adjoint_slots[*lhs_slot],
                            lhs_contrib,
                        );
                        adjoint_slots[*rhs_slot] = binary_in_arena(
                            arena,
                            BinaryOp::Add,
                            adjoint_slots[*rhs_slot],
                            rhs_contrib,
                        );
                    }
                });
            }
            ProgramInstruction::Call {
                result_slot,
                function,
                output_slot,
                output_offset,
                inputs,
                ..
            } => {
                let adj = adjoint_slots[*result_slot];
                if adj.is_zero() {
                    continue;
                }
                let profile = local_caches.dependency_profile(*function);
                let helper_outputs = if let Some(existing) = reverse_call_memo.get(result_slot) {
                    existing
                } else {
                    let helper = local_caches.reverse_scalar_helper(
                        *function,
                        *output_slot,
                        *output_offset,
                    )?;
                    let helper_outputs = helper.call(
                        &inputs
                            .iter()
                            .map(|input| input.matrix.clone())
                            .collect::<Vec<_>>(),
                    )?;
                    reverse_call_memo.insert(*result_slot, helper_outputs);
                    reverse_call_memo
                        .get(result_slot)
                        .expect("inserted reverse call helper output")
                };
                for (slot, input) in inputs.iter().enumerate() {
                    for (offset, &input_slot) in input.slots.iter().enumerate() {
                        if !profile.output_depends_on(*output_slot, *output_offset, slot, offset) {
                            continue;
                        }
                        adjoint_slots[input_slot] += adj * helper_outputs[slot].nz(offset);
                    }
                }
            }
        }
    }

    Ok(vars
        .iter()
        .copied()
        .map(|var| {
            program
                .slot_by_node
                .get(&var)
                .map(|&slot| adjoint_slots[slot])
                .unwrap_or_else(SX::zero)
        })
        .collect())
}

fn execute_program_reverse_batch(
    program: &SxProgram,
    vars: &[SX],
    seeds_by_direction: &[Vec<SX>],
) -> Result<Vec<Vec<SX>>> {
    let direction_count = seeds_by_direction.len();
    let mut adjoint_slots = vec![vec![SX::zero(); direction_count]; program.slot_exprs.len()];
    for (output_index, slot) in program.output_slots.iter().copied().enumerate() {
        for (direction, seeds) in seeds_by_direction.iter().enumerate() {
            adjoint_slots[slot][direction] += seeds[output_index];
        }
    }

    let mut local_caches = AdLocalCaches::default();
    for instruction in program.instructions.iter().rev() {
        match instruction {
            ProgramInstruction::Unary {
                result_slot,
                op,
                arg_slot,
                arg_expr,
            } => {
                let active_adjoints = adjoint_slots[*result_slot]
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(direction, adj)| (!adj.is_zero()).then_some((direction, adj)))
                    .collect::<Vec<_>>();
                if active_adjoints.is_empty() {
                    continue;
                }
                with_arena(|arena| {
                    let derivative = unary_derivative_in_arena(arena, *op, *arg_expr);
                    for (direction, adj) in &active_adjoints {
                        let contribution = binary_in_arena(arena, BinaryOp::Mul, *adj, derivative);
                        adjoint_slots[*arg_slot][*direction] = binary_in_arena(
                            arena,
                            BinaryOp::Add,
                            adjoint_slots[*arg_slot][*direction],
                            contribution,
                        );
                    }
                });
            }
            ProgramInstruction::Binary {
                result_slot,
                op,
                lhs_slot,
                rhs_slot,
                lhs_expr,
                rhs_expr,
            } => {
                let active_adjoints = adjoint_slots[*result_slot]
                    .iter()
                    .copied()
                    .enumerate()
                    .filter_map(|(direction, adj)| (!adj.is_zero()).then_some((direction, adj)))
                    .collect::<Vec<_>>();
                if active_adjoints.is_empty() {
                    continue;
                }
                with_arena(|arena| {
                    let partials = match op {
                        BinaryOp::Pow
                        | BinaryOp::Atan2
                        | BinaryOp::Hypot
                        | BinaryOp::Mod
                        | BinaryOp::Copysign => {
                            Some(binary_partials_in_arena(arena, *op, *lhs_expr, *rhs_expr))
                        }
                        _ => None,
                    };
                    for (direction, adj) in &active_adjoints {
                        match op {
                            BinaryOp::Add => {
                                adjoint_slots[*lhs_slot][*direction] = binary_in_arena(
                                    arena,
                                    BinaryOp::Add,
                                    adjoint_slots[*lhs_slot][*direction],
                                    *adj,
                                );
                                adjoint_slots[*rhs_slot][*direction] = binary_in_arena(
                                    arena,
                                    BinaryOp::Add,
                                    adjoint_slots[*rhs_slot][*direction],
                                    *adj,
                                );
                            }
                            BinaryOp::Sub => {
                                adjoint_slots[*lhs_slot][*direction] = binary_in_arena(
                                    arena,
                                    BinaryOp::Add,
                                    adjoint_slots[*lhs_slot][*direction],
                                    *adj,
                                );
                                adjoint_slots[*rhs_slot][*direction] = binary_in_arena(
                                    arena,
                                    BinaryOp::Sub,
                                    adjoint_slots[*rhs_slot][*direction],
                                    *adj,
                                );
                            }
                            BinaryOp::Mul => {
                                let lhs_contrib =
                                    binary_in_arena(arena, BinaryOp::Mul, *adj, *rhs_expr);
                                let rhs_contrib =
                                    binary_in_arena(arena, BinaryOp::Mul, *adj, *lhs_expr);
                                adjoint_slots[*lhs_slot][*direction] = binary_in_arena(
                                    arena,
                                    BinaryOp::Add,
                                    adjoint_slots[*lhs_slot][*direction],
                                    lhs_contrib,
                                );
                                adjoint_slots[*rhs_slot][*direction] = binary_in_arena(
                                    arena,
                                    BinaryOp::Add,
                                    adjoint_slots[*rhs_slot][*direction],
                                    rhs_contrib,
                                );
                            }
                            BinaryOp::Div => {
                                let lhs_contrib =
                                    binary_in_arena(arena, BinaryOp::Div, *adj, *rhs_expr);
                                let neg_one = append_constant_in_arena(arena, -1.0);
                                let neg_adj = binary_in_arena(arena, BinaryOp::Mul, neg_one, *adj);
                                let numer =
                                    binary_in_arena(arena, BinaryOp::Mul, neg_adj, *lhs_expr);
                                let rhs_sq =
                                    binary_in_arena(arena, BinaryOp::Mul, *rhs_expr, *rhs_expr);
                                let rhs_contrib =
                                    binary_in_arena(arena, BinaryOp::Div, numer, rhs_sq);
                                adjoint_slots[*lhs_slot][*direction] = binary_in_arena(
                                    arena,
                                    BinaryOp::Add,
                                    adjoint_slots[*lhs_slot][*direction],
                                    lhs_contrib,
                                );
                                adjoint_slots[*rhs_slot][*direction] = binary_in_arena(
                                    arena,
                                    BinaryOp::Add,
                                    adjoint_slots[*rhs_slot][*direction],
                                    rhs_contrib,
                                );
                            }
                            BinaryOp::Pow
                            | BinaryOp::Atan2
                            | BinaryOp::Hypot
                            | BinaryOp::Mod
                            | BinaryOp::Copysign => {
                                let (d_lhs, d_rhs) = partials
                                    .as_ref()
                                    .expect("partials computed for nontrivial binary op");
                                let lhs_contrib =
                                    binary_in_arena(arena, BinaryOp::Mul, *adj, *d_lhs);
                                let rhs_contrib =
                                    binary_in_arena(arena, BinaryOp::Mul, *adj, *d_rhs);
                                adjoint_slots[*lhs_slot][*direction] = binary_in_arena(
                                    arena,
                                    BinaryOp::Add,
                                    adjoint_slots[*lhs_slot][*direction],
                                    lhs_contrib,
                                );
                                adjoint_slots[*rhs_slot][*direction] = binary_in_arena(
                                    arena,
                                    BinaryOp::Add,
                                    adjoint_slots[*rhs_slot][*direction],
                                    rhs_contrib,
                                );
                            }
                        }
                    }
                });
            }
            ProgramInstruction::Call {
                result_slot,
                function,
                output_slot,
                output_offset,
                inputs,
                ..
            } => {
                let active_directions = adjoint_slots[*result_slot]
                    .iter()
                    .enumerate()
                    .filter_map(|(direction, adj)| (!adj.is_zero()).then_some(direction))
                    .collect::<Vec<_>>();
                if active_directions.is_empty() {
                    continue;
                }
                let mut helper_inputs = inputs
                    .iter()
                    .map(|input| input.matrix.clone())
                    .collect::<Vec<_>>();
                helper_inputs.extend(
                    active_directions
                        .iter()
                        .map(|&direction| SXMatrix::scalar(adjoint_slots[*result_slot][direction])),
                );
                let helper = local_caches.reverse_output_batch_helper(
                    *function,
                    *output_slot,
                    *output_offset,
                    active_directions.len(),
                )?;
                let helper_outputs = helper.call(&helper_inputs)?;
                for (slot, input) in inputs.iter().enumerate() {
                    for (active_index, &direction) in active_directions.iter().enumerate() {
                        let helper_output = &helper_outputs[active_index * inputs.len() + slot];
                        for &offset in &input.relevant_offsets {
                            let input_slot = input.slots[offset];
                            adjoint_slots[input_slot][direction] += helper_output.nz(offset);
                        }
                    }
                }
            }
        }
    }

    Ok((0..direction_count)
        .map(|direction| {
            vars.iter()
                .copied()
                .map(|var| {
                    program
                        .slot_by_node
                        .get(&var)
                        .map(|&slot| adjoint_slots[slot][direction])
                        .unwrap_or_else(SX::zero)
                })
                .collect::<Vec<_>>()
        })
        .collect())
}

#[expect(
    clippy::too_many_arguments,
    reason = "binary Hessian accumulation keeps derivative buffers explicit for allocation-free updates"
)]
fn accumulate_binary_hessian_from_partials<F>(
    expr_arena: &mut AdExprArena,
    direction_count: usize,
    zero: AdExpr,
    lhs_slot: usize,
    rhs_slot: usize,
    result_slot: usize,
    adj: AdExpr,
    adj_is_zero: bool,
    lhs_partial: AdExpr,
    rhs_partial: AdExpr,
    active_directions: &[usize],
    adjoint_slots: &mut [AdExpr],
    adjoint_tangent_slots: &mut [AdExpr],
    mut directional_partials: F,
) where
    F: FnMut(&mut AdExprArena, usize) -> (AdExpr, AdExpr),
{
    let slot_base = |slot: usize| slot * direction_count;
    if !adj_is_zero {
        let lhs_contrib = expr_arena.binary_ad(BinaryOp::Mul, adj, lhs_partial);
        let rhs_contrib = expr_arena.binary_ad(BinaryOp::Mul, adj, rhs_partial);
        adjoint_slots[lhs_slot] =
            expr_arena.binary_ad(BinaryOp::Add, adjoint_slots[lhs_slot], lhs_contrib);
        adjoint_slots[rhs_slot] =
            expr_arena.binary_ad(BinaryOp::Add, adjoint_slots[rhs_slot], rhs_contrib);
    }
    for &direction in active_directions {
        let adj_tangent = adjoint_tangent_slots[slot_base(result_slot) + direction];
        let adj_tangent_is_zero = expr_arena.is_zero(adj_tangent);
        let from_adjoint_lhs = if adj_tangent_is_zero {
            zero
        } else {
            expr_arena.binary_ad(BinaryOp::Mul, adj_tangent, lhs_partial)
        };
        let from_adjoint_rhs = if adj_tangent_is_zero {
            zero
        } else {
            expr_arena.binary_ad(BinaryOp::Mul, adj_tangent, rhs_partial)
        };
        let (lhs_partial_t, rhs_partial_t) = directional_partials(expr_arena, direction);
        let from_second_lhs = if adj_is_zero || expr_arena.is_zero(lhs_partial_t) {
            zero
        } else {
            expr_arena.binary_ad(BinaryOp::Mul, adj, lhs_partial_t)
        };
        let from_second_rhs = if adj_is_zero || expr_arena.is_zero(rhs_partial_t) {
            zero
        } else {
            expr_arena.binary_ad(BinaryOp::Mul, adj, rhs_partial_t)
        };
        let lhs_contribution =
            expr_arena.binary_ad(BinaryOp::Add, from_adjoint_lhs, from_second_lhs);
        let rhs_contribution =
            expr_arena.binary_ad(BinaryOp::Add, from_adjoint_rhs, from_second_rhs);
        let lhs_target = &mut adjoint_tangent_slots[slot_base(lhs_slot) + direction];
        *lhs_target = expr_arena.binary_ad(BinaryOp::Add, *lhs_target, lhs_contribution);
        let rhs_target = &mut adjoint_tangent_slots[slot_base(rhs_slot) + direction];
        *rhs_target = expr_arena.binary_ad(BinaryOp::Add, *rhs_target, rhs_contribution);
    }
}

fn execute_program_scalar_hessian_basis_batch(
    program: &SxProgram,
    vars: &[SX],
    active_var_groups: &[Vec<Index>],
) -> Result<Vec<Vec<SX>>> {
    let direction_count = active_var_groups.len();
    if direction_count == 0 {
        return Ok(Vec::new());
    }
    if program.output_slots.len() != 1 {
        return Err(SxError::Shape(
            "direct Hessian program expects a scalar output".into(),
        ));
    }

    let context_id = resolve_context_for_slices(&[("vars", vars)]);
    let mut expr_arena = AdExprArena::new(context_id);
    let mut expr_lower_memo = HashMap::<SX, AdExpr>::new();
    let tangent_slots = expr_arena.lower_sx_collection_with_memo(
        &execute_program_forward_basis_batch_slots(program, vars, active_var_groups)?,
        &mut expr_lower_memo,
    );
    let slot_expr_values =
        expr_arena.lower_sx_collection_with_memo(&program.slot_exprs, &mut expr_lower_memo);
    let slot_count = program.slot_exprs.len();
    let slot_base = |slot: usize| slot * direction_count;
    let output_slot = program.output_slots[0];
    let zero = expr_arena.zero();
    let one = expr_arena.one();
    let mut adjoint_slots = vec![zero; slot_count];
    adjoint_slots[output_slot] = one;
    let mut adjoint_tangent_slots = vec![zero; slot_count * direction_count];
    let mut local_caches = AdLocalCaches::default();
    let mut reverse_call_memo = HashMap::<usize, Vec<Vec<AdExpr>>>::new();
    let mut function_plan_cache = HashMap::<FunctionId, FunctionAdExprArenaPlan>::new();

    for instruction in program.instructions.iter().rev() {
        match instruction {
            ProgramInstruction::Unary {
                result_slot,
                op,
                arg_slot,
                ..
            } => {
                let adj = adjoint_slots[*result_slot];
                let adj_is_zero = expr_arena.is_zero(adj);
                let active_directions = if adj_is_zero {
                    (0..direction_count)
                        .filter(|&direction| {
                            !expr_arena
                                .is_zero(adjoint_tangent_slots[slot_base(*result_slot) + direction])
                        })
                        .collect::<Vec<_>>()
                } else {
                    (0..direction_count).collect::<Vec<_>>()
                };
                if adj_is_zero && active_directions.is_empty() {
                    continue;
                }
                let arg_expr = slot_expr_values[*arg_slot];
                let derivative = expr_arena.unary_derivative(*op, arg_expr);
                if !adj_is_zero {
                    let contribution = expr_arena.binary_ad(BinaryOp::Mul, adj, derivative);
                    adjoint_slots[*arg_slot] =
                        expr_arena.binary_ad(BinaryOp::Add, adjoint_slots[*arg_slot], contribution);
                }
                for direction in active_directions {
                    let from_adjoint = {
                        let adj_tangent =
                            adjoint_tangent_slots[slot_base(*result_slot) + direction];
                        if expr_arena.is_zero(adj_tangent) {
                            zero
                        } else {
                            expr_arena.binary_ad(BinaryOp::Mul, adj_tangent, derivative)
                        }
                    };
                    let from_second = if adj_is_zero {
                        zero
                    } else {
                        let second = expr_arena.unary_second_directional(
                            *op,
                            arg_expr,
                            tangent_slots[slot_base(*arg_slot) + direction],
                        );
                        if expr_arena.is_zero(second) {
                            zero
                        } else {
                            expr_arena.binary_ad(BinaryOp::Mul, adj, second)
                        }
                    };
                    let contribution =
                        expr_arena.binary_ad(BinaryOp::Add, from_adjoint, from_second);
                    let target = &mut adjoint_tangent_slots[slot_base(*arg_slot) + direction];
                    *target = expr_arena.binary_ad(BinaryOp::Add, *target, contribution);
                }
            }
            ProgramInstruction::Binary {
                result_slot,
                op,
                lhs_slot,
                rhs_slot,
                ..
            } => {
                let adj = adjoint_slots[*result_slot];
                let adj_is_zero = expr_arena.is_zero(adj);
                let active_directions = if adj_is_zero {
                    (0..direction_count)
                        .filter(|&direction| {
                            !expr_arena
                                .is_zero(adjoint_tangent_slots[slot_base(*result_slot) + direction])
                        })
                        .collect::<Vec<_>>()
                } else {
                    (0..direction_count).collect::<Vec<_>>()
                };
                if adj_is_zero && active_directions.is_empty() {
                    continue;
                }
                let lhs_expr = slot_expr_values[*lhs_slot];
                let rhs_expr = slot_expr_values[*rhs_slot];
                match *op {
                    BinaryOp::Add => {
                        if !adj_is_zero {
                            adjoint_slots[*lhs_slot] =
                                expr_arena.binary_ad(BinaryOp::Add, adjoint_slots[*lhs_slot], adj);
                            adjoint_slots[*rhs_slot] =
                                expr_arena.binary_ad(BinaryOp::Add, adjoint_slots[*rhs_slot], adj);
                        }
                        for direction in active_directions {
                            let adj_tangent =
                                adjoint_tangent_slots[slot_base(*result_slot) + direction];
                            if expr_arena.is_zero(adj_tangent) {
                                continue;
                            }
                            let lhs_target =
                                &mut adjoint_tangent_slots[slot_base(*lhs_slot) + direction];
                            *lhs_target =
                                expr_arena.binary_ad(BinaryOp::Add, *lhs_target, adj_tangent);
                            let rhs_target =
                                &mut adjoint_tangent_slots[slot_base(*rhs_slot) + direction];
                            *rhs_target =
                                expr_arena.binary_ad(BinaryOp::Add, *rhs_target, adj_tangent);
                        }
                    }
                    BinaryOp::Sub => {
                        if !adj_is_zero {
                            adjoint_slots[*lhs_slot] =
                                expr_arena.binary_ad(BinaryOp::Add, adjoint_slots[*lhs_slot], adj);
                            adjoint_slots[*rhs_slot] =
                                expr_arena.binary_ad(BinaryOp::Sub, adjoint_slots[*rhs_slot], adj);
                        }
                        for direction in active_directions {
                            let adj_tangent =
                                adjoint_tangent_slots[slot_base(*result_slot) + direction];
                            if expr_arena.is_zero(adj_tangent) {
                                continue;
                            }
                            let lhs_target =
                                &mut adjoint_tangent_slots[slot_base(*lhs_slot) + direction];
                            *lhs_target =
                                expr_arena.binary_ad(BinaryOp::Add, *lhs_target, adj_tangent);
                            let rhs_target =
                                &mut adjoint_tangent_slots[slot_base(*rhs_slot) + direction];
                            *rhs_target =
                                expr_arena.binary_ad(BinaryOp::Sub, *rhs_target, adj_tangent);
                        }
                    }
                    BinaryOp::Mul => {
                        if !adj_is_zero {
                            let lhs_contrib = expr_arena.binary_ad(BinaryOp::Mul, adj, rhs_expr);
                            let rhs_contrib = expr_arena.binary_ad(BinaryOp::Mul, adj, lhs_expr);
                            adjoint_slots[*lhs_slot] = expr_arena.binary_ad(
                                BinaryOp::Add,
                                adjoint_slots[*lhs_slot],
                                lhs_contrib,
                            );
                            adjoint_slots[*rhs_slot] = expr_arena.binary_ad(
                                BinaryOp::Add,
                                adjoint_slots[*rhs_slot],
                                rhs_contrib,
                            );
                        }
                        for direction in active_directions {
                            let lhs_tangent = tangent_slots[slot_base(*lhs_slot) + direction];
                            let rhs_tangent = tangent_slots[slot_base(*rhs_slot) + direction];
                            let adj_tangent =
                                adjoint_tangent_slots[slot_base(*result_slot) + direction];

                            let from_adjoint_lhs = if expr_arena.is_zero(adj_tangent) {
                                zero
                            } else {
                                expr_arena.binary_ad(BinaryOp::Mul, adj_tangent, rhs_expr)
                            };
                            let from_adjoint_rhs = if expr_arena.is_zero(adj_tangent) {
                                zero
                            } else {
                                expr_arena.binary_ad(BinaryOp::Mul, adj_tangent, lhs_expr)
                            };
                            let from_second_lhs = if adj_is_zero || expr_arena.is_zero(rhs_tangent)
                            {
                                zero
                            } else {
                                expr_arena.binary_ad(BinaryOp::Mul, adj, rhs_tangent)
                            };
                            let from_second_rhs = if adj_is_zero || expr_arena.is_zero(lhs_tangent)
                            {
                                zero
                            } else {
                                expr_arena.binary_ad(BinaryOp::Mul, adj, lhs_tangent)
                            };
                            let lhs_contribution = expr_arena.binary_ad(
                                BinaryOp::Add,
                                from_adjoint_lhs,
                                from_second_lhs,
                            );
                            let rhs_contribution = expr_arena.binary_ad(
                                BinaryOp::Add,
                                from_adjoint_rhs,
                                from_second_rhs,
                            );
                            let lhs_target =
                                &mut adjoint_tangent_slots[slot_base(*lhs_slot) + direction];
                            *lhs_target =
                                expr_arena.binary_ad(BinaryOp::Add, *lhs_target, lhs_contribution);
                            let rhs_target =
                                &mut adjoint_tangent_slots[slot_base(*rhs_slot) + direction];
                            *rhs_target =
                                expr_arena.binary_ad(BinaryOp::Add, *rhs_target, rhs_contribution);
                        }
                    }
                    BinaryOp::Div => {
                        let (lhs_partial, rhs_partial) =
                            expr_arena.binary_partials(BinaryOp::Div, lhs_expr, rhs_expr);
                        accumulate_binary_hessian_from_partials(
                            &mut expr_arena,
                            direction_count,
                            zero,
                            *lhs_slot,
                            *rhs_slot,
                            *result_slot,
                            adj,
                            adj_is_zero,
                            lhs_partial,
                            rhs_partial,
                            &active_directions,
                            &mut adjoint_slots,
                            &mut adjoint_tangent_slots,
                            |expr_arena, direction| {
                                expr_arena.binary_partial_directionals(
                                    BinaryOp::Div,
                                    lhs_expr,
                                    rhs_expr,
                                    tangent_slots[slot_base(*lhs_slot) + direction],
                                    tangent_slots[slot_base(*rhs_slot) + direction],
                                )
                            },
                        );
                    }
                    BinaryOp::Pow => {
                        let (lhs_partial, rhs_partial) =
                            expr_arena.binary_partials(BinaryOp::Pow, lhs_expr, rhs_expr);
                        accumulate_binary_hessian_from_partials(
                            &mut expr_arena,
                            direction_count,
                            zero,
                            *lhs_slot,
                            *rhs_slot,
                            *result_slot,
                            adj,
                            adj_is_zero,
                            lhs_partial,
                            rhs_partial,
                            &active_directions,
                            &mut adjoint_slots,
                            &mut adjoint_tangent_slots,
                            |expr_arena, direction| {
                                expr_arena.binary_partial_directionals(
                                    BinaryOp::Pow,
                                    lhs_expr,
                                    rhs_expr,
                                    tangent_slots[slot_base(*lhs_slot) + direction],
                                    tangent_slots[slot_base(*rhs_slot) + direction],
                                )
                            },
                        );
                    }
                    BinaryOp::Atan2 => {
                        let (lhs_partial, rhs_partial) =
                            expr_arena.binary_partials(BinaryOp::Atan2, lhs_expr, rhs_expr);
                        accumulate_binary_hessian_from_partials(
                            &mut expr_arena,
                            direction_count,
                            zero,
                            *lhs_slot,
                            *rhs_slot,
                            *result_slot,
                            adj,
                            adj_is_zero,
                            lhs_partial,
                            rhs_partial,
                            &active_directions,
                            &mut adjoint_slots,
                            &mut adjoint_tangent_slots,
                            |expr_arena, direction| {
                                expr_arena.binary_partial_directionals(
                                    BinaryOp::Atan2,
                                    lhs_expr,
                                    rhs_expr,
                                    tangent_slots[slot_base(*lhs_slot) + direction],
                                    tangent_slots[slot_base(*rhs_slot) + direction],
                                )
                            },
                        );
                    }
                    BinaryOp::Hypot => {
                        let (lhs_partial, rhs_partial) =
                            expr_arena.binary_partials(BinaryOp::Hypot, lhs_expr, rhs_expr);
                        accumulate_binary_hessian_from_partials(
                            &mut expr_arena,
                            direction_count,
                            zero,
                            *lhs_slot,
                            *rhs_slot,
                            *result_slot,
                            adj,
                            adj_is_zero,
                            lhs_partial,
                            rhs_partial,
                            &active_directions,
                            &mut adjoint_slots,
                            &mut adjoint_tangent_slots,
                            |expr_arena, direction| {
                                expr_arena.binary_partial_directionals(
                                    BinaryOp::Hypot,
                                    lhs_expr,
                                    rhs_expr,
                                    tangent_slots[slot_base(*lhs_slot) + direction],
                                    tangent_slots[slot_base(*rhs_slot) + direction],
                                )
                            },
                        );
                    }
                    BinaryOp::Mod => {
                        let (lhs_partial, rhs_partial) =
                            expr_arena.binary_partials(BinaryOp::Mod, lhs_expr, rhs_expr);
                        accumulate_binary_hessian_from_partials(
                            &mut expr_arena,
                            direction_count,
                            zero,
                            *lhs_slot,
                            *rhs_slot,
                            *result_slot,
                            adj,
                            adj_is_zero,
                            lhs_partial,
                            rhs_partial,
                            &active_directions,
                            &mut adjoint_slots,
                            &mut adjoint_tangent_slots,
                            |expr_arena, direction| {
                                expr_arena.binary_partial_directionals(
                                    BinaryOp::Mod,
                                    lhs_expr,
                                    rhs_expr,
                                    tangent_slots[slot_base(*lhs_slot) + direction],
                                    tangent_slots[slot_base(*rhs_slot) + direction],
                                )
                            },
                        );
                    }
                    BinaryOp::Copysign => {
                        let (lhs_partial, rhs_partial) =
                            expr_arena.binary_partials(BinaryOp::Copysign, lhs_expr, rhs_expr);
                        accumulate_binary_hessian_from_partials(
                            &mut expr_arena,
                            direction_count,
                            zero,
                            *lhs_slot,
                            *rhs_slot,
                            *result_slot,
                            adj,
                            adj_is_zero,
                            lhs_partial,
                            rhs_partial,
                            &active_directions,
                            &mut adjoint_slots,
                            &mut adjoint_tangent_slots,
                            |expr_arena, direction| {
                                expr_arena.binary_partial_directionals(
                                    BinaryOp::Copysign,
                                    lhs_expr,
                                    rhs_expr,
                                    tangent_slots[slot_base(*lhs_slot) + direction],
                                    tangent_slots[slot_base(*rhs_slot) + direction],
                                )
                            },
                        );
                    }
                }
            }
            ProgramInstruction::Call {
                result_slot,
                function,
                output_slot,
                output_offset,
                inputs,
                ..
            } => {
                let adj = adjoint_slots[*result_slot];
                let active_directions = if expr_arena.is_zero(adj) {
                    (0..direction_count)
                        .filter(|&direction| {
                            !expr_arena
                                .is_zero(adjoint_tangent_slots[slot_base(*result_slot) + direction])
                        })
                        .collect::<Vec<_>>()
                } else {
                    (0..direction_count).collect::<Vec<_>>()
                };
                if expr_arena.is_zero(adj) && active_directions.is_empty() {
                    continue;
                }
                let helper =
                    local_caches.reverse_scalar_helper(*function, *output_slot, *output_offset)?;
                let helper_inputs = inputs
                    .iter()
                    .map(|input| {
                        input
                            .slots
                            .iter()
                            .map(|&slot| slot_expr_values[slot])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();

                if !reverse_call_memo.contains_key(result_slot) {
                    let helper_outputs = execute_function_symbolically_adexpr_flat_by_id(
                        helper.id(),
                        &helper_inputs,
                        &mut expr_arena,
                        &mut expr_lower_memo,
                        &mut local_caches,
                        &mut function_plan_cache,
                    )?;
                    reverse_call_memo.insert(*result_slot, helper_outputs);
                }
                let helper_outputs = reverse_call_memo
                    .get(result_slot)
                    .expect("reverse helper outputs inserted");
                if !expr_arena.is_zero(adj) {
                    for (slot, input) in inputs.iter().enumerate() {
                        for &offset in &input.relevant_offsets {
                            let input_slot = input.slots[offset];
                            let contrib = expr_arena.binary_ad(
                                BinaryOp::Mul,
                                adj,
                                helper_outputs[slot][offset],
                            );
                            adjoint_slots[input_slot] = expr_arena.binary_ad(
                                BinaryOp::Add,
                                adjoint_slots[input_slot],
                                contrib,
                            );
                        }
                    }
                }
                if active_directions.is_empty() {
                    continue;
                }
                let helper_directionals = execute_function_forward_batch_adexpr_from_call_by_id(
                    helper.id(),
                    &helper_inputs,
                    inputs,
                    &active_directions,
                    direction_count,
                    &tangent_slots,
                    &mut expr_arena,
                    &mut expr_lower_memo,
                    &mut local_caches,
                    &mut function_plan_cache,
                )?;
                for (slot, input) in inputs.iter().enumerate() {
                    for (local_direction, &direction) in active_directions.iter().enumerate() {
                        let adj_tangent =
                            adjoint_tangent_slots[slot_base(*result_slot) + direction];
                        let helper_directional = &helper_directionals[local_direction][slot];
                        for &offset in &input.relevant_offsets {
                            let input_slot = input.slots[offset];
                            let partial = helper_outputs[slot][offset];
                            let from_adjoint = if expr_arena.is_zero(adj_tangent) {
                                zero
                            } else {
                                expr_arena.binary_ad(BinaryOp::Mul, adj_tangent, partial)
                            };
                            let from_second = if expr_arena.is_zero(adj) {
                                zero
                            } else {
                                let dpartial = helper_directional[offset];
                                if expr_arena.is_zero(dpartial) {
                                    zero
                                } else {
                                    expr_arena.binary_ad(BinaryOp::Mul, adj, dpartial)
                                }
                            };
                            let combined =
                                expr_arena.binary_ad(BinaryOp::Add, from_adjoint, from_second);
                            let target =
                                &mut adjoint_tangent_slots[slot_base(input_slot) + direction];
                            *target = expr_arena.binary_ad(BinaryOp::Add, *target, combined);
                        }
                    }
                }
            }
        }
    }

    (0..direction_count)
        .map(|direction| {
            let mut sx_memo = HashMap::new();
            vars.iter()
                .copied()
                .map(|var| {
                    program
                        .slot_by_node
                        .get(&var)
                        .map(|&slot| {
                            expr_arena.to_sx_with_simplify_depth(
                                adjoint_tangent_slots[slot_base(slot) + direction],
                                &mut sx_memo,
                                Some(HESSIAN_REIFY_SIMPLIFY_DEPTH),
                            )
                        })
                        .unwrap_or_else(|| Ok(SX::zero()))
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()
}

pub(crate) fn forward_directional(outputs: &[SX], vars: &[SX], seeds: &[SX]) -> Result<Vec<SX>> {
    if vars.len() != seeds.len() {
        return Err(SxError::Shape(format!(
            "forward seed length {} does not match variable length {}",
            seeds.len(),
            vars.len()
        )));
    }
    let context_id =
        resolve_context_for_slices(&[("outputs", outputs), ("vars", vars), ("seeds", seeds)]);
    with_sx_context_id(context_id, || {
        let program = program_for_outputs(outputs);
        execute_program_forward(&program, vars, seeds)
    })
}

pub(crate) fn forward_directional_batch(
    outputs: &[SX],
    vars: &[SX],
    seeds_by_direction: &[Vec<SX>],
) -> Result<Vec<Vec<SX>>> {
    if seeds_by_direction.is_empty() {
        return Ok(Vec::new());
    }
    if seeds_by_direction
        .iter()
        .any(|seeds| seeds.len() != vars.len())
    {
        return Err(SxError::Shape(format!(
            "forward seed length(s) do not match variable length {}",
            vars.len()
        )));
    }

    let seed_slices = seeds_by_direction
        .iter()
        .map(|seeds| ("seeds", seeds.as_slice()))
        .collect::<Vec<_>>();
    let mut slices = vec![("outputs", outputs), ("vars", vars)];
    slices.extend(seed_slices);
    let context_id = resolve_context_for_slices(&slices);
    with_sx_context_id(context_id, || {
        let program = program_for_outputs(outputs);
        execute_program_forward_batch(&program, vars, seeds_by_direction)
    })
}

pub(crate) fn forward_directional_basis_batch(
    outputs: &[SX],
    vars: &[SX],
    active_var_groups: &[Vec<Index>],
) -> Result<Vec<Vec<SX>>> {
    if active_var_groups.is_empty() {
        return Ok(Vec::new());
    }
    let context_id = resolve_context_for_slices(&[("outputs", outputs), ("vars", vars)]);
    with_sx_context_id(context_id, || {
        let program = program_for_outputs(outputs);
        execute_program_forward_basis_batch(&program, vars, active_var_groups)
    })
}

pub(crate) fn reverse_directional(outputs: &[SX], vars: &[SX], seeds: &[SX]) -> Result<Vec<SX>> {
    if outputs.len() != seeds.len() {
        return Err(SxError::Shape(format!(
            "reverse seed length {} does not match output length {}",
            seeds.len(),
            outputs.len()
        )));
    }
    let context_id =
        resolve_context_for_slices(&[("outputs", outputs), ("vars", vars), ("seeds", seeds)]);
    with_sx_context_id(context_id, || {
        let program = program_for_outputs(outputs);
        execute_program_reverse(&program, vars, seeds)
    })
}

pub(crate) fn reverse_directional_batch(
    outputs: &[SX],
    vars: &[SX],
    seeds_by_direction: &[Vec<SX>],
) -> Result<Vec<Vec<SX>>> {
    if seeds_by_direction.is_empty() {
        return Ok(Vec::new());
    }
    if seeds_by_direction
        .iter()
        .any(|seeds| seeds.len() != outputs.len())
    {
        return Err(SxError::Shape(format!(
            "reverse seed length(s) do not match output length {}",
            outputs.len()
        )));
    }

    let seed_slices = seeds_by_direction
        .iter()
        .map(|seeds| ("seeds", seeds.as_slice()))
        .collect::<Vec<_>>();
    let mut slices = vec![("outputs", outputs), ("vars", vars)];
    slices.extend(seed_slices);
    let context_id = resolve_context_for_slices(&slices);
    with_sx_context_id(context_id, || {
        let program = program_for_outputs(outputs);
        execute_program_reverse_batch(&program, vars, seeds_by_direction)
    })
}

pub(crate) fn scalar_hessian_basis_batch(
    expr: SX,
    vars: &[SX],
    active_var_groups: &[Vec<Index>],
) -> Result<Vec<Vec<SX>>> {
    let context_id = resolve_context_for_slices(&[("expr", &[expr]), ("vars", vars)]);
    with_sx_context_id(context_id, || {
        let program = program_for_outputs(&[expr]);
        execute_program_scalar_hessian_basis_batch(&program, vars, active_var_groups)
    })
}

#[allow(dead_code)]
pub(crate) fn depends_on(expr: SX, target: SX, memo: &mut HashMap<SX, bool>) -> bool {
    if let Some(existing) = memo.get(&expr) {
        return *existing;
    }
    let answer = match node_kind(expr) {
        NodeKind::Constant(_) => false,
        NodeKind::Symbol { .. } => expr == target,
        NodeKind::Unary { arg, .. } => depends_on(arg, target, memo),
        NodeKind::Binary { lhs, rhs, .. } => {
            depends_on(lhs, target, memo) || depends_on(rhs, target, memo)
        }
        NodeKind::Call {
            function,
            inputs,
            output_slot,
            output_offset,
        } => {
            let profile = dependency_profile(function);
            inputs.iter().enumerate().any(|(slot, input)| {
                input.nonzeros().iter().enumerate().any(|(offset, &value)| {
                    profile.output_depends_on(output_slot, output_offset, slot, offset)
                        && depends_on(value, target, memo)
                })
            })
        }
    };
    memo.insert(expr, answer);
    answer
}

impl SX {
    const fn from_parts(context_id: u32, node_id: u32) -> Self {
        Self(((context_id as u64) << 32) | node_id as u64)
    }

    pub fn context_id(self) -> u32 {
        (self.0 >> 32) as u32
    }

    fn node_id(self) -> u32 {
        self.0 as u32
    }

    fn constant_in_context(context_id: u32, value: f64) -> Self {
        with_arena_for_context(context_id, |arena| {
            arena.intern_node(NodeKind::Constant(value))
        })
    }

    pub fn sym(name: impl Into<String>) -> Self {
        with_arena(|arena| arena.fresh_symbol(name))
    }

    pub fn zero() -> Self {
        SX::from(0.0)
    }

    pub fn one() -> Self {
        SX::from(1.0)
    }

    pub fn sqr(self) -> Self {
        self * self
    }

    pub fn abs(self) -> Self {
        unary(UnaryOp::Abs, self)
    }

    pub fn sign(self) -> Self {
        unary(UnaryOp::Sign, self)
    }

    pub fn floor(self) -> Self {
        unary(UnaryOp::Floor, self)
    }

    pub fn ceil(self) -> Self {
        unary(UnaryOp::Ceil, self)
    }

    pub fn round(self) -> Self {
        unary(UnaryOp::Round, self)
    }

    pub fn trunc(self) -> Self {
        unary(UnaryOp::Trunc, self)
    }

    pub fn sqrt(self) -> Self {
        unary(UnaryOp::Sqrt, self)
    }

    pub fn exp(self) -> Self {
        unary(UnaryOp::Exp, self)
    }

    pub fn expm1(self) -> Self {
        self.exp() - SX::one()
    }

    pub fn exp2(self) -> Self {
        (std::f64::consts::LN_2 * self).exp()
    }

    pub fn exp10(self) -> Self {
        (std::f64::consts::LN_10 * self).exp()
    }

    pub fn log(self) -> Self {
        unary(UnaryOp::Log, self)
    }

    pub fn log1p(self) -> Self {
        (SX::one() + self).log()
    }

    pub fn log2(self) -> Self {
        self.log() / std::f64::consts::LN_2
    }

    pub fn log10(self) -> Self {
        self.log() / std::f64::consts::LN_10
    }

    pub fn log_base(self, base: impl Into<SX>) -> Self {
        self.log() / base.into().log()
    }

    pub fn sin(self) -> Self {
        unary(UnaryOp::Sin, self)
    }

    pub fn cos(self) -> Self {
        unary(UnaryOp::Cos, self)
    }

    pub fn tan(self) -> Self {
        unary(UnaryOp::Tan, self)
    }

    pub fn asin(self) -> Self {
        unary(UnaryOp::Asin, self)
    }

    pub fn acos(self) -> Self {
        unary(UnaryOp::Acos, self)
    }

    pub fn atan(self) -> Self {
        unary(UnaryOp::Atan, self)
    }

    pub fn sinh(self) -> Self {
        unary(UnaryOp::Sinh, self)
    }

    pub fn cosh(self) -> Self {
        unary(UnaryOp::Cosh, self)
    }

    pub fn tanh(self) -> Self {
        unary(UnaryOp::Tanh, self)
    }

    pub fn asinh(self) -> Self {
        unary(UnaryOp::Asinh, self)
    }

    pub fn acosh(self) -> Self {
        unary(UnaryOp::Acosh, self)
    }

    pub fn atanh(self) -> Self {
        unary(UnaryOp::Atanh, self)
    }

    pub fn pow(self, rhs: impl Into<SX>) -> Self {
        binary(BinaryOp::Pow, self, rhs.into())
    }

    pub fn powf(self, rhs: f64) -> Self {
        self.pow(rhs)
    }

    pub fn powi(self, exponent: i32) -> Self {
        if exponent == 0 {
            return SX::one();
        }
        if exponent < 0 {
            return SX::one() / self.powi(-exponent);
        }
        let mut result = SX::one();
        let mut base = self;
        let mut remaining = exponent as u32;
        while remaining > 0 {
            if remaining & 1 == 1 {
                result *= base;
            }
            remaining >>= 1;
            if remaining > 0 {
                base *= base;
            }
        }
        result
    }

    pub fn atan2(self, rhs: impl Into<SX>) -> Self {
        binary(BinaryOp::Atan2, self, rhs.into())
    }

    pub fn hypot(self, rhs: impl Into<SX>) -> Self {
        binary(BinaryOp::Hypot, self, rhs.into())
    }

    pub fn modulo(self, rhs: impl Into<SX>) -> Self {
        binary(BinaryOp::Mod, self, rhs.into())
    }

    pub fn copysign(self, rhs: impl Into<SX>) -> Self {
        binary(BinaryOp::Copysign, self, rhs.into())
    }

    pub fn min(self, rhs: impl Into<SX>) -> Self {
        let rhs = rhs.into();
        0.5 * (self + rhs - (self - rhs).abs())
    }

    pub fn max(self, rhs: impl Into<SX>) -> Self {
        let rhs = rhs.into();
        0.5 * (self + rhs + (self - rhs).abs())
    }

    pub fn inspect(self) -> NodeView {
        match node_kind(self) {
            NodeKind::Constant(v) => NodeView::Constant(v),
            NodeKind::Symbol { name, serial } => NodeView::Symbol {
                name: name.to_string(),
                serial,
            },
            NodeKind::Unary { op, arg } => NodeView::Unary { op, arg },
            NodeKind::Binary { op, lhs, rhs } => NodeView::Binary { op, lhs, rhs },
            NodeKind::Call {
                function,
                inputs,
                output_slot,
                output_offset,
            } => NodeView::Call {
                function_id: function,
                function_name: function_name(function).unwrap_or_else(|| format!("fn_{function}")),
                inputs: inputs.to_vec(),
                output_slot,
                output_offset,
            },
        }
    }

    pub fn is_symbolic(self) -> bool {
        matches!(node_kind(self), NodeKind::Symbol { .. })
    }

    pub fn is_zero(self) -> bool {
        matches!(node_kind(self), NodeKind::Constant(v) if v == 0.0)
    }

    pub fn is_one(self) -> bool {
        matches!(node_kind(self), NodeKind::Constant(v) if v == 1.0)
    }

    pub fn free_symbols(self) -> BTreeSet<SX> {
        let mut seen = HashSet::new();
        let mut stack = vec![self];
        let mut free = BTreeSet::new();
        while let Some(node) = stack.pop() {
            if !seen.insert(node) {
                continue;
            }
            match node_kind(node) {
                NodeKind::Symbol { .. } => {
                    free.insert(node);
                }
                NodeKind::Constant(_) => {}
                NodeKind::Unary { arg, .. } => {
                    stack.push(arg);
                }
                NodeKind::Binary { lhs, rhs, .. } => {
                    stack.push(lhs);
                    stack.push(rhs);
                }
                NodeKind::Call {
                    function,
                    inputs,
                    output_slot,
                    output_offset,
                } => {
                    let profile = dependency_profile(function);
                    for (slot, input) in inputs.iter().enumerate() {
                        for (offset, &expr) in input.nonzeros().iter().enumerate() {
                            if profile.output_depends_on(output_slot, output_offset, slot, offset) {
                                stack.push(expr);
                            }
                        }
                    }
                }
            }
        }
        free
    }

    pub fn symbol_name(self) -> Option<String> {
        match node_kind(self) {
            NodeKind::Symbol { name, .. } => Some(name.to_string()),
            NodeKind::Constant(_)
            | NodeKind::Unary { .. }
            | NodeKind::Binary { .. }
            | NodeKind::Call { .. } => None,
        }
    }

    pub fn id(self) -> u64 {
        self.0
    }
}

impl SXContext {
    pub fn root() -> Self {
        drop(lock_context_registry());
        Self {
            id: ROOT_CONTEXT_ID,
        }
    }

    pub fn new() -> Self {
        let mut registry = lock_context_registry();
        let id = registry.next_context_id;
        registry.next_context_id += 1;
        registry.arenas.insert(
            id,
            Arc::new(ContextArena {
                arena: Mutex::new(NodeArena::new(id)),
            }),
        );
        Self { id }
    }

    pub fn id(self) -> u32 {
        self.id
    }

    pub fn scoped<R>(self, f: impl FnOnce() -> R) -> R {
        with_sx_context_id(self.id, f)
    }

    pub fn make_sym(self, name: impl Into<String>) -> SX {
        self.scoped(|| SX::sym(name))
    }
}

impl Default for SXContext {
    fn default() -> Self {
        Self::new()
    }
}

impl From<f64> for SX {
    fn from(value: f64) -> Self {
        SX::constant_in_context(current_context_id(), value)
    }
}

impl fmt::Display for SX {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        with_arena_ref_for_sx(*self, |arena| {
            let rendered = format_expression(*self, arena, &mut HashMap::new());
            write!(f, "{rendered}")
        })
    }
}

impl Neg for SX {
    type Output = SX;

    fn neg(self) -> Self::Output {
        SX::constant_in_context(self.context_id(), -1.0) * self
    }
}

impl Add for SX {
    type Output = SX;

    fn add(self, rhs: Self) -> Self::Output {
        binary(BinaryOp::Add, self, rhs)
    }
}

impl AddAssign for SX {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for SX {
    type Output = SX;

    fn sub(self, rhs: Self) -> Self::Output {
        binary(BinaryOp::Sub, self, rhs)
    }
}

impl SubAssign for SX {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for SX {
    type Output = SX;

    fn mul(self, rhs: Self) -> Self::Output {
        binary(BinaryOp::Mul, self, rhs)
    }
}

impl MulAssign for SX {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for SX {
    type Output = SX;

    fn div(self, rhs: Self) -> Self::Output {
        binary(BinaryOp::Div, self, rhs)
    }
}

impl DivAssign for SX {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Rem for SX {
    type Output = SX;

    fn rem(self, rhs: Self) -> Self::Output {
        binary(BinaryOp::Mod, self, rhs)
    }
}

impl RemAssign for SX {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl Add<f64> for SX {
    type Output = SX;

    fn add(self, rhs: f64) -> Self::Output {
        self + SX::constant_in_context(self.context_id(), rhs)
    }
}

impl Sub<f64> for SX {
    type Output = SX;

    fn sub(self, rhs: f64) -> Self::Output {
        self - SX::constant_in_context(self.context_id(), rhs)
    }
}

impl Mul<f64> for SX {
    type Output = SX;

    fn mul(self, rhs: f64) -> Self::Output {
        self * SX::constant_in_context(self.context_id(), rhs)
    }
}

impl Div<f64> for SX {
    type Output = SX;

    fn div(self, rhs: f64) -> Self::Output {
        self / SX::constant_in_context(self.context_id(), rhs)
    }
}

impl Rem<f64> for SX {
    type Output = SX;

    fn rem(self, rhs: f64) -> Self::Output {
        self % SX::constant_in_context(self.context_id(), rhs)
    }
}

impl Add<SX> for f64 {
    type Output = SX;

    fn add(self, rhs: SX) -> Self::Output {
        SX::constant_in_context(rhs.context_id(), self) + rhs
    }
}

impl Sub<SX> for f64 {
    type Output = SX;

    fn sub(self, rhs: SX) -> Self::Output {
        SX::constant_in_context(rhs.context_id(), self) - rhs
    }
}

impl Mul<SX> for f64 {
    type Output = SX;

    fn mul(self, rhs: SX) -> Self::Output {
        SX::constant_in_context(rhs.context_id(), self) * rhs
    }
}

impl Div<SX> for f64 {
    type Output = SX;

    fn div(self, rhs: SX) -> Self::Output {
        SX::constant_in_context(rhs.context_id(), self) / rhs
    }
}

impl Rem<SX> for f64 {
    type Output = SX;

    fn rem(self, rhs: SX) -> Self::Output {
        SX::constant_in_context(rhs.context_id(), self) % rhs
    }
}
