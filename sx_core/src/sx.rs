use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::ccs::CCS;
use crate::error::{Result, SxError};
use crate::function::{
    DependencyProfile, FunctionId, SXFunction, dependency_profile, forward_batch_helper,
    forward_helper, function_name, reverse_output_batch_helper, reverse_scalar_helper,
};
use crate::{Index, SXMatrix};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SX(u32);

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

#[derive(Clone, Debug)]
struct Node {
    kind: NodeKind,
}

#[derive(Default)]
struct Interner {
    nodes: Vec<Node>,
    keyed: HashMap<NodeKey, SX>,
    next_symbol_serial: usize,
}

static INTERNER: OnceLock<Mutex<Interner>> = OnceLock::new();

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

#[derive(Default)]
struct AdLocalCaches {
    dependency_profiles: HashMap<FunctionId, Arc<DependencyProfile>>,
    forward_helpers: HashMap<FunctionId, Arc<SXFunction>>,
    forward_batch_helpers: HashMap<(FunctionId, usize), Arc<SXFunction>>,
    reverse_scalar_helpers: HashMap<(FunctionId, Index, Index), Arc<SXFunction>>,
    reverse_output_batch_helpers: HashMap<(FunctionId, Index, Index, usize), Arc<SXFunction>>,
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

    fn apply_constant(self, arg: f64) -> f64 {
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

    fn apply_constant(self, lhs: f64, rhs: f64) -> f64 {
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

    fn is_commutative(self) -> bool {
        matches!(self, Self::Add | Self::Mul | Self::Hypot)
    }
}

fn with_interner<R>(f: impl FnOnce(&mut Interner) -> R) -> R {
    let mutex = INTERNER.get_or_init(|| Mutex::new(Interner::default()));
    let mut guard = lock_interner(mutex);
    f(&mut guard)
}

fn with_interner_ref<R>(f: impl FnOnce(&Interner) -> R) -> R {
    let mutex = INTERNER.get_or_init(|| Mutex::new(Interner::default()));
    let guard = lock_interner(mutex);
    f(&guard)
}

fn lock_interner(mutex: &Mutex<Interner>) -> MutexGuard<'_, Interner> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
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

impl Interner {
    fn node(&self, sx: SX) -> &Node {
        &self.nodes[sx.0 as usize]
    }

    fn node_kind_ref(&self, sx: SX) -> &NodeKind {
        &self.node(sx).kind
    }

    fn fresh_symbol(&mut self, name: impl Into<String>) -> SX {
        let serial = self.next_symbol_serial;
        self.next_symbol_serial += 1;
        let id = SX(self.nodes.len() as u32);
        self.nodes.push(Node {
            kind: NodeKind::Symbol {
                serial,
                name: Arc::<str>::from(name.into()),
            },
        });
        id
    }

    fn intern_keyed(&mut self, key: NodeKey, kind: NodeKind) -> SX {
        if let Some(existing) = self.keyed.get(&key) {
            return *existing;
        }
        let id = SX(self.nodes.len() as u32);
        self.nodes.push(Node { kind });
        self.keyed.insert(key, id);
        id
    }
}

pub(crate) fn node_kind(sx: SX) -> NodeKind {
    with_interner_ref(|interner| interner.node_kind_ref(sx).clone())
}

fn is_zero_kind(kind: &NodeKind) -> bool {
    matches!(kind, NodeKind::Constant(value) if *value == 0.0)
}

fn is_one_kind(kind: &NodeKind) -> bool {
    matches!(kind, NodeKind::Constant(value) if *value == 1.0)
}

fn constant_value_with_interner(interner: &Interner, sx: SX) -> Option<f64> {
    match interner.node_kind_ref(sx) {
        NodeKind::Constant(value) => Some(*value),
        NodeKind::Symbol { .. }
        | NodeKind::Unary { .. }
        | NodeKind::Binary { .. }
        | NodeKind::Call { .. } => None,
    }
}

fn mul_constant_factor_with_interner(interner: &Interner, sx: SX) -> Option<(f64, SX)> {
    match interner.node_kind_ref(sx) {
        NodeKind::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
        } => {
            if let Some(value) = constant_value_with_interner(interner, *lhs) {
                Some((value, *rhs))
            } else {
                constant_value_with_interner(interner, *rhs).map(|value| (value, *lhs))
            }
        }
        NodeKind::Constant(_value) => None,
        NodeKind::Symbol { .. }
        | NodeKind::Unary { .. }
        | NodeKind::Binary { .. }
        | NodeKind::Call { .. } => None,
    }
}

#[derive(Clone, Debug)]
struct RationalFactors {
    coeff: f64,
    numerators: Vec<SX>,
    denominators: Vec<SX>,
}

fn intern_constant_with_interner(interner: &mut Interner, value: f64) -> SX {
    interner.intern_keyed(
        NodeKey::Constant(value.to_bits()),
        NodeKind::Constant(value),
    )
}

fn intern_binary_with_interner(interner: &mut Interner, op: BinaryOp, lhs: SX, rhs: SX) -> SX {
    use NodeKey as K;
    use NodeKind as N;

    let (lhs, rhs) = if op.is_commutative() {
        canonical_pair(lhs, rhs)
    } else {
        (lhs, rhs)
    };
    interner.intern_keyed(K::Binary { op, lhs, rhs }, N::Binary { op, lhs, rhs })
}

fn canonicalize_rational_factors(mut factors: RationalFactors) -> RationalFactors {
    if factors.coeff == 0.0 {
        factors.numerators.clear();
        factors.denominators.clear();
        return factors;
    }

    factors.numerators.sort_unstable();
    factors.denominators.sort_unstable();

    let mut numerators = Vec::with_capacity(factors.numerators.len());
    let mut denominators = Vec::with_capacity(factors.denominators.len());
    let mut numerator_idx = 0;
    let mut denominator_idx = 0;

    while numerator_idx < factors.numerators.len() && denominator_idx < factors.denominators.len() {
        let numerator = factors.numerators[numerator_idx];
        let denominator = factors.denominators[denominator_idx];
        match numerator.cmp(&denominator) {
            std::cmp::Ordering::Equal => {
                numerator_idx += 1;
                denominator_idx += 1;
            }
            std::cmp::Ordering::Less => {
                numerators.push(numerator);
                numerator_idx += 1;
            }
            std::cmp::Ordering::Greater => {
                denominators.push(denominator);
                denominator_idx += 1;
            }
        }
    }

    numerators.extend_from_slice(&factors.numerators[numerator_idx..]);
    denominators.extend_from_slice(&factors.denominators[denominator_idx..]);

    RationalFactors {
        coeff: factors.coeff,
        numerators,
        denominators,
    }
}

fn combine_rational_factors(lhs: RationalFactors, rhs: RationalFactors) -> RationalFactors {
    let mut numerators = lhs.numerators;
    numerators.extend(rhs.numerators);
    let mut denominators = lhs.denominators;
    denominators.extend(rhs.denominators);
    canonicalize_rational_factors(RationalFactors {
        coeff: lhs.coeff * rhs.coeff,
        numerators,
        denominators,
    })
}

fn divide_rational_factors(lhs: RationalFactors, rhs: RationalFactors) -> Option<RationalFactors> {
    if rhs.coeff == 0.0 {
        return None;
    }

    let mut numerators = lhs.numerators;
    numerators.extend(rhs.denominators);
    let mut denominators = lhs.denominators;
    denominators.extend(rhs.numerators);
    Some(canonicalize_rational_factors(RationalFactors {
        coeff: lhs.coeff / rhs.coeff,
        numerators,
        denominators,
    }))
}

fn rational_factors_with_interner(interner: &Interner, expr: SX) -> RationalFactors {
    match interner.node_kind_ref(expr) {
        NodeKind::Constant(value) => canonicalize_rational_factors(RationalFactors {
            coeff: *value,
            numerators: Vec::new(),
            denominators: Vec::new(),
        }),
        NodeKind::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
        } => combine_rational_factors(
            rational_factors_with_interner(interner, *lhs),
            rational_factors_with_interner(interner, *rhs),
        ),
        NodeKind::Binary {
            op: BinaryOp::Div,
            lhs,
            rhs,
        } => divide_rational_factors(
            rational_factors_with_interner(interner, *lhs),
            rational_factors_with_interner(interner, *rhs),
        )
        .unwrap_or_else(|| RationalFactors {
            coeff: 1.0,
            numerators: vec![expr],
            denominators: Vec::new(),
        }),
        NodeKind::Symbol { .. }
        | NodeKind::Unary { .. }
        | NodeKind::Binary { .. }
        | NodeKind::Call { .. } => RationalFactors {
            coeff: 1.0,
            numerators: vec![expr],
            denominators: Vec::new(),
        },
    }
}

fn rebuild_rational_factors_with_interner(interner: &mut Interner, factors: RationalFactors) -> SX {
    if factors.coeff == 0.0 {
        return intern_constant_with_interner(interner, 0.0);
    }

    let mut numerators = factors.numerators.into_iter();
    let mut expr = match numerators.next() {
        Some(first) if factors.coeff == 1.0 => first,
        Some(first) => {
            let coeff = intern_constant_with_interner(interner, factors.coeff);
            intern_binary_with_interner(interner, BinaryOp::Mul, coeff, first)
        }
        None => intern_constant_with_interner(interner, factors.coeff),
    };

    for numerator in numerators {
        expr = intern_binary_with_interner(interner, BinaryOp::Mul, expr, numerator);
    }

    for denominator in factors.denominators {
        expr = intern_binary_with_interner(interner, BinaryOp::Div, expr, denominator);
    }

    expr
}

fn combine_like_terms_with_interner(
    interner: &mut Interner,
    lhs: SX,
    rhs: SX,
    rhs_sign: f64,
) -> Option<SX> {
    let lhs_factors = rational_factors_with_interner(interner, lhs);
    let mut rhs_factors = rational_factors_with_interner(interner, rhs);
    rhs_factors.coeff *= rhs_sign;
    rhs_factors = canonicalize_rational_factors(rhs_factors);

    if lhs_factors.numerators != rhs_factors.numerators
        || lhs_factors.denominators != rhs_factors.denominators
    {
        return None;
    }

    Some(rebuild_rational_factors_with_interner(interner, canonicalize_rational_factors(
        RationalFactors {
            coeff: lhs_factors.coeff + rhs_factors.coeff,
            numerators: lhs_factors.numerators,
            denominators: lhs_factors.denominators,
        },
    )))
}

fn neg_with_interner(interner: &mut Interner, expr: SX) -> SX {
    let minus_one = intern_constant_with_interner(interner, -1.0);
    binary_with_interner(interner, BinaryOp::Mul, minus_one, expr)
}

fn unary_with_interner(interner: &mut Interner, op: UnaryOp, arg: SX) -> SX {
    use NodeKey as K;
    use NodeKind as N;

    let arg_kind = interner.node_kind_ref(arg).clone();
    if let Some(value) = constant_value_with_interner(interner, arg) {
        return intern_constant_with_interner(interner, op.apply_constant(value));
    }

    match op {
        UnaryOp::Abs => {
            if is_zero_kind(&arg_kind) {
                return intern_constant_with_interner(interner, 0.0);
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
                return intern_constant_with_interner(interner, 0.0);
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
                return intern_constant_with_interner(interner, 0.0);
            }
        }
        UnaryOp::Sqrt => {
            if is_zero_kind(&arg_kind) {
                return intern_constant_with_interner(interner, 0.0);
            }
            if is_one_kind(&arg_kind) {
                return intern_constant_with_interner(interner, 1.0);
            }
        }
        UnaryOp::Exp | UnaryOp::Cos | UnaryOp::Cosh => {
            if is_zero_kind(&arg_kind) {
                return intern_constant_with_interner(interner, 1.0);
            }
        }
        UnaryOp::Log => {
            if is_one_kind(&arg_kind) {
                return intern_constant_with_interner(interner, 0.0);
            }
        }
        UnaryOp::Acos | UnaryOp::Acosh | UnaryOp::Atanh => {}
    }

    interner.intern_keyed(K::Unary { op, arg }, N::Unary { op, arg })
}

fn binary_with_interner(interner: &mut Interner, op: BinaryOp, lhs: SX, rhs: SX) -> SX {
    use NodeKind as N;

    let lhs_kind = interner.node_kind_ref(lhs).clone();
    let rhs_kind = interner.node_kind_ref(rhs).clone();

    if let (N::Constant(a), N::Constant(b)) = (&lhs_kind, &rhs_kind) {
        return intern_constant_with_interner(interner, op.apply_constant(*a, *b));
    }

    match op {
        BinaryOp::Add => {
            if is_zero_kind(&lhs_kind) {
                return rhs;
            }
            if is_zero_kind(&rhs_kind) {
                return lhs;
            }
            if let Some(combined) = combine_like_terms_with_interner(interner, lhs, rhs, 1.0) {
                return combined;
            }
        }
        BinaryOp::Sub => {
            if is_zero_kind(&lhs_kind) {
                return neg_with_interner(interner, rhs);
            }
            if is_zero_kind(&rhs_kind) {
                return lhs;
            }
            if lhs == rhs {
                return intern_constant_with_interner(interner, 0.0);
            }
            if let Some(combined) = combine_like_terms_with_interner(interner, lhs, rhs, -1.0) {
                return combined;
            }
        }
        BinaryOp::Mul => {
            if is_zero_kind(&lhs_kind) || is_zero_kind(&rhs_kind) {
                return intern_constant_with_interner(interner, 0.0);
            }
            if is_one_kind(&lhs_kind) {
                return rhs;
            }
            if is_one_kind(&rhs_kind) {
                return lhs;
            }
            if let Some(lhs_value) = constant_value_with_interner(interner, lhs)
                && let Some((rhs_value, factor)) = mul_constant_factor_with_interner(interner, rhs)
            {
                let scaled = intern_constant_with_interner(interner, lhs_value * rhs_value);
                return binary_with_interner(
                    interner,
                    BinaryOp::Mul,
                    scaled,
                    factor,
                );
            }
            if let Some(rhs_value) = constant_value_with_interner(interner, rhs)
                && let Some((lhs_value, factor)) = mul_constant_factor_with_interner(interner, lhs)
            {
                let scaled = intern_constant_with_interner(interner, lhs_value * rhs_value);
                return binary_with_interner(
                    interner,
                    BinaryOp::Mul,
                    scaled,
                    factor,
                );
            }
            return rebuild_rational_factors_with_interner(
                interner,
                combine_rational_factors(
                    rational_factors_with_interner(interner, lhs),
                    rational_factors_with_interner(interner, rhs),
                ),
            );
        }
        BinaryOp::Div => {
            if is_zero_kind(&lhs_kind) {
                return intern_constant_with_interner(interner, 0.0);
            }
            if is_one_kind(&rhs_kind) {
                return lhs;
            }
            if let Some(divided) = divide_rational_factors(
                rational_factors_with_interner(interner, lhs),
                rational_factors_with_interner(interner, rhs),
            ) {
                return rebuild_rational_factors_with_interner(interner, divided);
            }
        }
        BinaryOp::Pow => {
            if is_zero_kind(&rhs_kind) {
                return intern_constant_with_interner(interner, 1.0);
            }
            if is_one_kind(&rhs_kind) {
                return lhs;
            }
            if let Some(exponent) = constant_value_with_interner(interner, rhs) {
                if exponent == 2.0 {
                    return binary_with_interner(interner, BinaryOp::Mul, lhs, lhs);
                }
                if exponent == 0.5 {
                    return unary_with_interner(interner, UnaryOp::Sqrt, lhs);
                }
                if is_zero_kind(&lhs_kind) && exponent > 0.0 {
                    return intern_constant_with_interner(interner, 0.0);
                }
            }
            if is_one_kind(&lhs_kind) {
                return intern_constant_with_interner(interner, 1.0);
            }
        }
        BinaryOp::Atan2 => {
            if is_zero_kind(&lhs_kind) && is_one_kind(&rhs_kind) {
                return intern_constant_with_interner(interner, 0.0);
            }
        }
        BinaryOp::Hypot => {
            if is_zero_kind(&lhs_kind) {
                return unary_with_interner(interner, UnaryOp::Abs, rhs);
            }
            if is_zero_kind(&rhs_kind) {
                return unary_with_interner(interner, UnaryOp::Abs, lhs);
            }
        }
        BinaryOp::Mod => {
            if is_zero_kind(&lhs_kind) {
                return intern_constant_with_interner(interner, 0.0);
            }
        }
        BinaryOp::Copysign => {
            if is_zero_kind(&lhs_kind) {
                return intern_constant_with_interner(interner, 0.0);
            }
            if is_zero_kind(&rhs_kind) {
                return unary_with_interner(interner, UnaryOp::Abs, lhs);
            }
        }
    }

    intern_binary_with_interner(interner, op, lhs, rhs)
}

fn format_expression(expr: SX, interner: &Interner, memo: &mut HashMap<SX, String>) -> String {
    if let Some(existing) = memo.get(&expr) {
        return existing.clone();
    }
    let formatted = match interner.node(expr).kind.clone() {
        NodeKind::Constant(v) => {
            if v.fract() == 0.0 {
                format!("{v:.1}")
            } else {
                format!("{v}")
            }
        }
        NodeKind::Symbol { name, .. } => name.to_string(),
        NodeKind::Unary { op, arg } => {
            format!("{}({})", op.name(), format_expression(arg, interner, memo))
        }
        NodeKind::Binary { op, lhs, rhs } => {
            if op == BinaryOp::Mul {
                if matches!(interner.node(lhs).kind, NodeKind::Constant(value) if value == -1.0) {
                    let rendered = format_expression(rhs, interner, memo);
                    let negated = format!("(-{rendered})");
                    memo.insert(expr, negated.clone());
                    return negated;
                }
                if matches!(interner.node(rhs).kind, NodeKind::Constant(value) if value == -1.0) {
                    let rendered = format_expression(lhs, interner, memo);
                    let negated = format!("(-{rendered})");
                    memo.insert(expr, negated.clone());
                    return negated;
                }
            }
            let lhs_rendered = format_expression(lhs, interner, memo);
            let rhs_rendered = format_expression(rhs, interner, memo);
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
    with_interner(|interner| {
        interner.intern_keyed(
            NodeKey::Call {
                function,
                inputs: inputs.clone(),
                output_slot,
                output_offset,
            },
            NodeKind::Call {
                function,
                inputs,
                output_slot,
                output_offset,
            },
        )
    })
}

fn canonical_pair(lhs: SX, rhs: SX) -> (SX, SX) {
    if lhs <= rhs { (lhs, rhs) } else { (rhs, lhs) }
}

fn unary(op: UnaryOp, arg: SX) -> SX {
    with_interner(|interner| unary_with_interner(interner, op, arg))
}

fn binary(op: BinaryOp, lhs: SX, rhs: SX) -> SX {
    with_interner(|interner| binary_with_interner(interner, op, lhs, rhs))
}

fn unary_derivative(op: UnaryOp, arg: SX) -> SX {
    with_interner(|interner| unary_derivative_with_interner(interner, op, arg))
}

fn unary_derivative_with_interner(interner: &mut Interner, op: UnaryOp, arg: SX) -> SX {
    match op {
        UnaryOp::Abs => unary_with_interner(interner, UnaryOp::Sign, arg),
        UnaryOp::Sign | UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Trunc => {
            intern_constant_with_interner(interner, 0.0)
        }
        UnaryOp::Sqrt => {
            let half = intern_constant_with_interner(interner, 0.5);
            let sqrt = unary_with_interner(interner, UnaryOp::Sqrt, arg);
            binary_with_interner(interner, BinaryOp::Div, half, sqrt)
        }
        UnaryOp::Exp => unary_with_interner(interner, UnaryOp::Exp, arg),
        UnaryOp::Log => {
            let one = intern_constant_with_interner(interner, 1.0);
            binary_with_interner(interner, BinaryOp::Div, one, arg)
        }
        UnaryOp::Sin => unary_with_interner(interner, UnaryOp::Cos, arg),
        UnaryOp::Cos => {
            let sin = unary_with_interner(interner, UnaryOp::Sin, arg);
            let neg_one = intern_constant_with_interner(interner, -1.0);
            binary_with_interner(interner, BinaryOp::Mul, neg_one, sin)
        }
        UnaryOp::Tan => {
            let cos = unary_with_interner(interner, UnaryOp::Cos, arg);
            let cos_sq = binary_with_interner(interner, BinaryOp::Mul, cos, cos);
            let one = intern_constant_with_interner(interner, 1.0);
            binary_with_interner(interner, BinaryOp::Div, one, cos_sq)
        }
        UnaryOp::Asin => {
            let arg_sq = binary_with_interner(interner, BinaryOp::Mul, arg, arg);
            let one = intern_constant_with_interner(interner, 1.0);
            let radicand = binary_with_interner(interner, BinaryOp::Sub, one, arg_sq);
            let denom = unary_with_interner(interner, UnaryOp::Sqrt, radicand);
            let one = intern_constant_with_interner(interner, 1.0);
            binary_with_interner(interner, BinaryOp::Div, one, denom)
        }
        UnaryOp::Acos => {
            let arg_sq = binary_with_interner(interner, BinaryOp::Mul, arg, arg);
            let one = intern_constant_with_interner(interner, 1.0);
            let radicand = binary_with_interner(interner, BinaryOp::Sub, one, arg_sq);
            let denom = unary_with_interner(interner, UnaryOp::Sqrt, radicand);
            let neg_one = intern_constant_with_interner(interner, -1.0);
            binary_with_interner(
                interner,
                BinaryOp::Div,
                neg_one,
                denom,
            )
        }
        UnaryOp::Atan => {
            let arg_sq = binary_with_interner(interner, BinaryOp::Mul, arg, arg);
            let one = intern_constant_with_interner(interner, 1.0);
            let denom = binary_with_interner(interner, BinaryOp::Add, one, arg_sq);
            let one = intern_constant_with_interner(interner, 1.0);
            binary_with_interner(interner, BinaryOp::Div, one, denom)
        }
        UnaryOp::Sinh => unary_with_interner(interner, UnaryOp::Cosh, arg),
        UnaryOp::Cosh => unary_with_interner(interner, UnaryOp::Sinh, arg),
        UnaryOp::Tanh => {
            let cosh = unary_with_interner(interner, UnaryOp::Cosh, arg);
            let cosh_sq = binary_with_interner(interner, BinaryOp::Mul, cosh, cosh);
            let one = intern_constant_with_interner(interner, 1.0);
            binary_with_interner(interner, BinaryOp::Div, one, cosh_sq)
        }
        UnaryOp::Asinh => {
            let arg_sq = binary_with_interner(interner, BinaryOp::Mul, arg, arg);
            let one = intern_constant_with_interner(interner, 1.0);
            let radicand = binary_with_interner(interner, BinaryOp::Add, arg_sq, one);
            let denom = unary_with_interner(interner, UnaryOp::Sqrt, radicand);
            let one = intern_constant_with_interner(interner, 1.0);
            binary_with_interner(interner, BinaryOp::Div, one, denom)
        }
        UnaryOp::Acosh => {
            let one = intern_constant_with_interner(interner, 1.0);
            let arg_minus_one = binary_with_interner(interner, BinaryOp::Sub, arg, one);
            let left = unary_with_interner(
                interner,
                UnaryOp::Sqrt,
                arg_minus_one,
            );
            let one = intern_constant_with_interner(interner, 1.0);
            let arg_plus_one = binary_with_interner(interner, BinaryOp::Add, arg, one);
            let right = unary_with_interner(
                interner,
                UnaryOp::Sqrt,
                arg_plus_one,
            );
            let denom = binary_with_interner(interner, BinaryOp::Mul, left, right);
            let one = intern_constant_with_interner(interner, 1.0);
            binary_with_interner(interner, BinaryOp::Div, one, denom)
        }
        UnaryOp::Atanh => {
            let arg_sq = binary_with_interner(interner, BinaryOp::Mul, arg, arg);
            let one = intern_constant_with_interner(interner, 1.0);
            let denom = binary_with_interner(interner, BinaryOp::Sub, one, arg_sq);
            let one = intern_constant_with_interner(interner, 1.0);
            binary_with_interner(interner, BinaryOp::Div, one, denom)
        }
    }
}

fn binary_partials(op: BinaryOp, lhs: SX, rhs: SX) -> (SX, SX) {
    with_interner(|interner| binary_partials_with_interner(interner, op, lhs, rhs))
}

fn binary_partials_with_interner(
    interner: &mut Interner,
    op: BinaryOp,
    lhs: SX,
    rhs: SX,
) -> (SX, SX) {
    match op {
        BinaryOp::Add => (
            intern_constant_with_interner(interner, 1.0),
            intern_constant_with_interner(interner, 1.0),
        ),
        BinaryOp::Sub => (
            intern_constant_with_interner(interner, 1.0),
            intern_constant_with_interner(interner, -1.0),
        ),
        BinaryOp::Mul => (rhs, lhs),
        BinaryOp::Div => {
            let one = intern_constant_with_interner(interner, 1.0);
            let lhs_partial = binary_with_interner(interner, BinaryOp::Div, one, rhs);
            let rhs_sq = binary_with_interner(interner, BinaryOp::Mul, rhs, rhs);
            let neg_one = intern_constant_with_interner(interner, -1.0);
            let neg_lhs = binary_with_interner(interner, BinaryOp::Mul, neg_one, lhs);
            let rhs_partial = binary_with_interner(interner, BinaryOp::Div, neg_lhs, rhs_sq);
            (lhs_partial, rhs_partial)
        }
        BinaryOp::Pow => {
            let pow = binary_with_interner(interner, BinaryOp::Pow, lhs, rhs);
            let one = intern_constant_with_interner(interner, 1.0);
            let rhs_minus_one = binary_with_interner(interner, BinaryOp::Sub, rhs, one);
            let lhs_pow = binary_with_interner(interner, BinaryOp::Pow, lhs, rhs_minus_one);
            let lhs_partial = binary_with_interner(
                interner,
                BinaryOp::Mul,
                rhs,
                lhs_pow,
            );
            let lhs_log = unary_with_interner(interner, UnaryOp::Log, lhs);
            let rhs_partial = binary_with_interner(interner, BinaryOp::Mul, pow, lhs_log);
            (lhs_partial, rhs_partial)
        }
        BinaryOp::Atan2 => {
            let lhs_sq = binary_with_interner(interner, BinaryOp::Mul, lhs, lhs);
            let rhs_sq = binary_with_interner(interner, BinaryOp::Mul, rhs, rhs);
            let denom = binary_with_interner(interner, BinaryOp::Add, lhs_sq, rhs_sq);
            let lhs_partial = binary_with_interner(interner, BinaryOp::Div, rhs, denom);
            let neg_one = intern_constant_with_interner(interner, -1.0);
            let neg_lhs = binary_with_interner(interner, BinaryOp::Mul, neg_one, lhs);
            let rhs_partial = binary_with_interner(interner, BinaryOp::Div, neg_lhs, denom);
            (lhs_partial, rhs_partial)
        }
        BinaryOp::Hypot => {
            let hypot = binary_with_interner(interner, BinaryOp::Hypot, lhs, rhs);
            let lhs_partial = binary_with_interner(interner, BinaryOp::Div, lhs, hypot);
            let rhs_partial = binary_with_interner(interner, BinaryOp::Div, rhs, hypot);
            (lhs_partial, rhs_partial)
        }
        BinaryOp::Mod => {
            let lhs_over_rhs = binary_with_interner(interner, BinaryOp::Div, lhs, rhs);
            let trunc = unary_with_interner(interner, UnaryOp::Trunc, lhs_over_rhs);
            let neg_one = intern_constant_with_interner(interner, -1.0);
            let rhs_partial = binary_with_interner(interner, BinaryOp::Mul, neg_one, trunc);
            let one = intern_constant_with_interner(interner, 1.0);
            (one, rhs_partial)
        }
        BinaryOp::Copysign => {
            let rhs_sign = unary_with_interner(interner, UnaryOp::Sign, rhs);
            let rhs_sign_abs = unary_with_interner(interner, UnaryOp::Abs, rhs_sign);
            let one = intern_constant_with_interner(interner, 1.0);
            let rhs_sign_term = binary_with_interner(interner, BinaryOp::Sub, one, rhs_sign_abs);
            let rhs_sign_full = binary_with_interner(interner, BinaryOp::Add, rhs_sign, rhs_sign_term);
            let lhs_sign = unary_with_interner(interner, UnaryOp::Sign, lhs);
            (
                binary_with_interner(interner, BinaryOp::Mul, lhs_sign, rhs_sign_full),
                intern_constant_with_interner(interner, 0.0),
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
}

fn execute_program_forward(
    program: &SxProgram,
    vars: &[SX],
    seeds: &[SX],
) -> Result<Vec<SX>> {
    let mut derivative_slots = vec![SX::zero(); program.slot_exprs.len()];
    for (var, seed) in vars.iter().copied().zip(seeds.iter().copied()) {
        if let Some(&slot) = program.slot_by_node.get(&var) {
            derivative_slots[slot] = seed;
        }
    }

    let mut local_caches = AdLocalCaches::default();
    let mut call_memo = HashMap::<ForwardCallCacheKey, Vec<SXMatrix>>::new();
    for instruction in &program.instructions {
        match instruction {
            ProgramInstruction::Unary {
                result_slot,
                op,
                arg_slot,
                arg_expr,
            } => {
                with_interner(|interner| {
                    let derivative = unary_derivative_with_interner(interner, *op, *arg_expr);
                    derivative_slots[*result_slot] = binary_with_interner(
                        interner,
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
                with_interner(|interner| {
                    derivative_slots[*result_slot] = match op {
                        BinaryOp::Add => binary_with_interner(
                            interner,
                            BinaryOp::Add,
                            derivative_slots[*lhs_slot],
                            derivative_slots[*rhs_slot],
                        ),
                        BinaryOp::Sub => binary_with_interner(
                            interner,
                            BinaryOp::Sub,
                            derivative_slots[*lhs_slot],
                            derivative_slots[*rhs_slot],
                        ),
                        BinaryOp::Mul => {
                            let left = binary_with_interner(
                                interner,
                                BinaryOp::Mul,
                                derivative_slots[*lhs_slot],
                                *rhs_expr,
                            );
                            let right = binary_with_interner(
                                interner,
                                BinaryOp::Mul,
                                *lhs_expr,
                                derivative_slots[*rhs_slot],
                            );
                            binary_with_interner(interner, BinaryOp::Add, left, right)
                        }
                        BinaryOp::Div => {
                            let left = binary_with_interner(
                                interner,
                                BinaryOp::Mul,
                                derivative_slots[*lhs_slot],
                                *rhs_expr,
                            );
                            let right = binary_with_interner(
                                interner,
                                BinaryOp::Mul,
                                *lhs_expr,
                                derivative_slots[*rhs_slot],
                            );
                            let numer = binary_with_interner(interner, BinaryOp::Sub, left, right);
                            let rhs_sq =
                                binary_with_interner(interner, BinaryOp::Mul, *rhs_expr, *rhs_expr);
                            binary_with_interner(interner, BinaryOp::Div, numer, rhs_sq)
                        }
                        BinaryOp::Pow
                        | BinaryOp::Atan2
                        | BinaryOp::Hypot
                        | BinaryOp::Mod
                        | BinaryOp::Copysign => {
                            let (d_lhs, d_rhs) =
                                binary_partials_with_interner(interner, *op, *lhs_expr, *rhs_expr);
                            let left = binary_with_interner(
                                interner,
                                BinaryOp::Mul,
                                derivative_slots[*lhs_slot],
                                d_lhs,
                            );
                            let right = binary_with_interner(
                                interner,
                                BinaryOp::Mul,
                                derivative_slots[*rhs_slot],
                                d_rhs,
                            );
                            binary_with_interner(interner, BinaryOp::Add, left, right)
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
                } else if let Some(existing) = call_memo.get(site_key) {
                    existing[*output_slot].nz(*output_offset)
                } else {
                    let mut seed_inputs = Vec::with_capacity(inputs.len());
                    for input in inputs {
                        let mut seed_nonzeros = vec![SX::zero(); input.slots.len()];
                        for &offset in &input.relevant_offsets {
                            seed_nonzeros[offset] = derivative_slots[input.slots[offset]];
                        }
                        seed_inputs.push(SXMatrix::new(input.matrix.ccs().clone(), seed_nonzeros)?);
                    }
                    let helper = local_caches.forward_helper(*function)?;
                    let mut helper_inputs =
                        inputs.iter().map(|input| input.matrix.clone()).collect::<Vec<_>>();
                    helper_inputs.extend(seed_inputs);
                    let helper_outputs = helper.call(&helper_inputs)?;
                    let selected = helper_outputs[*output_slot].nz(*output_offset);
                    call_memo.insert(site_key.clone(), helper_outputs);
                    selected
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

    execute_program_forward_batch_with_slots(
        program,
        derivative_slots,
        active_masks,
        direction_count,
    )
}

fn execute_program_forward_basis_batch(
    program: &SxProgram,
    vars: &[SX],
    active_var_groups: &[Vec<Index>],
) -> Result<Vec<Vec<SX>>> {
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

fn execute_program_forward_batch_with_slots(
    program: &SxProgram,
    mut derivative_slots: Vec<SX>,
    mut active_masks: Option<Vec<u64>>,
    direction_count: usize,
) -> Result<Vec<Vec<SX>>> {
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
                    with_interner(|interner| {
                        let derivative = unary_derivative_with_interner(interner, *op, *arg_expr);
                        for direction in iter_direction_bits(arg_mask) {
                            derivative_slots[result_base + direction] = binary_with_interner(
                                interner,
                                BinaryOp::Mul,
                                derivative_slots[arg_base + direction],
                                derivative,
                            );
                        }
                    });
                } else {
                    with_interner(|interner| {
                        let derivative = unary_derivative_with_interner(interner, *op, *arg_expr);
                        for direction in 0..direction_count {
                            derivative_slots[result_base + direction] = binary_with_interner(
                                interner,
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
                    with_interner(|interner| match op {
                        BinaryOp::Add => {
                            for direction in iter_direction_bits(result_mask) {
                                derivative_slots[result_base + direction] = binary_with_interner(
                                    interner,
                                    BinaryOp::Add,
                                    derivative_slots[lhs_base + direction],
                                    derivative_slots[rhs_base + direction],
                                );
                            }
                        }
                        BinaryOp::Sub => {
                            for direction in iter_direction_bits(result_mask) {
                                derivative_slots[result_base + direction] = binary_with_interner(
                                    interner,
                                    BinaryOp::Sub,
                                    derivative_slots[lhs_base + direction],
                                    derivative_slots[rhs_base + direction],
                                );
                            }
                        }
                        BinaryOp::Mul => {
                            for direction in iter_direction_bits(result_mask) {
                                let left = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    derivative_slots[lhs_base + direction],
                                    *rhs_expr,
                                );
                                let right = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    *lhs_expr,
                                    derivative_slots[rhs_base + direction],
                                );
                                derivative_slots[result_base + direction] =
                                    binary_with_interner(interner, BinaryOp::Add, left, right);
                            }
                        }
                        BinaryOp::Div => {
                            let rhs_sq = binary_with_interner(
                                interner,
                                BinaryOp::Mul,
                                *rhs_expr,
                                *rhs_expr,
                            );
                            for direction in iter_direction_bits(result_mask) {
                                let left = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    derivative_slots[lhs_base + direction],
                                    *rhs_expr,
                                );
                                let right = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    *lhs_expr,
                                    derivative_slots[rhs_base + direction],
                                );
                                let numer =
                                    binary_with_interner(interner, BinaryOp::Sub, left, right);
                                derivative_slots[result_base + direction] =
                                    binary_with_interner(interner, BinaryOp::Div, numer, rhs_sq);
                            }
                        }
                        BinaryOp::Pow
                        | BinaryOp::Atan2
                        | BinaryOp::Hypot
                        | BinaryOp::Mod
                        | BinaryOp::Copysign => {
                            let (d_lhs, d_rhs) = binary_partials_with_interner(
                                interner,
                                *op,
                                *lhs_expr,
                                *rhs_expr,
                            );
                            for direction in iter_direction_bits(result_mask) {
                                let left = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    derivative_slots[lhs_base + direction],
                                    d_lhs,
                                );
                                let right = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    derivative_slots[rhs_base + direction],
                                    d_rhs,
                                );
                                derivative_slots[result_base + direction] =
                                    binary_with_interner(interner, BinaryOp::Add, left, right);
                            }
                        }
                    });
                } else {
                    with_interner(|interner| match op {
                        BinaryOp::Add => {
                            for direction in 0..direction_count {
                                derivative_slots[result_base + direction] = binary_with_interner(
                                    interner,
                                    BinaryOp::Add,
                                    derivative_slots[lhs_base + direction],
                                    derivative_slots[rhs_base + direction],
                                );
                            }
                        }
                        BinaryOp::Sub => {
                            for direction in 0..direction_count {
                                derivative_slots[result_base + direction] = binary_with_interner(
                                    interner,
                                    BinaryOp::Sub,
                                    derivative_slots[lhs_base + direction],
                                    derivative_slots[rhs_base + direction],
                                );
                            }
                        }
                        BinaryOp::Mul => {
                            for direction in 0..direction_count {
                                let left = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    derivative_slots[lhs_base + direction],
                                    *rhs_expr,
                                );
                                let right = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    *lhs_expr,
                                    derivative_slots[rhs_base + direction],
                                );
                                derivative_slots[result_base + direction] =
                                    binary_with_interner(interner, BinaryOp::Add, left, right);
                            }
                        }
                        BinaryOp::Div => {
                            let rhs_sq = binary_with_interner(
                                interner,
                                BinaryOp::Mul,
                                *rhs_expr,
                                *rhs_expr,
                            );
                            for direction in 0..direction_count {
                                let left = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    derivative_slots[lhs_base + direction],
                                    *rhs_expr,
                                );
                                let right = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    *lhs_expr,
                                    derivative_slots[rhs_base + direction],
                                );
                                let numer =
                                    binary_with_interner(interner, BinaryOp::Sub, left, right);
                                derivative_slots[result_base + direction] =
                                    binary_with_interner(interner, BinaryOp::Div, numer, rhs_sq);
                            }
                        }
                        BinaryOp::Pow
                        | BinaryOp::Atan2
                        | BinaryOp::Hypot
                        | BinaryOp::Mod
                        | BinaryOp::Copysign => {
                            let (d_lhs, d_rhs) = binary_partials_with_interner(
                                interner,
                                *op,
                                *lhs_expr,
                                *rhs_expr,
                            );
                            for direction in 0..direction_count {
                                let left = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    derivative_slots[lhs_base + direction],
                                    d_lhs,
                                );
                                let right = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    derivative_slots[rhs_base + direction],
                                    d_rhs,
                                );
                                derivative_slots[result_base + direction] =
                                    binary_with_interner(interner, BinaryOp::Add, left, right);
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
                        let key = ForwardBatchCallCacheKey {
                            site_key: site_key.clone(),
                            direction_count: active_count,
                            direction_mask,
                        };
                        if let Some(existing) = call_memo.get(&key) {
                            let mut selected = vec![SX::zero(); direction_count];
                            for (local_direction, &direction) in active_directions.iter().enumerate()
                            {
                                selected[direction] =
                                    existing[*output_slot * active_count + local_direction]
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
                                selected[direction] =
                                    helper_outputs[*output_slot * active_count + local_direction]
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
                        let key = ForwardBatchCallCacheKey {
                            site_key: site_key.clone(),
                            direction_count,
                            direction_mask: u64::MAX,
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
                for direction in 0..direction_count {
                    derivative_slots[result_base + direction] = result_values[direction];
                }
            }
        }
    }

    Ok((0..direction_count)
        .map(|direction| {
            program
                .output_slots
                .iter()
                .map(|&slot| derivative_slots[slot_base(slot) + direction])
                .collect::<Vec<_>>()
        })
        .collect())
}

fn execute_program_reverse(
    program: &SxProgram,
    vars: &[SX],
    seeds: &[SX],
) -> Result<Vec<SX>> {
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
                    with_interner(|interner| {
                        let derivative = unary_derivative_with_interner(interner, *op, *arg_expr);
                        let contribution =
                            binary_with_interner(interner, BinaryOp::Mul, adj, derivative);
                        adjoint_slots[*arg_slot] = binary_with_interner(
                            interner,
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
                with_interner(|interner| match op {
                    BinaryOp::Add => {
                        adjoint_slots[*lhs_slot] = binary_with_interner(
                            interner,
                            BinaryOp::Add,
                            adjoint_slots[*lhs_slot],
                            adj,
                        );
                        adjoint_slots[*rhs_slot] = binary_with_interner(
                            interner,
                            BinaryOp::Add,
                            adjoint_slots[*rhs_slot],
                            adj,
                        );
                    }
                    BinaryOp::Sub => {
                        adjoint_slots[*lhs_slot] = binary_with_interner(
                            interner,
                            BinaryOp::Add,
                            adjoint_slots[*lhs_slot],
                            adj,
                        );
                        adjoint_slots[*rhs_slot] = binary_with_interner(
                            interner,
                            BinaryOp::Sub,
                            adjoint_slots[*rhs_slot],
                            adj,
                        );
                    }
                    BinaryOp::Mul => {
                        let lhs_contrib =
                            binary_with_interner(interner, BinaryOp::Mul, adj, *rhs_expr);
                        let rhs_contrib =
                            binary_with_interner(interner, BinaryOp::Mul, adj, *lhs_expr);
                        adjoint_slots[*lhs_slot] = binary_with_interner(
                            interner,
                            BinaryOp::Add,
                            adjoint_slots[*lhs_slot],
                            lhs_contrib,
                        );
                        adjoint_slots[*rhs_slot] = binary_with_interner(
                            interner,
                            BinaryOp::Add,
                            adjoint_slots[*rhs_slot],
                            rhs_contrib,
                        );
                    }
                    BinaryOp::Div => {
                        let lhs_contrib =
                            binary_with_interner(interner, BinaryOp::Div, adj, *rhs_expr);
                        let neg_one = intern_constant_with_interner(interner, -1.0);
                        let neg_adj = binary_with_interner(interner, BinaryOp::Mul, neg_one, adj);
                        let numer =
                            binary_with_interner(interner, BinaryOp::Mul, neg_adj, *lhs_expr);
                        let rhs_sq =
                            binary_with_interner(interner, BinaryOp::Mul, *rhs_expr, *rhs_expr);
                        let rhs_contrib =
                            binary_with_interner(interner, BinaryOp::Div, numer, rhs_sq);
                        adjoint_slots[*lhs_slot] = binary_with_interner(
                            interner,
                            BinaryOp::Add,
                            adjoint_slots[*lhs_slot],
                            lhs_contrib,
                        );
                        adjoint_slots[*rhs_slot] = binary_with_interner(
                            interner,
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
                            binary_partials_with_interner(interner, *op, *lhs_expr, *rhs_expr);
                        let lhs_contrib =
                            binary_with_interner(interner, BinaryOp::Mul, adj, d_lhs);
                        let rhs_contrib =
                            binary_with_interner(interner, BinaryOp::Mul, adj, d_rhs);
                        adjoint_slots[*lhs_slot] = binary_with_interner(
                            interner,
                            BinaryOp::Add,
                            adjoint_slots[*lhs_slot],
                            lhs_contrib,
                        );
                        adjoint_slots[*rhs_slot] = binary_with_interner(
                            interner,
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
                    let helper =
                        local_caches.reverse_scalar_helper(*function, *output_slot, *output_offset)?;
                    let helper_outputs =
                        helper.call(&inputs.iter().map(|input| input.matrix.clone()).collect::<Vec<_>>())?;
                    reverse_call_memo.insert(*result_slot, helper_outputs);
                    reverse_call_memo.get(result_slot).expect("inserted reverse call helper output")
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
                with_interner(|interner| {
                    let derivative = unary_derivative_with_interner(interner, *op, *arg_expr);
                    for (direction, adj) in &active_adjoints {
                        let contribution =
                            binary_with_interner(interner, BinaryOp::Mul, *adj, derivative);
                        adjoint_slots[*arg_slot][*direction] = binary_with_interner(
                            interner,
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
                with_interner(|interner| {
                    let partials = match op {
                        BinaryOp::Pow
                        | BinaryOp::Atan2
                        | BinaryOp::Hypot
                        | BinaryOp::Mod
                        | BinaryOp::Copysign => {
                            Some(binary_partials_with_interner(interner, *op, *lhs_expr, *rhs_expr))
                        }
                        _ => None,
                    };
                    for (direction, adj) in &active_adjoints {
                        match op {
                            BinaryOp::Add => {
                                adjoint_slots[*lhs_slot][*direction] = binary_with_interner(
                                    interner,
                                    BinaryOp::Add,
                                    adjoint_slots[*lhs_slot][*direction],
                                    *adj,
                                );
                                adjoint_slots[*rhs_slot][*direction] = binary_with_interner(
                                    interner,
                                    BinaryOp::Add,
                                    adjoint_slots[*rhs_slot][*direction],
                                    *adj,
                                );
                            }
                            BinaryOp::Sub => {
                                adjoint_slots[*lhs_slot][*direction] = binary_with_interner(
                                    interner,
                                    BinaryOp::Add,
                                    adjoint_slots[*lhs_slot][*direction],
                                    *adj,
                                );
                                adjoint_slots[*rhs_slot][*direction] = binary_with_interner(
                                    interner,
                                    BinaryOp::Sub,
                                    adjoint_slots[*rhs_slot][*direction],
                                    *adj,
                                );
                            }
                            BinaryOp::Mul => {
                                let lhs_contrib = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    *adj,
                                    *rhs_expr,
                                );
                                let rhs_contrib = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    *adj,
                                    *lhs_expr,
                                );
                                adjoint_slots[*lhs_slot][*direction] = binary_with_interner(
                                    interner,
                                    BinaryOp::Add,
                                    adjoint_slots[*lhs_slot][*direction],
                                    lhs_contrib,
                                );
                                adjoint_slots[*rhs_slot][*direction] = binary_with_interner(
                                    interner,
                                    BinaryOp::Add,
                                    adjoint_slots[*rhs_slot][*direction],
                                    rhs_contrib,
                                );
                            }
                            BinaryOp::Div => {
                                let lhs_contrib =
                                    binary_with_interner(interner, BinaryOp::Div, *adj, *rhs_expr);
                                let neg_one = intern_constant_with_interner(interner, -1.0);
                                let neg_adj =
                                    binary_with_interner(interner, BinaryOp::Mul, neg_one, *adj);
                                let numer =
                                    binary_with_interner(interner, BinaryOp::Mul, neg_adj, *lhs_expr);
                                let rhs_sq = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    *rhs_expr,
                                    *rhs_expr,
                                );
                                let rhs_contrib =
                                    binary_with_interner(interner, BinaryOp::Div, numer, rhs_sq);
                                adjoint_slots[*lhs_slot][*direction] = binary_with_interner(
                                    interner,
                                    BinaryOp::Add,
                                    adjoint_slots[*lhs_slot][*direction],
                                    lhs_contrib,
                                );
                                adjoint_slots[*rhs_slot][*direction] = binary_with_interner(
                                    interner,
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
                                let lhs_contrib = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    *adj,
                                    *d_lhs,
                                );
                                let rhs_contrib = binary_with_interner(
                                    interner,
                                    BinaryOp::Mul,
                                    *adj,
                                    *d_rhs,
                                );
                                adjoint_slots[*lhs_slot][*direction] = binary_with_interner(
                                    interner,
                                    BinaryOp::Add,
                                    adjoint_slots[*lhs_slot][*direction],
                                    lhs_contrib,
                                );
                                adjoint_slots[*rhs_slot][*direction] = binary_with_interner(
                                    interner,
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

pub(crate) fn forward_directional(outputs: &[SX], vars: &[SX], seeds: &[SX]) -> Result<Vec<SX>> {
    if vars.len() != seeds.len() {
        return Err(SxError::Shape(format!(
            "forward seed length {} does not match variable length {}",
            seeds.len(),
            vars.len()
        )));
    }
    let program = program_for_outputs(outputs);
    execute_program_forward(&program, vars, seeds)
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

    let program = program_for_outputs(outputs);
    execute_program_forward_batch(&program, vars, seeds_by_direction)
}

pub(crate) fn forward_directional_basis_batch(
    outputs: &[SX],
    vars: &[SX],
    active_var_groups: &[Vec<Index>],
) -> Result<Vec<Vec<SX>>> {
    if active_var_groups.is_empty() {
        return Ok(Vec::new());
    }
    let program = program_for_outputs(outputs);
    execute_program_forward_basis_batch(&program, vars, active_var_groups)
}

pub(crate) fn reverse_directional(outputs: &[SX], vars: &[SX], seeds: &[SX]) -> Result<Vec<SX>> {
    if outputs.len() != seeds.len() {
        return Err(SxError::Shape(format!(
            "reverse seed length {} does not match output length {}",
            seeds.len(),
            outputs.len()
        )));
    }
    let program = program_for_outputs(outputs);
    execute_program_reverse(&program, vars, seeds)
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

    let program = program_for_outputs(outputs);
    execute_program_reverse_batch(&program, vars, seeds_by_direction)
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
    pub fn sym(name: impl Into<String>) -> Self {
        with_interner(|interner| interner.fresh_symbol(name))
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

    pub fn id(self) -> u32 {
        self.0
    }
}

impl From<f64> for SX {
    fn from(value: f64) -> Self {
        with_interner(|interner| {
            interner.intern_keyed(
                NodeKey::Constant(value.to_bits()),
                NodeKind::Constant(value),
            )
        })
    }
}

impl fmt::Display for SX {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        with_interner_ref(|interner| {
            let rendered = format_expression(*self, interner, &mut HashMap::new());
            write!(f, "{rendered}")
        })
    }
}

impl Neg for SX {
    type Output = SX;

    fn neg(self) -> Self::Output {
        SX::from(-1.0) * self
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
        self + SX::from(rhs)
    }
}

impl Sub<f64> for SX {
    type Output = SX;

    fn sub(self, rhs: f64) -> Self::Output {
        self - SX::from(rhs)
    }
}

impl Mul<f64> for SX {
    type Output = SX;

    fn mul(self, rhs: f64) -> Self::Output {
        self * SX::from(rhs)
    }
}

impl Div<f64> for SX {
    type Output = SX;

    fn div(self, rhs: f64) -> Self::Output {
        self / SX::from(rhs)
    }
}

impl Rem<f64> for SX {
    type Output = SX;

    fn rem(self, rhs: f64) -> Self::Output {
        self % SX::from(rhs)
    }
}

impl Add<SX> for f64 {
    type Output = SX;

    fn add(self, rhs: SX) -> Self::Output {
        SX::from(self) + rhs
    }
}

impl Sub<SX> for f64 {
    type Output = SX;

    fn sub(self, rhs: SX) -> Self::Output {
        SX::from(self) - rhs
    }
}

impl Mul<SX> for f64 {
    type Output = SX;

    fn mul(self, rhs: SX) -> Self::Output {
        SX::from(self) * rhs
    }
}

impl Div<SX> for f64 {
    type Output = SX;

    fn div(self, rhs: SX) -> Self::Output {
        SX::from(self) / rhs
    }
}

impl Rem<SX> for f64 {
    type Output = SX;

    fn rem(self, rhs: SX) -> Self::Output {
        SX::from(self) % rhs
    }
}
