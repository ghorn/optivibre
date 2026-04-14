use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use crate::error::{Result, SxError};
use crate::function::{CallPolicy, FunctionId, NamedMatrix, SXFunction, function_by_id};
use crate::sx::with_sx_context_id;
use crate::{BinaryOp, CCS, Index, NodeView, SX, SXContext, SXMatrix, UnaryOp};

static NEXT_EXPR_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub struct SXExpr {
    node: Arc<SXExprNode>,
}

#[derive(Clone, Debug)]
struct SXExprNode {
    id: u64,
    context_id: u32,
    source: Option<SX>,
    kind: SXExprKind,
}

#[derive(Clone, Debug)]
enum SXExprKind {
    Constant(f64),
    Symbol {
        name: Arc<str>,
    },
    Unary {
        op: UnaryOp,
        arg: SXExpr,
    },
    Binary {
        op: BinaryOp,
        lhs: SXExpr,
        rhs: SXExpr,
    },
    Call {
        function_id: FunctionId,
        inputs: Vec<SXExprMatrix>,
        output_slot: Index,
        output_offset: Index,
    },
}

#[derive(Clone, Debug)]
pub struct SXExprMatrix {
    ccs: CCS,
    nonzeros: Vec<SXExpr>,
}

#[derive(Clone, Debug)]
pub struct ExprNamedMatrix {
    name: String,
    matrix: SXExprMatrix,
}

#[derive(Clone, Debug)]
pub struct SXExprFunction {
    name: String,
    inputs: Vec<ExprNamedMatrix>,
    outputs: Vec<ExprNamedMatrix>,
    call_policy_override: Option<CallPolicy>,
}

impl PartialEq for SXExpr {
    fn eq(&self, other: &Self) -> bool {
        self.id() == other.id()
    }
}

impl Eq for SXExpr {}

impl Hash for SXExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id().hash(state);
    }
}

impl PartialOrd for SXExpr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SXExpr {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id().cmp(&other.id())
    }
}

impl fmt::Debug for SXExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SXExpr")
            .field("id", &self.id())
            .field("context_id", &self.context_id())
            .field("display", &self.to_string())
            .finish()
    }
}

impl fmt::Display for SXExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind() {
            SXExprKind::Constant(value) => {
                if value.fract() == 0.0 {
                    write!(f, "{value:.1}")
                } else {
                    write!(f, "{value}")
                }
            }
            SXExprKind::Symbol { name, .. } => write!(f, "{name}"),
            SXExprKind::Unary { op, arg } => write!(f, "{}({arg})", op.name()),
            SXExprKind::Binary { op, lhs, rhs } => {
                if let Some(symbol) = op.symbol() {
                    write!(f, "({lhs} {symbol} {rhs})")
                } else {
                    write!(f, "{}({lhs}, {rhs})", op.name())
                }
            }
            SXExprKind::Call {
                function_id,
                output_slot,
                output_offset,
                ..
            } => write!(f, "fn_{function_id}(..)[{output_slot}:{output_offset}]"),
        }
    }
}

impl SXExpr {
    fn new(context_id: u32, source: Option<SX>, kind: SXExprKind) -> Self {
        Self {
            node: Arc::new(SXExprNode {
                id: NEXT_EXPR_ID.fetch_add(1, AtomicOrdering::Relaxed),
                context_id,
                source,
                kind,
            }),
        }
    }

    fn kind(&self) -> &SXExprKind {
        &self.node.kind
    }

    pub fn id(&self) -> u64 {
        self.node.id
    }

    pub fn context_id(&self) -> u32 {
        self.node.context_id
    }

    fn ensure_same_context(lhs: &Self, rhs: &Self) -> u32 {
        let lhs_context = lhs.context_id();
        let rhs_context = rhs.context_id();
        assert!(
            lhs_context == rhs_context,
            "mixed SXExpr contexts are not supported: lhs context {} vs rhs context {}",
            lhs_context,
            rhs_context
        );
        lhs_context
    }

    pub fn constant_in_context(context_id: u32, value: f64) -> Self {
        Self::new(context_id, None, SXExprKind::Constant(value))
    }

    pub fn sym_in_context(context: SXContext, name: impl Into<String>) -> Self {
        Self::new(
            context.id(),
            None,
            SXExprKind::Symbol {
                name: Arc::<str>::from(name.into()),
            },
        )
    }

    pub fn sym(name: impl Into<String>) -> Self {
        Self::sym_in_context(SXContext::root(), name)
    }

    pub fn zero_in_context(context_id: u32) -> Self {
        Self::constant_in_context(context_id, 0.0)
    }

    pub fn one_in_context(context_id: u32) -> Self {
        Self::constant_in_context(context_id, 1.0)
    }

    pub fn zero() -> Self {
        Self::zero_in_context(SXContext::root().id())
    }

    pub fn one() -> Self {
        Self::one_in_context(SXContext::root().id())
    }

    pub fn is_zero(&self) -> bool {
        matches!(self.kind(), SXExprKind::Constant(value) if *value == 0.0)
    }

    pub fn is_one(&self) -> bool {
        matches!(self.kind(), SXExprKind::Constant(value) if *value == 1.0)
    }

    pub fn constant_value(&self) -> Option<f64> {
        match self.kind() {
            SXExprKind::Constant(value) => Some(*value),
            SXExprKind::Symbol { .. }
            | SXExprKind::Unary { .. }
            | SXExprKind::Binary { .. }
            | SXExprKind::Call { .. } => None,
        }
    }

    fn canonical_pair(lhs: Self, rhs: Self) -> (Self, Self) {
        if lhs <= rhs { (lhs, rhs) } else { (rhs, lhs) }
    }

    pub(crate) fn from_sx_with_memo(expr: SX, memo: &mut HashMap<SX, SXExpr>) -> Self {
        if let Some(existing) = memo.get(&expr) {
            return existing.clone();
        }
        let lowered = match expr.inspect() {
            NodeView::Constant(value) => {
                SXExpr::new(expr.context_id(), Some(expr), SXExprKind::Constant(value))
            }
            NodeView::Symbol { name, serial: _ } => SXExpr::new(
                expr.context_id(),
                Some(expr),
                SXExprKind::Symbol {
                    name: Arc::<str>::from(name),
                },
            ),
            NodeView::Unary { op, arg } => SXExpr::new(
                expr.context_id(),
                Some(expr),
                SXExprKind::Unary {
                    op,
                    arg: SXExpr::from_sx_with_memo(arg, memo),
                },
            ),
            NodeView::Binary { op, lhs, rhs } => SXExpr::new(
                expr.context_id(),
                Some(expr),
                SXExprKind::Binary {
                    op,
                    lhs: SXExpr::from_sx_with_memo(lhs, memo),
                    rhs: SXExpr::from_sx_with_memo(rhs, memo),
                },
            ),
            NodeView::Call {
                function_id,
                inputs,
                output_slot,
                output_offset,
                ..
            } => SXExpr::new(
                expr.context_id(),
                Some(expr),
                SXExprKind::Call {
                    function_id,
                    inputs: inputs
                        .into_iter()
                        .map(|input| SXExprMatrix::from_sx_matrix_with_memo(&input, memo))
                        .collect(),
                    output_slot,
                    output_offset,
                },
            ),
        };
        memo.insert(expr, lowered.clone());
        lowered
    }

    pub fn from_sx(expr: SX) -> Self {
        SXExpr::from_sx_with_memo(expr, &mut HashMap::new())
    }

    pub fn to_sx(&self) -> Result<SX> {
        fn lift(expr: &SXExpr, memo: &mut HashMap<u64, SX>) -> Result<SX> {
            if let Some(existing) = memo.get(&expr.id()).copied() {
                return Ok(existing);
            }
            if let Some(source) = expr.node.source {
                memo.insert(expr.id(), source);
                return Ok(source);
            }

            let context_id = expr.context_id();
            let lowered = match expr.kind() {
                SXExprKind::Constant(value) => with_sx_context_id(context_id, || SX::from(*value)),
                SXExprKind::Symbol { name, .. } => {
                    with_sx_context_id(context_id, || SX::sym(name.to_string()))
                }
                SXExprKind::Unary { op, arg } => {
                    let arg = lift(arg, memo)?;
                    apply_sx_unary(*op, arg)
                }
                SXExprKind::Binary { op, lhs, rhs } => {
                    let lhs = lift(lhs, memo)?;
                    let rhs = lift(rhs, memo)?;
                    apply_sx_binary(*op, lhs, rhs)
                }
                SXExprKind::Call {
                    function_id,
                    inputs,
                    output_slot,
                    output_offset,
                } => {
                    let function = function_by_id(*function_id).ok_or_else(|| {
                        SxError::Graph(format!("unknown function id {}", function_id))
                    })?;
                    let inputs = inputs
                        .iter()
                        .map(|input| input.to_sx_matrix())
                        .collect::<Result<Vec<_>>>()?;
                    let outputs = function.call(&inputs)?;
                    outputs[*output_slot].nz(*output_offset)
                }
            };
            memo.insert(expr.id(), lowered);
            Ok(lowered)
        }

        lift(self, &mut HashMap::new())
    }

    pub fn unary(op: UnaryOp, arg: Self) -> Self {
        if let Some(value) = arg.constant_value() {
            return Self::constant_in_context(arg.context_id(), op.apply_constant(value));
        }

        match op {
            UnaryOp::Abs => {
                if arg.is_zero() {
                    return Self::zero_in_context(arg.context_id());
                }
                if matches!(
                    arg.kind(),
                    SXExprKind::Unary {
                        op: UnaryOp::Abs,
                        ..
                    }
                ) {
                    return arg;
                }
            }
            UnaryOp::Sign => {
                if arg.is_zero() {
                    return Self::zero_in_context(arg.context_id());
                }
            }
            UnaryOp::Sqrt => {
                if arg.is_zero() {
                    return Self::zero_in_context(arg.context_id());
                }
                if arg.is_one() {
                    return Self::one_in_context(arg.context_id());
                }
            }
            UnaryOp::Exp | UnaryOp::Cos | UnaryOp::Cosh => {
                if arg.is_zero() {
                    return Self::one_in_context(arg.context_id());
                }
            }
            UnaryOp::Log => {
                if arg.is_one() {
                    return Self::zero_in_context(arg.context_id());
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
                if arg.is_zero() {
                    return Self::zero_in_context(arg.context_id());
                }
            }
            UnaryOp::Acos | UnaryOp::Acosh | UnaryOp::Atanh => {}
        }

        Self::new(arg.context_id(), None, SXExprKind::Unary { op, arg })
    }

    pub fn binary(op: BinaryOp, lhs: Self, rhs: Self) -> Self {
        let context_id = Self::ensure_same_context(&lhs, &rhs);
        if let (Some(a), Some(b)) = (lhs.constant_value(), rhs.constant_value()) {
            return Self::constant_in_context(context_id, op.apply_constant(a, b));
        }

        match op {
            BinaryOp::Add => {
                if lhs.is_zero() {
                    return rhs;
                }
                if rhs.is_zero() {
                    return lhs;
                }
                let (lhs, rhs) = Self::canonical_pair(lhs, rhs);
                return Self::new(context_id, None, SXExprKind::Binary { op, lhs, rhs });
            }
            BinaryOp::Sub => {
                if rhs.is_zero() {
                    return lhs;
                }
                if lhs.is_zero() {
                    return -rhs;
                }
                if lhs == rhs {
                    return Self::zero_in_context(context_id);
                }
            }
            BinaryOp::Mul => {
                if lhs.is_zero() || rhs.is_zero() {
                    return Self::zero_in_context(context_id);
                }
                if lhs.is_one() {
                    return rhs;
                }
                if rhs.is_one() {
                    return lhs;
                }
                let (lhs, rhs) = Self::canonical_pair(lhs, rhs);
                return Self::new(context_id, None, SXExprKind::Binary { op, lhs, rhs });
            }
            BinaryOp::Div => {
                if lhs.is_zero() {
                    return Self::zero_in_context(context_id);
                }
                if rhs.is_one() {
                    return lhs;
                }
            }
            BinaryOp::Pow => {
                if rhs.is_zero() {
                    return Self::one_in_context(context_id);
                }
                if rhs.is_one() {
                    return lhs;
                }
                if lhs.is_one() {
                    return Self::one_in_context(context_id);
                }
            }
            BinaryOp::Atan2 => {
                if lhs.is_zero() && rhs.is_one() {
                    return Self::zero_in_context(context_id);
                }
            }
            BinaryOp::Hypot => {
                if lhs.is_zero() {
                    return rhs.abs();
                }
                if rhs.is_zero() {
                    return lhs.abs();
                }
                let (lhs, rhs) = Self::canonical_pair(lhs, rhs);
                return Self::new(context_id, None, SXExprKind::Binary { op, lhs, rhs });
            }
            BinaryOp::Mod => {
                if lhs.is_zero() {
                    return Self::zero_in_context(context_id);
                }
            }
            BinaryOp::Copysign => {
                if lhs.is_zero() {
                    return Self::zero_in_context(context_id);
                }
                if rhs.is_zero() {
                    return lhs.abs();
                }
            }
        }

        Self::new(context_id, None, SXExprKind::Binary { op, lhs, rhs })
    }

    pub fn binary_ad(op: BinaryOp, lhs: Self, rhs: Self) -> Self {
        let context_id = Self::ensure_same_context(&lhs, &rhs);
        if let (Some(a), Some(b)) = (lhs.constant_value(), rhs.constant_value()) {
            return Self::constant_in_context(context_id, op.apply_constant(a, b));
        }

        match op {
            BinaryOp::Add => {
                if lhs.is_zero() {
                    return rhs;
                }
                if rhs.is_zero() {
                    return lhs;
                }
                let (lhs, rhs) = Self::canonical_pair(lhs, rhs);
                Self::new(context_id, None, SXExprKind::Binary { op, lhs, rhs })
            }
            BinaryOp::Sub => {
                if rhs.is_zero() {
                    lhs
                } else if lhs.is_zero() {
                    -rhs
                } else {
                    Self::new(context_id, None, SXExprKind::Binary { op, lhs, rhs })
                }
            }
            BinaryOp::Mul => {
                if lhs.is_zero() || rhs.is_zero() {
                    Self::zero_in_context(context_id)
                } else if lhs.is_one() {
                    rhs
                } else if rhs.is_one() {
                    lhs
                } else {
                    let (lhs, rhs) = Self::canonical_pair(lhs, rhs);
                    Self::new(context_id, None, SXExprKind::Binary { op, lhs, rhs })
                }
            }
            BinaryOp::Div => {
                if lhs.is_zero() {
                    Self::zero_in_context(context_id)
                } else if rhs.is_one() {
                    lhs
                } else {
                    Self::new(context_id, None, SXExprKind::Binary { op, lhs, rhs })
                }
            }
            BinaryOp::Pow => {
                if rhs.is_zero() {
                    Self::one_in_context(context_id)
                } else if rhs.is_one() {
                    lhs
                } else if lhs.is_one() {
                    Self::one_in_context(context_id)
                } else {
                    Self::new(context_id, None, SXExprKind::Binary { op, lhs, rhs })
                }
            }
            BinaryOp::Atan2 | BinaryOp::Hypot | BinaryOp::Mod | BinaryOp::Copysign => {
                if op.is_commutative() {
                    let (lhs, rhs) = Self::canonical_pair(lhs, rhs);
                    Self::new(context_id, None, SXExprKind::Binary { op, lhs, rhs })
                } else {
                    Self::new(context_id, None, SXExprKind::Binary { op, lhs, rhs })
                }
            }
        }
    }

    pub fn abs(self) -> Self {
        Self::unary(UnaryOp::Abs, self)
    }
    pub fn sign(self) -> Self {
        Self::unary(UnaryOp::Sign, self)
    }
    pub fn floor(self) -> Self {
        Self::unary(UnaryOp::Floor, self)
    }
    pub fn ceil(self) -> Self {
        Self::unary(UnaryOp::Ceil, self)
    }
    pub fn round(self) -> Self {
        Self::unary(UnaryOp::Round, self)
    }
    pub fn trunc(self) -> Self {
        Self::unary(UnaryOp::Trunc, self)
    }
    pub fn sqrt(self) -> Self {
        Self::unary(UnaryOp::Sqrt, self)
    }
    pub fn exp(self) -> Self {
        Self::unary(UnaryOp::Exp, self)
    }
    pub fn log(self) -> Self {
        Self::unary(UnaryOp::Log, self)
    }
    pub fn sin(self) -> Self {
        Self::unary(UnaryOp::Sin, self)
    }
    pub fn cos(self) -> Self {
        Self::unary(UnaryOp::Cos, self)
    }
    pub fn tan(self) -> Self {
        Self::unary(UnaryOp::Tan, self)
    }
    pub fn asin(self) -> Self {
        Self::unary(UnaryOp::Asin, self)
    }
    pub fn acos(self) -> Self {
        Self::unary(UnaryOp::Acos, self)
    }
    pub fn atan(self) -> Self {
        Self::unary(UnaryOp::Atan, self)
    }
    pub fn sinh(self) -> Self {
        Self::unary(UnaryOp::Sinh, self)
    }
    pub fn cosh(self) -> Self {
        Self::unary(UnaryOp::Cosh, self)
    }
    pub fn tanh(self) -> Self {
        Self::unary(UnaryOp::Tanh, self)
    }
    pub fn asinh(self) -> Self {
        Self::unary(UnaryOp::Asinh, self)
    }
    pub fn acosh(self) -> Self {
        Self::unary(UnaryOp::Acosh, self)
    }
    pub fn atanh(self) -> Self {
        Self::unary(UnaryOp::Atanh, self)
    }
    pub fn pow(self, rhs: Self) -> Self {
        Self::binary(BinaryOp::Pow, self, rhs)
    }
    pub fn atan2(self, rhs: Self) -> Self {
        Self::binary(BinaryOp::Atan2, self, rhs)
    }
    pub fn hypot(self, rhs: Self) -> Self {
        Self::binary(BinaryOp::Hypot, self, rhs)
    }
    pub fn modulo(self, rhs: Self) -> Self {
        Self::binary(BinaryOp::Mod, self, rhs)
    }
    pub fn copysign(self, rhs: Self) -> Self {
        Self::binary(BinaryOp::Copysign, self, rhs)
    }

    #[allow(dead_code)]
    pub(crate) fn unary_derivative(op: UnaryOp, arg: Self) -> Self {
        let context_id = arg.context_id();
        match op {
            UnaryOp::Abs => arg.sign(),
            UnaryOp::Sign | UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round | UnaryOp::Trunc => {
                Self::zero_in_context(context_id)
            }
            UnaryOp::Sqrt => Self::constant_in_context(context_id, 0.5) / arg.sqrt(),
            UnaryOp::Exp => arg.exp(),
            UnaryOp::Log => Self::one_in_context(context_id) / arg,
            UnaryOp::Sin => arg.cos(),
            UnaryOp::Cos => Self::constant_in_context(context_id, -1.0) * arg.sin(),
            UnaryOp::Tan => {
                let cos = arg.clone().cos();
                Self::one_in_context(context_id) / (cos.clone() * cos)
            }
            UnaryOp::Asin => {
                let one = Self::one_in_context(context_id);
                one.clone() / (one.clone() - arg.clone() * arg).sqrt()
            }
            UnaryOp::Acos => {
                let one = Self::one_in_context(context_id);
                Self::constant_in_context(context_id, -1.0)
                    / (one.clone() - arg.clone() * arg).sqrt()
            }
            UnaryOp::Atan => {
                let one = Self::one_in_context(context_id);
                one.clone() / (one + arg.clone() * arg)
            }
            UnaryOp::Sinh => arg.cosh(),
            UnaryOp::Cosh => arg.sinh(),
            UnaryOp::Tanh => {
                let cosh = arg.clone().cosh();
                Self::one_in_context(context_id) / (cosh.clone() * cosh)
            }
            UnaryOp::Asinh => {
                let one = Self::one_in_context(context_id);
                one.clone() / (arg.clone() * arg + one).sqrt()
            }
            UnaryOp::Acosh => {
                let one = Self::one_in_context(context_id);
                one.clone() / ((arg.clone() - one.clone()).sqrt() * (arg + one).sqrt())
            }
            UnaryOp::Atanh => {
                let one = Self::one_in_context(context_id);
                one.clone() / (one - arg.clone() * arg)
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn binary_partials(op: BinaryOp, lhs: Self, rhs: Self) -> (Self, Self) {
        let context_id = Self::ensure_same_context(&lhs, &rhs);
        match op {
            BinaryOp::Add => (
                Self::one_in_context(context_id),
                Self::one_in_context(context_id),
            ),
            BinaryOp::Sub => (
                Self::one_in_context(context_id),
                Self::constant_in_context(context_id, -1.0),
            ),
            BinaryOp::Mul => (rhs, lhs),
            BinaryOp::Div => {
                let lhs_partial = Self::one_in_context(context_id) / rhs.clone();
                let rhs_partial =
                    (Self::constant_in_context(context_id, -1.0) * lhs) / (rhs.clone() * rhs);
                (lhs_partial, rhs_partial)
            }
            BinaryOp::Pow => {
                let pow = lhs.clone().pow(rhs.clone());
                let one = Self::one_in_context(context_id);
                let lhs_partial = rhs.clone() * lhs.clone().pow(rhs.clone() - one.clone());
                let rhs_partial = pow * lhs.log();
                (lhs_partial, rhs_partial)
            }
            BinaryOp::Atan2 => {
                let denom = lhs.clone() * lhs.clone() + rhs.clone() * rhs.clone();
                (
                    rhs.clone() / denom.clone(),
                    (Self::constant_in_context(context_id, -1.0) * lhs) / denom,
                )
            }
            BinaryOp::Hypot => {
                let hypot = lhs.clone().hypot(rhs.clone());
                (lhs / hypot.clone(), rhs / hypot)
            }
            BinaryOp::Mod => {
                let trunc = (lhs.clone() / rhs.clone()).trunc();
                (
                    Self::one_in_context(context_id),
                    Self::constant_in_context(context_id, -1.0) * trunc,
                )
            }
            BinaryOp::Copysign => (
                lhs.sign()
                    * (rhs.clone().sign() + (Self::one_in_context(context_id) - rhs.sign().abs())),
                Self::zero_in_context(context_id),
            ),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn unary_second_directional(op: UnaryOp, arg: Self, arg_tangent: Self) -> Self {
        let context_id = arg.context_id();
        if arg_tangent.is_zero() {
            return Self::zero_in_context(context_id);
        }
        match op {
            UnaryOp::Abs
            | UnaryOp::Sign
            | UnaryOp::Floor
            | UnaryOp::Ceil
            | UnaryOp::Round
            | UnaryOp::Trunc => Self::zero_in_context(context_id),
            UnaryOp::Sqrt => {
                let neg_quarter = Self::constant_in_context(context_id, -0.25);
                let sqrt_arg = arg.clone().sqrt();
                let denom = SXExpr::binary_ad(BinaryOp::Mul, arg.clone(), sqrt_arg);
                let scaled = SXExpr::binary_ad(BinaryOp::Mul, neg_quarter, arg_tangent);
                SXExpr::binary_ad(BinaryOp::Div, scaled, denom)
            }
            UnaryOp::Exp => SXExpr::binary_ad(BinaryOp::Mul, arg.exp(), arg_tangent),
            UnaryOp::Log => {
                let numer = SXExpr::binary_ad(
                    BinaryOp::Mul,
                    Self::constant_in_context(context_id, -1.0),
                    arg_tangent,
                );
                let denom = SXExpr::binary_ad(BinaryOp::Mul, arg.clone(), arg);
                SXExpr::binary_ad(BinaryOp::Div, numer, denom)
            }
            UnaryOp::Sin => {
                let neg_sin = -(arg.sin());
                SXExpr::binary_ad(BinaryOp::Mul, neg_sin, arg_tangent)
            }
            UnaryOp::Cos => {
                let neg_cos = -(arg.cos());
                SXExpr::binary_ad(BinaryOp::Mul, neg_cos, arg_tangent)
            }
            UnaryOp::Tan => {
                let two = Self::constant_in_context(context_id, 2.0);
                let tan_arg = arg.clone().tan();
                let cos_arg = arg.cos();
                let cos_sq = SXExpr::binary_ad(BinaryOp::Mul, cos_arg.clone(), cos_arg);
                let factor = SXExpr::binary_ad(BinaryOp::Mul, two, tan_arg);
                let factor = SXExpr::binary_ad(BinaryOp::Div, factor, cos_sq);
                SXExpr::binary_ad(BinaryOp::Mul, factor, arg_tangent)
            }
            UnaryOp::Asin => {
                let one = Self::one_in_context(context_id);
                let arg_sq = SXExpr::binary_ad(BinaryOp::Mul, arg.clone(), arg.clone());
                let radicand = SXExpr::binary_ad(BinaryOp::Sub, one.clone(), arg_sq);
                let sqrt = radicand.clone().sqrt();
                let denom = SXExpr::binary_ad(BinaryOp::Mul, radicand, sqrt);
                let numer = SXExpr::binary_ad(BinaryOp::Mul, arg, arg_tangent);
                SXExpr::binary_ad(BinaryOp::Div, numer, denom)
            }
            UnaryOp::Acos => -SXExpr::unary_second_directional(UnaryOp::Asin, arg, arg_tangent),
            UnaryOp::Atan => {
                let neg_two = Self::constant_in_context(context_id, -2.0);
                let one = Self::one_in_context(context_id);
                let arg_sq = SXExpr::binary_ad(BinaryOp::Mul, arg.clone(), arg.clone());
                let denom_base = SXExpr::binary_ad(BinaryOp::Add, one.clone(), arg_sq);
                let denom = SXExpr::binary_ad(BinaryOp::Mul, denom_base.clone(), denom_base);
                let numer = SXExpr::binary_ad(BinaryOp::Mul, arg.clone(), arg_tangent);
                let numer = SXExpr::binary_ad(BinaryOp::Mul, neg_two, numer);
                SXExpr::binary_ad(BinaryOp::Div, numer, denom)
            }
            UnaryOp::Sinh => SXExpr::binary_ad(BinaryOp::Mul, arg.sinh(), arg_tangent),
            UnaryOp::Cosh => SXExpr::binary_ad(BinaryOp::Mul, arg.cosh(), arg_tangent),
            UnaryOp::Tanh => {
                let neg_two = Self::constant_in_context(context_id, -2.0);
                let tanh_arg = arg.clone().tanh();
                let cosh_arg = arg.cosh();
                let cosh_sq = SXExpr::binary_ad(BinaryOp::Mul, cosh_arg.clone(), cosh_arg);
                let factor = SXExpr::binary_ad(BinaryOp::Mul, neg_two, tanh_arg);
                let factor = SXExpr::binary_ad(BinaryOp::Div, factor, cosh_sq);
                SXExpr::binary_ad(BinaryOp::Mul, factor, arg_tangent)
            }
            UnaryOp::Asinh => {
                let one = Self::one_in_context(context_id);
                let arg_sq = SXExpr::binary_ad(BinaryOp::Mul, arg.clone(), arg.clone());
                let radicand = SXExpr::binary_ad(BinaryOp::Add, arg_sq, one.clone());
                let sqrt = radicand.clone().sqrt();
                let denom = SXExpr::binary_ad(BinaryOp::Mul, radicand, sqrt);
                let numer = -(SXExpr::binary_ad(BinaryOp::Mul, arg, arg_tangent));
                SXExpr::binary_ad(BinaryOp::Div, numer, denom)
            }
            UnaryOp::Acosh => {
                let one = Self::one_in_context(context_id);
                let arg_sq = SXExpr::binary_ad(BinaryOp::Mul, arg.clone(), arg.clone());
                let denom_base = SXExpr::binary_ad(BinaryOp::Sub, arg_sq, one);
                let sqrt = denom_base.clone().sqrt();
                let denom = SXExpr::binary_ad(BinaryOp::Mul, denom_base, sqrt);
                let numer = -(SXExpr::binary_ad(BinaryOp::Mul, arg, arg_tangent));
                SXExpr::binary_ad(BinaryOp::Div, numer, denom)
            }
            UnaryOp::Atanh => {
                let two = Self::constant_in_context(context_id, 2.0);
                let one = Self::one_in_context(context_id);
                let arg_sq = SXExpr::binary_ad(BinaryOp::Mul, arg.clone(), arg.clone());
                let denom_base = SXExpr::binary_ad(BinaryOp::Sub, one, arg_sq);
                let denom = SXExpr::binary_ad(BinaryOp::Mul, denom_base.clone(), denom_base);
                let numer = SXExpr::binary_ad(BinaryOp::Mul, arg, arg_tangent);
                let numer = SXExpr::binary_ad(BinaryOp::Mul, two, numer);
                SXExpr::binary_ad(BinaryOp::Div, numer, denom)
            }
        }
    }

    #[allow(dead_code)]
    pub(crate) fn binary_partial_directionals(
        op: BinaryOp,
        lhs: Self,
        rhs: Self,
        lhs_tangent: Self,
        rhs_tangent: Self,
    ) -> (Self, Self) {
        let context_id = Self::ensure_same_context(&lhs, &rhs);
        let zero = Self::zero_in_context(context_id);
        match op {
            BinaryOp::Add | BinaryOp::Sub => (zero.clone(), zero),
            BinaryOp::Mul => (rhs_tangent, lhs_tangent),
            BinaryOp::Div => {
                let rhs_sq = SXExpr::binary_ad(BinaryOp::Mul, rhs.clone(), rhs.clone());
                let lhs_dir = {
                    let neg_rhs_t = -rhs_tangent.clone();
                    SXExpr::binary_ad(BinaryOp::Div, neg_rhs_t, rhs_sq.clone())
                };
                let rhs_dir = {
                    let lhs_over_rhs_sq =
                        SXExpr::binary_ad(BinaryOp::Div, lhs_tangent.clone(), rhs_sq.clone());
                    let left = -lhs_over_rhs_sq;
                    let two = Self::constant_in_context(context_id, 2.0);
                    let rhs_cu = SXExpr::binary_ad(BinaryOp::Mul, rhs_sq, rhs.clone());
                    let numer = SXExpr::binary_ad(BinaryOp::Mul, lhs.clone(), rhs_tangent);
                    let numer = SXExpr::binary_ad(BinaryOp::Mul, two, numer);
                    let right = SXExpr::binary_ad(BinaryOp::Div, numer, rhs_cu);
                    SXExpr::binary_ad(BinaryOp::Add, left, right)
                };
                (lhs_dir, rhs_dir)
            }
            BinaryOp::Pow => {
                let pow = SXExpr::binary_ad(BinaryOp::Pow, lhs.clone(), rhs.clone());
                let log_lhs = lhs.clone().log();
                let one = Self::one_in_context(context_id);
                let lhs_inv = SXExpr::binary_ad(BinaryOp::Div, one.clone(), lhs.clone());
                let rhs_lhs = SXExpr::binary_ad(BinaryOp::Mul, rhs.clone(), lhs_tangent.clone());
                let rhs_lhs = SXExpr::binary_ad(BinaryOp::Mul, rhs_lhs, lhs_inv.clone());
                let rhs_log =
                    SXExpr::binary_ad(BinaryOp::Mul, rhs_tangent.clone(), log_lhs.clone());
                let pow_factor = SXExpr::binary_ad(BinaryOp::Add, rhs_log, rhs_lhs);
                let pow_t = SXExpr::binary_ad(BinaryOp::Mul, pow.clone(), pow_factor);

                let rhs_lhs_inv = SXExpr::binary_ad(BinaryOp::Mul, rhs.clone(), lhs_inv.clone());
                let lhs_term1 = SXExpr::binary_ad(BinaryOp::Mul, pow_t.clone(), rhs_lhs_inv);
                let rhs_tangent_lhs_inv =
                    SXExpr::binary_ad(BinaryOp::Mul, rhs_tangent, lhs_inv.clone());
                let lhs_term2 = SXExpr::binary_ad(BinaryOp::Mul, pow.clone(), rhs_tangent_lhs_inv);
                let lhs_inv_sq = SXExpr::binary_ad(BinaryOp::Mul, lhs_inv.clone(), lhs_inv);
                let rhs_lhs_tangent =
                    SXExpr::binary_ad(BinaryOp::Mul, rhs.clone(), lhs_tangent.clone());
                let rhs_lhs_tangent_inv_sq =
                    SXExpr::binary_ad(BinaryOp::Mul, rhs_lhs_tangent, lhs_inv_sq);
                let lhs_term3_inner =
                    SXExpr::binary_ad(BinaryOp::Mul, pow.clone(), rhs_lhs_tangent_inv_sq);
                let lhs_terms = SXExpr::binary_ad(BinaryOp::Add, lhs_term1, lhs_term2);
                let lhs_dir = SXExpr::binary_ad(BinaryOp::Add, lhs_terms, -lhs_term3_inner);

                let rhs_term1 = SXExpr::binary_ad(BinaryOp::Mul, pow_t, log_lhs);
                let lhs_tangent_lhs_inv =
                    SXExpr::binary_ad(BinaryOp::Mul, lhs_tangent, one / lhs.clone());
                let rhs_term2 = SXExpr::binary_ad(BinaryOp::Mul, pow, lhs_tangent_lhs_inv);
                let rhs_dir = SXExpr::binary_ad(BinaryOp::Add, rhs_term1, rhs_term2);
                (lhs_dir, rhs_dir)
            }
            BinaryOp::Atan2 => {
                let two = Self::constant_in_context(context_id, 2.0);
                let lhs_sq = SXExpr::binary_ad(BinaryOp::Mul, lhs.clone(), lhs.clone());
                let rhs_sq = SXExpr::binary_ad(BinaryOp::Mul, rhs.clone(), rhs.clone());
                let denom = SXExpr::binary_ad(BinaryOp::Add, lhs_sq, rhs_sq);
                let denom_sq = SXExpr::binary_ad(BinaryOp::Mul, denom.clone(), denom.clone());
                let two_lhs_lhs_tangent =
                    SXExpr::binary_ad(BinaryOp::Mul, lhs.clone(), lhs_tangent.clone());
                let two_lhs_lhs_tangent =
                    SXExpr::binary_ad(BinaryOp::Mul, two.clone(), two_lhs_lhs_tangent);
                let two_rhs_rhs_tangent =
                    SXExpr::binary_ad(BinaryOp::Mul, rhs.clone(), rhs_tangent.clone());
                let two_rhs_rhs_tangent =
                    SXExpr::binary_ad(BinaryOp::Mul, two, two_rhs_rhs_tangent);
                let denom_t =
                    SXExpr::binary_ad(BinaryOp::Add, two_lhs_lhs_tangent, two_rhs_rhs_tangent);
                let lhs_dir = {
                    let left = SXExpr::binary_ad(BinaryOp::Div, rhs_tangent.clone(), denom.clone());
                    let rhs_denom_t =
                        SXExpr::binary_ad(BinaryOp::Mul, rhs.clone(), denom_t.clone());
                    let right = SXExpr::binary_ad(BinaryOp::Div, rhs_denom_t, denom_sq.clone());
                    SXExpr::binary_ad(BinaryOp::Sub, left, right)
                };
                let rhs_dir = {
                    let lhs_tangent_over_denom =
                        SXExpr::binary_ad(BinaryOp::Div, lhs_tangent, denom.clone());
                    let left = -lhs_tangent_over_denom;
                    let lhs_denom_t = SXExpr::binary_ad(BinaryOp::Mul, lhs, denom_t);
                    let right = SXExpr::binary_ad(BinaryOp::Div, lhs_denom_t, denom_sq);
                    SXExpr::binary_ad(BinaryOp::Add, left, right)
                };
                (lhs_dir, rhs_dir)
            }
            BinaryOp::Hypot => {
                let hypot = SXExpr::binary_ad(BinaryOp::Hypot, lhs.clone(), rhs.clone());
                let lhs_lhs_tangent =
                    SXExpr::binary_ad(BinaryOp::Mul, lhs.clone(), lhs_tangent.clone());
                let rhs_rhs_tangent =
                    SXExpr::binary_ad(BinaryOp::Mul, rhs.clone(), rhs_tangent.clone());
                let numer_t = SXExpr::binary_ad(BinaryOp::Add, lhs_lhs_tangent, rhs_rhs_tangent);
                let hypot_t = SXExpr::binary_ad(BinaryOp::Div, numer_t, hypot.clone());
                let hypot_sq = SXExpr::binary_ad(BinaryOp::Mul, hypot.clone(), hypot.clone());
                let lhs_dir = {
                    let left = SXExpr::binary_ad(BinaryOp::Div, lhs_tangent, hypot.clone());
                    let lhs_hypot_t =
                        SXExpr::binary_ad(BinaryOp::Mul, lhs.clone(), hypot_t.clone());
                    let right = SXExpr::binary_ad(BinaryOp::Div, lhs_hypot_t, hypot_sq.clone());
                    SXExpr::binary_ad(BinaryOp::Sub, left, right)
                };
                let rhs_dir = {
                    let left = SXExpr::binary_ad(BinaryOp::Div, rhs_tangent, hypot.clone());
                    let rhs_hypot_t = SXExpr::binary_ad(BinaryOp::Mul, rhs, hypot_t);
                    let right = SXExpr::binary_ad(BinaryOp::Div, rhs_hypot_t, hypot_sq);
                    SXExpr::binary_ad(BinaryOp::Sub, left, right)
                };
                (lhs_dir, rhs_dir)
            }
            BinaryOp::Mod | BinaryOp::Copysign => (zero.clone(), zero),
        }
    }
}

impl SXExprMatrix {
    pub fn new(ccs: CCS, nonzeros: Vec<SXExpr>) -> Result<Self> {
        if ccs.nnz() != nonzeros.len() {
            return Err(SxError::Shape(format!(
                "CCS nnz {} does not match value nnz {}",
                ccs.nnz(),
                nonzeros.len()
            )));
        }
        Ok(Self { ccs, nonzeros })
    }

    pub(crate) fn from_sx_matrix_with_memo(
        matrix: &SXMatrix,
        memo: &mut HashMap<SX, SXExpr>,
    ) -> Self {
        Self {
            ccs: matrix.ccs().clone(),
            nonzeros: matrix
                .nonzeros()
                .iter()
                .copied()
                .map(|expr| SXExpr::from_sx_with_memo(expr, memo))
                .collect(),
        }
    }

    pub fn from_sx_matrix(matrix: &SXMatrix) -> Self {
        Self::from_sx_matrix_with_memo(matrix, &mut HashMap::new())
    }

    pub fn to_sx_matrix(&self) -> Result<SXMatrix> {
        SXMatrix::new(
            self.ccs.clone(),
            self.nonzeros
                .iter()
                .map(SXExpr::to_sx)
                .collect::<Result<Vec<_>>>()?,
        )
    }

    pub fn ccs(&self) -> &CCS {
        &self.ccs
    }

    pub fn nonzeros(&self) -> &[SXExpr] {
        &self.nonzeros
    }

    pub fn nnz(&self) -> Index {
        self.nonzeros.len()
    }

    pub fn nz(&self, offset: Index) -> SXExpr {
        self.nonzeros[offset].clone()
    }
}

impl ExprNamedMatrix {
    pub fn new(name: impl Into<String>, matrix: SXExprMatrix) -> Result<Self> {
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

    pub fn matrix(&self) -> &SXExprMatrix {
        &self.matrix
    }
}

impl SXExprFunction {
    pub fn from_sx_function(function: &SXFunction) -> Result<Self> {
        Ok(Self {
            name: function.name().to_string(),
            inputs: function
                .inputs()
                .iter()
                .map(|input| {
                    ExprNamedMatrix::new(input.name(), SXExprMatrix::from_sx_matrix(input.matrix()))
                })
                .collect::<Result<Vec<_>>>()?,
            outputs: function
                .outputs()
                .iter()
                .map(|output| {
                    ExprNamedMatrix::new(
                        output.name(),
                        SXExprMatrix::from_sx_matrix(output.matrix()),
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            call_policy_override: function.call_policy_override(),
        })
    }

    pub fn to_sx_function(&self) -> Result<SXFunction> {
        let inputs = self
            .inputs
            .iter()
            .map(|input| NamedMatrix::new(input.name(), input.matrix().to_sx_matrix()?))
            .collect::<Result<Vec<_>>>()?;
        let outputs = self
            .outputs
            .iter()
            .map(|output| NamedMatrix::new(output.name(), output.matrix().to_sx_matrix()?))
            .collect::<Result<Vec<_>>>()?;
        let function = SXFunction::new(self.name.clone(), inputs, outputs)?;
        Ok(match self.call_policy_override {
            Some(policy) => function.with_call_policy_override(policy),
            None => function,
        })
    }
}

fn apply_sx_unary(op: UnaryOp, arg: SX) -> SX {
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

fn apply_sx_binary(op: BinaryOp, lhs: SX, rhs: SX) -> SX {
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

impl Neg for SXExpr {
    type Output = SXExpr;

    fn neg(self) -> Self::Output {
        SXExpr::constant_in_context(self.context_id(), -1.0) * self
    }
}

impl Add for SXExpr {
    type Output = SXExpr;

    fn add(self, rhs: Self) -> Self::Output {
        SXExpr::binary(BinaryOp::Add, self, rhs)
    }
}

impl Sub for SXExpr {
    type Output = SXExpr;

    fn sub(self, rhs: Self) -> Self::Output {
        SXExpr::binary(BinaryOp::Sub, self, rhs)
    }
}

impl Mul for SXExpr {
    type Output = SXExpr;

    fn mul(self, rhs: Self) -> Self::Output {
        SXExpr::binary(BinaryOp::Mul, self, rhs)
    }
}

impl Div for SXExpr {
    type Output = SXExpr;

    fn div(self, rhs: Self) -> Self::Output {
        SXExpr::binary(BinaryOp::Div, self, rhs)
    }
}

impl Rem for SXExpr {
    type Output = SXExpr;

    fn rem(self, rhs: Self) -> Self::Output {
        SXExpr::binary(BinaryOp::Mod, self, rhs)
    }
}
