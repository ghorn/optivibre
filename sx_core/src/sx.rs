use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::error::{Result, SxError};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SX(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug)]
enum NodeKind {
    Constant(f64),
    Symbol { serial: usize, name: String },
    Binary { op: BinaryOp, lhs: SX, rhs: SX },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum NodeKey {
    Constant(u64),
    Binary { op: BinaryOp, lhs: SX, rhs: SX },
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
    Symbol { name: String, serial: usize },
    Binary { op: BinaryOp, lhs: SX, rhs: SX },
}

impl BinaryOp {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
        }
    }

    fn apply_constants(self, lhs: f64, rhs: f64) -> f64 {
        match self {
            Self::Add => lhs + rhs,
            Self::Sub => lhs - rhs,
            Self::Mul => lhs * rhs,
            Self::Div => lhs / rhs,
        }
    }

    fn is_commutative(self) -> bool {
        matches!(self, Self::Add | Self::Mul)
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

impl Interner {
    fn node(&self, sx: SX) -> &Node {
        &self.nodes[sx.0 as usize]
    }

    fn node_kind(&self, sx: SX) -> NodeKind {
        self.node(sx).kind.clone()
    }

    fn fresh_symbol(&mut self, name: impl Into<String>) -> SX {
        let serial = self.next_symbol_serial;
        self.next_symbol_serial += 1;
        let id = SX(self.nodes.len() as u32);
        self.nodes.push(Node {
            kind: NodeKind::Symbol {
                serial,
                name: name.into(),
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

fn node_kind(sx: SX) -> NodeKind {
    with_interner_ref(|interner| interner.node_kind(sx))
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
        NodeKind::Symbol { name, .. } => name,
        NodeKind::Binary { op, lhs, rhs } => {
            format!(
                "({} {} {})",
                format_expression(lhs, interner, memo),
                op.symbol(),
                format_expression(rhs, interner, memo),
            )
        }
    };
    memo.insert(expr, formatted.clone());
    formatted
}

fn canonical_pair(lhs: SX, rhs: SX) -> (SX, SX) {
    if lhs <= rhs { (lhs, rhs) } else { (rhs, lhs) }
}

fn binary(op: BinaryOp, lhs: SX, rhs: SX) -> SX {
    use NodeKey as K;
    use NodeKind as N;

    let lhs_kind = node_kind(lhs);
    let rhs_kind = node_kind(rhs);

    if let (N::Constant(a), N::Constant(b)) = (&lhs_kind, &rhs_kind) {
        return SX::from(op.apply_constants(*a, *b));
    }

    match op {
        BinaryOp::Add => {
            if lhs.is_zero() {
                return rhs;
            }
            if rhs.is_zero() {
                return lhs;
            }
        }
        BinaryOp::Sub => {
            if rhs.is_zero() {
                return lhs;
            }
            if lhs == rhs {
                return SX::zero();
            }
        }
        BinaryOp::Mul => {
            if lhs.is_zero() || rhs.is_zero() {
                return SX::zero();
            }
            if lhs.is_one() {
                return rhs;
            }
            if rhs.is_one() {
                return lhs;
            }
        }
        BinaryOp::Div => {
            if lhs.is_zero() {
                return SX::zero();
            }
            if rhs.is_one() {
                return lhs;
            }
        }
    }

    let (lhs, rhs) = if op.is_commutative() {
        canonical_pair(lhs, rhs)
    } else {
        (lhs, rhs)
    };
    with_interner(|interner| {
        interner.intern_keyed(K::Binary { op, lhs, rhs }, N::Binary { op, lhs, rhs })
    })
}

fn topo_visit(node: SX, seen: &mut HashSet<SX>, order: &mut Vec<SX>) {
    if !seen.insert(node) {
        return;
    }
    match node_kind(node) {
        NodeKind::Binary { lhs, rhs, .. } => {
            topo_visit(lhs, seen, order);
            topo_visit(rhs, seen, order);
            order.push(node);
        }
        NodeKind::Constant(_) | NodeKind::Symbol { .. } => {}
    }
}

fn directional_forward(expr: SX, seeds: &HashMap<SX, SX>, memo: &mut HashMap<SX, SX>) -> SX {
    if let Some(existing) = memo.get(&expr) {
        return *existing;
    }
    let derivative = match node_kind(expr) {
        NodeKind::Constant(_) => SX::zero(),
        NodeKind::Symbol { .. } => seeds.get(&expr).copied().unwrap_or_else(SX::zero),
        NodeKind::Binary { op, lhs, rhs } => match op {
            BinaryOp::Add => {
                directional_forward(lhs, seeds, memo) + directional_forward(rhs, seeds, memo)
            }
            BinaryOp::Sub => {
                directional_forward(lhs, seeds, memo) - directional_forward(rhs, seeds, memo)
            }
            BinaryOp::Mul => {
                directional_forward(lhs, seeds, memo) * rhs
                    + lhs * directional_forward(rhs, seeds, memo)
            }
            BinaryOp::Div => {
                let dl = directional_forward(lhs, seeds, memo);
                let dr = directional_forward(rhs, seeds, memo);
                (dl * rhs - lhs * dr) / (rhs * rhs)
            }
        },
    };
    memo.insert(expr, derivative);
    derivative
}

pub(crate) fn forward_directional(outputs: &[SX], vars: &[SX], seeds: &[SX]) -> Result<Vec<SX>> {
    if vars.len() != seeds.len() {
        return Err(SxError::Shape(format!(
            "forward seed length {} does not match variable length {}",
            seeds.len(),
            vars.len()
        )));
    }
    let seed_map = vars
        .iter()
        .copied()
        .zip(seeds.iter().copied())
        .collect::<HashMap<_, _>>();
    let mut memo = HashMap::new();
    Ok(outputs
        .iter()
        .copied()
        .map(|output| directional_forward(output, &seed_map, &mut memo))
        .collect())
}

pub(crate) fn reverse_directional(outputs: &[SX], vars: &[SX], seeds: &[SX]) -> Result<Vec<SX>> {
    if outputs.len() != seeds.len() {
        return Err(SxError::Shape(format!(
            "reverse seed length {} does not match output length {}",
            seeds.len(),
            outputs.len()
        )));
    }
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    for output in outputs.iter().copied() {
        topo_visit(output, &mut seen, &mut order);
    }

    let mut adjoints = HashMap::<SX, SX>::new();
    for (output, seed) in outputs.iter().copied().zip(seeds.iter().copied()) {
        adjoints
            .entry(output)
            .and_modify(|entry| *entry = *entry + seed)
            .or_insert(seed);
    }

    for node in order.into_iter().rev() {
        let Some(adj) = adjoints.get(&node).copied() else {
            continue;
        };
        match node_kind(node) {
            NodeKind::Binary { op, lhs, rhs } => match op {
                BinaryOp::Add => {
                    adjoints
                        .entry(lhs)
                        .and_modify(|entry| *entry = *entry + adj)
                        .or_insert(adj);
                    adjoints
                        .entry(rhs)
                        .and_modify(|entry| *entry = *entry + adj)
                        .or_insert(adj);
                }
                BinaryOp::Sub => {
                    adjoints
                        .entry(lhs)
                        .and_modify(|entry| *entry = *entry + adj)
                        .or_insert(adj);
                    adjoints
                        .entry(rhs)
                        .and_modify(|entry| *entry = *entry - adj)
                        .or_insert(-adj);
                }
                BinaryOp::Mul => {
                    let lhs_contrib = adj * rhs;
                    let rhs_contrib = adj * lhs;
                    adjoints
                        .entry(lhs)
                        .and_modify(|entry| *entry = *entry + lhs_contrib)
                        .or_insert(lhs_contrib);
                    adjoints
                        .entry(rhs)
                        .and_modify(|entry| *entry = *entry + rhs_contrib)
                        .or_insert(rhs_contrib);
                }
                BinaryOp::Div => {
                    let lhs_contrib = adj / rhs;
                    let rhs_contrib = -(adj * lhs) / (rhs * rhs);
                    adjoints
                        .entry(lhs)
                        .and_modify(|entry| *entry = *entry + lhs_contrib)
                        .or_insert(lhs_contrib);
                    adjoints
                        .entry(rhs)
                        .and_modify(|entry| *entry = *entry + rhs_contrib)
                        .or_insert(rhs_contrib);
                }
            },
            NodeKind::Constant(_) | NodeKind::Symbol { .. } => {}
        }
    }

    Ok(vars
        .iter()
        .copied()
        .map(|var| adjoints.get(&var).copied().unwrap_or_else(SX::zero))
        .collect())
}

pub(crate) fn depends_on(expr: SX, target: SX, memo: &mut HashMap<SX, bool>) -> bool {
    if let Some(existing) = memo.get(&expr) {
        return *existing;
    }
    let answer = match node_kind(expr) {
        NodeKind::Constant(_) => false,
        NodeKind::Symbol { .. } => expr == target,
        NodeKind::Binary { lhs, rhs, .. } => {
            depends_on(lhs, target, memo) || depends_on(rhs, target, memo)
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

    pub fn inspect(self) -> NodeView {
        match node_kind(self) {
            NodeKind::Constant(v) => NodeView::Constant(v),
            NodeKind::Symbol { name, serial } => NodeView::Symbol { name, serial },
            NodeKind::Binary { op, lhs, rhs } => NodeView::Binary { op, lhs, rhs },
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
                NodeKind::Binary { lhs, rhs, .. } => {
                    stack.push(lhs);
                    stack.push(rhs);
                }
            }
        }
        free
    }

    pub fn symbol_name(self) -> Option<String> {
        match node_kind(self) {
            NodeKind::Symbol { name, .. } => Some(name),
            _ => None,
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

impl Sub for SX {
    type Output = SX;

    fn sub(self, rhs: Self) -> Self::Output {
        binary(BinaryOp::Sub, self, rhs)
    }
}

impl Mul for SX {
    type Output = SX;

    fn mul(self, rhs: Self) -> Self::Output {
        binary(BinaryOp::Mul, self, rhs)
    }
}

impl Div for SX {
    type Output = SX;

    fn div(self, rhs: Self) -> Self::Output {
        binary(BinaryOp::Div, self, rhs)
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
