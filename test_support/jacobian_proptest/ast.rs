use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::jacobian_proptest::domain::{InputBox, RangeCert};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperatorTier {
    Tier1,
    Tier2Domain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CaseProfile {
    Mixed,
    CallHeavy,
    BinaryHeavy,
    UnaryChain,
    RepeatedCalls,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOpAst {
    Neg,
    Sin,
    Cos,
    Exp,
    Sqrt,
    Log,
    Square,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOpAst {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExprAst {
    Const(f64),
    Input(usize),
    Unary {
        op: UnaryOpAst,
        arg: Box<ExprAst>,
    },
    Binary {
        op: BinaryOpAst,
        lhs: Box<ExprAst>,
        rhs: Box<ExprAst>,
    },
    Call {
        helper: usize,
        output: usize,
        args: Vec<ExprAst>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct FunctionAst {
    pub input_count: usize,
    pub outputs: Vec<ExprAst>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CaseFeatures {
    pub input_count: usize,
    pub root_output_count: usize,
    pub helper_count: usize,
    pub helper_multi_output_count: usize,
    pub call_count: usize,
    pub max_call_depth: usize,
    pub repeated_helper_calls: usize,
    pub nested_helper_calls: usize,
    pub tier2_op_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedCase {
    pub profile: CaseProfile,
    pub operator_tier: OperatorTier,
    pub input_box: InputBox,
    pub sample_input: Vec<f64>,
    pub helpers: Vec<FunctionAst>,
    pub root: FunctionAst,
    pub certified_output_ranges: Vec<RangeCert>,
    pub features: CaseFeatures,
}

impl GeneratedCase {
    pub fn has_calls(&self) -> bool {
        self.features.call_count > 0
    }
}

impl fmt::Display for ExprAst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const(value) => write!(f, "{value:.6}"),
            Self::Input(index) => write!(f, "x[{index}]"),
            Self::Unary { op, arg } => match op {
                UnaryOpAst::Neg => write!(f, "-({arg})"),
                UnaryOpAst::Sin => write!(f, "sin({arg})"),
                UnaryOpAst::Cos => write!(f, "cos({arg})"),
                UnaryOpAst::Exp => write!(f, "exp({arg})"),
                UnaryOpAst::Sqrt => write!(f, "sqrt({arg})"),
                UnaryOpAst::Log => write!(f, "log({arg})"),
                UnaryOpAst::Square => write!(f, "square({arg})"),
            },
            Self::Binary { op, lhs, rhs } => {
                let op_text = match op {
                    BinaryOpAst::Add => "+",
                    BinaryOpAst::Sub => "-",
                    BinaryOpAst::Mul => "*",
                    BinaryOpAst::Div => "/",
                };
                write!(f, "({lhs} {op_text} {rhs})")
            }
            Self::Call {
                helper,
                output,
                args,
            } => {
                write!(f, "h{helper}[{output}](")?;
                for (index, arg) in args.iter().enumerate() {
                    if index > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for FunctionAst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "inputs={}", self.input_count)?;
        for (index, output) in self.outputs.iter().enumerate() {
            writeln!(f, "  y[{index}] = {output}")?;
        }
        Ok(())
    }
}

impl fmt::Display for GeneratedCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "profile={:?} tier={:?} inputs={} outputs={} helpers={}",
            self.profile,
            self.operator_tier,
            self.features.input_count,
            self.features.root_output_count,
            self.features.helper_count
        )?;
        writeln!(f, "input_box={}", self.input_box)?;
        writeln!(f, "sample_input={:?}", self.sample_input)?;
        for (index, helper) in self.helpers.iter().enumerate() {
            writeln!(f, "helper h{index}:")?;
            write!(f, "{helper}")?;
        }
        writeln!(f, "root:")?;
        write!(f, "{}", self.root)?;
        if !self.certified_output_ranges.is_empty() {
            writeln!(f, "output_ranges:")?;
            for (index, range) in self.certified_output_ranges.iter().enumerate() {
                writeln!(f, "  y[{index}] -> {range}")?;
            }
        }
        writeln!(
            f,
            "features: calls={} repeated_helper_calls={} nested_helper_calls={} multi_output_helpers={} max_call_depth={} tier2_ops={}",
            self.features.call_count,
            self.features.repeated_helper_calls,
            self.features.nested_helper_calls,
            self.features.helper_multi_output_count,
            self.features.max_call_depth,
            self.features.tier2_op_count
        )
    }
}

pub fn compute_case_features(
    root: &FunctionAst,
    helpers: &[FunctionAst],
    operator_tier: OperatorTier,
    input_box: &InputBox,
    certified_output_ranges: Vec<RangeCert>,
    profile: CaseProfile,
    sample_input: Vec<f64>,
) -> GeneratedCase {
    let mut helper_call_counts = BTreeMap::<usize, usize>::new();
    let mut reached_helpers = BTreeSet::<usize>::new();
    let mut features = CaseFeatures {
        input_count: root.input_count,
        root_output_count: root.outputs.len(),
        helper_count: 0,
        helper_multi_output_count: 0,
        ..CaseFeatures::default()
    };

    fn visit_expr(
        expr: &ExprAst,
        depth: usize,
        helpers: &[FunctionAst],
        reached_helpers: &mut BTreeSet<usize>,
        features: &mut CaseFeatures,
        helper_call_counts: &mut BTreeMap<usize, usize>,
    ) {
        match expr {
            ExprAst::Const(_) | ExprAst::Input(_) => {}
            ExprAst::Unary { op, arg } => {
                if matches!(op, UnaryOpAst::Sqrt | UnaryOpAst::Log) {
                    features.tier2_op_count += 1;
                }
                visit_expr(
                    arg,
                    depth,
                    helpers,
                    reached_helpers,
                    features,
                    helper_call_counts,
                );
            }
            ExprAst::Binary { op, lhs, rhs } => {
                if matches!(op, BinaryOpAst::Div) {
                    features.tier2_op_count += 1;
                }
                visit_expr(
                    lhs,
                    depth,
                    helpers,
                    reached_helpers,
                    features,
                    helper_call_counts,
                );
                visit_expr(
                    rhs,
                    depth,
                    helpers,
                    reached_helpers,
                    features,
                    helper_call_counts,
                );
            }
            ExprAst::Call {
                helper,
                args,
                output,
                ..
            } => {
                features.call_count += 1;
                features.max_call_depth = features.max_call_depth.max(depth + 1);
                *helper_call_counts.entry(*helper).or_default() += 1;
                for arg in args {
                    visit_expr(
                        arg,
                        depth + 1,
                        helpers,
                        reached_helpers,
                        features,
                        helper_call_counts,
                    );
                }
                if let Some(function) = helpers.get(*helper) {
                    if reached_helpers.insert(*helper) {
                        features.helper_count += 1;
                        if function.outputs.len() > 1 {
                            features.helper_multi_output_count += 1;
                        }
                    }
                    if let Some(output_expr) = function.outputs.get(*output) {
                        features.nested_helper_calls += 1;
                        visit_expr(
                            output_expr,
                            depth + 1,
                            helpers,
                            reached_helpers,
                            features,
                            helper_call_counts,
                        );
                    }
                }
            }
        }
    }

    for expr in &root.outputs {
        visit_expr(
            expr,
            0,
            helpers,
            &mut reached_helpers,
            &mut features,
            &mut helper_call_counts,
        );
    }

    features.repeated_helper_calls = helper_call_counts
        .values()
        .map(|count| count.saturating_sub(1))
        .sum();

    GeneratedCase {
        profile,
        operator_tier,
        input_box: input_box.clone(),
        sample_input,
        helpers: helpers.to_vec(),
        root: root.clone(),
        certified_output_ranges,
        features,
    }
}
