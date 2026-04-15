#![allow(dead_code, unused_imports)]

mod ast;
mod compare;
mod domain;
mod generate;
mod lower;
mod oracle;
mod seed;

#[path = "../symbolic_eval.rs"]
mod symbolic_eval;

pub use ast::{
    BinaryOpAst, CaseFeatures, CaseProfile, ExprAst, FunctionAst, GeneratedCase, OperatorTier,
    UnaryOpAst,
};
pub use compare::{
    DenseMatrix, JacobianMismatchSummary, MatrixMismatchEntry, compare_dense_matrices,
    dense_from_sx_ccs_values, finite_difference_jacobian,
};
pub use domain::{InputBox, InputBoxFamily, RangeCert};
pub use generate::{
    CoverageCounters, CoverageSnapshot, GeneratedCaseRequirements, GeneratorConfig, ProfileMode,
    PropertyScenario, RejectReason, generate_case_from_seed,
};
pub use lower::{LoweredCase, instantiate_case, lower_case_to_sx_functions};
pub use oracle::{
    eval_ast_outputs, eval_lowered_function_outputs, eval_symbolic_function_nonzeros,
    symbolic_eval_expr,
};
pub use seed::{CaseSeed, ExprSeed, FunctionSeed, case_seed_strategy};
