use std::collections::HashMap;

use sx_core::{NamedMatrix, SX, SXFunction, SXMatrix};

use crate::jacobian_proptest::ast::{BinaryOpAst, ExprAst, FunctionAst, GeneratedCase, UnaryOpAst};
use crate::jacobian_proptest::symbolic_eval;

pub fn eval_ast_outputs(case: &GeneratedCase, inputs: &[f64]) -> Vec<f64> {
    assert_eq!(inputs.len(), case.root.input_count);
    case.root
        .outputs
        .iter()
        .map(|expr| eval_expr(expr, inputs, &case.helpers))
        .collect()
}

fn eval_expr(expr: &ExprAst, inputs: &[f64], helpers: &[FunctionAst]) -> f64 {
    match expr {
        ExprAst::Const(value) => *value,
        ExprAst::Input(index) => inputs[*index],
        ExprAst::Unary { op, arg } => {
            let value = eval_expr(arg, inputs, helpers);
            match op {
                UnaryOpAst::Neg => -value,
                UnaryOpAst::Sin => value.sin(),
                UnaryOpAst::Cos => value.cos(),
                UnaryOpAst::Exp => value.exp(),
                UnaryOpAst::Sqrt => value.sqrt(),
                UnaryOpAst::Log => value.ln(),
                UnaryOpAst::Square => value * value,
            }
        }
        ExprAst::Binary { op, lhs, rhs } => {
            let lhs = eval_expr(lhs, inputs, helpers);
            let rhs = eval_expr(rhs, inputs, helpers);
            match op {
                BinaryOpAst::Add => lhs + rhs,
                BinaryOpAst::Sub => lhs - rhs,
                BinaryOpAst::Mul => lhs * rhs,
                BinaryOpAst::Div => lhs / rhs,
            }
        }
        ExprAst::Call {
            helper,
            output,
            args,
        } => {
            let function = &helpers[*helper];
            let arg_values = args
                .iter()
                .map(|arg| eval_expr(arg, inputs, helpers))
                .collect::<Vec<_>>();
            eval_expr(&function.outputs[*output], &arg_values, helpers)
        }
    }
}

pub fn eval_symbolic_function_nonzeros(function: &SXFunction, inputs: &[&[f64]]) -> Vec<Vec<f64>> {
    assert_eq!(function.inputs().len(), inputs.len());
    let mut vars = HashMap::new();
    for (input, values) in function.inputs().iter().zip(inputs) {
        assert_eq!(input.matrix().nonzeros().len(), values.len());
        for (&symbol, &value) in input.matrix().nonzeros().iter().zip(values.iter()) {
            vars.insert(symbol.id(), value);
        }
    }
    function
        .outputs()
        .iter()
        .map(|output| {
            output
                .matrix()
                .nonzeros()
                .iter()
                .map(|&expr| symbolic_eval::eval(expr, &vars))
                .collect()
        })
        .collect()
}

pub fn symbolic_eval_expr(expr: SX, vars: &HashMap<u64, f64>) -> f64 {
    symbolic_eval::eval(expr, vars)
}

pub fn eval_lowered_function_outputs(function: &SXFunction, x: &[f64]) -> Vec<f64> {
    let outputs = eval_symbolic_function_nonzeros(function, &[x]);
    outputs.into_iter().next().unwrap_or_default()
}

pub fn named_matrix(name: &str, matrix: SXMatrix) -> NamedMatrix {
    NamedMatrix::new(name, matrix).expect("named matrix should build")
}
