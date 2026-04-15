use sx_core::{CCS, NamedMatrix, SX, SXContext, SXFunction, SXMatrix};

use crate::jacobian_proptest::ast::{BinaryOpAst, ExprAst, FunctionAst, GeneratedCase, UnaryOpAst};

#[derive(Clone)]
pub struct LoweredCase {
    pub primal: SXFunction,
    pub jacobian: SXFunction,
    pub jacobian_ccs: CCS,
}

pub fn instantiate_case(case: &GeneratedCase, root_inputs: &[SX]) -> (Vec<SXFunction>, Vec<SX>) {
    let mut helpers = Vec::with_capacity(case.helpers.len());
    for (index, helper_ast) in case.helpers.iter().enumerate() {
        let helper_inputs = (0..helper_ast.input_count)
            .map(|slot| SX::sym(format!("h{index}_x{slot}")))
            .collect::<Vec<_>>();
        let outputs = lower_function_outputs(helper_ast, &helper_inputs, &helpers);
        let input_matrix = SXMatrix::dense_column(helper_inputs).expect("helper input column");
        let output_matrix = SXMatrix::dense_column(outputs).expect("helper output column");
        let function = SXFunction::new(
            format!("generated_helper_{index}"),
            vec![NamedMatrix::new("x", input_matrix).expect("helper input")],
            vec![NamedMatrix::new("y", output_matrix).expect("helper output")],
        )
        .expect("generated helper should build");
        helpers.push(function);
    }
    let root_outputs = lower_function_outputs(&case.root, root_inputs, &helpers);
    (helpers, root_outputs)
}

pub fn lower_case_to_sx_functions(case: &GeneratedCase) -> LoweredCase {
    let context = SXContext::root();
    context.scoped(|| {
        let inputs = (0..case.root.input_count)
            .map(|index| SX::sym(format!("x{index}")))
            .collect::<Vec<_>>();
        let (_helpers, outputs) = instantiate_case(case, &inputs);
        let input_matrix = SXMatrix::dense_column(inputs.clone()).expect("root input");
        let output_matrix = SXMatrix::dense_column(outputs).expect("root output");
        let primal = SXFunction::new(
            "generated_root",
            vec![NamedMatrix::new("x", input_matrix.clone()).expect("root input")],
            vec![NamedMatrix::new("y", output_matrix.clone()).expect("root output")],
        )
        .expect("primal function should build");
        let jacobian_matrix = output_matrix
            .jacobian(&input_matrix)
            .expect("jacobian should build");
        let jacobian_ccs = jacobian_matrix.ccs().clone();
        let jacobian = SXFunction::new(
            "generated_root_jacobian",
            vec![NamedMatrix::new("x", input_matrix).expect("jacobian input")],
            vec![NamedMatrix::new("jac", jacobian_matrix).expect("jacobian output")],
        )
        .expect("jacobian function should build");
        LoweredCase {
            primal,
            jacobian,
            jacobian_ccs,
        }
    })
}

fn lower_function_outputs(
    function: &FunctionAst,
    inputs: &[SX],
    helpers: &[SXFunction],
) -> Vec<SX> {
    function
        .outputs
        .iter()
        .map(|expr| lower_expr(expr, inputs, helpers))
        .collect()
}

fn lower_expr(expr: &ExprAst, inputs: &[SX], helpers: &[SXFunction]) -> SX {
    match expr {
        ExprAst::Const(value) => SX::from(*value),
        ExprAst::Input(index) => inputs[*index],
        ExprAst::Unary { op, arg } => {
            let arg = lower_expr(arg, inputs, helpers);
            match op {
                UnaryOpAst::Neg => -arg,
                UnaryOpAst::Sin => arg.sin(),
                UnaryOpAst::Cos => arg.cos(),
                UnaryOpAst::Exp => arg.exp(),
                UnaryOpAst::Sqrt => arg.sqrt(),
                UnaryOpAst::Log => arg.log(),
                UnaryOpAst::Square => arg * arg,
            }
        }
        ExprAst::Binary { op, lhs, rhs } => {
            let lhs = lower_expr(lhs, inputs, helpers);
            let rhs = lower_expr(rhs, inputs, helpers);
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
            let arg_values = args
                .iter()
                .map(|arg| lower_expr(arg, inputs, helpers))
                .collect::<Vec<_>>();
            let arg_matrix = SXMatrix::dense_column(arg_values).expect("call input");
            helpers[*helper]
                .call_output(&[arg_matrix])
                .expect("call output")
                .nz(*output)
        }
    }
}
