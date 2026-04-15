use std::collections::HashMap;

use sx_core::{BinaryOp, NodeView, SX, UnaryOp, lookup_function};

pub fn eval(expr: SX, vars: &HashMap<u64, f64>) -> f64 {
    match expr.inspect() {
        NodeView::Constant(value) => value,
        NodeView::Symbol { .. } => vars[&expr.id()],
        NodeView::Unary { op, arg } => {
            let arg = eval(arg, vars);
            match op {
                UnaryOp::Abs => arg.abs(),
                UnaryOp::Sign => {
                    if arg > 0.0 {
                        1.0
                    } else if arg < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                }
                UnaryOp::Floor => arg.floor(),
                UnaryOp::Ceil => arg.ceil(),
                UnaryOp::Round => arg.round(),
                UnaryOp::Trunc => arg.trunc(),
                UnaryOp::Sqrt => arg.sqrt(),
                UnaryOp::Exp => arg.exp(),
                UnaryOp::Log => arg.ln(),
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
        NodeView::Binary { op, lhs, rhs } => {
            let lhs = eval(lhs, vars);
            let rhs = eval(rhs, vars);
            match op {
                BinaryOp::Add => lhs + rhs,
                BinaryOp::Sub => lhs - rhs,
                BinaryOp::Mul => lhs * rhs,
                BinaryOp::Div => lhs / rhs,
                BinaryOp::Pow => lhs.powf(rhs),
                BinaryOp::Atan2 => lhs.atan2(rhs),
                BinaryOp::Hypot => lhs.hypot(rhs),
                BinaryOp::Mod => lhs % rhs,
                BinaryOp::Copysign => lhs.copysign(rhs),
            }
        }
        NodeView::Call {
            function_id,
            inputs,
            output_slot,
            output_offset,
            ..
        } => {
            let function = lookup_function(function_id).expect("call should reference a function");
            let mut call_vars = HashMap::new();
            for (formal, actual) in function.inputs().iter().zip(inputs.iter()) {
                for (&symbol, &expr) in formal
                    .matrix()
                    .nonzeros()
                    .iter()
                    .zip(actual.nonzeros().iter())
                {
                    call_vars.insert(symbol.id(), eval(expr, vars));
                }
            }
            eval(
                function.outputs()[output_slot].matrix().nonzeros()[output_offset],
                &call_vars,
            )
        }
    }
}
