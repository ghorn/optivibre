use std::collections::HashSet;

use anyhow::{Result as AnyResult, bail};
use approx::assert_abs_diff_eq;
use sx_codegen::{InstructionKind, LoweredFunction, ValueRef, lower_function};
use sx_codegen_llvm::{
    AotWrapperOptions, CompiledJitFunction, JitOptimizationLevel, LlvmOptimizationLevel,
    LlvmTarget, emit_object_bytes_lowered, generate_aot_wrapper_module,
};
use sx_core::{BinaryOp, CCS, NamedMatrix, NodeView, SX, SXFunction, SXMatrix, UnaryOp};

#[derive(Clone, Debug, PartialEq)]
struct DenseValue {
    nrow: usize,
    ncol: usize,
    values: Vec<f64>,
}

impl DenseValue {
    fn scalar(value: f64) -> Self {
        Self {
            nrow: 1,
            ncol: 1,
            values: vec![value],
        }
    }

    fn from_rows(rows: &[&[f64]]) -> Self {
        let nrow = rows.len();
        let ncol = rows.first().map_or(0, |row| row.len());
        let mut values = Vec::with_capacity(nrow * ncol);
        for col in 0..ncol {
            for row in rows {
                values.push(row[col]);
            }
        }
        Self { nrow, ncol, values }
    }
}

fn must_ok<T: core::fmt::Debug, E: core::fmt::Debug>(result: Result<T, E>) -> T {
    assert!(result.is_ok(), "expected Ok(..), got {result:?}");
    match result {
        Ok(value) => value,
        Err(_) => unreachable!(),
    }
}

fn dense_value_to_slot(ccs: &CCS, dense: &DenseValue) -> AnyResult<Vec<f64>> {
    if dense.nrow != ccs.nrow() || dense.ncol != ccs.ncol() {
        bail!(
            "shape mismatch: expected {}x{}, got {}x{}",
            ccs.nrow(),
            ccs.ncol(),
            dense.nrow,
            dense.ncol
        );
    }
    Ok(ccs
        .positions()
        .into_iter()
        .map(|(row, col)| dense.values[row + col * ccs.nrow()])
        .collect())
}

fn dense_value_from_slot(ccs: &CCS, values: &[f64]) -> DenseValue {
    let mut dense = DenseValue {
        nrow: ccs.nrow(),
        ncol: ccs.ncol(),
        values: vec![0.0; ccs.nrow() * ccs.ncol()],
    };
    for ((row, col), value) in ccs.positions().into_iter().zip(values.iter().copied()) {
        dense.values[row + col * ccs.nrow()] = value;
    }
    dense
}

fn jit_eval_function(function: &SXFunction, inputs: &[DenseValue]) -> AnyResult<Vec<DenseValue>> {
    if inputs.len() != function.n_in() {
        bail!(
            "input arity mismatch: expected {}, got {}",
            function.n_in(),
            inputs.len()
        );
    }

    let compiled = CompiledJitFunction::compile_function(function, JitOptimizationLevel::O0)?;
    let mut context = compiled.create_context();

    for (slot_idx, (slot, dense)) in function.inputs().iter().zip(inputs).enumerate() {
        let projected = dense_value_to_slot(slot.matrix().ccs(), dense)?;
        context.input_mut(slot_idx).copy_from_slice(&projected);
    }

    compiled.eval(&mut context);

    Ok(function
        .outputs()
        .iter()
        .enumerate()
        .map(|(slot_idx, slot)| {
            dense_value_from_slot(slot.matrix().ccs(), context.output(slot_idx))
        })
        .collect())
}

fn eval_unary(op: UnaryOp, value: f64) -> f64 {
    match op {
        UnaryOp::Abs => value.abs(),
        UnaryOp::Sign => {
            if value > 0.0 {
                1.0
            } else if value < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        UnaryOp::Floor => value.floor(),
        UnaryOp::Ceil => value.ceil(),
        UnaryOp::Round => value.round(),
        UnaryOp::Trunc => value.trunc(),
        UnaryOp::Sqrt => value.sqrt(),
        UnaryOp::Exp => value.exp(),
        UnaryOp::Log => value.ln(),
        UnaryOp::Sin => value.sin(),
        UnaryOp::Cos => value.cos(),
        UnaryOp::Tan => value.tan(),
        UnaryOp::Asin => value.asin(),
        UnaryOp::Acos => value.acos(),
        UnaryOp::Atan => value.atan(),
        UnaryOp::Sinh => value.sinh(),
        UnaryOp::Cosh => value.cosh(),
        UnaryOp::Tanh => value.tanh(),
        UnaryOp::Asinh => value.asinh(),
        UnaryOp::Acosh => value.acosh(),
        UnaryOp::Atanh => value.atanh(),
    }
}

fn eval_binary(op: BinaryOp, lhs: f64, rhs: f64) -> f64 {
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

fn lowered_eval_function(
    lowered: &LoweredFunction,
    inputs: &[DenseValue],
) -> AnyResult<Vec<DenseValue>> {
    let projected_inputs = lowered
        .inputs
        .iter()
        .zip(inputs)
        .map(|(slot, dense)| dense_value_to_slot(&slot.ccs, dense))
        .collect::<AnyResult<Vec<_>>>()?;

    let mut temps = vec![0.0; lowered.instructions.len()];
    for instruction in &lowered.instructions {
        let value_ref = |value: ValueRef, temps: &[f64]| -> f64 {
            match value {
                ValueRef::Const(value) => value,
                ValueRef::Temp(temp) => temps[temp],
                ValueRef::Input { slot, offset } => projected_inputs[slot][offset],
            }
        };

        temps[instruction.temp] = match instruction.kind {
            InstructionKind::Unary { op, input } => eval_unary(op, value_ref(input, &temps)),
            InstructionKind::Binary { op, lhs, rhs } => {
                eval_binary(op, value_ref(lhs, &temps), value_ref(rhs, &temps))
            }
        };
    }

    Ok(lowered
        .outputs
        .iter()
        .zip(&lowered.output_values)
        .map(|(slot, values)| {
            let slot_values = values
                .iter()
                .map(|value| match *value {
                    ValueRef::Const(value) => value,
                    ValueRef::Temp(temp) => temps[temp],
                    ValueRef::Input { slot, offset } => projected_inputs[slot][offset],
                })
                .collect::<Vec<_>>();
            dense_value_from_slot(&slot.ccs, &slot_values)
        })
        .collect())
}

fn node_count(expr: SX) -> usize {
    fn visit(expr: SX, seen: &mut HashSet<SX>) {
        if !seen.insert(expr) {
            return;
        }
        match expr.inspect() {
            NodeView::Constant(_) | NodeView::Symbol { .. } => {}
            NodeView::Unary { arg, .. } => visit(arg, seen),
            NodeView::Binary { lhs, rhs, .. } => {
                visit(lhs, seen);
                visit(rhs, seen);
            }
        }
    }

    let mut seen = HashSet::new();
    visit(expr, &mut seen);
    seen.len()
}

fn named_matrix(name: &str, matrix: SXMatrix) -> NamedMatrix {
    must_ok(NamedMatrix::new(name, matrix))
}

fn rk4_reference_step(x0: [f64; 2], u: f64, dt: f64, stages: usize) -> ([f64; 2], f64) {
    fn dynamics(x: [f64; 2], u: f64) -> ([f64; 2], f64) {
        let xdot = [(1.0 - x[1] * x[1]) * x[0] - x[1] + u, x[0]];
        let stage_cost = x[0] * x[0] + x[1] * x[1] + u * u;
        (xdot, stage_cost)
    }

    let mut x = x0;
    let mut q = 0.0;
    for _ in 0..stages {
        let (k1, k1_q) = dynamics(x, u);
        let (k2, k2_q) = dynamics([x[0] + dt * 0.5 * k1[0], x[1] + dt * 0.5 * k1[1]], u);
        let (k3, k3_q) = dynamics([x[0] + dt * 0.5 * k2[0], x[1] + dt * 0.5 * k2[1]], u);
        let (k4, k4_q) = dynamics([x[0] + dt * k3[0], x[1] + dt * k3[1]], u);
        x = [
            x[0] + dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
            x[1] + dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        ];
        q += dt / 6.0 * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q);
    }
    (x, q)
}

#[test]
fn casadi_ssym_example_constructs_symbolic_shapes() {
    let scalar = SX::sym("x");
    let NodeView::Symbol { name, .. } = scalar.inspect() else {
        unreachable!();
    };
    assert_eq!(name, "x");

    let column = must_ok(SXMatrix::sym_dense("x", 2, 1));
    let row = must_ok(SXMatrix::sym_dense("x", 1, 2));
    let matrix = must_ok(SXMatrix::sym_dense("x", 2, 3));

    assert_eq!(column.shape(), (2, 1));
    assert_eq!(row.shape(), (1, 2));
    assert_eq!(matrix.shape(), (2, 3));
    assert!(column.nonzeros().iter().all(|expr| expr.is_symbolic()));
    assert!(row.nonzeros().iter().all(|expr| expr.is_symbolic()));
    assert!(matrix.nonzeros().iter().all(|expr| expr.is_symbolic()));
}

#[test]
fn casadi_sxfunction_constr_example_constructs_and_evaluates() {
    let x = SX::sym("x");
    let y = must_ok(SXMatrix::sym_dense("y", 2, 1));
    let _z = must_ok(SXMatrix::sym_dense("z", 2, 3));

    let concatenated = must_ok(SXMatrix::dense_column(vec![x, y.nz(0), y.nz(1)]));
    let scaled = y.map_nonzeros(|entry| entry * x);

    let function = must_ok(SXFunction::new(
        "f",
        vec![
            named_matrix("x", SXMatrix::scalar(x)),
            named_matrix("y", y.clone()),
        ],
        vec![
            named_matrix("out_x", SXMatrix::scalar(x)),
            named_matrix("out_y", y),
            named_matrix("out_cat", concatenated),
            named_matrix("out_scaled", scaled),
            named_matrix("out_zero", SXMatrix::scalar(0.0)),
        ],
    ));

    assert_eq!(function.n_in(), 2);
    assert_eq!(function.n_out(), 5);

    let outputs = must_ok(jit_eval_function(
        &function,
        &[
            DenseValue::scalar(2.0),
            DenseValue::from_rows(&[&[3.0], &[4.0]]),
        ],
    ));

    assert_eq!(outputs[0], DenseValue::scalar(2.0));
    assert_eq!(outputs[1], DenseValue::from_rows(&[&[3.0], &[4.0]]));
    assert_eq!(outputs[2], DenseValue::from_rows(&[&[2.0], &[3.0], &[4.0]]));
    assert_eq!(outputs[3], DenseValue::from_rows(&[&[6.0], &[8.0]]));
    assert_eq!(outputs[4], DenseValue::scalar(0.0));
}

#[test]
fn casadi_accessing_sx_algorithm_example_manual_eval_matches_jit() {
    let a = SX::sym("a");
    let b = must_ok(SXMatrix::sym_dense("b", 2, 1));
    let outputs = must_ok(SXMatrix::dense_column(vec![
        2.0 * a + b.nz(0),
        2.0 * a + b.nz(1),
    ]));
    let function = must_ok(SXFunction::new(
        "f",
        vec![named_matrix("a", SXMatrix::scalar(a)), named_matrix("b", b)],
        vec![named_matrix("r", outputs)],
    ));

    let inputs = [
        DenseValue::scalar(2.0),
        DenseValue::from_rows(&[&[3.0], &[4.0]]),
    ];
    let lowered = must_ok(lower_function(&function));
    let manual_outputs = must_ok(lowered_eval_function(&lowered, &inputs));
    let jit_outputs = must_ok(jit_eval_function(&function, &inputs));

    assert_eq!(manual_outputs, jit_outputs);
    assert_eq!(jit_outputs[0], DenseValue::from_rows(&[&[7.0], &[8.0]]));
}

#[test]
fn casadi_generate_code_example_emits_wrapper_and_object() {
    let x = SX::sym("x");
    let y = SX::sym("y");
    let mut z = x * y + 2.0 * y;
    z += 4.0 * z;

    let function = must_ok(SXFunction::new(
        "f",
        vec![
            named_matrix("x", SXMatrix::scalar(x)),
            named_matrix("y", SXMatrix::scalar(y)),
        ],
        vec![named_matrix("z", SXMatrix::scalar(z))],
    ));

    let lowered = must_ok(lower_function(&function));
    let wrapper = must_ok(generate_aot_wrapper_module(
        &lowered,
        &AotWrapperOptions {
            emit_doc_comments: true,
        },
    ));
    let object = must_ok(emit_object_bytes_lowered(
        &lowered,
        LlvmOptimizationLevel::O0,
        &LlvmTarget::Native,
    ));
    let outputs = must_ok(jit_eval_function(
        &function,
        &[DenseValue::scalar(3.0), DenseValue::scalar(2.0)],
    ));

    assert!(wrapper.contains("pub mod f_llvm_aot"));
    assert!(wrapper.contains("Generated LLVM AOT wrapper"));
    assert!(!object.is_empty());
    assert_eq!(outputs[0], DenseValue::scalar(50.0));
}

#[test]
fn casadi_n_nodes_example_tracks_graph_growth() {
    let x = SX::sym("x");
    let y = SX::sym("y");

    let mut z = x * y + 2.0 * y;
    assert_eq!(node_count(z), 6);

    z += 4.0 * z;
    assert_eq!(node_count(z), 8);

    z *= z + 1.0;
    assert_eq!(node_count(z), 11);
}

#[test]
fn casadi_sparsity_jac_example_is_tridiagonal() {
    let x = must_ok(SXMatrix::sym_dense("x", 40, 1));
    let y = must_ok(SXMatrix::dense_column(
        (0..38)
            .map(|idx| x.nz(idx) - 2.0 * x.nz(idx + 1) + x.nz(idx + 2))
            .collect(),
    ));
    let jacobian_ccs = must_ok(y.jacobian_ccs(&x));
    let mut expected = Vec::new();
    for col in 0_usize..40 {
        let row_start = col.saturating_sub(2);
        let row_end = col.min(37);
        for row in row_start..=row_end {
            if row + 2 >= col && row <= col {
                expected.push((row, col));
            }
        }
    }

    assert_eq!(jacobian_ccs.nrow(), 38);
    assert_eq!(jacobian_ccs.ncol(), 40);
    assert_eq!(jacobian_ccs.positions(), expected);
}

#[test]
fn casadi_direct_single_shooting_rk4_example_matches_numeric_reference() {
    let x0 = must_ok(SXMatrix::sym_dense("x0", 2, 1));
    let u = SXMatrix::scalar(SX::sym("u"));
    let dt = 10.0 / 20.0 / 4.0;

    let stage = |state0: SX, state1: SX, control: SX| -> (SX, SX, SX) {
        (
            (1.0 - state1.sqr()) * state0 - state1 + control,
            state0,
            state0.sqr() + state1.sqr() + control.sqr(),
        )
    };

    let mut xk0 = x0.nz(0);
    let mut xk1 = x0.nz(1);
    let mut q = SX::zero();
    for _ in 0..4 {
        let (k1_0, k1_1, k1_q) = stage(xk0, xk1, u.nz(0));
        let (k2_0, k2_1, k2_q) = stage(xk0 + dt * 0.5 * k1_0, xk1 + dt * 0.5 * k1_1, u.nz(0));
        let (k3_0, k3_1, k3_q) = stage(xk0 + dt * 0.5 * k2_0, xk1 + dt * 0.5 * k2_1, u.nz(0));
        let (k4_0, k4_1, k4_q) = stage(xk0 + dt * k3_0, xk1 + dt * k3_1, u.nz(0));

        xk0 += dt / 6.0 * (k1_0 + 2.0 * k2_0 + 2.0 * k3_0 + k4_0);
        xk1 += dt / 6.0 * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1);
        q += dt / 6.0 * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q);
    }

    let function = must_ok(SXFunction::new(
        "direct_single_shooting_rk4",
        vec![named_matrix("x0", x0), named_matrix("p", u)],
        vec![
            named_matrix("xf", must_ok(SXMatrix::dense_column(vec![xk0, xk1]))),
            named_matrix("qf", SXMatrix::scalar(q)),
        ],
    ));

    let outputs = must_ok(jit_eval_function(
        &function,
        &[
            DenseValue::from_rows(&[&[0.2], &[0.3]]),
            DenseValue::scalar(0.4),
        ],
    ));
    let (xf_ref, qf_ref) = rk4_reference_step([0.2, 0.3], 0.4, dt, 4);

    assert_abs_diff_eq!(outputs[0].values[0], xf_ref[0], epsilon = 1e-12);
    assert_abs_diff_eq!(outputs[0].values[1], xf_ref[1], epsilon = 1e-12);
    assert_abs_diff_eq!(outputs[1].values[0], qf_ref, epsilon = 1e-12);
}
