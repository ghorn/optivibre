use std::collections::HashMap;

use approx::assert_abs_diff_eq;
use sx_core::{CallPolicy, HessianStrategy, NamedMatrix, NodeView, SX, SXFunction, SXMatrix};

#[path = "../../test_support/symbolic_eval.rs"]
mod symbolic_eval;

use symbolic_eval::eval;

fn named(name: &str, matrix: SXMatrix) -> NamedMatrix {
    NamedMatrix::new(name, matrix).expect("named matrix should be valid")
}

fn eval_scalar(expr: SX, symbols: &[SX], point: &[f64]) -> f64 {
    let vars = symbols
        .iter()
        .zip(point.iter().copied())
        .map(|(symbol, value)| (symbol.id(), value))
        .collect::<HashMap<_, _>>();
    eval(expr, &vars)
}

fn central_hessian_entry(expr: SX, symbols: &[SX], point: &[f64], row: usize, col: usize, eps: f64) -> f64 {
    let mut shifted = point.to_vec();
    if row == col {
        shifted[row] = point[row] + eps;
        let forward = eval_scalar(expr, symbols, &shifted);
        shifted[row] = point[row] - eps;
        let backward = eval_scalar(expr, symbols, &shifted);
        let center = eval_scalar(expr, symbols, point);
        (forward - 2.0 * center + backward) / (eps * eps)
    } else {
        shifted[row] = point[row] + eps;
        shifted[col] = point[col] + eps;
        let pp = eval_scalar(expr, symbols, &shifted);

        shifted[col] = point[col] - eps;
        let pm = eval_scalar(expr, symbols, &shifted);

        shifted[row] = point[row] - eps;
        shifted[col] = point[col] + eps;
        let mp = eval_scalar(expr, symbols, &shifted);

        shifted[col] = point[col] - eps;
        let mm = eval_scalar(expr, symbols, &shifted);

        (pp - pm - mp + mm) / (4.0 * eps * eps)
    }
}

#[test]
fn sx_function_call_supports_multi_output_matrices() {
    let z = SXMatrix::sym_dense("z", 2, 1).expect("symbolic input should build");
    let sum = SXMatrix::scalar(z.nz(0) + z.nz(1));
    let features = SXMatrix::dense_column(vec![z.nz(0) * z.nz(1), z.nz(0) - z.nz(1)])
        .expect("feature vector should build");
    let function = SXFunction::new(
        "pair_features",
        vec![named("z", z)],
        vec![named("sum", sum), named("features", features)],
    )
    .expect("function should build");

    let inputs = [SXMatrix::dense_column(vec![SX::from(3.0), SX::from(2.0)])
        .expect("numeric input should build")];
    let outputs = function.call(&inputs).expect("call should succeed");

    assert_eq!(outputs.len(), 2);
    match outputs[0].nz(0).inspect() {
        NodeView::Call {
            function_name,
            output_slot,
            output_offset,
            ..
        } => {
            assert_eq!(function_name, "pair_features");
            assert_eq!(output_slot, 0);
            assert_eq!(output_offset, 0);
        }
        other => panic!("expected call node, got {other:?}"),
    }
    for (offset, value) in outputs[1].nonzeros().iter().enumerate() {
        match value.inspect() {
            NodeView::Call {
                function_name,
                output_slot,
                output_offset,
                ..
            } => {
                assert_eq!(function_name, "pair_features");
                assert_eq!(output_slot, 1);
                assert_eq!(output_offset, offset);
            }
            other => panic!("expected call node, got {other:?}"),
        }
    }
}

#[test]
fn sx_function_call_policy_override_round_trips() {
    let x = SXMatrix::sym_dense("x", 1, 1).expect("symbolic input should build");
    let function = SXFunction::new(
        "identity_scalar",
        vec![named("x", x.clone())],
        vec![named("y", x)],
    )
    .expect("function should build");

    assert_eq!(function.call_policy_override(), None);
    let overridden = function.with_call_policy_override(CallPolicy::NoInlineLLVM);
    assert_eq!(
        overridden.call_policy_override(),
        Some(CallPolicy::NoInlineLLVM)
    );
    assert_eq!(function.call_policy_override(), None);
}

#[test]
fn call_aware_derivatives_keep_irrelevant_inputs_zero() {
    let z = SXMatrix::sym_dense("z", 2, 1).expect("symbolic input should build");
    let outputs =
        SXMatrix::dense_column(vec![z.nz(0), z.nz(0).sqr()]).expect("output vector should build");
    let function = SXFunction::new("first_only", vec![named("z", z)], vec![named("y", outputs)])
        .expect("function should build");

    let x = SXMatrix::sym_dense("x", 2, 1).expect("symbolic call input should build");
    let called = function
        .call_output(&[x.clone()])
        .expect("call output should succeed");

    let jacobian = called.jacobian(&x).expect("jacobian should build");
    assert_eq!(jacobian.ccs().nnz(), 2);
    assert!(jacobian.get(0, 1).is_zero());
    assert!(jacobian.get(1, 1).is_zero());

    let gradient = SXMatrix::scalar(called.nz(1))
        .gradient(&x)
        .expect("gradient should build");
    assert!(gradient.nz(1).is_zero());

    let hessian = SXMatrix::scalar(called.nz(1))
        .hessian(&x)
        .expect("hessian should build");
    assert_eq!(hessian.ccs().nnz(), 1);
    assert!(hessian.get(1, 1).is_zero());
}

#[test]
fn call_aware_jacobian_drops_zero_seed_reverse_entries() {
    let z = SXMatrix::sym_dense("z", 2, 1).expect("symbolic input should build");
    let function = SXFunction::new(
        "identity_pair",
        vec![named("z", z.clone())],
        vec![named("y", z)],
    )
    .expect("function should build");

    let x = SXMatrix::sym_dense("x", 2, 1).expect("symbolic call input should build");
    let first_output = SXMatrix::scalar(
        function
            .call_output(&[x.clone()])
            .expect("call output should succeed")
            .nz(0),
    );

    let jacobian = first_output.jacobian(&x).expect("jacobian should build");
    assert_eq!(jacobian.ccs().nnz(), 1);
    assert!(!jacobian.get(0, 0).is_zero());
    assert!(jacobian.get(0, 1).is_zero());
}

#[test]
fn sx_function_derivative_helpers_expose_forward_and_reverse_programs() {
    let x = SXMatrix::sym_dense("x", 2, 1).expect("symbolic input should build");
    let outputs = SXMatrix::dense_column(vec![x.nz(0) * x.nz(1), x.nz(0) + x.nz(1)])
        .expect("output vector should build");
    let function = SXFunction::new("pair_ops", vec![named("x", x)], vec![named("y", outputs)])
        .expect("function should build");

    let forward = function.forward(2).expect("forward helper should build");
    assert_eq!(forward.n_in(), 3);
    assert_eq!(forward.n_out(), 2);

    let reverse = function.reverse(2).expect("reverse helper should build");
    assert_eq!(reverse.n_in(), 3);
    assert_eq!(reverse.n_out(), 2);
}

#[test]
fn call_aware_reverse_batch_helper_keeps_irrelevant_inputs_zero() {
    let z = SXMatrix::sym_dense("z", 2, 1).expect("symbolic input should build");
    let inner = SXFunction::new(
        "first_square",
        vec![named("z", z.clone())],
        vec![named("y", SXMatrix::scalar(z.nz(0).sqr()))],
    )
    .expect("inner function should build");

    let x = SXMatrix::sym_dense("x", 2, 1).expect("symbolic input should build");
    let outer_output = inner
        .call_output(&[x.clone()])
        .expect("nested call output should build");
    let outer = SXFunction::new(
        "outer",
        vec![named("x", x.clone())],
        vec![named("y", outer_output)],
    )
    .expect("outer function should build");

    let reverse = outer.reverse(2).expect("reverse helper should build");
    let outputs = reverse.outputs();
    assert_eq!(outputs.len(), 2);
    assert!(
        outputs[0].matrix().nz(1).is_zero(),
        "direction 0 irrelevant slot: {:?}",
        outputs[0].matrix().nz(1).inspect()
    );
    assert!(
        outputs[1].matrix().nz(1).is_zero(),
        "direction 1 irrelevant slot: {:?}",
        outputs[1].matrix().nz(1).inspect()
    );
    let helper_x = reverse.inputs()[0].matrix();
    let helper_s0 = reverse.inputs()[1].matrix();
    let helper_s1 = reverse.inputs()[2].matrix();
    let direction_0_free = outputs[0].matrix().nz(0).free_symbols();
    assert!(direction_0_free.contains(&helper_x.nz(0)));
    assert!(direction_0_free.contains(&helper_s0.nz(0)));
    assert!(!direction_0_free.contains(&helper_x.nz(1)));
    assert!(!direction_0_free.contains(&helper_s1.nz(0)));

    let direction_1_free = outputs[1].matrix().nz(0).free_symbols();
    assert!(direction_1_free.contains(&helper_x.nz(0)));
    assert!(direction_1_free.contains(&helper_s1.nz(0)));
    assert!(!direction_1_free.contains(&helper_x.nz(1)));
    assert!(!direction_1_free.contains(&helper_s0.nz(0)));
}

#[test]
fn preserved_call_jacobian_keeps_direct_identity_terms_alongside_constant_helpers() {
    let z = SXMatrix::sym_dense("z", 1, 1).expect("symbolic input should build");
    let helper0 = SXFunction::new(
        "const_leaf",
        vec![named("z", z.clone())],
        vec![named(
            "y",
            SXMatrix::dense_column(vec![SX::from(0.0), SX::from(0.75)])
                .expect("helper output should build"),
        )],
    )
    .expect("helper0 should build")
    .with_call_policy_override(CallPolicy::NoInlineLLVM);
    let helper1 = SXFunction::new(
        "const_mid",
        vec![named("z", z.clone())],
        vec![named(
            "y",
            SXMatrix::scalar(
                helper0
                    .call_output(&[z.clone()])
                    .expect("helper0 call should build")
                    .nz(0),
            ),
        )],
    )
    .expect("helper1 should build")
    .with_call_policy_override(CallPolicy::NoInlineLLVM);
    let helper2 = SXFunction::new(
        "const_root",
        vec![named("z", z.clone())],
        vec![named(
            "y",
            SXMatrix::scalar(
                helper1
                    .call_output(&[z.clone()])
                    .expect("helper1 call should build")
                    .nz(0),
            ),
        )],
    )
    .expect("helper2 should build")
    .with_call_policy_override(CallPolicy::NoInlineLLVM);

    let x = SXMatrix::sym_dense("x", 2, 1).expect("symbolic input should build");
    let helper_outputs = helper0
        .call_output(&[SXMatrix::dense_column(vec![x.nz(0)]).expect("helper input should build")])
        .expect("helper call should build");
    let outputs = SXMatrix::dense_column(vec![
        helper2
            .call_output(&[SXMatrix::dense_column(vec![x.nz(0).sin()]).expect("nested input")])
            .expect("nested helper call should build")
            .nz(0),
        x.nz(1) + helper_outputs.nz(1),
    ])
    .expect("outer outputs should build");

    let jacobian = outputs.jacobian(&x).expect("jacobian should build");
    assert!(
        !jacobian.get(1, 1).is_zero(),
        "expected direct identity derivative to survive alongside constant helper calls: {:?}",
        jacobian.get(1, 1).inspect()
    );
}

#[test]
fn preserved_call_hessian_matches_central_difference_for_nested_helpers() {
    let z = SXMatrix::sym_dense("z", 2, 1).expect("inner input should build");
    let inner = SXFunction::new(
        "nested_inner_hessian",
        vec![named("z", z.clone())],
        vec![named(
            "y",
            SXMatrix::dense_column(vec![
                z.nz(0) * z.nz(1) + z.nz(0).sin(),
                z.nz(1).exp() + z.nz(0).sqr(),
            ])
            .expect("inner outputs should build"),
        )],
    )
    .expect("inner should build")
    .with_call_policy_override(CallPolicy::NoInlineLLVM);

    let w = SXMatrix::sym_dense("w", 2, 1).expect("outer input should build");
    let inner_called = inner
        .call_output(&[w.clone()])
        .expect("inner call should build");
    let outer_scalar = SXMatrix::scalar(
        inner_called.nz(0) * w.nz(0) + inner_called.nz(1).sin() + w.nz(1).sqr(),
    );
    let outer = SXFunction::new(
        "nested_outer_hessian",
        vec![named("w", w.clone())],
        vec![named("y", outer_scalar)],
    )
    .expect("outer should build")
    .with_call_policy_override(CallPolicy::NoInlineLLVM);

    let x = SXMatrix::sym_dense("x", 3, 1).expect("root input should build");
    let outer_input = SXMatrix::dense_column(vec![
        x.nz(0) + x.nz(2),
        x.nz(1) * x.nz(2) + x.nz(0).sin(),
    ])
    .expect("outer call input should build");
    let scalar = SXMatrix::scalar(
        outer
            .call_output(&[outer_input])
            .expect("outer call should build")
            .nz(0)
            + x.nz(0) * x.nz(1) * x.nz(2),
    );

    let default_hessian = scalar.hessian(&x).expect("default hessian should build");
    let by_column = scalar
        .hessian_with_strategy(&x, HessianStrategy::LowerTriangleByColumn)
        .expect("by-column hessian should build");
    assert_eq!(default_hessian.ccs(), by_column.ccs());

    let symbols = [x.nz(0), x.nz(1), x.nz(2)];
    let point = [0.2, -0.3, 0.4];
    let vars = HashMap::from([
        (symbols[0].id(), point[0]),
        (symbols[1].id(), point[1]),
        (symbols[2].id(), point[2]),
    ]);
    let eps = 1.0e-5;

    for col in 0..3 {
        for row in col..3 {
            let analytic_default = eval(default_hessian.get(row, col), &vars);
            let analytic_by_column = eval(by_column.get(row, col), &vars);
            let finite_difference = central_hessian_entry(scalar.nz(0), &symbols, &point, row, col, eps);

            assert!(
                (analytic_default - analytic_by_column).abs() <= 1.0e-10,
                "strategy mismatch at ({row}, {col}): default={analytic_default:.12e} by_column={analytic_by_column:.12e}",
            );
            assert!(
                (analytic_default - finite_difference).abs() <= 2.0e-4,
                "finite-difference mismatch at ({row}, {col}): analytic={analytic_default:.12e} fd={finite_difference:.12e}",
            );
        }
    }
}

#[test]
fn preserved_call_jacobian_keeps_identity_terms_through_called_function_boundary() {
    let z = SXMatrix::sym_dense("z", 1, 1).expect("symbolic input should build");
    let helper0 = SXFunction::new(
        "const_leaf_boundary",
        vec![named("z", z.clone())],
        vec![named(
            "y",
            SXMatrix::dense_column(vec![SX::from(0.0), SX::from(0.75)])
                .expect("helper output should build"),
        )],
    )
    .expect("helper0 should build")
    .with_call_policy_override(CallPolicy::NoInlineLLVM);
    let helper1 = SXFunction::new(
        "const_mid_boundary",
        vec![named("z", z.clone())],
        vec![named(
            "y",
            SXMatrix::scalar(
                helper0
                    .call_output(&[z.clone()])
                    .expect("helper0 call should build")
                    .nz(0),
            ),
        )],
    )
    .expect("helper1 should build")
    .with_call_policy_override(CallPolicy::NoInlineLLVM);
    let helper2 = SXFunction::new(
        "const_root_boundary",
        vec![named("z", z.clone())],
        vec![named(
            "y",
            SXMatrix::scalar(
                helper1
                    .call_output(&[z.clone()])
                    .expect("helper1 call should build")
                    .nz(0),
            ),
        )],
    )
    .expect("helper2 should build")
    .with_call_policy_override(CallPolicy::NoInlineLLVM);

    let x_inner = SXMatrix::sym_dense("x_inner", 2, 1).expect("inner input should build");
    let inner_outputs = SXMatrix::dense_column(vec![
        helper2
            .call_output(
                &[SXMatrix::dense_column(vec![x_inner.nz(0).sin()]).expect("nested input")],
            )
            .expect("nested helper call should build")
            .nz(0),
        x_inner.nz(1)
            + helper0
                .call_output(&[
                    SXMatrix::dense_column(vec![x_inner.nz(0)]).expect("helper input should build")
                ])
                .expect("helper call should build")
                .nz(1),
    ])
    .expect("inner outputs should build");
    let inner = SXFunction::new(
        "integrator_like_boundary",
        vec![named("x", x_inner)],
        vec![named("y", inner_outputs)],
    )
    .expect("inner function should build")
    .with_call_policy_override(CallPolicy::NoInlineLLVM);

    let x = SXMatrix::sym_dense("x", 2, 1).expect("outer input should build");
    let outer_outputs = inner
        .call_output(&[x.clone()])
        .expect("outer call should build");
    let jacobian = outer_outputs.jacobian(&x).expect("jacobian should build");
    assert!(
        !jacobian.get(1, 1).is_zero(),
        "expected called-function identity derivative to survive preserved call boundary: {:?}",
        jacobian.get(1, 1).inspect()
    );
}

#[test]
fn preserved_call_jacobian_keeps_identity_terms_with_multi_input_called_function() {
    let z = SXMatrix::sym_dense("z", 1, 1).expect("symbolic input should build");
    let helper0 = SXFunction::new(
        "const_leaf_multi_input",
        vec![named("z", z.clone())],
        vec![named(
            "y",
            SXMatrix::dense_column(vec![SX::from(0.0), SX::from(0.75)])
                .expect("helper output should build"),
        )],
    )
    .expect("helper0 should build")
    .with_call_policy_override(CallPolicy::NoInlineLLVM);
    let helper1 = SXFunction::new(
        "const_mid_multi_input",
        vec![named("z", z.clone())],
        vec![named(
            "y",
            SXMatrix::scalar(
                helper0
                    .call_output(&[z.clone()])
                    .expect("helper0 call should build")
                    .nz(0),
            ),
        )],
    )
    .expect("helper1 should build")
    .with_call_policy_override(CallPolicy::NoInlineLLVM);

    let x_inner = SXMatrix::sym_dense("x_inner", 2, 1).expect("x input should build");
    let u_inner = SXMatrix::sym_dense("u_inner", 2, 1).expect("u input should build");
    let dudt_inner = SXMatrix::sym_dense("dudt_inner", 2, 1).expect("dudt input should build");
    let dt_inner = SXMatrix::sym_dense("dt_inner", 1, 1).expect("dt input should build");
    let inner_outputs = SXMatrix::dense_column(vec![
        helper1
            .call_output(&[SXMatrix::dense_column(vec![x_inner.nz(0).sin()])
                .expect("nested input should build")])
            .expect("nested helper call should build")
            .nz(0),
        x_inner.nz(1)
            + dt_inner.nz(0)
                * helper0
                    .call_output(&[SXMatrix::dense_column(vec![
                        x_inner.nz(0) + u_inner.nz(0) + dudt_inner.nz(0),
                    ])
                    .expect("helper input should build")])
                    .expect("helper call should build")
                    .nz(1),
    ])
    .expect("inner outputs should build");
    let inner = SXFunction::new(
        "integrator_like_multi_input",
        vec![
            named("x", x_inner),
            named("u", u_inner),
            named("dudt", dudt_inner),
            named("dt", dt_inner),
        ],
        vec![named("y", inner_outputs)],
    )
    .expect("inner function should build")
    .with_call_policy_override(CallPolicy::NoInlineLLVM);

    let x = SXMatrix::sym_dense("x", 2, 1).expect("outer x input should build");
    let outer_outputs = inner
        .call_output(&[
            x.clone(),
            SXMatrix::dense_column(vec![SX::from(0.1), SX::from(0.2)]).expect("u const"),
            SXMatrix::dense_column(vec![SX::from(0.3), SX::from(0.4)]).expect("dudt const"),
            SXMatrix::dense_column(vec![SX::from(0.5)]).expect("dt const"),
        ])
        .expect("outer call should build");
    let jacobian = outer_outputs.jacobian(&x).expect("jacobian should build");
    assert!(
        !jacobian.get(1, 1).is_zero(),
        "expected called-function identity derivative to survive multi-input preserved call boundary: {:?}",
        jacobian.get(1, 1).inspect()
    );
}

#[test]
fn preserved_call_forward_combined_seed_uses_seed_specific_call_memoization() {
    let z = SXMatrix::sym_dense("z", 2, 1).expect("symbolic input should build");
    let inner = SXFunction::new(
        "identity_pair_seed_memo",
        vec![named("z", z.clone())],
        vec![named("y", z)],
    )
    .expect("inner function should build")
    .with_call_policy_override(CallPolicy::NoInlineLLVM);

    let x = SXMatrix::sym_dense("x", 2, 1).expect("outer input should build");
    let called = inner
        .call_output(&[x.clone()])
        .expect("call output should build");
    let combined_seed =
        SXMatrix::dense_column(vec![SX::from(1.0), SX::from(1.0)]).expect("seed should build");
    let forward = called
        .forward(&x, &combined_seed)
        .expect("combined forward should build");

    let vars = HashMap::from([(x.nz(0).id(), 2.0), (x.nz(1).id(), 3.0)]);
    assert_abs_diff_eq!(eval(forward.nz(0), &vars), 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(eval(forward.nz(1), &vars), 1.0, epsilon = 1e-12);

    let jacobian = called.jacobian(&x).expect("jacobian should build");
    assert_abs_diff_eq!(eval(jacobian.get(0, 0), &vars), 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(eval(jacobian.get(1, 1), &vars), 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(eval(jacobian.get(0, 1), &vars), 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(eval(jacobian.get(1, 0), &vars), 0.0, epsilon = 1e-12);
}
