use sx_core::{CallPolicy, NamedMatrix, NodeView, SX, SXFunction, SXMatrix};

fn named(name: &str, matrix: SXMatrix) -> NamedMatrix {
    NamedMatrix::new(name, matrix).expect("named matrix should be valid")
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
