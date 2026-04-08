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
