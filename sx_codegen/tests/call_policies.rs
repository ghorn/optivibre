use sx_codegen::lower_function_with_policies;
use sx_core::{CallPolicy, CallPolicyConfig, NamedMatrix, SXFunction, SXMatrix};

fn named(name: &str, matrix: SXMatrix) -> NamedMatrix {
    NamedMatrix::new(name, matrix).expect("named matrix should be valid")
}

fn build_reused_call_function(callee_policy: Option<CallPolicy>) -> SXFunction {
    let z = SXMatrix::sym_dense("z", 1, 1).expect("symbolic input should build");
    let callee = SXFunction::new(
        "affine_shift",
        vec![named("z", z.clone())],
        vec![named("value", SXMatrix::scalar(2.0 * z.nz(0) + 1.0))],
    )
    .expect("callee should build");
    let callee = if let Some(policy) = callee_policy {
        callee.with_call_policy_override(policy)
    } else {
        callee
    };

    let x = SXMatrix::sym_dense("x", 1, 1).expect("symbolic root input should build");
    let call0 = callee
        .call_output(&[SXMatrix::scalar(x.nz(0))])
        .expect("first call should build");
    let call1 = callee
        .call_output(&[SXMatrix::scalar(x.nz(0) + 2.0)])
        .expect("second call should build");
    SXFunction::new(
        "reused_calls",
        vec![named("x", x)],
        vec![named("y", SXMatrix::scalar(call0.nz(0) + call1.nz(0)))],
    )
    .expect("root function should build")
}

#[test]
fn ignored_function_overrides_emit_warning_and_stats() {
    let function = build_reused_call_function(Some(CallPolicy::NoInlineLLVM));
    let lowered = lower_function_with_policies(
        &function,
        CallPolicyConfig {
            default_policy: CallPolicy::InlineAtLowering,
            respect_function_overrides: false,
        },
    )
    .expect("lowering should succeed");

    assert_eq!(lowered.warnings.len(), 2);
    assert_eq!(lowered.stats.overrides_ignored, 4);
    assert_eq!(lowered.stats.inlines_at_lowering, 2);
    assert!(lowered.subfunctions.is_empty());
}

#[test]
fn noinline_policy_preserves_subfunction_and_call_stats() {
    let function = build_reused_call_function(Some(CallPolicy::NoInlineLLVM));
    let lowered = lower_function_with_policies(
        &function,
        CallPolicyConfig {
            default_policy: CallPolicy::InlineAtLowering,
            respect_function_overrides: true,
        },
    )
    .expect("lowering should succeed");

    assert_eq!(lowered.subfunctions.len(), 1);
    assert_eq!(
        lowered.subfunctions[0].call_policy,
        CallPolicy::NoInlineLLVM
    );
    assert_eq!(lowered.stats.call_site_count, 2);
    assert_eq!(lowered.stats.overrides_applied, 4);
    assert_eq!(lowered.stats.llvm_subfunctions_emitted, 1);
    assert_eq!(lowered.stats.llvm_call_instructions_emitted, 2);
    assert!(lowered.stats.no_inline_llvm_policy_count >= 2);
}

#[test]
fn inline_at_call_matches_legacy_flattening_behavior() {
    let function = build_reused_call_function(None);
    let lowered = lower_function_with_policies(
        &function,
        CallPolicyConfig {
            default_policy: CallPolicy::InlineAtCall,
            respect_function_overrides: true,
        },
    )
    .expect("lowering should succeed");

    assert!(lowered.subfunctions.is_empty());
    assert_eq!(lowered.stats.inlines_at_call, 2);
    assert_eq!(lowered.stats.call_site_count, 0);
    assert_eq!(lowered.stats.llvm_call_instructions_emitted, 0);
}
