use sx_codegen_llvm::{CompiledJitFunction, FunctionCompileOptions, LlvmOptimizationLevel};
use sx_core::{CCS, CallPolicy, CallPolicyConfig, NamedMatrix, SX, SXFunction, SXMatrix};

fn named(name: &str, matrix: SXMatrix) -> NamedMatrix {
    NamedMatrix::new(name, matrix).expect("named matrix should build")
}

fn compile_with_policy(function: &SXFunction, policy: CallPolicy) -> CompiledJitFunction {
    CompiledJitFunction::compile_function_with_options(
        function,
        FunctionCompileOptions {
            opt_level: LlvmOptimizationLevel::O0,
            call_policy: CallPolicyConfig {
                default_policy: policy,
                respect_function_overrides: true,
            },
        },
    )
    .expect("compilation should succeed")
}

fn build_repeated_step_constraints() -> (SXFunction, SXFunction, CCS) {
    let z = SXMatrix::dense_column(vec![SX::sym("x"), SX::sym("u")]).expect("helper input");
    let step_outputs = SXMatrix::dense_column(vec![
        z.nz(0) + 0.3 * (z.nz(0).sin() + z.nz(1)),
        z.nz(0) * z.nz(1) + z.nz(1).sin(),
    ])
    .expect("helper output");
    let step = SXFunction::new(
        "ms_step",
        vec![named("z", z)],
        vec![named("y", step_outputs)],
    )
    .expect("helper function should build");

    let x = SXMatrix::dense_column(vec![
        SX::sym("x0"),
        SX::sym("x1"),
        SX::sym("x2"),
        SX::sym("u0"),
        SX::sym("u1"),
    ])
    .expect("root input");
    let step0 = step
        .call_output(&[SXMatrix::dense_column(vec![x.nz(0), x.nz(3)]).expect("step0 input")])
        .expect("step0 output");
    let step1 = step
        .call_output(&[SXMatrix::dense_column(vec![x.nz(1), x.nz(4)]).expect("step1 input")])
        .expect("step1 output");
    let constraints = SXMatrix::dense_column(vec![
        step0.nz(0) - x.nz(1),
        step1.nz(0) - x.nz(2),
        step0.nz(1) - 0.1,
        step1.nz(1) - 0.2,
    ])
    .expect("constraints");
    let jacobian = constraints.jacobian(&x).expect("jacobian");
    let jacobian_ccs = jacobian.ccs().clone();

    let primal = SXFunction::new(
        "mini_ms_constraints",
        vec![named("x", x.clone())],
        vec![named("g", constraints)],
    )
    .expect("primal function");
    let jac = SXFunction::new(
        "mini_ms_constraints_jacobian",
        vec![named("x", x)],
        vec![named("jac", jacobian)],
    )
    .expect("jacobian function");
    (primal, jac, jacobian_ccs)
}

fn eval_compiled(function: &CompiledJitFunction, x: &[f64]) -> Vec<f64> {
    let mut context = function.create_context();
    context.input_mut(0).copy_from_slice(x);
    function.eval(&mut context);
    context.output(0).to_vec()
}

fn finite_difference_jacobian(
    function: &CompiledJitFunction,
    x: &[f64],
    ccs: &CCS,
    step: f64,
) -> Vec<f64> {
    let base = eval_compiled(function, x);
    let rows = base.len();
    let cols = x.len();
    let mut dense = vec![0.0; rows * cols];
    for col in 0..cols {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[col] += step;
        xm[col] -= step;
        let fp = eval_compiled(function, &xp);
        let fm = eval_compiled(function, &xm);
        for row in 0..rows {
            dense[row * cols + col] = (fp[row] - fm[row]) / (2.0 * step);
        }
    }
    let mut jac = Vec::with_capacity(ccs.nnz());
    for col in 0..ccs.ncol() {
        for idx in ccs.col_ptrs()[col]..ccs.col_ptrs()[col + 1] {
            let row = ccs.row_indices()[idx];
            jac.push(dense[row * cols + col]);
        }
    }
    jac
}

#[test]
#[ignore = "manual diagnostic for call-policy Jacobian mismatch below NLP layer"]
fn diagnose_repeated_step_jacobian_by_call_policy() {
    let (primal, jacobian, jacobian_ccs) = build_repeated_step_constraints();
    let x = [0.3, -0.2, 0.1, 0.4, -0.6];

    for (label, policy) in [
        ("inline_at_call", CallPolicy::InlineAtCall),
        ("inline_at_lowering", CallPolicy::InlineAtLowering),
        ("inline_in_llvm", CallPolicy::InlineInLLVM),
        ("noinline_llvm", CallPolicy::NoInlineLLVM),
    ] {
        let primal_compiled = compile_with_policy(&primal, policy);
        let jacobian_compiled = compile_with_policy(&jacobian, policy);
        let analytic = eval_compiled(&jacobian_compiled, &x);
        let finite_difference =
            finite_difference_jacobian(&primal_compiled, &x, &jacobian_ccs, 1.0e-6);
        let max_abs_error = analytic
            .iter()
            .zip(finite_difference.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        let missing_nonzeros = analytic
            .iter()
            .zip(finite_difference.iter())
            .filter(|(lhs, rhs)| lhs.abs() <= 1.0e-7 && rhs.abs() > 1.0e-7)
            .count();

        println!(
            "{label}: max_abs_error={max_abs_error:.3e}, missing_nonzeros={missing_nonzeros}"
        );
    }
}
