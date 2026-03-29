use examples_run::{
    constrained_rosenbrock_constraints_llvm_aot, rosenbrock_gradient_llvm_aot,
    rosenbrock_objective_llvm_aot,
};

#[test]
fn rosenbrock_generated_objective_and_gradient_evaluate() {
    let mut objective = rosenbrock_objective_llvm_aot::RosenbrockObjectiveLlvmAotContext::default();
    objective.x.copy_from_slice(&[1.0, 1.0]);
    objective.eval();
    assert!((objective.objective - 0.0).abs() < 1e-12);

    let mut gradient = rosenbrock_gradient_llvm_aot::RosenbrockGradientLlvmAotContext::default();
    gradient.x.copy_from_slice(&[1.0, 1.0]);
    gradient.eval();
    assert_eq!(gradient.gradient.len(), 2);
    assert!(gradient.gradient.iter().all(|entry| entry.abs() < 1e-12));
}

#[test]
fn constrained_rosenbrock_constraint_context_has_correct_shape() {
    let mut ctx = constrained_rosenbrock_constraints_llvm_aot::ConstrainedRosenbrockConstraintsLlvmAotContext::default();
    ctx.x.copy_from_slice(&[0.5, 0.5]);
    ctx.eval();
    assert!((ctx.constraints - 0.0).abs() < 1e-12);
}
