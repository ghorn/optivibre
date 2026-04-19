use std::cell::RefCell;
use std::collections::HashMap;

use proptest::test_runner::{Config, TestCaseError, TestCaseResult, TestRunner};
use sx_codegen_llvm::{CompiledJitFunction, FunctionCompileOptions, LlvmOptimizationLevel};
use sx_core::{CallPolicy, CallPolicyConfig, SXFunction};

#[path = "../../test_support/jacobian_proptest/mod.rs"]
mod jacobian_proptest;

use jacobian_proptest::{
    CoverageCounters, DenseMatrix, GeneratedCase, GeneratedCaseRequirements, GeneratorConfig,
    OperatorTier, ProfileMode, PropertyScenario, case_seed_strategy, compare_dense_matrices,
    dense_from_sx_ccs_values, eval_ast_outputs, eval_lowered_function_outputs,
    eval_symbolic_function_nonzeros, finite_difference_jacobian, generate_case_from_seed,
    lower_case_to_sx_functions,
};

const CI_CONFIG: GeneratorConfig = GeneratorConfig {
    max_inputs: 4,
    max_outputs: 3,
    max_helpers: 3,
    max_helper_outputs: 3,
    max_nodes_per_expr: 10,
    max_call_depth: 2,
    exp_input_cap: 2.0,
    positive_margin: 1.0e-3,
    nonzero_margin: 1.0e-3,
    output_abs_cap: 12.0,
    fd_step: 1.0e-6,
};

const STRESS_CONFIG: GeneratorConfig = GeneratorConfig {
    max_inputs: 6,
    max_outputs: 5,
    max_helpers: 5,
    max_helper_outputs: 4,
    max_nodes_per_expr: 20,
    max_call_depth: 3,
    exp_input_cap: 2.0,
    positive_margin: 1.0e-3,
    nonzero_margin: 1.0e-3,
    output_abs_cap: 16.0,
    fd_step: 1.0e-6,
};

const BALANCED_CI: PropertyScenario = PropertyScenario {
    name: "balanced_ci",
    accepted_cases: 8,
    max_global_rejects: 48,
    operator_tier: OperatorTier::Tier1,
    generator: CI_CONFIG,
    profile_mode: ProfileMode::Mixed,
    requirements: GeneratedCaseRequirements {
        min_root_outputs: 1,
        require_calls: false,
        require_multi_output_helper: false,
        require_repeated_helper_calls: false,
        require_nested_helper_calls: false,
    },
};

const CALL_HEAVY_CI: PropertyScenario = PropertyScenario {
    name: "call_heavy_ci",
    accepted_cases: 8,
    max_global_rejects: 48,
    operator_tier: OperatorTier::Tier1,
    generator: CI_CONFIG,
    profile_mode: ProfileMode::ForceCallHeavy,
    requirements: GeneratedCaseRequirements {
        min_root_outputs: 2,
        require_calls: true,
        require_multi_output_helper: true,
        require_repeated_helper_calls: true,
        require_nested_helper_calls: true,
    },
};

const DOMAIN_OPS_CI: PropertyScenario = PropertyScenario {
    name: "domain_ops_ci",
    accepted_cases: 4,
    max_global_rejects: 24,
    operator_tier: OperatorTier::Tier2Domain,
    generator: CI_CONFIG,
    profile_mode: ProfileMode::Mixed,
    requirements: GeneratedCaseRequirements {
        min_root_outputs: 1,
        require_calls: false,
        require_multi_output_helper: false,
        require_repeated_helper_calls: false,
        require_nested_helper_calls: false,
    },
};

const STRESS: PropertyScenario = PropertyScenario {
    name: "stress",
    accepted_cases: 64,
    max_global_rejects: 1280,
    operator_tier: OperatorTier::Tier2Domain,
    generator: STRESS_CONFIG,
    profile_mode: ProfileMode::Mixed,
    requirements: GeneratedCaseRequirements {
        min_root_outputs: 2,
        require_calls: false,
        require_multi_output_helper: false,
        require_repeated_helper_calls: false,
        require_nested_helper_calls: false,
    },
};

const CALL_HEAVY_STRESS: PropertyScenario = PropertyScenario {
    name: "call_heavy_stress",
    accepted_cases: 48,
    max_global_rejects: 960,
    operator_tier: OperatorTier::Tier2Domain,
    generator: STRESS_CONFIG,
    profile_mode: ProfileMode::ForceCallHeavy,
    requirements: GeneratedCaseRequirements {
        min_root_outputs: 3,
        require_calls: true,
        require_multi_output_helper: true,
        require_repeated_helper_calls: true,
        require_nested_helper_calls: true,
    },
};

const POLICIES: [CallPolicy; 4] = [
    CallPolicy::InlineAtCall,
    CallPolicy::InlineAtLowering,
    CallPolicy::InlineInLLVM,
    CallPolicy::NoInlineLLVM,
];

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

enum CaseEvaluation {
    Pass(PolicyArtifacts),
    Reject(&'static str),
    Fail(String),
}

struct PolicyArtifacts {
    jit_primal: Vec<f64>,
    symbolic_jacobian: DenseMatrix,
    jit_jacobian: DenseMatrix,
    fd_jacobian: DenseMatrix,
    compile_stats: String,
}

fn eval_compiled_output(function: &CompiledJitFunction, x: &[f64]) -> Vec<f64> {
    let mut context = function.create_context();
    context.input_mut(0).copy_from_slice(x);
    function.eval(&mut context);
    context.output(0).to_vec()
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn all_dense_finite(matrix: &DenseMatrix) -> bool {
    matrix.values.iter().all(|value| value.is_finite())
}

fn require_primal_match(
    case: &GeneratedCase,
    policy: CallPolicy,
    ast: &[f64],
    symbolic: &[f64],
    jit: &[f64],
) -> TestCaseResult {
    for ((index, ast_value), (symbolic_value, jit_value)) in
        ast.iter().enumerate().zip(symbolic.iter().zip(jit.iter()))
    {
        let symbolic_error = (ast_value - symbolic_value).abs();
        let jit_error = (ast_value - jit_value).abs();
        if symbolic_error > 1.0e-10 || jit_error > 1.0e-10 {
            return Err(TestCaseError::fail(format!(
                "primal mismatch\ncase:\n{case}\npolicy={policy:?}\noutput_index={index}\nast={ast_value:.6e}\nsymbolic={symbolic_value:.6e}\njit={jit_value:.6e}"
            )));
        }
    }
    Ok(())
}

fn require_jacobian_match(
    case: &GeneratedCase,
    policy: CallPolicy,
    symbolic: &DenseMatrix,
    jit: &DenseMatrix,
    fd: &DenseMatrix,
    compile_stats: &sx_core::CompileStats,
) -> TestCaseResult {
    let symbolic_vs_jit = compare_dense_matrices(symbolic, jit, 1.0e-9);
    if !symbolic_vs_jit.within_tolerances(1.0e-9, 1.0e-9) {
        return Err(TestCaseError::fail(format!(
            "symbolic analytic vs JIT analytic mismatch\ncase:\n{case}\npolicy={policy:?}\nsummary:\n{symbolic_vs_jit}\ncompile_stats={compile_stats:?}"
        )));
    }

    let jit_vs_fd = compare_dense_matrices(jit, fd, 1.0e-7);
    if !jit_vs_fd.within_tolerances(5.0e-5, 5.0e-4) {
        return Err(TestCaseError::fail(format!(
            "JIT analytic vs finite difference mismatch\ncase:\n{case}\npolicy={policy:?}\nsymbolic_vs_jit:\n{symbolic_vs_jit}\njit_vs_fd:\n{jit_vs_fd}\ncompile_stats={compile_stats:?}"
        )));
    }
    Ok(())
}

fn assert_call_policy_hooks(
    case: &GeneratedCase,
    policy: CallPolicy,
    primal: &CompiledJitFunction,
) -> TestCaseResult {
    let lowered = primal.lowered();
    if lowered.stats.call_site_count == 0 {
        return Ok(());
    }
    match policy {
        CallPolicy::InlineAtCall | CallPolicy::InlineAtLowering => {
            if !lowered.subfunctions.is_empty() {
                return Err(TestCaseError::fail(format!(
                    "expected inlined lowering to produce no subfunctions\npolicy={policy:?}\ncase:\n{case}\nstats={:?}",
                    lowered.stats
                )));
            }
        }
        CallPolicy::InlineInLLVM | CallPolicy::NoInlineLLVM => {
            if lowered.subfunctions.is_empty() || lowered.stats.llvm_call_instructions_emitted == 0
            {
                return Err(TestCaseError::fail(format!(
                    "expected preserved calls to survive lowering\npolicy={policy:?}\ncase:\n{case}\nstats={:?}",
                    lowered.stats
                )));
            }
        }
    }
    Ok(())
}

fn require_policy_consistency(
    seed_debug: &str,
    case: &GeneratedCase,
    baseline_policy: CallPolicy,
    baseline: &PolicyArtifacts,
    policy: CallPolicy,
    other: &PolicyArtifacts,
) -> TestCaseResult {
    for (index, (lhs, rhs)) in baseline
        .jit_primal
        .iter()
        .zip(other.jit_primal.iter())
        .enumerate()
    {
        if (lhs - rhs).abs() > 1.0e-10 {
            return Err(TestCaseError::fail(format!(
                "policy-differential primal mismatch\nseed={seed_debug}\ncase:\n{case}\nbaseline_policy={baseline_policy:?}\npolicy={policy:?}\noutput_index={index}\nbaseline={lhs:.6e}\nother={rhs:.6e}"
            )));
        }
    }

    let symbolic_summary = compare_dense_matrices(
        &baseline.symbolic_jacobian,
        &other.symbolic_jacobian,
        1.0e-9,
    );
    if !symbolic_summary.within_tolerances(1.0e-9, 1.0e-9) {
        return Err(TestCaseError::fail(format!(
            "policy-differential symbolic Jacobian mismatch\nseed={seed_debug}\ncase:\n{case}\nbaseline_policy={baseline_policy:?}\npolicy={policy:?}\nsummary:\n{symbolic_summary}\nbaseline_compile_stats={}\nother_compile_stats={}",
            baseline.compile_stats, other.compile_stats
        )));
    }

    let jit_summary = compare_dense_matrices(&baseline.jit_jacobian, &other.jit_jacobian, 1.0e-9);
    if !jit_summary.within_tolerances(1.0e-9, 1.0e-9) {
        return Err(TestCaseError::fail(format!(
            "policy-differential JIT Jacobian mismatch\nseed={seed_debug}\ncase:\n{case}\nbaseline_policy={baseline_policy:?}\npolicy={policy:?}\nsummary:\n{jit_summary}\nbaseline_compile_stats={}\nother_compile_stats={}",
            baseline.compile_stats, other.compile_stats
        )));
    }

    let fd_summary = compare_dense_matrices(&baseline.fd_jacobian, &other.fd_jacobian, 1.0e-9);
    if !fd_summary.within_tolerances(1.0e-9, 1.0e-9) {
        return Err(TestCaseError::fail(format!(
            "policy-differential finite-difference Jacobian mismatch\nseed={seed_debug}\ncase:\n{case}\nbaseline_policy={baseline_policy:?}\npolicy={policy:?}\nsummary:\n{fd_summary}"
        )));
    }
    Ok(())
}

fn run_scenario(scenario: &'static PropertyScenario) {
    let strategy = case_seed_strategy(scenario);
    let mut runner = TestRunner::new(Config {
        cases: scenario.accepted_cases,
        max_global_rejects: scenario.max_global_rejects,
        ..Config::default()
    });
    let counters = RefCell::new(CoverageCounters::default());
    let result = runner.run(&strategy, |seed| {
        let seed_debug = format!("{seed:#?}");
        let case = match generate_case_from_seed(scenario, &seed) {
            Ok(case) => case,
            Err(_) => {
                counters.borrow_mut().static_rejects += 1;
                return Err(TestCaseError::reject(
                    "static generation/certification reject",
                ));
            }
        };
        let mut artifacts = HashMap::new();
        for policy in POLICIES {
            match evaluate_case(&case, policy, scenario.generator.fd_step) {
                CaseEvaluation::Pass(artifact) => {
                    artifacts.insert(policy, artifact);
                }
                CaseEvaluation::Reject(reason) => {
                    if reason == "finite_difference_nonfinite" {
                        counters.borrow_mut().fd_nonfinite_rejects += 1;
                    } else {
                        counters.borrow_mut().runtime_nonfinite_rejects += 1;
                    }
                    return Err(TestCaseError::reject(reason));
                }
                CaseEvaluation::Fail(message) => {
                    let coverage = counters.borrow().snapshot();
                    return Err(TestCaseError::fail(format!(
                        "seed={seed_debug}\ncoverage_before_failure={coverage:?}\n{message}"
                    )));
                }
            }
        }
        let inline_baseline = artifacts
            .get(&CallPolicy::InlineAtCall)
            .expect("baseline policy artifacts should exist");
        for policy in POLICIES
            .into_iter()
            .filter(|policy| *policy != CallPolicy::InlineAtCall)
        {
            let artifact = artifacts
                .get(&policy)
                .expect("policy artifacts should exist for all policies");
            require_policy_consistency(
                &seed_debug,
                &case,
                CallPolicy::InlineAtCall,
                inline_baseline,
                policy,
                artifact,
            )?;
        }
        counters.borrow_mut().record_accept(&case);
        Ok(())
    });
    let counters = counters.into_inner();
    if let Err(err) = result {
        panic!(
            "scenario={} failed: {err}\ncoverage: accepted={} static_rejects={} runtime_nonfinite_rejects={} fd_nonfinite_rejects={} profiles={:?}",
            scenario.name,
            counters.accepted_valid_cases,
            counters.static_rejects,
            counters.runtime_nonfinite_rejects,
            counters.fd_nonfinite_rejects,
            counters.accepted_by_profile
        );
    }
    eprintln!(
        "scenario={} coverage accepted={} static_rejects={} runtime_nonfinite_rejects={} fd_nonfinite_rejects={} profiles={:?} call_cases={} repeated_helper_cases={} multi_output_helpers={} tier2_cases={}",
        scenario.name,
        counters.accepted_valid_cases,
        counters.static_rejects,
        counters.runtime_nonfinite_rejects,
        counters.fd_nonfinite_rejects,
        counters.accepted_by_profile,
        counters.accepted_call_cases,
        counters.accepted_repeated_helper_cases,
        counters.accepted_multi_output_helper_cases,
        counters.accepted_tier2_cases
    );
}

fn evaluate_case(case: &GeneratedCase, policy: CallPolicy, fd_step: f64) -> CaseEvaluation {
    let lowered = lower_case_to_sx_functions(case);
    let ast_primal = eval_ast_outputs(case, &case.sample_input);
    if !all_finite(&ast_primal) {
        return CaseEvaluation::Reject("ast_nonfinite");
    }

    let symbolic_primal = eval_lowered_function_outputs(&lowered.primal, &case.sample_input);
    if !all_finite(&symbolic_primal) {
        return CaseEvaluation::Reject("symbolic_primal_nonfinite");
    }

    let primal_compiled = compile_with_policy(&lowered.primal, policy);
    let jacobian_compiled = compile_with_policy(&lowered.jacobian, policy);
    let jit_primal = eval_compiled_output(&primal_compiled, &case.sample_input);
    if !all_finite(&jit_primal) {
        return CaseEvaluation::Reject("jit_primal_nonfinite");
    }
    if let Err(err) = require_primal_match(case, policy, &ast_primal, &symbolic_primal, &jit_primal)
    {
        return CaseEvaluation::Fail(err.to_string());
    }

    let symbolic_jac = eval_symbolic_function_nonzeros(&lowered.jacobian, &[&case.sample_input]);
    let symbolic_jac_values = symbolic_jac.into_iter().next().unwrap_or_default();
    if !all_finite(&symbolic_jac_values) {
        return CaseEvaluation::Reject("symbolic_jacobian_nonfinite");
    }

    let jit_jac_values = eval_compiled_output(&jacobian_compiled, &case.sample_input);
    if !all_finite(&jit_jac_values) {
        return CaseEvaluation::Reject("jit_jacobian_nonfinite");
    }
    let symbolic_dense = dense_from_sx_ccs_values(&lowered.jacobian_ccs, &symbolic_jac_values);
    let jit_dense = dense_from_sx_ccs_values(&lowered.jacobian_ccs, &jit_jac_values);
    let fd_dense = finite_difference_jacobian(&primal_compiled, &case.sample_input, fd_step);
    if !all_dense_finite(&fd_dense) {
        return CaseEvaluation::Reject("finite_difference_nonfinite");
    }
    if let Err(err) = require_jacobian_match(
        case,
        policy,
        &symbolic_dense,
        &jit_dense,
        &fd_dense,
        &primal_compiled.compile_report().stats,
    ) {
        return CaseEvaluation::Fail(err.to_string());
    }
    if let Err(err) = assert_call_policy_hooks(case, policy, &primal_compiled) {
        return CaseEvaluation::Fail(err.to_string());
    }
    CaseEvaluation::Pass(PolicyArtifacts {
        jit_primal,
        symbolic_jacobian: symbolic_dense,
        jit_jacobian: jit_dense,
        fd_jacobian: fd_dense,
        compile_stats: format!("{:?}", primal_compiled.compile_report().stats),
    })
}

#[test]
fn generated_jacobians_balanced_ci() {
    run_scenario(&BALANCED_CI);
}

#[test]
fn generated_jacobians_call_heavy_ci() {
    run_scenario(&CALL_HEAVY_CI);
}

#[test]
fn generated_jacobians_domain_ops_ci() {
    run_scenario(&DOMAIN_OPS_CI);
}

fn require_release_mode_for_manual_property_runs() {
    #[cfg(debug_assertions)]
    panic!(
        "manual Jacobian property stress runs must be executed in release mode\n\ntry:\n  cargo test -p sx_codegen_llvm --release --test generated_jacobian_props generated_jacobians_stress -- --ignored"
    );
}

#[test]
#[ignore = "manual property stress run"]
fn generated_jacobians_stress() {
    require_release_mode_for_manual_property_runs();
    run_scenario(&STRESS);
}

#[test]
#[ignore = "manual property stress run"]
fn generated_jacobians_call_heavy_stress() {
    require_release_mode_for_manual_property_runs();
    run_scenario(&CALL_HEAVY_STRESS);
}
