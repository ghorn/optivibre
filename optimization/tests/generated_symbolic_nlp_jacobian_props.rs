#![allow(clippy::large_enum_variant)]

use std::cell::RefCell;
use std::collections::HashMap;

use optimization::{
    CallPolicy, CallPolicyConfig, CompiledNlpProblem, FunctionCompileOptions,
    LlvmOptimizationLevel, SymbolicNlpOutputs, TypedRuntimeNlpBounds, symbolic_nlp,
};
use proptest::test_runner::{Config, TestCaseError, TestCaseResult, TestRunner};
use sx_core::SX;

#[path = "../../test_support/jacobian_proptest/mod.rs"]
mod jacobian_proptest;

use jacobian_proptest::{
    CoverageCounters, DenseMatrix, GeneratedCase, GeneratedCaseRequirements, GeneratorConfig,
    OperatorTier, ProfileMode, PropertyScenario, case_seed_strategy, compare_dense_matrices,
    eval_ast_outputs, generate_case_from_seed, instantiate_case,
};

const CI_CONFIG: GeneratorConfig = GeneratorConfig {
    max_inputs: 4,
    max_outputs: 5,
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
    max_inputs: 4,
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
    name: "nlp_balanced_ci",
    accepted_cases: 4,
    max_global_rejects: 24,
    operator_tier: OperatorTier::Tier1,
    generator: CI_CONFIG,
    profile_mode: ProfileMode::Mixed,
    requirements: GeneratedCaseRequirements {
        min_root_outputs: 2,
        require_calls: false,
        require_multi_output_helper: false,
        require_repeated_helper_calls: false,
        require_nested_helper_calls: false,
    },
};

const CALL_HEAVY_CI: PropertyScenario = PropertyScenario {
    name: "nlp_call_heavy_ci",
    accepted_cases: 4,
    max_global_rejects: 24,
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

const DOMAIN_OPS_STRESS: PropertyScenario = PropertyScenario {
    name: "nlp_domain_ops_stress",
    accepted_cases: 24,
    max_global_rejects: 480,
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
    name: "nlp_call_heavy_stress",
    accepted_cases: 24,
    max_global_rejects: 480,
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

const POLICIES: [CallPolicy; 2] = [CallPolicy::InlineAtLowering, CallPolicy::NoInlineLLVM];

enum CaseEvaluation {
    Pass(PolicyArtifacts),
    Reject(&'static str),
    Fail(String),
}

struct PolicyArtifacts {
    equality_values: Vec<f64>,
    inequality_values: Vec<f64>,
    equality_jacobian: DenseMatrix,
    inequality_jacobian: DenseMatrix,
    equality_fd: DenseMatrix,
    inequality_fd: DenseMatrix,
    compile_stats: String,
}

fn padded_input(case: &GeneratedCase) -> [f64; 4] {
    let mut values = [0.0; 4];
    for (slot, value) in case.sample_input.iter().copied().enumerate() {
        if slot < values.len() {
            values[slot] = value;
        }
    }
    values
}

fn dense_from_optimization_ccs(ccs: &optimization::CCS, values: &[f64]) -> DenseMatrix {
    assert_eq!(ccs.nnz(), values.len());
    let mut dense = DenseMatrix::zeros(ccs.nrow, ccs.ncol);
    for col in 0..ccs.ncol {
        for (index, value) in values
            .iter()
            .enumerate()
            .take(ccs.col_ptrs[col + 1])
            .skip(ccs.col_ptrs[col])
        {
            let row = ccs.row_indices[index];
            dense.set(row, col, *value);
        }
    }
    dense
}

fn analytic_constraint_jacobian(
    problem: &impl CompiledNlpProblem,
    x: &[f64],
    equality: bool,
) -> DenseMatrix {
    let parameters = Vec::<optimization::ParameterMatrix<'_>>::new();
    if equality {
        let ccs = problem.equality_jacobian_ccs();
        let mut values = vec![0.0; ccs.nnz()];
        problem.equality_jacobian_values(x, &parameters, &mut values);
        dense_from_optimization_ccs(ccs, &values)
    } else {
        let ccs = problem.inequality_jacobian_ccs();
        let mut values = vec![0.0; ccs.nnz()];
        problem.inequality_jacobian_values(x, &parameters, &mut values);
        dense_from_optimization_ccs(ccs, &values)
    }
}

fn finite_difference_constraint_jacobian(
    problem: &impl CompiledNlpProblem,
    x: &[f64],
    equality: bool,
    step: f64,
) -> DenseMatrix {
    let parameters = Vec::<optimization::ParameterMatrix<'_>>::new();
    let rows = if equality {
        problem.equality_count()
    } else {
        problem.inequality_count()
    };
    let cols = x.len();
    let mut dense = DenseMatrix::zeros(rows, cols);
    let mut plus = vec![0.0; rows];
    let mut minus = vec![0.0; rows];
    for col in 0..cols {
        let mut xp = x.to_vec();
        let mut xm = x.to_vec();
        xp[col] += step;
        xm[col] -= step;
        if equality {
            problem.equality_values(&xp, &parameters, &mut plus);
            problem.equality_values(&xm, &parameters, &mut minus);
        } else {
            problem.inequality_values(&xp, &parameters, &mut plus);
            problem.inequality_values(&xm, &parameters, &mut minus);
        }
        for row in 0..rows {
            dense.set(row, col, (plus[row] - minus[row]) / (2.0 * step));
        }
    }
    dense
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn all_dense_finite(matrix: &DenseMatrix) -> bool {
    matrix.values.iter().all(|value| value.is_finite())
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
        .equality_values
        .iter()
        .zip(other.equality_values.iter())
        .enumerate()
    {
        if (lhs - rhs).abs() > 1.0e-10 {
            return Err(TestCaseError::fail(format!(
                "policy-differential equality values mismatch\nseed={seed_debug}\ncase:\n{case}\nbaseline_policy={baseline_policy:?}\npolicy={policy:?}\nrow={index}\nbaseline={lhs:.6e}\nother={rhs:.6e}"
            )));
        }
    }
    for (index, (lhs, rhs)) in baseline
        .inequality_values
        .iter()
        .zip(other.inequality_values.iter())
        .enumerate()
    {
        if (lhs - rhs).abs() > 1.0e-10 {
            return Err(TestCaseError::fail(format!(
                "policy-differential inequality values mismatch\nseed={seed_debug}\ncase:\n{case}\nbaseline_policy={baseline_policy:?}\npolicy={policy:?}\nrow={index}\nbaseline={lhs:.6e}\nother={rhs:.6e}"
            )));
        }
    }

    let eq_summary = compare_dense_matrices(
        &baseline.equality_jacobian,
        &other.equality_jacobian,
        1.0e-9,
    );
    if !eq_summary.within_tolerances(1.0e-9, 1.0e-9) {
        return Err(TestCaseError::fail(format!(
            "policy-differential equality Jacobian mismatch\nseed={seed_debug}\ncase:\n{case}\nbaseline_policy={baseline_policy:?}\npolicy={policy:?}\nsummary:\n{eq_summary}\nbaseline_compile_stats={}\nother_compile_stats={}",
            baseline.compile_stats, other.compile_stats
        )));
    }

    let ineq_summary = compare_dense_matrices(
        &baseline.inequality_jacobian,
        &other.inequality_jacobian,
        1.0e-9,
    );
    if !ineq_summary.within_tolerances(1.0e-9, 1.0e-9) {
        return Err(TestCaseError::fail(format!(
            "policy-differential inequality Jacobian mismatch\nseed={seed_debug}\ncase:\n{case}\nbaseline_policy={baseline_policy:?}\npolicy={policy:?}\nsummary:\n{ineq_summary}\nbaseline_compile_stats={}\nother_compile_stats={}",
            baseline.compile_stats, other.compile_stats
        )));
    }

    let eq_fd_summary = compare_dense_matrices(&baseline.equality_fd, &other.equality_fd, 1.0e-9);
    let ineq_fd_summary =
        compare_dense_matrices(&baseline.inequality_fd, &other.inequality_fd, 1.0e-9);
    if !eq_fd_summary.within_tolerances(1.0e-9, 1.0e-9)
        || !ineq_fd_summary.within_tolerances(1.0e-9, 1.0e-9)
    {
        return Err(TestCaseError::fail(format!(
            "policy-differential finite-difference mismatch\nseed={seed_debug}\ncase:\n{case}\nbaseline_policy={baseline_policy:?}\npolicy={policy:?}\neq_summary:\n{eq_fd_summary}\nineq_summary:\n{ineq_fd_summary}"
        )));
    }

    Ok(())
}

fn build_symbolic_nlp_from_case(
    case: &GeneratedCase,
) -> optimization::TypedSymbolicNlp<[SX; 4], (), [SX; 3], [SX; 3]> {
    symbolic_nlp::<[SX; 4], (), [SX; 3], [SX; 3], _>("generated_symbolic_case", move |x, _| {
        let root_inputs = (0..case.root.input_count.min(4))
            .map(|index| x[index])
            .collect::<Vec<_>>();
        let (_helpers, outputs) = instantiate_case(case, &root_inputs);
        let mut equalities = [SX::from(0.0); 3];
        let mut inequalities = [SX::from(0.0); 3];
        for (index, output) in outputs.into_iter().enumerate() {
            if index % 2 == 0 {
                let slot = index / 2;
                if slot < equalities.len() {
                    equalities[slot] = output;
                }
            } else {
                let slot = index / 2;
                if slot < inequalities.len() {
                    inequalities[slot] = output;
                }
            }
        }
        SymbolicNlpOutputs {
            objective: SX::from(0.0),
            equalities,
            inequalities,
        }
    })
    .expect("symbolic nlp should build")
}

fn evaluate_case(case: &GeneratedCase, policy: CallPolicy, fd_step: f64) -> CaseEvaluation {
    let symbolic = build_symbolic_nlp_from_case(case);
    let compiled = symbolic
        .compile_jit_with_options(FunctionCompileOptions {
            opt_level: LlvmOptimizationLevel::O0,
            call_policy: CallPolicyConfig {
                default_policy: policy,
                respect_function_overrides: true,
            },
        })
        .expect("NLP compile should succeed");
    let x = padded_input(case);

    let expected_outputs = eval_ast_outputs(case, &case.sample_input);
    if !all_finite(&expected_outputs) {
        return CaseEvaluation::Reject("ast_nonfinite");
    }

    let equality_values = compiled.evaluate_equalities_flat(&x, &());
    let inequality_values = compiled.evaluate_inequalities_flat(&x, &());
    if !all_finite(&equality_values) || !all_finite(&inequality_values) {
        return CaseEvaluation::Reject("nlp_values_nonfinite");
    }

    let bound = compiled
        .bind_runtime_bounds(&TypedRuntimeNlpBounds::default())
        .expect("runtime bounds should bind");
    let eq_analytic = analytic_constraint_jacobian(&bound, &x, true);
    let ineq_analytic = analytic_constraint_jacobian(&bound, &x, false);
    if !all_dense_finite(&eq_analytic) || !all_dense_finite(&ineq_analytic) {
        return CaseEvaluation::Reject("analytic_jacobian_nonfinite");
    }
    let eq_fd = finite_difference_constraint_jacobian(&bound, &x, true, fd_step);
    let ineq_fd = finite_difference_constraint_jacobian(&bound, &x, false, fd_step);
    if !all_dense_finite(&eq_fd) || !all_dense_finite(&ineq_fd) {
        return CaseEvaluation::Reject("finite_difference_nonfinite");
    }

    let eq_summary = compare_dense_matrices(&eq_analytic, &eq_fd, 1.0e-7);
    if !eq_summary.within_tolerances(5.0e-5, 5.0e-4) {
        return CaseEvaluation::Fail(format!(
            "equality Jacobian mismatch\ncase:\n{case}\npolicy={policy:?}\ninput={x:?}\nsummary:\n{eq_summary}\ncompile_stats={:?}",
            compiled.backend_compile_report().stats
        ));
    }
    let ineq_summary = compare_dense_matrices(&ineq_analytic, &ineq_fd, 1.0e-7);
    if !ineq_summary.within_tolerances(5.0e-5, 5.0e-4) {
        return CaseEvaluation::Fail(format!(
            "inequality Jacobian mismatch\ncase:\n{case}\npolicy={policy:?}\ninput={x:?}\nsummary:\n{ineq_summary}\ncompile_stats={:?}",
            compiled.backend_compile_report().stats
        ));
    }

    CaseEvaluation::Pass(PolicyArtifacts {
        equality_values,
        inequality_values,
        equality_jacobian: eq_analytic,
        inequality_jacobian: ineq_analytic,
        equality_fd: eq_fd,
        inequality_fd: ineq_fd,
        compile_stats: format!("{:?}", compiled.backend_compile_report().stats),
    })
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
                    return Err(TestCaseError::fail(format!(
                        "seed={seed_debug}\ncoverage_before_failure={:?}\n{message}",
                        counters.borrow().snapshot()
                    )));
                }
            }
        }
        let baseline = artifacts
            .get(&CallPolicy::InlineAtLowering)
            .expect("baseline artifacts should exist");
        let preserved = artifacts
            .get(&CallPolicy::NoInlineLLVM)
            .expect("preserved-call artifacts should exist");
        require_policy_consistency(
            &seed_debug,
            &case,
            CallPolicy::InlineAtLowering,
            baseline,
            CallPolicy::NoInlineLLVM,
            preserved,
        )?;
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

#[test]
fn generated_symbolic_nlp_jacobians_ci() {
    run_scenario(&BALANCED_CI);
}

#[test]
fn generated_symbolic_nlp_call_heavy_ci() {
    run_scenario(&CALL_HEAVY_CI);
}

fn require_release_mode_for_manual_property_runs() {
    assert!(
        !matches!(option_env!("OPTIVIBRE_OPT_LEVEL"), Some("0")),
        "manual symbolic NLP Jacobian stress runs must be executed with an optimized binary; current opt-level=0\n\ntry:\n  cargo test -p optimization --release --test generated_symbolic_nlp_jacobian_props generated_symbolic_nlp_call_heavy_stress -- --ignored"
    );
}

#[test]
#[ignore = "manual property stress run"]
fn generated_symbolic_nlp_domain_ops_stress() {
    require_release_mode_for_manual_property_runs();
    run_scenario(&DOMAIN_OPS_STRESS);
}

#[test]
#[ignore = "manual property stress run"]
fn generated_symbolic_nlp_call_heavy_stress() {
    require_release_mode_for_manual_property_runs();
    run_scenario(&CALL_HEAVY_STRESS);
}
