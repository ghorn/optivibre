use std::cell::RefCell;

use optimization::{
    CallPolicy, CallPolicyConfig, CompiledNlpProblem, FunctionCompileOptions,
    LlvmOptimizationLevel, ParameterMatrix, Vectorize, flatten_value, symbolic_column,
    symbolic_value,
};
use proptest::test_runner::{Config, TestCaseError, TestCaseResult, TestRunner};
use sx_codegen_llvm::CompiledJitFunction;
use sx_core::{NamedMatrix, SX, SXFunction, SXMatrix};

use crate::{
    Bounds1D, CollocationFamily, DirectCollocation, DirectCollocationInitialGuess,
    DirectCollocationRuntimeValues, MultipleShooting, MultipleShootingInitialGuess,
    MultipleShootingRuntimeValues, Ocp, OcpCompileOptions, OcpSymbolicFunctionOptions,
};

use crate::jacobian_proptest::{
    CaseFeatures, CaseProfile, CoverageCounters, DenseMatrix, ExprAst, FunctionAst, GeneratedCase,
    GeneratedCaseRequirements, GeneratorConfig, InputBox, JacobianMismatchSummary, OperatorTier,
    ProfileMode, PropertyScenario, RangeCert, UnaryOpAst, case_seed_strategy,
    compare_dense_matrices, dense_from_sx_ccs_values, eval_symbolic_function_nonzeros,
    finite_difference_jacobian, generate_case_from_seed, instantiate_case,
};

const MS_INTERVALS: usize = 2;
const MS_RK4_SUBSTEPS: usize = 2;
const DC_INTERVALS: usize = 2;
const DC_COLLOCATION_ROOTS: usize = 2;

const OCP_CI_CONFIG: GeneratorConfig = GeneratorConfig {
    max_inputs: 4,
    max_outputs: 4,
    max_helpers: 4,
    max_helper_outputs: 3,
    max_nodes_per_expr: 12,
    max_call_depth: 2,
    exp_input_cap: 2.0,
    positive_margin: 1.0e-3,
    nonzero_margin: 1.0e-3,
    output_abs_cap: 12.0,
    fd_step: 1.0e-6,
};

const OCP_STRESS_CONFIG: GeneratorConfig = GeneratorConfig {
    max_inputs: 4,
    max_outputs: 4,
    max_helpers: 5,
    max_helper_outputs: 4,
    max_nodes_per_expr: 18,
    max_call_depth: 3,
    exp_input_cap: 2.0,
    positive_margin: 1.0e-3,
    nonzero_margin: 1.0e-3,
    output_abs_cap: 14.0,
    fd_step: 1.0e-6,
};

static INLINE_SMOKE: PropertyScenario = PropertyScenario {
    name: "ocp_inline_smoke",
    accepted_cases: 4,
    max_global_rejects: 24,
    operator_tier: OperatorTier::Tier1,
    generator: OCP_CI_CONFIG,
    profile_mode: ProfileMode::ForceCallHeavy,
    requirements: GeneratedCaseRequirements {
        min_root_outputs: 3,
        require_calls: true,
        require_multi_output_helper: true,
        require_repeated_helper_calls: true,
        require_nested_helper_calls: true,
    },
};

static POLICY_DIFFERENTIAL_STRESS: PropertyScenario = PropertyScenario {
    name: "ocp_policy_differential_stress",
    accepted_cases: 16,
    max_global_rejects: 320,
    operator_tier: OperatorTier::Tier1,
    generator: OCP_STRESS_CONFIG,
    profile_mode: ProfileMode::ForceCallHeavy,
    requirements: GeneratedCaseRequirements {
        min_root_outputs: 3,
        require_calls: true,
        require_multi_output_helper: true,
        require_repeated_helper_calls: true,
        require_nested_helper_calls: true,
    },
};

static DIRECT_COLLOCATION_INEQUALITY_STRESS: PropertyScenario = PropertyScenario {
    name: "dc_inequality_stress",
    accepted_cases: 24,
    max_global_rejects: 480,
    operator_tier: OperatorTier::Tier2Domain,
    generator: OCP_STRESS_CONFIG,
    profile_mode: ProfileMode::ForceCallHeavy,
    requirements: GeneratedCaseRequirements {
        min_root_outputs: 4,
        require_calls: true,
        require_multi_output_helper: true,
        require_repeated_helper_calls: true,
        require_nested_helper_calls: true,
    },
};

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct StageState<T> {
    x0: T,
    x1: T,
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct StageControl<T> {
    u0: T,
    u1: T,
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct PathIneq<T> {
    c0: T,
    c1: T,
    c2: T,
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct BoundaryIneq<T> {
    b0: T,
    b1: T,
}

enum CaseEvaluation {
    Pass(OcpPolicyArtifacts),
    Reject(&'static str),
    Fail(String),
}

struct OcpPolicyArtifacts {
    equality_values: Vec<f64>,
    inequality_values: Vec<f64>,
    equality_jacobian: DenseMatrix,
    inequality_jacobian: DenseMatrix,
    equality_fd: DenseMatrix,
    inequality_fd: DenseMatrix,
    compile_stats: String,
}

fn symmetric_bounds(limit: f64) -> Bounds1D {
    Bounds1D {
        lower: Some(-limit),
        upper: Some(limit),
    }
}

fn sample(case: &GeneratedCase, index: usize) -> f64 {
    case.sample_input[index % case.sample_input.len()]
}

fn constant_state(case: &GeneratedCase) -> StageState<f64> {
    StageState {
        x0: sample(case, 0),
        x1: sample(case, 1),
    }
}

fn constant_control(case: &GeneratedCase) -> StageControl<f64> {
    StageControl {
        u0: sample(case, 2),
        u1: sample(case, 3),
    }
}

fn stage_pool_sx(x: &StageState<SX>, u: &StageControl<SX>) -> [SX; 4] {
    [x.x0, x.x1, u.u0, u.u1]
}

fn boundary_pool_sx(initial: &StageState<SX>, terminal: &StageState<SX>) -> [SX; 4] {
    [initial.x0, initial.x1, terminal.x0, terminal.x1]
}

fn generated_outputs(case: &GeneratedCase, pool: &[SX; 4]) -> Vec<SX> {
    assert!(case.root.input_count <= pool.len());
    let (_helpers, outputs) = instantiate_case(case, &pool[..case.root.input_count]);
    outputs
}

fn pick_output(outputs: &[SX], index: usize) -> SX {
    outputs[index % outputs.len()]
}

fn deterministic_ms_bug_case() -> GeneratedCase {
    let helper0 = FunctionAst {
        input_count: 1,
        outputs: vec![ExprAst::Const(0.0), ExprAst::Const(6.5)],
    };
    let helper1 = FunctionAst {
        input_count: 1,
        outputs: vec![ExprAst::Call {
            helper: 0,
            output: 0,
            args: vec![ExprAst::Input(0)],
        }],
    };
    let helper2 = FunctionAst {
        input_count: 1,
        outputs: vec![ExprAst::Call {
            helper: 1,
            output: 0,
            args: vec![ExprAst::Input(0)],
        }],
    };
    let helper3 = FunctionAst {
        input_count: 1,
        outputs: vec![ExprAst::Call {
            helper: 2,
            output: 0,
            args: vec![ExprAst::Input(0)],
        }],
    };
    GeneratedCase {
        profile: CaseProfile::CallHeavy,
        operator_tier: OperatorTier::Tier1,
        input_box: InputBox::new(vec![RangeCert::any_finite(0.2, 1.2)]),
        sample_input: vec![0.494_210_024_353_398_95],
        helpers: vec![helper0, helper1, helper2, helper3],
        root: FunctionAst {
            input_count: 1,
            outputs: vec![
                ExprAst::Call {
                    helper: 3,
                    output: 0,
                    args: vec![ExprAst::Unary {
                        op: UnaryOpAst::Sin,
                        arg: Box::new(ExprAst::Input(0)),
                    }],
                },
                ExprAst::Const(0.943_818_209_374_633_7),
                ExprAst::Const(0.665_129_424_648_048_3),
            ],
        },
        certified_output_ranges: vec![
            RangeCert::exact(0.0),
            RangeCert::exact(0.943_818_209_374_633_7),
            RangeCert::exact(0.665_129_424_648_048_3),
        ],
        features: CaseFeatures {
            input_count: 1,
            root_output_count: 3,
            helper_count: 4,
            helper_multi_output_count: 1,
            call_count: 4,
            max_call_depth: 4,
            repeated_helper_calls: 0,
            nested_helper_calls: 4,
            tier2_op_count: 0,
        },
    }
}

fn ms_decision_layout_names() -> Vec<String> {
    let mut names = Vec::new();
    <crate::MsVars<StageState<SX>, StageControl<SX>, MS_INTERVALS> as Vectorize<SX>>::flat_layout_names(
        "w",
        &mut names,
    );
    names
}

fn ms_equality_layout_names() -> Vec<String> {
    let mut names = Vec::new();
    <crate::MsEqualities<StageState<SX>, StageControl<SX>, MS_INTERVALS> as Vectorize<SX>>::flat_layout_names(
        "eq",
        &mut names,
    );
    names
}

fn named_worst_entry(
    summary: &JacobianMismatchSummary,
    row_names: &[String],
    col_names: &[String],
) -> Option<String> {
    summary.worst_entry.as_ref().map(|entry| {
        let row_name = row_names
            .get(entry.row)
            .map(String::as_str)
            .unwrap_or("<row?>");
        let col_name = col_names
            .get(entry.col)
            .map(String::as_str)
            .unwrap_or("<col?>");
        format!(
            "worst_named_entry row={} ({}) col={} ({}) lhs={:.6e} rhs={:.6e} abs={:.3e} rel={:.3e}",
            entry.row,
            row_name,
            entry.col,
            col_name,
            entry.lhs,
            entry.rhs,
            entry.abs_error,
            entry.rel_error
        )
    })
}

fn generated_ms_ocp(
    case: &GeneratedCase,
) -> Ocp<
    StageState<SX>,
    StageControl<SX>,
    (),
    PathIneq<SX>,
    (),
    BoundaryIneq<SX>,
    MultipleShooting<MS_INTERVALS, MS_RK4_SUBSTEPS>,
> {
    let lag_case = case.clone();
    let mayer_case = case.clone();
    let ode_case = case.clone();
    let path_case = case.clone();
    let bineq_case = case.clone();
    Ocp::new(
        "generated_ocp_ms",
        MultipleShooting::<MS_INTERVALS, MS_RK4_SUBSTEPS>,
    )
    .objective_lagrange(
        move |x: &StageState<SX>, u: &StageControl<SX>, _: &StageControl<SX>, _: &()| {
            let outputs = generated_outputs(&lag_case, &stage_pool_sx(x, u));
            let y0 = pick_output(&outputs, 0);
            let y1 = pick_output(&outputs, 1);
            let y2 = pick_output(&outputs, 2);
            0.5 * y0.sqr() + 0.25 * y1.sqr() + 0.1 * y2.sqr()
        },
    )
    .objective_mayer(
        move |x0: &StageState<SX>,
              _: &StageControl<SX>,
              xt: &StageState<SX>,
              _: &StageControl<SX>,
              _: &(),
              _: &SX| {
            let outputs = generated_outputs(&mayer_case, &boundary_pool_sx(x0, xt));
            let y0 = pick_output(&outputs, 0);
            let y1 = pick_output(&outputs, 1);
            0.5 * y0.sqr() + 0.25 * y1.sqr()
        },
    )
    .ode(move |x: &StageState<SX>, u: &StageControl<SX>, _: &()| {
        let outputs = generated_outputs(&ode_case, &stage_pool_sx(x, u));
        StageState {
            x0: pick_output(&outputs, 0),
            x1: pick_output(&outputs, 1),
        }
    })
    .path_constraints(
        move |x: &StageState<SX>, u: &StageControl<SX>, _: &StageControl<SX>, _: &()| {
            let outputs = generated_outputs(&path_case, &stage_pool_sx(x, u));
            let y0 = pick_output(&outputs, 0);
            let y1 = pick_output(&outputs, 1);
            let y2 = pick_output(&outputs, 2);
            let y3 = pick_output(&outputs, 3);
            PathIneq {
                c0: y0 + 0.125 * y3,
                c1: y1 - 0.2 * y0 + 0.1 * y2,
                c2: y0 + 0.25 * y2 + 0.15 * y3,
            }
        },
    )
    .boundary_equalities(
        |_: &StageState<SX>,
         _: &StageControl<SX>,
         _: &StageState<SX>,
         _: &StageControl<SX>,
         _: &(),
         _: &SX| (),
    )
    .boundary_inequalities(
        move |x0: &StageState<SX>,
              _: &StageControl<SX>,
              xt: &StageState<SX>,
              _: &StageControl<SX>,
              _: &(),
              _: &SX| {
            let outputs = generated_outputs(&bineq_case, &boundary_pool_sx(x0, xt));
            let y0 = pick_output(&outputs, 0);
            let y1 = pick_output(&outputs, 1);
            let y2 = pick_output(&outputs, 2);
            let y3 = pick_output(&outputs, 3);
            BoundaryIneq {
                b0: y0 + 0.2 * y2,
                b1: y1 - 0.1 * y0 + 0.1 * y3,
            }
        },
    )
    .build()
    .expect("generated multiple-shooting OCP should build")
}

fn generated_dc_ocp(
    case: &GeneratedCase,
) -> Ocp<
    StageState<SX>,
    StageControl<SX>,
    (),
    PathIneq<SX>,
    (),
    BoundaryIneq<SX>,
    DirectCollocation<DC_INTERVALS, DC_COLLOCATION_ROOTS>,
> {
    let lag_case = case.clone();
    let mayer_case = case.clone();
    let ode_case = case.clone();
    let path_case = case.clone();
    let bineq_case = case.clone();
    Ocp::new(
        "generated_ocp_dc",
        DirectCollocation::<DC_INTERVALS, DC_COLLOCATION_ROOTS> {
            family: CollocationFamily::RadauIIA,
        },
    )
    .objective_lagrange(
        move |x: &StageState<SX>, u: &StageControl<SX>, _: &StageControl<SX>, _: &()| {
            let outputs = generated_outputs(&lag_case, &stage_pool_sx(x, u));
            let y0 = pick_output(&outputs, 0);
            let y1 = pick_output(&outputs, 1);
            let y2 = pick_output(&outputs, 2);
            0.5 * y0.sqr() + 0.25 * y1.sqr() + 0.1 * y2.sqr()
        },
    )
    .objective_mayer(
        move |x0: &StageState<SX>,
              _: &StageControl<SX>,
              xt: &StageState<SX>,
              _: &StageControl<SX>,
              _: &(),
              _: &SX| {
            let outputs = generated_outputs(&mayer_case, &boundary_pool_sx(x0, xt));
            let y0 = pick_output(&outputs, 0);
            let y1 = pick_output(&outputs, 1);
            0.5 * y0.sqr() + 0.25 * y1.sqr()
        },
    )
    .ode(move |x: &StageState<SX>, u: &StageControl<SX>, _: &()| {
        let outputs = generated_outputs(&ode_case, &stage_pool_sx(x, u));
        StageState {
            x0: pick_output(&outputs, 0),
            x1: pick_output(&outputs, 1),
        }
    })
    .path_constraints(
        move |x: &StageState<SX>, u: &StageControl<SX>, _: &StageControl<SX>, _: &()| {
            let outputs = generated_outputs(&path_case, &stage_pool_sx(x, u));
            let y0 = pick_output(&outputs, 0);
            let y1 = pick_output(&outputs, 1);
            let y2 = pick_output(&outputs, 2);
            PathIneq {
                c0: y0,
                c1: y1,
                c2: y0 + 0.25 * y2,
            }
        },
    )
    .boundary_equalities(
        |_: &StageState<SX>,
         _: &StageControl<SX>,
         _: &StageState<SX>,
         _: &StageControl<SX>,
         _: &(),
         _: &SX| (),
    )
    .boundary_inequalities(
        move |x0: &StageState<SX>,
              _: &StageControl<SX>,
              xt: &StageState<SX>,
              _: &StageControl<SX>,
              _: &(),
              _: &SX| {
            let outputs = generated_outputs(&bineq_case, &boundary_pool_sx(x0, xt));
            BoundaryIneq {
                b0: pick_output(&outputs, 0),
                b1: pick_output(&outputs, 1),
            }
        },
    )
    .build()
    .expect("generated direct-collocation OCP should build")
}

fn runtime_limit(case: &GeneratedCase) -> f64 {
    case.certified_output_ranges
        .iter()
        .fold(1.0_f64, |acc, range| {
            acc.max(range.lower.abs()).max(range.upper.abs())
        })
        + 1.0
}

fn ms_runtime(
    case: &GeneratedCase,
) -> MultipleShootingRuntimeValues<
    (),
    PathIneq<Bounds1D>,
    (),
    BoundaryIneq<Bounds1D>,
    StageState<f64>,
    StageControl<f64>,
    MS_INTERVALS,
> {
    let x = constant_state(case);
    let u = constant_control(case);
    let dudt = StageControl {
        u0: sample(case, 0),
        u1: sample(case, 1),
    };
    let limit = runtime_limit(case);
    MultipleShootingRuntimeValues {
        parameters: (),
        beq: (),
        bineq_bounds: BoundaryIneq {
            b0: symmetric_bounds(limit),
            b1: symmetric_bounds(limit),
        },
        path_bounds: PathIneq {
            c0: symmetric_bounds(limit),
            c1: symmetric_bounds(limit),
            c2: symmetric_bounds(limit),
        },
        tf_bounds: Bounds1D {
            lower: Some(0.5),
            upper: Some(1.5),
        },
        initial_guess: MultipleShootingInitialGuess::Constant {
            x,
            u,
            dudt,
            tf: 1.0,
        },
    }
}

fn dc_runtime(
    case: &GeneratedCase,
) -> DirectCollocationRuntimeValues<
    (),
    PathIneq<Bounds1D>,
    (),
    BoundaryIneq<Bounds1D>,
    StageState<f64>,
    StageControl<f64>,
    DC_INTERVALS,
    DC_COLLOCATION_ROOTS,
> {
    let x = constant_state(case);
    let u = constant_control(case);
    let dudt = StageControl {
        u0: sample(case, 0),
        u1: sample(case, 1),
    };
    let limit = runtime_limit(case);
    DirectCollocationRuntimeValues {
        parameters: (),
        beq: (),
        bineq_bounds: BoundaryIneq {
            b0: symmetric_bounds(limit),
            b1: symmetric_bounds(limit),
        },
        path_bounds: PathIneq {
            c0: symmetric_bounds(limit),
            c1: symmetric_bounds(limit),
            c2: symmetric_bounds(limit),
        },
        tf_bounds: Bounds1D {
            lower: Some(0.5),
            upper: Some(1.5),
        },
        initial_guess: DirectCollocationInitialGuess::Constant {
            x,
            u,
            dudt,
            tf: 1.0,
        },
    }
}

fn dense_from_optimization_ccs(ccs: &optimization::CCS, values: &[f64]) -> DenseMatrix {
    assert_eq!(ccs.nnz(), values.len());
    let mut dense = DenseMatrix::zeros(ccs.nrow, ccs.ncol);
    for col in 0..ccs.ncol {
        for index in ccs.col_ptrs[col]..ccs.col_ptrs[col + 1] {
            let row = ccs.row_indices[index];
            dense.set(row, col, values[index]);
        }
    }
    dense
}

fn analytic_constraint_jacobian(
    problem: &impl CompiledNlpProblem,
    x: &[f64],
    equality: bool,
) -> DenseMatrix {
    let parameters = Vec::<ParameterMatrix<'_>>::new();
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
    let parameters = Vec::<ParameterMatrix<'_>>::new();
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

fn format_ms_named_summary(summary: &JacobianMismatchSummary) -> String {
    let row_names = ms_equality_layout_names();
    let col_names = ms_decision_layout_names();
    named_worst_entry(summary, &row_names, &col_names)
        .unwrap_or_else(|| "worst_named_entry=<none>".to_string())
}

fn compile_function_with_policy(function: &SXFunction, policy: CallPolicy) -> CompiledJitFunction {
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
    .expect("function compilation should succeed")
}

fn inline_options() -> OcpCompileOptions {
    OcpCompileOptions {
        function_options: FunctionCompileOptions::from(LlvmOptimizationLevel::O0),
        symbolic_functions: OcpSymbolicFunctionOptions::inline_all(),
        hessian_strategy: sx_core::HessianStrategy::LowerTriangleByColumn,
    }
}

fn preserved_options() -> OcpCompileOptions {
    OcpCompileOptions {
        function_options: FunctionCompileOptions::from(LlvmOptimizationLevel::O0),
        symbolic_functions: OcpSymbolicFunctionOptions::function_all_with_call_policy(
            CallPolicy::NoInlineLLVM,
        ),
        hessian_strategy: sx_core::HessianStrategy::LowerTriangleByColumn,
    }
}

fn constant_column<T>(value: &T) -> SXMatrix
where
    T: Vectorize<f64>,
{
    let flat = flatten_value(value);
    SXMatrix::dense_column(flat.into_iter().map(SX::from).collect()).expect("constant column")
}

fn require_policy_consistency(
    seed_debug: &str,
    transcription: &str,
    case: &GeneratedCase,
    baseline: &OcpPolicyArtifacts,
    preserved: &OcpPolicyArtifacts,
) -> TestCaseResult {
    for (index, (lhs, rhs)) in baseline
        .equality_values
        .iter()
        .zip(preserved.equality_values.iter())
        .enumerate()
    {
        if (lhs - rhs).abs() > 1.0e-10 {
            return Err(TestCaseError::fail(format!(
                "{transcription}: policy-differential equality value mismatch\nseed={seed_debug}\ncase:\n{case}\nrow={index}\ninline_all={lhs:.6e}\npreserved={rhs:.6e}"
            )));
        }
    }
    for (index, (lhs, rhs)) in baseline
        .inequality_values
        .iter()
        .zip(preserved.inequality_values.iter())
        .enumerate()
    {
        if (lhs - rhs).abs() > 1.0e-10 {
            return Err(TestCaseError::fail(format!(
                "{transcription}: policy-differential inequality value mismatch\nseed={seed_debug}\ncase:\n{case}\nrow={index}\ninline_all={lhs:.6e}\npreserved={rhs:.6e}"
            )));
        }
    }

    let eq_summary = compare_dense_matrices(
        &baseline.equality_jacobian,
        &preserved.equality_jacobian,
        1.0e-9,
    );
    if !eq_summary.within_tolerances(1.0e-9, 1.0e-9) {
        return Err(TestCaseError::fail(format!(
            "{transcription}: policy-differential equality Jacobian mismatch\nseed={seed_debug}\ncase:\n{case}\nsummary:\n{eq_summary}\ninline_all_stats={}\npreserved_stats={}",
            baseline.compile_stats, preserved.compile_stats
        )));
    }

    let ineq_summary = compare_dense_matrices(
        &baseline.inequality_jacobian,
        &preserved.inequality_jacobian,
        1.0e-9,
    );
    if !ineq_summary.within_tolerances(1.0e-9, 1.0e-9) {
        return Err(TestCaseError::fail(format!(
            "{transcription}: policy-differential inequality Jacobian mismatch\nseed={seed_debug}\ncase:\n{case}\nsummary:\n{ineq_summary}\ninline_all_stats={}\npreserved_stats={}",
            baseline.compile_stats, preserved.compile_stats
        )));
    }

    let eq_fd_summary =
        compare_dense_matrices(&baseline.equality_fd, &preserved.equality_fd, 1.0e-9);
    let ineq_fd_summary =
        compare_dense_matrices(&baseline.inequality_fd, &preserved.inequality_fd, 1.0e-9);
    if !eq_fd_summary.within_tolerances(1.0e-9, 1.0e-9)
        || !ineq_fd_summary.within_tolerances(1.0e-9, 1.0e-9)
    {
        return Err(TestCaseError::fail(format!(
            "{transcription}: policy-differential finite-difference mismatch\nseed={seed_debug}\ncase:\n{case}\neq_summary:\n{eq_fd_summary}\nineq_summary:\n{ineq_fd_summary}"
        )));
    }

    Ok(())
}

fn evaluate_multiple_shooting(
    case: &GeneratedCase,
    options: OcpCompileOptions,
    policy_label: &str,
    expect_preserved_calls: bool,
    fd_step: f64,
) -> CaseEvaluation {
    let compiled = generated_ms_ocp(case)
        .compile_jit_with_ocp_options(options)
        .expect("multiple-shooting compile should succeed");
    let stats = &compiled.backend_compile_report().stats;
    if expect_preserved_calls {
        if stats.call_site_count == 0 {
            return CaseEvaluation::Reject("ms_preserved_call_sites_optimized_away");
        }
    } else if stats.call_site_count != 0 {
        return CaseEvaluation::Fail(format!(
            "multiple shooting inline_all should remove call sites\ncase:\n{case}\npolicy={policy_label}\nstats={stats:?}"
        ));
    }

    let values = ms_runtime(case);
    let x0 = compiled
        .build_initial_guess(&values)
        .expect("multiple-shooting initial guess should build");
    let bounds = compiled
        .build_runtime_bounds(&values)
        .expect("multiple-shooting bounds should build");
    let runtime_params = ((), ());

    let equality_values = compiled
        .compiled
        .evaluate_equalities_flat(&x0, &runtime_params);
    let inequality_values = compiled
        .compiled
        .evaluate_inequalities_flat(&x0, &runtime_params);
    if !all_finite(&equality_values) || !all_finite(&inequality_values) {
        return CaseEvaluation::Reject("ms_values_nonfinite");
    }

    let x = flatten_value(&x0);
    let bound_problem = compiled
        .compiled
        .bind_runtime_bounds(&bounds)
        .expect("multiple-shooting runtime bounds should bind");
    let equality_jacobian = analytic_constraint_jacobian(&bound_problem, &x, true);
    let inequality_jacobian = analytic_constraint_jacobian(&bound_problem, &x, false);
    if !all_dense_finite(&equality_jacobian) || !all_dense_finite(&inequality_jacobian) {
        return CaseEvaluation::Reject("ms_analytic_jacobian_nonfinite");
    }
    let equality_fd = finite_difference_constraint_jacobian(&bound_problem, &x, true, fd_step);
    let inequality_fd = finite_difference_constraint_jacobian(&bound_problem, &x, false, fd_step);
    if !all_dense_finite(&equality_fd) || !all_dense_finite(&inequality_fd) {
        return CaseEvaluation::Reject("finite_difference_nonfinite");
    }

    let eq_summary = compare_dense_matrices(&equality_jacobian, &equality_fd, 1.0e-7);
    if !eq_summary.within_tolerances(5.0e-5, 5.0e-4) {
        return CaseEvaluation::Fail(format!(
            "multiple shooting equality Jacobian mismatch\ncase:\n{case}\npolicy={policy_label}\nsummary:\n{eq_summary}\n{}\nstats={stats:?}",
            format_ms_named_summary(&eq_summary)
        ));
    }
    let ineq_summary = compare_dense_matrices(&inequality_jacobian, &inequality_fd, 1.0e-7);
    if !ineq_summary.within_tolerances(5.0e-5, 5.0e-4) {
        return CaseEvaluation::Fail(format!(
            "multiple shooting inequality Jacobian mismatch\ncase:\n{case}\npolicy={policy_label}\nsummary:\n{ineq_summary}\nstats={stats:?}"
        ));
    }

    CaseEvaluation::Pass(OcpPolicyArtifacts {
        equality_values,
        inequality_values,
        equality_jacobian,
        inequality_jacobian,
        equality_fd,
        inequality_fd,
        compile_stats: format!("{stats:?}"),
    })
}

fn evaluate_direct_collocation(
    case: &GeneratedCase,
    options: OcpCompileOptions,
    policy_label: &str,
    expect_preserved_calls: bool,
    fd_step: f64,
) -> CaseEvaluation {
    let compiled = generated_dc_ocp(case)
        .compile_jit_with_ocp_options(options)
        .expect("direct-collocation compile should succeed");
    let stats = &compiled.backend_compile_report().stats;
    if expect_preserved_calls {
        if stats.call_site_count == 0 {
            return CaseEvaluation::Reject("dc_preserved_call_sites_optimized_away");
        }
    } else if stats.call_site_count != 0 {
        return CaseEvaluation::Fail(format!(
            "direct collocation inline_all should remove call sites\ncase:\n{case}\npolicy={policy_label}\nstats={stats:?}"
        ));
    }

    let values = dc_runtime(case);
    let x0 = compiled
        .build_initial_guess(&values)
        .expect("direct-collocation initial guess should build");
    let bounds = compiled
        .build_runtime_bounds(&values)
        .expect("direct-collocation bounds should build");
    let runtime_params = ((), ());

    let equality_values = compiled
        .compiled
        .evaluate_equalities_flat(&x0, &runtime_params);
    let inequality_values = compiled
        .compiled
        .evaluate_inequalities_flat(&x0, &runtime_params);
    if !all_finite(&equality_values) || !all_finite(&inequality_values) {
        return CaseEvaluation::Reject("dc_values_nonfinite");
    }

    let x = flatten_value(&x0);
    let bound_problem = compiled
        .compiled
        .bind_runtime_bounds(&bounds)
        .expect("direct-collocation runtime bounds should bind");
    let equality_jacobian = analytic_constraint_jacobian(&bound_problem, &x, true);
    let inequality_jacobian = analytic_constraint_jacobian(&bound_problem, &x, false);
    if !all_dense_finite(&equality_jacobian) || !all_dense_finite(&inequality_jacobian) {
        return CaseEvaluation::Reject("dc_analytic_jacobian_nonfinite");
    }
    let equality_fd = finite_difference_constraint_jacobian(&bound_problem, &x, true, fd_step);
    let inequality_fd = finite_difference_constraint_jacobian(&bound_problem, &x, false, fd_step);
    if !all_dense_finite(&equality_fd) || !all_dense_finite(&inequality_fd) {
        return CaseEvaluation::Reject("finite_difference_nonfinite");
    }

    let eq_summary = compare_dense_matrices(&equality_jacobian, &equality_fd, 1.0e-7);
    if !eq_summary.within_tolerances(5.0e-5, 5.0e-4) {
        return CaseEvaluation::Fail(format!(
            "direct collocation equality Jacobian mismatch\ncase:\n{case}\npolicy={policy_label}\nsummary:\n{eq_summary}\nstats={stats:?}"
        ));
    }
    let ineq_summary = compare_dense_matrices(&inequality_jacobian, &inequality_fd, 1.0e-7);
    if !ineq_summary.within_tolerances(5.0e-5, 5.0e-4) {
        return CaseEvaluation::Fail(format!(
            "direct collocation inequality Jacobian mismatch\ncase:\n{case}\npolicy={policy_label}\nsummary:\n{ineq_summary}\nstats={stats:?}"
        ));
    }

    CaseEvaluation::Pass(OcpPolicyArtifacts {
        equality_values,
        inequality_values,
        equality_jacobian,
        inequality_jacobian,
        equality_fd,
        inequality_fd,
        compile_stats: format!("{stats:?}"),
    })
}

fn run_inline_smoke(scenario: &'static PropertyScenario) {
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
                return Err(TestCaseError::reject("static generation/certification reject"));
            }
        };

        for (transcription, evaluation) in [
            (
                "multiple_shooting",
                evaluate_multiple_shooting(
                    &case,
                    inline_options(),
                    "inline_all",
                    false,
                    scenario.generator.fd_step,
                ),
            ),
            (
                "direct_collocation",
                evaluate_direct_collocation(
                    &case,
                    inline_options(),
                    "inline_all",
                    false,
                    scenario.generator.fd_step,
                ),
            ),
        ] {
            match evaluation {
                CaseEvaluation::Pass(_) => {}
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
                        "{transcription}\nseed={seed_debug}\ncoverage_before_failure={:?}\n{message}",
                        counters.borrow().snapshot()
                    )));
                }
            }
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
}

fn run_policy_differential(scenario: &'static PropertyScenario) {
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
                return Err(TestCaseError::reject("static generation/certification reject"));
            }
        };

        let ms_inline = match evaluate_multiple_shooting(
            &case,
            inline_options(),
            "inline_all",
            false,
            scenario.generator.fd_step,
        ) {
            CaseEvaluation::Pass(artifact) => artifact,
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
                    "multiple_shooting\nseed={seed_debug}\ncoverage_before_failure={:?}\n{message}",
                    counters.borrow().snapshot()
                )));
            }
        };
        let ms_preserved = match evaluate_multiple_shooting(
            &case,
            preserved_options(),
            "noinline_llvm",
            true,
            scenario.generator.fd_step,
        ) {
            CaseEvaluation::Pass(artifact) => artifact,
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
                    "multiple_shooting\nseed={seed_debug}\ncoverage_before_failure={:?}\n{message}",
                    counters.borrow().snapshot()
                )));
            }
        };
        require_policy_consistency(
            &seed_debug,
            "multiple_shooting",
            &case,
            &ms_inline,
            &ms_preserved,
        )?;

        let dc_inline = match evaluate_direct_collocation(
            &case,
            inline_options(),
            "inline_all",
            false,
            scenario.generator.fd_step,
        ) {
            CaseEvaluation::Pass(artifact) => artifact,
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
                    "direct_collocation\nseed={seed_debug}\ncoverage_before_failure={:?}\n{message}",
                    counters.borrow().snapshot()
                )));
            }
        };
        let dc_preserved = match evaluate_direct_collocation(
            &case,
            preserved_options(),
            "noinline_llvm",
            true,
            scenario.generator.fd_step,
        ) {
            CaseEvaluation::Pass(artifact) => artifact,
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
                    "direct_collocation\nseed={seed_debug}\ncoverage_before_failure={:?}\n{message}",
                    counters.borrow().snapshot()
                )));
            }
        };
        require_policy_consistency(
            &seed_debug,
            "direct_collocation",
            &case,
            &dc_inline,
            &dc_preserved,
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
}

fn require_release_mode_for_manual_property_runs() {
    assert!(
        !cfg!(debug_assertions),
        "manual OCP Jacobian property stress runs must be executed in release mode\n\ntry:\n  cargo test -p optimal_control --release generated_direct_collocation_inequality_stress -- --ignored --nocapture"
    );
}

fn run_direct_collocation_inequality_stress(scenario: &'static PropertyScenario) {
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
                return Err(TestCaseError::reject("static generation/certification reject"));
            }
        };

        let inline = match evaluate_direct_collocation(
            &case,
            inline_options(),
            "inline_all",
            false,
            scenario.generator.fd_step,
        ) {
            CaseEvaluation::Pass(artifact) => artifact,
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
                    "direct_collocation_inline_all\nseed={seed_debug}\ncoverage_before_failure={:?}\n{message}",
                    counters.borrow().snapshot()
                )));
            }
        };

        let preserved = match evaluate_direct_collocation(
            &case,
            preserved_options(),
            "noinline_llvm",
            true,
            scenario.generator.fd_step,
        ) {
            CaseEvaluation::Pass(artifact) => artifact,
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
                    "direct_collocation_preserved\nseed={seed_debug}\ncoverage_before_failure={:?}\n{message}",
                    counters.borrow().snapshot()
                )));
            }
        };

        require_policy_consistency(&seed_debug, "direct_collocation", &case, &inline, &preserved)?;
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
}

#[test]
fn generated_ocp_assembly_inline_all_smoke() {
    run_inline_smoke(&INLINE_SMOKE);
}

#[test]
fn generated_multiple_shooting_preserved_call_equality_regression_is_fixed() {
    let case = deterministic_ms_bug_case();
    match evaluate_multiple_shooting(
        &case,
        inline_options(),
        "inline_all",
        false,
        OCP_CI_CONFIG.fd_step,
    ) {
        CaseEvaluation::Pass(_) => {}
        CaseEvaluation::Reject(reason) => {
            panic!("inline_all unexpectedly rejected deterministic reproducer: {reason}");
        }
        CaseEvaluation::Fail(message) => {
            panic!("inline_all unexpectedly failed deterministic reproducer:\n{message}");
        }
    }

    match evaluate_multiple_shooting(
        &case,
        preserved_options(),
        "noinline_llvm",
        true,
        OCP_CI_CONFIG.fd_step,
    ) {
        CaseEvaluation::Pass(_) => {}
        CaseEvaluation::Fail(message) => {
            panic!("preserved-call deterministic regression still fails:\n{message}");
        }
        CaseEvaluation::Reject(reason) => {
            panic!("preserved-call deterministic regression unexpectedly rejected: {reason}");
        }
    }
}

#[test]
fn preserved_call_multiple_shooting_integrator_wrapper_identity_jacobian_is_fixed() {
    let case = deterministic_ms_bug_case();
    let ocp = generated_ms_ocp(&case);
    let library = ocp
        .build_multiple_shooting_symbolic_function_library(preserved_options().symbolic_functions)
        .expect("multiple-shooting symbolic library should build");
    let integrator = library
        .multiple_shooting_integrator
        .expect("multiple-shooting integrator helper should be present");

    let x = symbolic_value::<StageState<SX>>("x").expect("symbolic state should build");
    let x_column = symbolic_column(&x).expect("symbolic state column should build");
    let u_const = constant_control(&case);
    let dudt_const = StageControl {
        u0: sample(&case, 0),
        u1: sample(&case, 1),
    };
    let outputs = integrator
        .call(&[
            x_column.clone(),
            constant_column(&u_const),
            constant_column(&dudt_const),
            constant_column(&()),
            SXMatrix::dense_column(vec![SX::from(0.5)]).expect("dt column should build"),
        ])
        .expect("integrator wrapper call should build");
    let primal = SXFunction::new(
        "ms_integrator_wrapper_primal",
        vec![NamedMatrix::new("x", x_column.clone()).expect("wrapper input")],
        vec![NamedMatrix::new("x_next", outputs[0].clone()).expect("wrapper output")],
    )
    .expect("integrator wrapper primal should build");
    let jacobian_matrix = outputs[0]
        .jacobian(&x_column)
        .expect("integrator wrapper Jacobian should build");
    let jacobian_ccs = jacobian_matrix.ccs().clone();
    let jacobian = SXFunction::new(
        "ms_integrator_wrapper_jacobian",
        vec![NamedMatrix::new("x", x_column).expect("wrapper input")],
        vec![NamedMatrix::new("jac", jacobian_matrix).expect("wrapper jacobian output")],
    )
    .expect("integrator wrapper Jacobian function should build");

    let x_input = flatten_value(&constant_state(&case));
    let primal_compiled = compile_function_with_policy(&primal, CallPolicy::NoInlineLLVM);
    let jacobian_compiled = compile_function_with_policy(&jacobian, CallPolicy::NoInlineLLVM);
    let symbolic_jac = eval_symbolic_function_nonzeros(&jacobian, &[&x_input]);
    let symbolic_jac_values = symbolic_jac.into_iter().next().unwrap_or_default();
    let jit_jac_values = {
        let mut context = jacobian_compiled.create_context();
        context.input_mut(0).copy_from_slice(&x_input);
        jacobian_compiled.eval(&mut context);
        context.output(0).to_vec()
    };
    let symbolic_dense = dense_from_sx_ccs_values(&jacobian_ccs, &symbolic_jac_values);
    let jit_dense = dense_from_sx_ccs_values(&jacobian_ccs, &jit_jac_values);
    let fd_dense = finite_difference_jacobian(&primal_compiled, &x_input, OCP_CI_CONFIG.fd_step);

    let symbolic_vs_jit = compare_dense_matrices(&symbolic_dense, &jit_dense, 1.0e-9);
    assert!(
        symbolic_vs_jit.within_tolerances(1.0e-9, 1.0e-9),
        "integrator wrapper symbolic vs JIT Jacobian mismatch\ncase:\n{case}\nsummary:\n{symbolic_vs_jit}"
    );
    let jit_vs_fd = compare_dense_matrices(&jit_dense, &fd_dense, 1.0e-7);
    assert!(
        jit_vs_fd.within_tolerances(5.0e-5, 5.0e-4),
        "preserved-call integrator wrapper regression still fails\ncase:\n{case}\nsummary:\n{jit_vs_fd}"
    );
}

#[test]
#[ignore = "diagnose preserved-call multiple-shooting integrator forward helper"]
fn diagnose_preserved_call_multiple_shooting_integrator_forward_helper_identity_term() {
    let case = deterministic_ms_bug_case();
    let ocp = generated_ms_ocp(&case);
    let library = ocp
        .build_multiple_shooting_symbolic_function_library(preserved_options().symbolic_functions)
        .expect("multiple-shooting symbolic library should build");
    let integrator = library
        .multiple_shooting_integrator
        .expect("multiple-shooting integrator helper should be present");
    let forward = integrator.forward(1).expect("forward helper should build");

    let x_input = flatten_value(&constant_state(&case));
    let u_input = flatten_value(&constant_control(&case));
    let dudt_input = flatten_value(&StageControl {
        u0: sample(&case, 0),
        u1: sample(&case, 1),
    });
    let x_seed = vec![0.0, 1.0];
    let u_seed = vec![0.0, 0.0];
    let dudt_seed = vec![0.0, 0.0];
    let dt_seed = vec![0.0];

    let outputs = eval_symbolic_function_nonzeros(
        &forward,
        &[
            &x_input,
            &u_input,
            &dudt_input,
            &[],
            &[0.5],
            &x_seed,
            &u_seed,
            &dudt_seed,
            &[],
            &dt_seed,
        ],
    );
    let x_next_directional = outputs
        .first()
        .expect("forward helper should return x_next directional output");
    assert!(
        (x_next_directional[1] - 1.0).abs() <= 1.0e-9,
        "expected forward helper to preserve x_next.x1 wrt x.x1 identity term, got {:?}",
        x_next_directional
    );

    let combined_outputs = eval_symbolic_function_nonzeros(
        &forward,
        &[
            &x_input,
            &u_input,
            &dudt_input,
            &[],
            &[0.5],
            &[1.0, 1.0],
            &u_seed,
            &dudt_seed,
            &[],
            &dt_seed,
        ],
    );
    let combined_x_next = combined_outputs
        .first()
        .expect("forward helper should return x_next directional output");
    assert!(
        (combined_x_next[0] - 1.0).abs() <= 1.0e-9 && (combined_x_next[1] - 1.0).abs() <= 1.0e-9,
        "expected combined forward seed to preserve both identity terms, got {:?}",
        combined_x_next
    );
}

#[test]
#[ignore = "diagnose preserved-call multiple-shooting integrator Jacobian sparsity"]
fn diagnose_preserved_call_multiple_shooting_integrator_wrapper_jacobian_structure() {
    let case = deterministic_ms_bug_case();
    let ocp = generated_ms_ocp(&case);
    let library = ocp
        .build_multiple_shooting_symbolic_function_library(preserved_options().symbolic_functions)
        .expect("multiple-shooting symbolic library should build");
    let integrator = library
        .multiple_shooting_integrator
        .expect("multiple-shooting integrator helper should be present");

    let x = symbolic_value::<StageState<SX>>("x").expect("symbolic state should build");
    let x_column = symbolic_column(&x).expect("symbolic state column should build");
    let outputs = integrator
        .call(&[
            x_column.clone(),
            constant_column(&constant_control(&case)),
            constant_column(&StageControl {
                u0: sample(&case, 0),
                u1: sample(&case, 1),
            }),
            constant_column(&()),
            SXMatrix::dense_column(vec![SX::from(0.5)]).expect("dt column should build"),
        ])
        .expect("integrator wrapper call should build");
    let jacobian = outputs[0]
        .jacobian(&x_column)
        .expect("integrator wrapper Jacobian should build");

    assert!(
        jacobian.ccs().nz_index(1, 1).is_some(),
        "expected Jacobian structure to include x_next.x1 wrt x.x1, positions={:?}",
        jacobian.ccs().positions()
    );
}

#[test]
#[ignore = "diagnose preserved-call multiple-shooting integrator wrapper forward path"]
fn diagnose_preserved_call_multiple_shooting_integrator_wrapper_forward_path() {
    let case = deterministic_ms_bug_case();
    let ocp = generated_ms_ocp(&case);
    let library = ocp
        .build_multiple_shooting_symbolic_function_library(preserved_options().symbolic_functions)
        .expect("multiple-shooting symbolic library should build");
    let integrator = library
        .multiple_shooting_integrator
        .expect("multiple-shooting integrator helper should be present");

    let x = symbolic_value::<StageState<SX>>("x").expect("symbolic state should build");
    let x_column = symbolic_column(&x).expect("symbolic state column should build");
    let outputs = integrator
        .call(&[
            x_column.clone(),
            constant_column(&constant_control(&case)),
            constant_column(&StageControl {
                u0: sample(&case, 0),
                u1: sample(&case, 1),
            }),
            constant_column(&()),
            SXMatrix::dense_column(vec![SX::from(0.5)]).expect("dt column should build"),
        ])
        .expect("integrator wrapper call should build");

    let seed_x1 = SXMatrix::dense_column(vec![SX::from(0.0), SX::from(1.0)]).expect("seed");
    let wrapper_forward_x1 = outputs[0]
        .forward(&x_column, &seed_x1)
        .expect("wrapper forward should build");
    let seed_combined = SXMatrix::dense_column(vec![SX::from(1.0), SX::from(1.0)]).expect("seed");
    let wrapper_forward_combined = outputs[0]
        .forward(&x_column, &seed_combined)
        .expect("wrapper forward should build");
    let combined_expr = wrapper_forward_combined.nz(1).inspect();

    let x_input = flatten_value(&constant_state(&case));
    let x1_values = eval_symbolic_function_nonzeros(
        &SXFunction::new(
            "wrapper_forward_x1",
            vec![NamedMatrix::new("x", x_column.clone()).expect("wrapper input")],
            vec![NamedMatrix::new("dx", wrapper_forward_x1).expect("wrapper output")],
        )
        .expect("forward wrapper function"),
        &[&x_input],
    );
    let combined_values = eval_symbolic_function_nonzeros(
        &SXFunction::new(
            "wrapper_forward_combined",
            vec![NamedMatrix::new("x", x_column).expect("wrapper input")],
            vec![NamedMatrix::new("dx", wrapper_forward_combined).expect("wrapper output")],
        )
        .expect("forward wrapper function"),
        &[&x_input],
    );

    assert!(
        (x1_values[0][1] - 1.0).abs() <= 1.0e-9,
        "expected wrapper forward seed [0,1] to preserve identity term, got {:?}",
        x1_values[0]
    );
    assert!(
        (combined_values[0][0] - 1.0).abs() <= 1.0e-9
            && (combined_values[0][1] - 1.0).abs() <= 1.0e-9,
        "expected wrapper forward seed [1,1] to preserve both identity terms, got {:?}, expr={:?}",
        combined_values[0],
        combined_expr
    );
}

#[test]
#[ignore = "manual OCP assembly policy differential search"]
fn generated_ocp_assembly_policy_differential_stress() {
    require_release_mode_for_manual_property_runs();
    run_policy_differential(&POLICY_DIFFERENTIAL_STRESS);
}

#[test]
#[ignore = "manual direct-collocation inequality stress search"]
fn generated_direct_collocation_inequality_stress() {
    require_release_mode_for_manual_property_runs();
    run_direct_collocation_inequality_stress(&DIRECT_COLLOCATION_INEQUALITY_STRESS);
}
