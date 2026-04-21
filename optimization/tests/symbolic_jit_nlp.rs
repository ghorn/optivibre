use approx::assert_abs_diff_eq;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use optimization::{
    CallPolicy, CallPolicyConfig, ClarabelSqpOptions, ConstraintBounds,
    FiniteDifferenceValidationOptions, FunctionCompileOptions, InteriorPointLinearSolver,
    InteriorPointOptions, LlvmOptimizationLevel, NlpEvaluationBenchmarkOptions, RuntimeNlpBounds,
    RuntimeNlpScaling, SqpGlobalization, SymbolicCompileProgress, SymbolicCompileStage,
    SymbolicNlpOutputs, TypedNlpScaling, TypedRuntimeNlpBounds, clear_optivibre_jit_cache,
    flat_view, symbolic_nlp, symbolic_nlp_dynamic,
};
#[cfg(feature = "ipopt")]
use optimization::{IpoptOptions, IpoptRawStatus};
use sx_core::{NamedMatrix, SX, SXFunction, SXMatrix};
use tempfile::TempDir;

#[derive(Clone, optimization::Vectorize)]
struct Pair<T> {
    x: T,
    y: T,
}

#[derive(Clone, optimization::Vectorize)]
struct Point<T> {
    x: T,
    y: T,
}

#[derive(Clone, optimization::Vectorize)]
struct Scalar<T> {
    x: T,
}

#[derive(Clone, optimization::Vectorize)]
struct Chain<T, const N: usize> {
    points: [Point<T>; N],
}

fn named(name: &str, matrix: SXMatrix) -> NamedMatrix {
    NamedMatrix::new(name, matrix).expect("named matrix should be valid")
}

fn dynamic_parameterized_problem() -> optimization::DynamicSymbolicNlp {
    let x = Pair {
        x: SX::sym("x0"),
        y: SX::sym("x1"),
    };
    let p = Pair {
        x: SX::sym("p0"),
        y: SX::sym("p1"),
    };
    symbolic_nlp_dynamic(
        "parameterized_quadratic_dynamic",
        optimization::symbolic_column(&x).expect("x column"),
        Some(optimization::symbolic_column(&p).expect("p column")),
        (x.x - p.x).sqr() + (x.y - p.y).sqr(),
        Some(SXMatrix::dense_column(vec![x.x + x.y - 1.0]).expect("equality column")),
        Some(SXMatrix::dense_column(vec![x.x.sqr() + x.y.sqr(), x.y]).expect("ineq column")),
    )
    .expect("dynamic symbolic NLP should build")
}

#[derive(Clone, optimization::Vectorize)]
struct TinyMs<T> {
    x0: T,
    x1: T,
    x2: T,
    u0: T,
    u1: T,
}

#[test]
fn typed_symbolic_rosenbrock_solves_end_to_end_with_jit() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, (), (), (), _>("rosenbrock", |x, _| SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: (),
            inequalities: (),
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let timing = compiled.backend_timing_metadata();
    let summary = compiled
        .solve_sqp(
            &Pair { x: -1.2, y: 1.0 },
            &(),
            &TypedRuntimeNlpBounds::default(),
            &ClarabelSqpOptions {
                max_iters: 80,
                dual_tol: 1e-7,
                verbose: false,
                ..ClarabelSqpOptions::default()
            },
        )
        .expect("SQP solve should succeed");

    assert!(timing.function_creation_time.is_some());
    assert!(timing.derivative_generation_time.is_some());
    assert!(timing.jit_time.is_some());
    assert_abs_diff_eq!(summary.x[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[1], 1.0, epsilon = 1e-6);
    assert!(summary.objective <= 1e-10);
    assert_eq!(summary.equality_inf_norm, None);
    assert_eq!(summary.inequality_inf_norm, None);
    assert_eq!(summary.complementarity_inf_norm, None);

    let final_view: PairView<'_, f64> =
        flat_view::<Pair<f64>, f64>(&summary.x).expect("solver output should project into a view");
    assert_abs_diff_eq!(*final_view.x, 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(*final_view.y, 1.0, epsilon = 1e-6);
    let snapshot_view: PairView<'_, f64> = flat_view::<Pair<f64>, f64>(&summary.final_state.x)
        .expect("final snapshot should project into a view");
    assert_abs_diff_eq!(*snapshot_view.x, 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(*snapshot_view.y, 1.0, epsilon = 1e-6);
}

#[test]
fn typed_symbolic_disk_constrained_rosenbrock_solves_with_runtime_constraint_bounds() {
    let symbolic = symbolic_nlp::<Pair<SX>, (), (), Pair<SX>, _>("disk_rosenbrock", |x, _| {
        SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: (),
            inequalities: Pair {
                x: x.x.sqr() + x.y.sqr(),
                y: x.y,
            },
        }
    })
    .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let summary = compiled
        .solve_sqp(
            &Pair { x: -1.2, y: 1.0 },
            &(),
            &TypedRuntimeNlpBounds {
                variable_lower: None,
                variable_upper: None,
                inequality_lower: Some(Pair { x: None, y: None }),
                inequality_upper: Some(Pair {
                    x: Some(1.5),
                    y: Some(2.0),
                }),
                scaling: None,
            },
            &ClarabelSqpOptions {
                max_iters: 80,
                merit_penalty: 25.0,
                verbose: false,
                ..ClarabelSqpOptions::default()
            },
        )
        .expect("SQP solve should succeed");

    assert!(summary.primal_inf_norm <= 1e-6);
    assert!(summary.dual_inf_norm <= 1e-6);
    assert!(
        summary
            .complementarity_inf_norm
            .is_some_and(|value| value <= 1e-6)
    );
    assert!(summary.objective < 10.0);
    assert!(summary.x[0].powi(2) + summary.x[1].powi(2) <= 1.5 + 1e-5);
}

#[test]
fn typed_symbolic_hanging_chain_solves_end_to_end() {
    const N: usize = 4;
    let span = 3.0;
    let link_length = 0.75;
    let symbolic = symbolic_nlp::<Chain<SX, N>, (), [SX; N + 1], (), _>("hanging_chain", |q, _| {
        let objective = q.points.iter().fold(SX::zero(), |acc, point| acc + point.y);
        let mut constraints = std::array::from_fn(|_| SX::zero());
        let mut prev_x = SX::from(0.0);
        let mut prev_y = SX::from(0.0);
        let link_length_sq = link_length * link_length;
        for (index, point) in q.points.iter().enumerate() {
            constraints[index] =
                (point.x - prev_x).sqr() + (point.y - prev_y).sqr() - link_length_sq;
            prev_x = point.x;
            prev_y = point.y;
        }
        constraints[N] = (prev_x - span).sqr() + prev_y.sqr() - link_length_sq;
        SymbolicNlpOutputs {
            objective,
            equalities: constraints,
            inequalities: (),
        }
    })
    .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let summary = compiled
        .solve_sqp(
            &Chain {
                points: [
                    Point { x: 0.75, y: 0.0 },
                    Point {
                        x: 1.125,
                        y: -0.649_519_052_8,
                    },
                    Point {
                        x: 1.875,
                        y: -0.649_519_052_8,
                    },
                    Point { x: 2.25, y: 0.0 },
                ],
            },
            &(),
            &TypedRuntimeNlpBounds {
                variable_lower: None,
                variable_upper: None,
                inequality_lower: None,
                inequality_upper: None,
                scaling: None,
            },
            &ClarabelSqpOptions {
                max_iters: 120,
                merit_penalty: 50.0,
                dual_tol: 1e-5,
                overall_tol: 1e-5,
                verbose: false,
                ..ClarabelSqpOptions::default()
            },
        )
        .expect("SQP solve should succeed");

    assert!(summary.equality_inf_norm.is_some_and(|value| value <= 1e-6));
    assert_eq!(summary.inequality_inf_norm, None);
    assert_eq!(summary.complementarity_inf_norm, None);
    assert!(summary.objective < -1.35);
    assert_abs_diff_eq!(summary.x[0] + summary.x[6], 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[2] + summary.x[4], 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[1], summary.x[7], epsilon = 1e-5);
    assert_abs_diff_eq!(summary.x[3], summary.x[5], epsilon = 1e-5);
}

#[test]
fn typed_symbolic_parameterized_nlp_solves_end_to_end() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, Pair<SX>, SX, (), _>("parameterized_quadratic", |x, p| {
            SymbolicNlpOutputs {
                objective: (x.x - p.x).sqr() + (x.y - p.y).sqr(),
                equalities: x.x + x.y,
                inequalities: (),
            }
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let summary = compiled
        .solve_sqp(
            &Pair { x: 0.9, y: 0.1 },
            &Pair { x: 0.25, y: 0.75 },
            &TypedRuntimeNlpBounds {
                variable_lower: None,
                variable_upper: None,
                inequality_lower: None,
                inequality_upper: None,
                scaling: None,
            },
            &ClarabelSqpOptions {
                dual_tol: 1e-3,
                complementarity_tol: 1e-5,
                overall_tol: 1e-3,
                globalization: SqpGlobalization::LineSearchMerit(Default::default()),
                verbose: false,
                ..ClarabelSqpOptions::default()
            },
        )
        .expect("SQP solve should succeed");

    assert_abs_diff_eq!(summary.x[0], -0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[1], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.objective, 0.5, epsilon = 1e-9);
    assert!(summary.equality_inf_norm.is_some_and(|value| value <= 1e-8));
}

#[test]
fn dynamic_symbolic_builder_matches_typed_compile_surface() {
    let typed = symbolic_nlp::<Pair<SX>, Pair<SX>, SX, Pair<SX>, _>(
        "parameterized_quadratic_dynamic",
        |x, p| SymbolicNlpOutputs {
            objective: (x.x - p.x).sqr() + (x.y - p.y).sqr(),
            equalities: x.x + x.y - 1.0,
            inequalities: Pair {
                x: x.x.sqr() + x.y.sqr(),
                y: x.y,
            },
        },
    )
    .expect("typed symbolic NLP should build")
    .compile_jit()
    .expect("typed JIT compile should succeed");
    let dynamic = dynamic_parameterized_problem()
        .compile_jit()
        .expect("dynamic JIT compile should succeed");

    assert_eq!(typed.compile_stats(), dynamic.compile_stats());
}

#[test]
fn dynamic_symbolic_nlp_solves_with_flat_bounds_and_scaling() {
    let compiled = dynamic_parameterized_problem()
        .compile_jit()
        .expect("dynamic JIT compile should succeed");
    let summary = compiled
        .solve_sqp(
            &[-1.2, 1.0],
            Some(&[0.25, 0.75]),
            &RuntimeNlpBounds {
                variables: ConstraintBounds::default(),
                inequalities: ConstraintBounds {
                    lower: Some(vec![None, None]),
                    upper: Some(vec![Some(2.0), Some(1.0)]),
                },
            },
            Some(&RuntimeNlpScaling {
                variables: vec![2.0, 0.5],
                constraints: vec![1.0, 0.25, 0.5],
                objective: 2.0,
            }),
            &ClarabelSqpOptions {
                verbose: false,
                max_iters: 80,
                ..ClarabelSqpOptions::default()
            },
        )
        .expect("dynamic SQP solve should succeed");

    assert!(summary.equality_inf_norm.is_some_and(|value| value <= 1e-6));
    assert!(
        summary
            .inequality_inf_norm
            .is_some_and(|value| value <= 1e-6)
    );
    assert_abs_diff_eq!(summary.x[0] + summary.x[1], 1.0, epsilon = 1e-6);
    assert!(summary.x[0].powi(2) + summary.x[1].powi(2) <= 2.0 + 1e-6);
    assert!(summary.x[1] <= 1.0 + 1e-6);
}

#[test]
fn dynamic_symbolic_nlp_uses_reference_scales_like_typed_api() {
    let compiled = dynamic_parameterized_problem()
        .compile_jit()
        .expect("dynamic JIT compile should succeed");
    let benchmark = compiled
        .benchmark_bounded_evaluations(
            &[-1.2, 1.0],
            Some(&[0.25, 0.75]),
            &RuntimeNlpBounds {
                variables: ConstraintBounds::default(),
                inequalities: ConstraintBounds {
                    lower: Some(vec![None, None]),
                    upper: Some(vec![Some(2.0), Some(1.0)]),
                },
            },
            Some(&RuntimeNlpScaling {
                variables: vec![2.0, 0.5],
                constraints: vec![1.0, 0.25, 0.5],
                objective: 2.0,
            }),
            NlpEvaluationBenchmarkOptions {
                warmup_iterations: 1,
                measured_iterations: 1,
            },
        )
        .expect("dynamic benchmark should succeed");

    assert_abs_diff_eq!(
        benchmark.benchmark_point.objective_value,
        1.0825,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        benchmark.benchmark_point.equality_inf_norm.unwrap(),
        1.2,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        benchmark.benchmark_point.inequality_inf_norm.unwrap(),
        1.76,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        benchmark.benchmark_point.decision_inf_norm,
        2.0,
        epsilon = 1e-12
    );
}

#[test]
fn typed_symbolic_scaling_keeps_callbacks_in_original_units() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, Pair<SX>, SX, (), _>("parameterized_quadratic_scaled", |x, p| {
            SymbolicNlpOutputs {
                objective: (x.x - p.x).sqr() + (x.y - p.y).sqr(),
                equalities: x.x + x.y,
                inequalities: (),
            }
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let mut first_snapshot = None;
    let summary = compiled
        .solve_sqp_with_callback(
            &Pair { x: 0.9, y: 0.1 },
            &Pair { x: 0.25, y: 0.75 },
            &TypedRuntimeNlpBounds {
                variable_lower: None,
                variable_upper: None,
                inequality_lower: None,
                inequality_upper: None,
                scaling: Some(TypedNlpScaling {
                    variable: Pair { x: 2.0, y: 0.5 },
                    constraints: vec![10.0],
                    objective: 100.0,
                }),
            },
            &ClarabelSqpOptions {
                dual_tol: 1e-3,
                complementarity_tol: 1e-5,
                overall_tol: 1e-3,
                globalization: SqpGlobalization::LineSearchMerit(Default::default()),
                verbose: false,
                ..ClarabelSqpOptions::default()
            },
            |snapshot| {
                first_snapshot.get_or_insert_with(|| snapshot.clone());
            },
        )
        .expect("scaled SQP solve should succeed");

    let snapshot = first_snapshot.expect("callback should see at least one iterate");
    assert_abs_diff_eq!(snapshot.x[0], 0.9, epsilon = 1e-12);
    assert_abs_diff_eq!(snapshot.x[1], 0.1, epsilon = 1e-12);
    assert_abs_diff_eq!(snapshot.objective, 0.845, epsilon = 1e-12);
    assert_abs_diff_eq!(
        snapshot
            .eq_inf
            .expect("equality residual should be present"),
        0.1,
        epsilon = 1e-12
    );
    assert!(snapshot.overall_inf >= 0.1);

    assert_abs_diff_eq!(summary.x[0], -0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[1], 0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.objective, 0.5, epsilon = 1e-9);
    assert!(summary.equality_inf_norm.is_some_and(|value| value <= 1e-8));
    assert!(summary.overall_inf_norm <= 1e-3);
    assert_abs_diff_eq!(summary.final_state.x[0], -0.25, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.final_state.x[1], 0.25, epsilon = 1e-6);
}

#[test]
fn typed_symbolic_compile_exposes_timing_metadata() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, (), (), (), _>("timed_rosenbrock", |x, _| SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: (),
            inequalities: (),
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let timing = compiled.backend_timing_metadata();

    assert!(timing.function_creation_time.is_some());
    assert!(timing.derivative_generation_time.is_some());
    assert!(timing.jit_time.is_some());
}

#[test]
fn typed_symbolic_compile_callback_reports_full_pre_jit_symbolic_timing() {
    let symbolic = symbolic_nlp::<Pair<SX>, (), (), (), _>("timed_rosenbrock_callback", |x, _| {
        SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: (),
            inequalities: (),
        }
    })
    .expect("symbolic NLP should build");
    let mut callback_metadata = None;
    let compiled = symbolic
        .compile_jit_with_symbolic_callback(|metadata| {
            callback_metadata = Some(metadata);
        })
        .expect("JIT compile should succeed");
    let callback_metadata = callback_metadata.expect("callback timing should be captured");
    let callback_timing = callback_metadata.timing;
    let final_timing = compiled.backend_timing_metadata();

    assert!(callback_timing.function_creation_time.is_some());
    assert!(callback_timing.derivative_generation_time.is_some());
    assert_eq!(callback_timing.jit_time, None);
    assert_eq!(
        callback_timing.function_creation_time,
        final_timing.function_creation_time
    );
    assert_eq!(
        callback_timing.derivative_generation_time,
        final_timing.derivative_generation_time
    );
    assert!(final_timing.jit_time.is_some());
    assert_eq!(callback_metadata.stats.variable_count, 2);
    assert_eq!(callback_metadata.stats.equality_count, 0);
    assert_eq!(callback_metadata.stats.inequality_count, 0);
    assert!(callback_metadata.stats.hessian_nnz > 0);
}

#[test]
fn typed_symbolic_compile_progress_reports_symbolic_stages_before_ready() {
    let symbolic = symbolic_nlp::<Pair<SX>, (), SX, SX, _>("timed_stage_progress", |x, _| {
        SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: x.x + x.y,
            inequalities: x.x.sqr() + x.y.sqr(),
        }
    })
    .expect("symbolic NLP should build");

    let mut stages = Vec::new();
    let mut saw_ready = false;
    symbolic
        .compile_jit_with_options_and_symbolic_progress_callback(
            FunctionCompileOptions::from(LlvmOptimizationLevel::O0),
            |progress| match progress {
                SymbolicCompileProgress::Stage(progress) => stages.push(progress.stage),
                SymbolicCompileProgress::Ready(metadata) => {
                    saw_ready = true;
                    assert!(metadata.setup_profile.hessian_generation.is_some());
                }
            },
        )
        .expect("JIT compile should succeed");

    assert_eq!(
        stages,
        vec![
            SymbolicCompileStage::BuildProblem,
            SymbolicCompileStage::ObjectiveGradient,
            SymbolicCompileStage::EqualityJacobian,
            SymbolicCompileStage::InequalityJacobian,
            SymbolicCompileStage::LagrangianAssembly,
            SymbolicCompileStage::HessianGeneration,
        ]
    );
    assert!(saw_ready);
}

#[test]
fn typed_symbolic_compile_exposes_backend_compile_report() {
    let symbolic = symbolic_nlp::<Pair<SX>, (), SX, SX, _>("timed_compile_report", |x, _| {
        SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: x.x + x.y,
            inequalities: x.x.sqr() + x.y.sqr(),
        }
    })
    .expect("symbolic NLP should build");
    let compiled = symbolic
        .compile_jit_with_options(FunctionCompileOptions {
            opt_level: LlvmOptimizationLevel::O0,
            call_policy: CallPolicyConfig {
                default_policy: CallPolicy::InlineAtLowering,
                respect_function_overrides: true,
            },
        })
        .expect("JIT compile should succeed");
    let report = compiled.backend_compile_report();

    assert_eq!(report.timing, compiled.backend_timing_metadata());
    assert!(report.setup_profile.symbolic_construction.is_some());
    assert!(report.setup_profile.objective_gradient.is_some());
    assert!(report.setup_profile.equality_jacobian.is_some());
    assert!(report.setup_profile.inequality_jacobian.is_some());
    assert!(report.setup_profile.lagrangian_assembly.is_some());
    assert!(report.setup_profile.hessian_generation.is_some());
    assert!(report.setup_profile.lowering.is_some());
    assert!(report.setup_profile.llvm_jit.is_some());
}

fn cache_env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("cache env lock")
}

fn collect_cache_manifests(root: &Path) -> Vec<String> {
    fn visit(dir: &Path, manifests: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                visit(&path, manifests);
            } else if path.file_name().and_then(|name| name.to_str()) == Some("manifest.json") {
                let manifest = std::fs::read_to_string(&path).expect("read manifest");
                manifests.push(manifest);
            }
        }
    }

    let mut manifests = Vec::new();
    visit(root, &mut manifests);
    manifests.sort();
    manifests
}

fn collect_cache_manifests_with_prefix(root: &Path, lowered_name_prefix: &str) -> Vec<String> {
    let needle = format!("\"lowered_name\": \"{lowered_name_prefix}");
    collect_cache_manifests(root)
        .into_iter()
        .filter(|manifest| manifest.contains(&needle))
        .collect()
}

#[test]
fn typed_symbolic_compile_reports_llvm_disk_cache_hits_on_second_compile() {
    let _guard = cache_env_lock();
    let cache_root = TempDir::new().expect("temp cache root");
    unsafe { std::env::set_var("OPTIVIBRE_JIT_CACHE_DIR", cache_root.path()) };
    clear_optivibre_jit_cache().expect("clear temp cache");

    let symbolic = symbolic_nlp::<Pair<SX>, (), SX, SX, _>("timed_compile_cache_report", |x, _| {
        SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: x.x + x.y,
            inequalities: x.x.sqr() + x.y.sqr(),
        }
    })
    .expect("symbolic NLP should build");

    let options = FunctionCompileOptions {
        opt_level: LlvmOptimizationLevel::O0,
        call_policy: CallPolicyConfig {
            default_policy: CallPolicy::InlineAtLowering,
            respect_function_overrides: true,
        },
    };

    let first = symbolic
        .compile_jit_with_options(options)
        .expect("first JIT compile should succeed");
    let first_report = first.backend_compile_report();
    assert_eq!(first_report.llvm_jit_cache.hits, 0);
    assert!(first_report.llvm_jit_cache.misses > 0);
    let manifests_after_first =
        collect_cache_manifests_with_prefix(cache_root.path(), "timed_compile_cache_report");

    let second = symbolic
        .compile_jit_with_options(options)
        .expect("second JIT compile should succeed");
    let second_report = second.backend_compile_report();
    assert!(second_report.llvm_jit_cache.hits > 0);
    let manifests_after_second =
        collect_cache_manifests_with_prefix(cache_root.path(), "timed_compile_cache_report");
    assert_eq!(
        manifests_after_second, manifests_after_first,
        "second compile should not create new cache entries"
    );
    assert_eq!(
        second_report.llvm_jit_cache.hits,
        manifests_after_first.len(),
        "all cached kernels should be reused on the second compile"
    );
    assert_eq!(second_report.llvm_jit_cache.misses, 0);

    unsafe { std::env::remove_var("OPTIVIBRE_JIT_CACHE_DIR") };
}

#[test]
fn typed_symbolic_compiled_nlp_derivatives_match_finite_difference() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, (), SX, SX, _>("fd_validation", |x, _| SymbolicNlpOutputs {
            objective: x.x.powi(3) + x.x * x.y + 2.0 * x.y.sqr(),
            equalities: x.x + x.y - 0.5,
            inequalities: x.x.sqr() - x.y + 0.25,
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let report = compiled
        .validate_derivatives(
            &Pair { x: 0.3, y: -0.2 },
            &(),
            &0.7,
            &0.4,
            FiniteDifferenceValidationOptions {
                first_order_step: 1.0e-6,
                second_order_step: 1.0e-4,
                zero_tolerance: 1.0e-7,
            },
        )
        .expect("finite-difference validation should succeed");

    assert!(
        report.objective_gradient.max_abs_error <= 1.0e-8,
        "objective gradient max abs error too large: {:?}",
        report.objective_gradient
    );
    let equality_jacobian = report
        .equality_jacobian
        .as_ref()
        .expect("equality Jacobian report should exist");
    assert!(
        equality_jacobian.max_abs_error <= 1.0e-9,
        "equality Jacobian max abs error too large: {:?}",
        equality_jacobian
    );
    assert_eq!(equality_jacobian.sparsity.missing_from_analytic, 0);
    assert_eq!(equality_jacobian.sparsity.extra_in_analytic, 0);

    let inequality_jacobian = report
        .inequality_jacobian
        .as_ref()
        .expect("inequality Jacobian report should exist");
    assert!(
        inequality_jacobian.max_abs_error <= 1.0e-7,
        "inequality Jacobian max abs error too large: {:?}",
        inequality_jacobian
    );
    assert_eq!(inequality_jacobian.sparsity.missing_from_analytic, 0);
    assert_eq!(inequality_jacobian.sparsity.extra_in_analytic, 0);

    assert!(
        report.lagrangian_hessian.max_abs_error <= 5.0e-7,
        "lagrangian Hessian max abs error too large: {:?}",
        report.lagrangian_hessian
    );
    assert_eq!(report.lagrangian_hessian.sparsity.missing_from_analytic, 0);
    assert_eq!(report.lagrangian_hessian.sparsity.extra_in_analytic, 0);
}

#[test]
fn typed_symbolic_derivative_validation_rejects_nonfinite_derivatives() {
    let symbolic =
        symbolic_nlp::<Scalar<SX>, (), (), (), _>("sqrt_singularity", |x, _| SymbolicNlpOutputs {
            objective: x.x.sqrt(),
            equalities: (),
            inequalities: (),
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let error = compiled
        .validate_derivatives(
            &Scalar { x: 0.0 },
            &(),
            &(),
            &(),
            FiniteDifferenceValidationOptions {
                first_order_step: 1.0e-6,
                second_order_step: 1.0e-4,
                zero_tolerance: 1.0e-7,
            },
        )
        .expect_err("non-finite derivatives should be rejected");
    let message = format!("{error:#}");
    assert!(
        message.contains("objective_gradient") && message.contains("non-finite"),
        "unexpected validation error: {message}"
    );
}

#[test]
#[ignore = "manual diagnostic for helper-call derivative validation by call policy"]
fn diagnose_helper_call_derivatives_by_call_policy() {
    let z = SXMatrix::dense_column(vec![SX::sym("z0"), SX::sym("z1")]).expect("helper input");
    let helper_outputs = SXMatrix::dense_column(vec![
        z.nz(0).sin() + z.nz(1).powi(2),
        z.nz(0) * z.nz(1) + z.nz(1).sin(),
    ])
    .expect("helper output");
    let helper = SXFunction::new(
        "pair_features",
        vec![named("z", z)],
        vec![named("y", helper_outputs)],
    )
    .expect("helper function should build");

    let symbolic = symbolic_nlp::<Pair<SX>, (), SX, Pair<SX>, _>("helper_call_fd", |x, _| {
        let input = SXMatrix::dense_column(vec![x.x, x.y]).expect("call input");
        let called = helper
            .call_output(&[input])
            .expect("helper call output should build");
        SymbolicNlpOutputs {
            objective: x.x.sqr() + 2.0 * x.y.sqr(),
            equalities: called.nz(0) + x.x,
            inequalities: Pair {
                x: called.nz(0) - 0.1,
                y: called.nz(1) + x.x * x.y,
            },
        }
    })
    .expect("symbolic NLP should build");

    for (label, policy) in [
        ("inline_at_call", CallPolicy::InlineAtCall),
        ("inline_at_lowering", CallPolicy::InlineAtLowering),
        ("inline_in_llvm", CallPolicy::InlineInLLVM),
        ("noinline_llvm", CallPolicy::NoInlineLLVM),
    ] {
        let compiled = symbolic
            .compile_jit_with_options(FunctionCompileOptions {
                opt_level: LlvmOptimizationLevel::O0,
                call_policy: CallPolicyConfig {
                    default_policy: policy,
                    respect_function_overrides: true,
                },
            })
            .expect("JIT compile should succeed");
        let report = compiled
            .validate_derivatives(
                &Pair { x: 0.3, y: -0.2 },
                &(),
                &0.7,
                &Pair { x: 0.4, y: -0.6 },
                FiniteDifferenceValidationOptions {
                    first_order_step: 1.0e-6,
                    second_order_step: 1.0e-4,
                    zero_tolerance: 1.0e-7,
                },
            )
            .expect("finite-difference validation should succeed");

        println!(
            "{label}: eq={:?}\n  ineq={:?}\n  hess={:?}",
            report.equality_jacobian, report.inequality_jacobian, report.lagrangian_hessian
        );
    }
}

#[test]
#[ignore = "manual diagnostic for repeated helper-call derivative validation by call policy"]
fn diagnose_repeated_helper_call_derivatives_by_call_policy() {
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

    let symbolic =
        symbolic_nlp::<TinyMs<SX>, (), [SX; 2], [SX; 2], _>("mini_ms_helper_call", |v, _| {
            let step0 = step
                .call_output(&[SXMatrix::dense_column(vec![v.x0, v.u0]).expect("step0 input")])
                .expect("step0 output should build");
            let step1 = step
                .call_output(&[SXMatrix::dense_column(vec![v.x1, v.u1]).expect("step1 input")])
                .expect("step1 output should build");

            SymbolicNlpOutputs {
                objective: v.u0.sqr() + 0.5 * v.u1.sqr(),
                equalities: [step0.nz(0) - v.x1, step1.nz(0) - v.x2],
                inequalities: [step0.nz(1) - 0.1, step1.nz(1) - 0.2],
            }
        })
        .expect("symbolic NLP should build");

    for (label, policy) in [
        ("inline_at_call", CallPolicy::InlineAtCall),
        ("inline_at_lowering", CallPolicy::InlineAtLowering),
        ("inline_in_llvm", CallPolicy::InlineInLLVM),
        ("noinline_llvm", CallPolicy::NoInlineLLVM),
    ] {
        let compiled = symbolic
            .compile_jit_with_options(FunctionCompileOptions {
                opt_level: LlvmOptimizationLevel::O0,
                call_policy: CallPolicyConfig {
                    default_policy: policy,
                    respect_function_overrides: true,
                },
            })
            .expect("JIT compile should succeed");
        let report = compiled
            .validate_derivatives(
                &TinyMs {
                    x0: 0.3,
                    x1: -0.2,
                    x2: 0.1,
                    u0: 0.4,
                    u1: -0.6,
                },
                &(),
                &[0.7, -0.3],
                &[0.4, -0.5],
                FiniteDifferenceValidationOptions {
                    first_order_step: 1.0e-6,
                    second_order_step: 1.0e-4,
                    zero_tolerance: 1.0e-7,
                },
            )
            .expect("finite-difference validation should succeed");

        println!(
            "{label}: eq={:?}\n  ineq={:?}\n  hess={:?}",
            report.equality_jacobian, report.inequality_jacobian, report.lagrangian_hessian
        );
    }
}

#[cfg(feature = "ipopt")]
#[test]
fn typed_symbolic_rosenbrock_solves_with_ipopt_without_box_bounds() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, (), (), (), _>("rosenbrock_ipopt", |x, _| SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: (),
            inequalities: (),
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let summary = compiled
        .solve_ipopt(
            &Pair { x: -1.2, y: 1.0 },
            &(),
            &TypedRuntimeNlpBounds::default(),
            &IpoptOptions {
                max_iters: 120,
                tol: 1e-9,
                ..IpoptOptions::default()
            },
        )
        .expect("Ipopt solve should succeed");

    assert_abs_diff_eq!(summary.x[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(summary.x[1], 1.0, epsilon = 1e-6);
    assert!(summary.objective <= 1e-10);
    assert!(summary.dual_inf_norm <= 1e-6);
}

#[cfg(feature = "ipopt")]
#[test]
fn typed_symbolic_inequality_only_problem_solves_with_ipopt_without_box_bounds() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, (), (), Pair<SX>, _>("disk_rosenbrock_ipopt", |x, _| {
            SymbolicNlpOutputs {
                objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
                equalities: (),
                inequalities: Pair {
                    x: x.x.sqr() + x.y.sqr(),
                    y: x.y,
                },
            }
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let summary = compiled
        .solve_ipopt(
            &Pair { x: -1.2, y: 1.0 },
            &(),
            &TypedRuntimeNlpBounds {
                variable_lower: None,
                variable_upper: None,
                inequality_lower: Some(Pair { x: None, y: None }),
                inequality_upper: Some(Pair {
                    x: Some(1.5),
                    y: Some(2.0),
                }),
                scaling: None,
            },
            &IpoptOptions {
                max_iters: 200,
                tol: 1e-9,
                ..IpoptOptions::default()
            },
        )
        .expect("Ipopt solve should succeed");

    assert!(summary.primal_inf_norm <= 1e-7);
    assert!(summary.dual_inf_norm <= 1e-6);
    assert!(summary.complementarity_inf_norm <= 1e-6);
    assert!(summary.objective < 10.0);
    assert!(summary.x[0].powi(2) + summary.x[1].powi(2) <= 1.5 + 1e-5);
}

#[test]
fn typed_symbolic_problem_reports_adapter_timing_with_interior_point() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, (), (), (), _>("rosenbrock_ip", |x, _| SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: (),
            inequalities: (),
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let mut snapshots = Vec::new();
    let summary = compiled
        .solve_interior_point_with_callback(
            &Pair { x: -1.2, y: 1.0 },
            &(),
            &TypedRuntimeNlpBounds::default(),
            &InteriorPointOptions {
                max_iters: 120,
                dual_tol: 1e-6,
                linear_solver: InteriorPointLinearSolver::Auto,
                verbose: false,
                ..InteriorPointOptions::default()
            },
            |snapshot| snapshots.push(snapshot.clone()),
        )
        .expect("interior-point solve should succeed");

    assert!(summary.profiling.adapter_timing.is_some());
    assert!(
        snapshots
            .iter()
            .any(|snapshot| snapshot.timing.adapter_timing.is_some())
    );
}

#[cfg(feature = "ipopt")]
#[test]
fn typed_symbolic_problem_reports_adapter_timing_with_ipopt() {
    let symbolic =
        symbolic_nlp::<Pair<SX>, (), (), (), _>("rosenbrock_ipopt_cb", |x, _| SymbolicNlpOutputs {
            objective: (1.0 - x.x).sqr() + 100.0 * (x.y - x.x.sqr()).sqr(),
            equalities: (),
            inequalities: (),
        })
        .expect("symbolic NLP should build");
    let compiled = symbolic.compile_jit().expect("JIT compile should succeed");
    let mut snapshots = Vec::new();
    let summary = compiled
        .solve_ipopt_with_callback(
            &Pair { x: -1.2, y: 1.0 },
            &(),
            &TypedRuntimeNlpBounds::default(),
            &IpoptOptions {
                max_iters: 120,
                tol: 1e-9,
                ..IpoptOptions::default()
            },
            |snapshot| snapshots.push(snapshot.clone()),
        )
        .expect("Ipopt solve should succeed");

    assert!(matches!(
        summary.status,
        IpoptRawStatus::SolveSucceeded
            | IpoptRawStatus::SolvedToAcceptableLevel
            | IpoptRawStatus::FeasiblePointFound
    ));
    assert!(summary.profiling.adapter_timing.is_some());
    assert!(
        snapshots
            .iter()
            .any(|snapshot| snapshot.timing.adapter_timing.is_some())
    );
}
