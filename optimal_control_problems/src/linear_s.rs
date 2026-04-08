use crate::common::{
    CachedCompile, CompileCacheStatus, CompileProgressInfo, CompileProgressUpdate, FromMap,
    LatexSection, MetricKey, OcpCompileProgressState, PlotMode, ProblemId, ProblemSpec, Scene2D,
    ScenePath, SolveArtifact, SolveStreamEvent, SolverMethod, SolverReport, SqpConfig,
    TranscriptionConfig, TranscriptionMethod, cached_compile_with_progress, chart,
    compile_progress_info, default_solver_method, default_sqp_config, default_transcription,
    direct_collocation_compile_key as dc_compile_key, expect_finite,
    interactive_multiple_shooting_opt_level, interval_arc_bound_series, interval_arc_series,
    metric_with_key, numeric_metric_with_key, ocp_compile_progress_update, problem_controls,
    problem_scientific_slider_control, problem_slider_control, problem_spec, sample_or_default,
    segmented_bound_series, segmented_series, solve_cached_direct_collocation_problem,
    solve_cached_direct_collocation_problem_with_progress, solve_cached_multiple_shooting_problem,
    solve_cached_multiple_shooting_problem_with_progress, solver_config_from_map,
    solver_method_from_map, transcription_from_map, transcription_metrics,
};
use anyhow::Result;
use optimal_control::{
    Bounds1D, CompiledDirectCollocationOcp, CompiledMultipleShootingOcp, DirectCollocation,
    DirectCollocationInitialGuess, DirectCollocationRuntimeValues, DirectCollocationTimeGrid,
    DirectCollocationTrajectories, InterpolatedTrajectory, IntervalArc, MultipleShooting,
    MultipleShootingInitialGuess, MultipleShootingRuntimeValues, MultipleShootingTrajectories, Ocp,
    direct_collocation_root_arcs, direct_collocation_state_like_arcs,
};
use serde::Serialize;
use std::cell::RefCell;
use std::f64::consts::PI;
use std::rc::Rc;
use sx_core::SX;
const RK4_SUBSTEPS: usize = 2;
const DEFAULT_INTERVALS: usize = 30;
const DEFAULT_COLLOCATION_DEGREE: usize = 3;
const SUPPORTED_INTERVALS: [usize; 1] = [DEFAULT_INTERVALS];
const SUPPORTED_DEGREES: [usize; 1] = [DEFAULT_COLLOCATION_DEGREE];
#[derive(Clone, Debug, PartialEq, Serialize, optimization::Vectorize)]
pub struct State<T> {
    pub x: T,
    pub y: T,
    pub vx: T,
    pub vy: T,
}
#[derive(Clone, Debug, PartialEq, Serialize, optimization::Vectorize)]
pub struct Control<T> {
    pub ax: T,
    pub ay: T,
}
#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct Path<T> {
    y: T,
    ax: T,
    ay: T,
    jx: T,
    jy: T,
}
#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct Boundary<T> {
    x0: T,
    y0: T,
    vx0: T,
    vy0: T,
    x_t: T,
    y_t: T,
    vx_t: T,
    vy_t: T,
}
#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct ModelParams<T> {
    target_x: T,
    lateral_amplitude: T,
    jerk_weight: T,
}

type MsCompiled<const N: usize> = CompiledMultipleShootingOcp<
    State<SX>,
    Control<SX>,
    ModelParams<SX>,
    Path<SX>,
    Boundary<SX>,
    (),
    N,
    RK4_SUBSTEPS,
>;

type DcCompiled<const N: usize, const K: usize> = CompiledDirectCollocationOcp<
    State<SX>,
    Control<SX>,
    ModelParams<SX>,
    Path<SX>,
    Boundary<SX>,
    (),
    N,
    K,
>;

crate::standard_ocp_compile_caches!(
    MULTIPLE_SHOOTING_CACHE: MsCompiled<DEFAULT_INTERVALS>,
    DIRECT_COLLOCATION_CACHE: DcCompiled<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>
);

const PROBLEM_NAME: &str = "Linear Point-to-Point S Maneuver";
#[derive(Clone, Debug)]
pub struct Params {
    pub target_x_m: f64,
    pub tf_s: f64,
    pub lateral_amplitude_m: f64,
    pub corridor_half_width_m: f64,
    pub accel_limit_mps2: f64,
    pub jerk_limit_mps3: f64,
    pub jerk_regularization: f64,
    pub solver_method: SolverMethod,
    pub solver: SqpConfig,
    pub transcription: TranscriptionConfig,
}
impl Default for Params {
    fn default() -> Self {
        Self {
            target_x_m: 12.0,
            tf_s: 5.5,
            lateral_amplitude_m: 1.8,
            corridor_half_width_m: 2.8,
            accel_limit_mps2: 4.0,
            jerk_limit_mps3: 7.0,
            jerk_regularization: 1.5e-2,
            solver_method: default_solver_method(),
            solver: default_sqp_config(),
            transcription: default_transcription(DEFAULT_INTERVALS),
        }
    }
}
impl FromMap for Params {
    fn from_map(values: &std::collections::BTreeMap<String, f64>) -> Result<Self> {
        let defaults = Self::default();
        Ok(Self {
            target_x_m: expect_finite(
                sample_or_default(values, "target_x_m", defaults.target_x_m),
                "target_x_m",
            )?,
            tf_s: expect_finite(sample_or_default(values, "tf_s", defaults.tf_s), "tf_s")?,
            lateral_amplitude_m: expect_finite(
                sample_or_default(values, "lateral_amplitude_m", defaults.lateral_amplitude_m),
                "lateral_amplitude_m",
            )?,
            corridor_half_width_m: expect_finite(
                sample_or_default(
                    values,
                    "corridor_half_width_m",
                    defaults.corridor_half_width_m,
                ),
                "corridor_half_width_m",
            )?,
            accel_limit_mps2: expect_finite(
                sample_or_default(values, "accel_limit_mps2", defaults.accel_limit_mps2),
                "accel_limit_mps2",
            )?,
            jerk_limit_mps3: expect_finite(
                sample_or_default(values, "jerk_limit_mps3", defaults.jerk_limit_mps3),
                "jerk_limit_mps3",
            )?,
            jerk_regularization: expect_finite(
                sample_or_default(values, "jerk_regularization", defaults.jerk_regularization),
                "jerk_regularization",
            )?,
            solver_method: solver_method_from_map(values, defaults.solver_method)?,
            solver: solver_config_from_map(values, defaults.solver)?,
            transcription: transcription_from_map(
                values,
                defaults.transcription,
                &SUPPORTED_INTERVALS,
                &SUPPORTED_DEGREES,
            )?,
        })
    }
}
pub fn spec() -> ProblemSpec {
    let defaults = Params::default();
    problem_spec(
        ProblemId::LinearSManeuver,
        "Linear Point-to-Point S Maneuver",
        "A jerk-limited planar point mass driven between endpoints while tracking a smooth S-shaped lateral reference inside a bounded corridor.",
        problem_controls(
            defaults.transcription,
            &SUPPORTED_INTERVALS,
            &SUPPORTED_DEGREES,
            defaults.solver_method,
            defaults.solver,
            vec![
                problem_slider_control(
                    "target_x_m",
                    "Target X",
                    8.0,
                    20.0,
                    0.5,
                    defaults.target_x_m,
                    "m",
                    "Downrange target position.",
                ),
                problem_slider_control(
                    "tf_s",
                    "Transfer Time",
                    3.0,
                    10.0,
                    0.25,
                    defaults.tf_s,
                    "s",
                    "Fixed maneuver duration.",
                ),
                problem_slider_control(
                    "lateral_amplitude_m",
                    "S Amplitude",
                    0.4,
                    2.5,
                    0.1,
                    defaults.lateral_amplitude_m,
                    "m",
                    "Peak lateral excursion encouraged by the stage cost.",
                ),
                problem_slider_control(
                    "corridor_half_width_m",
                    "Corridor Half-Width",
                    0.8,
                    4.0,
                    0.1,
                    defaults.corridor_half_width_m,
                    "m",
                    "Absolute bound on lateral displacement.",
                ),
                problem_slider_control(
                    "accel_limit_mps2",
                    "Accel Limit",
                    1.0,
                    8.0,
                    0.1,
                    defaults.accel_limit_mps2,
                    "m/s²",
                    "Absolute bound on both acceleration components.",
                ),
                problem_slider_control(
                    "jerk_limit_mps3",
                    "Jerk Limit",
                    1.0,
                    16.0,
                    0.2,
                    defaults.jerk_limit_mps3,
                    "m/s³",
                    "Absolute bound on both jerk components.",
                ),
                problem_scientific_slider_control(
                    "jerk_regularization",
                    "Jerk Weight",
                    0.0,
                    1.0e-1,
                    1.0e-3,
                    defaults.jerk_regularization,
                    "",
                    "Quadratic stage-cost weight on jerk magnitude.",
                ),
            ],
        ),
        vec![
            LatexSection {
                title: "Physical State".to_string(),
                entries: vec![r"\mathbf{x} = \begin{bmatrix} x & y & v_x & v_y \end{bmatrix}^{\mathsf T}".to_string()],
            },
            LatexSection {
                title: "Control-State".to_string(),
                entries: vec![
                    r"\mathbf{u} = \begin{bmatrix} a_x & a_y \end{bmatrix}^{\mathsf T}".to_string(),
                    r"\dot{\mathbf{u}} = \begin{bmatrix} j_x & j_y \end{bmatrix}^{\mathsf T}".to_string(),
                ],
            },
            LatexSection {
                title: "Reference Path".to_string(),
                entries: vec![
                    r"s = \frac{x}{x_T}".to_string(),
                    r"g(s) = 16 s^2 (1 - s)^2".to_string(),
                    r"y_{\mathrm{ref}}(x) = A \, g(s) \sin(2 \pi s)".to_string(),
                ],
            },
            LatexSection {
                title: "Objective".to_string(),
                entries: vec![
                    r"J = \int_0^T \left(1.8\,(y - y_{\mathrm{ref}}(x))^2 + w_j\,(j_x^2 + j_y^2)\right) \, dt".to_string(),
                ],
            },
            LatexSection {
                title: "Differential Equations".to_string(),
                entries: vec![
                    r"\dot{x} = v_x".to_string(),
                    r"\dot{y} = v_y".to_string(),
                    r"\dot{v}_x = a_x".to_string(),
                    r"\dot{v}_y = a_y".to_string(),
                    r"\dot{a}_x = j_x".to_string(),
                    r"\dot{a}_y = j_y".to_string(),
                ],
            },
        ],
        vec![
            "The dynamics are linear in position, velocity, acceleration, and jerk.".to_string(),
            "The S-shape is produced by a smooth lateral-reference objective, while acceleration, jerk, and lateral position remain hard-constrained.".to_string(),
        ],
    )
}
fn s_reference(x: f64, params: &Params) -> f64 {
    let s = (x / params.target_x_m).clamp(0.0, 1.0);
    let gate = 16.0 * s.powi(2) * (1.0 - s).powi(2);
    params.lateral_amplitude_m * gate * (2.0 * PI * s).sin()
}

fn cached_multiple_shooting() -> Result<CachedCompile<MsCompiled<DEFAULT_INTERVALS>>> {
    MULTIPLE_SHOOTING_CACHE.with(|cache| {
        cache.borrow_mut().get_or_try_init(DEFAULT_INTERVALS, || {
            Ok(model(MultipleShooting::<DEFAULT_INTERVALS, RK4_SUBSTEPS>)
                .compile_jit_with_opt_level(interactive_multiple_shooting_opt_level())?)
        })
    })
}

fn cached_direct_collocation(
    family: optimal_control::CollocationFamily,
) -> Result<CachedCompile<DcCompiled<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>>> {
    DIRECT_COLLOCATION_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .get_or_try_init(dc_compile_key(family), || {
                Ok(model(
                    DirectCollocation::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE> { family },
                )
                .compile_jit()?)
            })
    })
}

fn compile_multiple_shooting_with_progress(
    callback: &mut dyn FnMut(CompileProgressUpdate),
) -> Result<(
    Rc<RefCell<MsCompiled<DEFAULT_INTERVALS>>>,
    CompileProgressInfo,
)> {
    MULTIPLE_SHOOTING_CACHE.with(|cache| {
        cached_compile_with_progress(
            &mut cache.borrow_mut(),
            DEFAULT_INTERVALS,
            callback,
            |on_compile_progress| {
                let mut progress_state = OcpCompileProgressState::default();
                Ok(model(MultipleShooting::<DEFAULT_INTERVALS, RK4_SUBSTEPS>)
                    .compile_jit_with_opt_level_and_progress_callback(
                        interactive_multiple_shooting_opt_level(),
                        |progress| {
                            on_compile_progress(ocp_compile_progress_update(
                                progress,
                                &mut progress_state,
                            ));
                        },
                    )?)
            },
            |compiled| {
                compile_progress_info(
                    compiled.backend_timing_metadata(),
                    compiled.nlp_compile_stats(),
                    compiled.helper_kernel_count(),
                    compiled.helper_compile_stats(),
                )
            },
        )
    })
}

fn compile_direct_collocation_with_progress(
    family: optimal_control::CollocationFamily,
    callback: &mut dyn FnMut(CompileProgressUpdate),
) -> Result<(
    Rc<RefCell<DcCompiled<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>>>,
    CompileProgressInfo,
)> {
    DIRECT_COLLOCATION_CACHE.with(|cache| {
        cached_compile_with_progress(
            &mut cache.borrow_mut(),
            dc_compile_key(family),
            callback,
            |on_compile_progress| {
                let mut progress_state = OcpCompileProgressState::default();
                Ok(model(
                    DirectCollocation::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE> { family },
                )
                .compile_jit_with_progress_callback(|progress| {
                    on_compile_progress(ocp_compile_progress_update(progress, &mut progress_state));
                })?)
            },
            |compiled| {
                compile_progress_info(
                    compiled.backend_timing_metadata(),
                    compiled.nlp_compile_stats(),
                    compiled.helper_kernel_count(),
                    compiled.helper_compile_stats(),
                )
            },
        )
    })
}

pub fn prewarm(params: &Params) -> Result<()> {
    match params.transcription.method {
        TranscriptionMethod::MultipleShooting => cached_multiple_shooting().map(|_| ()),
        TranscriptionMethod::DirectCollocation => {
            cached_direct_collocation(params.transcription.collocation_family).map(|_| ())
        }
    }
}

pub fn prewarm_with_progress<F>(params: &Params, emit: F) -> Result<()>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    match params.transcription.method {
        TranscriptionMethod::MultipleShooting => {
            let mut lifecycle =
                crate::common::SolveLifecycleReporter::new(emit, params.solver_method);
            lifecycle.prewarm_with_progress(compile_multiple_shooting_with_progress)
        }
        TranscriptionMethod::DirectCollocation => {
            let mut lifecycle =
                crate::common::SolveLifecycleReporter::new(emit, params.solver_method);
            lifecycle.prewarm_with_progress(|callback| {
                compile_direct_collocation_with_progress(
                    params.transcription.collocation_family,
                    callback,
                )
            })
        }
    }
}

pub fn compile_cache_statuses() -> Vec<CompileCacheStatus> {
    crate::standard_ocp_compile_cache_statuses!(
        ProblemId::LinearSManeuver,
        PROBLEM_NAME,
        MULTIPLE_SHOOTING_CACHE,
        DIRECT_COLLOCATION_CACHE
    )
}

fn model<Scheme>(
    scheme: Scheme,
) -> Ocp<State<SX>, Control<SX>, ModelParams<SX>, Path<SX>, Boundary<SX>, (), Scheme> {
    Ocp::new("linear_s_problem", scheme)
        .objective_lagrange(
            |state: &State<SX>, _: &Control<SX>, jerk: &Control<SX>, runtime: &ModelParams<SX>| {
                let s = state.x / runtime.target_x;
                let gate = 16.0 * s.sqr() * (1.0 - s).sqr();
                let y_ref = runtime.lateral_amplitude * gate * (2.0 * PI * s).sin();
                1.8 * (state.y - y_ref).sqr()
                    + runtime.jerk_weight * (jerk.ax.sqr() + jerk.ay.sqr())
            },
        )
        .objective_mayer(
            |_: &State<SX>,
             _: &Control<SX>,
             _: &State<SX>,
             _: &Control<SX>,
             _: &ModelParams<SX>,
             _: &SX| { SX::zero() },
        )
        .ode(
            |state: &State<SX>, control: &Control<SX>, _: &ModelParams<SX>| State {
                x: state.vx,
                y: state.vy,
                vx: control.ax,
                vy: control.ay,
            },
        )
        .path_constraints(
            |state: &State<SX>, control: &Control<SX>, jerk: &Control<SX>, _: &ModelParams<SX>| {
                Path {
                    y: state.y,
                    ax: control.ax,
                    ay: control.ay,
                    jx: jerk.ax,
                    jy: jerk.ay,
                }
            },
        )
        .boundary_equalities(
            |initial: &State<SX>,
             _: &Control<SX>,
             terminal: &State<SX>,
             _: &Control<SX>,
             _: &ModelParams<SX>,
             _: &SX| Boundary {
                x0: initial.x,
                y0: initial.y,
                vx0: initial.vx,
                vy0: initial.vy,
                x_t: terminal.x,
                y_t: terminal.y,
                vx_t: terminal.vx,
                vy_t: terminal.vy,
            },
        )
        .boundary_inequalities(
            |_: &State<SX>,
             _: &Control<SX>,
             _: &State<SX>,
             _: &Control<SX>,
             _: &ModelParams<SX>,
             _: &SX| (),
        )
        .build()
        .expect("linear_s model should build")
}
fn smoothstep(s: f64) -> f64 {
    10.0 * s.powi(3) - 15.0 * s.powi(4) + 6.0 * s.powi(5)
}
fn lateral_shape(s: f64, amplitude: f64) -> f64 {
    let gate = 16.0 * s.powi(2) * (1.0 - s).powi(2);
    amplitude * gate * (2.0 * PI * s).sin()
}
fn finite_difference(values: &[f64], dt: f64) -> Vec<f64> {
    (0..values.len())
        .map(|index| {
            if index == 0 {
                (values[1] - values[0]) / dt
            } else if index + 1 == values.len() {
                (values[index] - values[index - 1]) / dt
            } else {
                (values[index + 1] - values[index - 1]) / (2.0 * dt)
            }
        })
        .collect()
}
fn guess<const N: usize>(params: &Params) -> InterpolatedTrajectory<State<f64>, Control<f64>> {
    let sample_count = 2 * N + 1;
    let dt = params.tf_s / (sample_count as f64 - 1.0);
    let times = (0..sample_count)
        .map(|index| index as f64 * dt)
        .collect::<Vec<_>>();
    let x_values = times
        .iter()
        .map(|time| params.target_x_m * smoothstep(time / params.tf_s))
        .collect::<Vec<_>>();
    let y_values = times
        .iter()
        .map(|time| lateral_shape(time / params.tf_s, params.lateral_amplitude_m))
        .collect::<Vec<_>>();
    let vx = finite_difference(&x_values, dt);
    let vy = finite_difference(&y_values, dt);
    let ax = finite_difference(&vx, dt);
    let ay = finite_difference(&vy, dt);
    let jx = finite_difference(&ax, dt);
    let jy = finite_difference(&ay, dt);
    InterpolatedTrajectory {
        sample_times: times,
        x_samples: x_values
            .iter()
            .zip(y_values.iter())
            .zip(vx.iter().zip(vy.iter()))
            .map(|((x, y), (vx, vy))| State {
                x: *x,
                y: *y,
                vx: *vx,
                vy: *vy,
            })
            .collect(),
        u_samples: ax
            .iter()
            .zip(ay.iter())
            .map(|(ax, ay)| Control { ax: *ax, ay: *ay })
            .collect(),
        dudt_samples: jx
            .iter()
            .zip(jy.iter())
            .map(|(ax, ay)| Control { ax: *ax, ay: *ay })
            .collect(),
        tf: params.tf_s,
    }
}
pub fn solve(params: &Params) -> Result<SolveArtifact> {
    match params.transcription.method {
        TranscriptionMethod::MultipleShooting => solve_multiple_shooting(params),
        TranscriptionMethod::DirectCollocation => solve_direct_collocation(params),
    }
}
pub fn solve_with_progress<F>(params: &Params, emit: F) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    match params.transcription.method {
        TranscriptionMethod::MultipleShooting => {
            solve_multiple_shooting_with_progress(params, emit)
        }
        TranscriptionMethod::DirectCollocation => {
            solve_direct_collocation_with_progress(params, emit)
        }
    }
}
fn ms_runtime<const N: usize>(
    params: &Params,
) -> MultipleShootingRuntimeValues<
    ModelParams<f64>,
    Path<Bounds1D>,
    Boundary<f64>,
    (),
    State<f64>,
    Control<f64>,
    N,
> {
    MultipleShootingRuntimeValues {
        parameters: ModelParams {
            target_x: params.target_x_m,
            lateral_amplitude: params.lateral_amplitude_m,
            jerk_weight: params.jerk_regularization,
        },
        beq: Boundary {
            x0: 0.0,
            y0: 0.0,
            vx0: 0.0,
            vy0: 0.0,
            x_t: params.target_x_m,
            y_t: 0.0,
            vx_t: 0.0,
            vy_t: 0.0,
        },
        bineq_bounds: (),
        path_bounds: Path {
            y: Bounds1D {
                lower: Some(-params.corridor_half_width_m),
                upper: Some(params.corridor_half_width_m),
            },
            ax: Bounds1D {
                lower: Some(-params.accel_limit_mps2),
                upper: Some(params.accel_limit_mps2),
            },
            ay: Bounds1D {
                lower: Some(-params.accel_limit_mps2),
                upper: Some(params.accel_limit_mps2),
            },
            jx: Bounds1D {
                lower: Some(-params.jerk_limit_mps3),
                upper: Some(params.jerk_limit_mps3),
            },
            jy: Bounds1D {
                lower: Some(-params.jerk_limit_mps3),
                upper: Some(params.jerk_limit_mps3),
            },
        },
        tf_bounds: Bounds1D {
            lower: Some(params.tf_s),
            upper: Some(params.tf_s),
        },
        initial_guess: MultipleShootingInitialGuess::Interpolated(guess::<N>(params)),
    }
}
fn dc_runtime<const N: usize, const K: usize>(
    params: &Params,
) -> DirectCollocationRuntimeValues<
    ModelParams<f64>,
    Path<Bounds1D>,
    Boundary<f64>,
    (),
    State<f64>,
    Control<f64>,
    N,
    K,
> {
    DirectCollocationRuntimeValues {
        parameters: ModelParams {
            target_x: params.target_x_m,
            lateral_amplitude: params.lateral_amplitude_m,
            jerk_weight: params.jerk_regularization,
        },
        beq: Boundary {
            x0: 0.0,
            y0: 0.0,
            vx0: 0.0,
            vy0: 0.0,
            x_t: params.target_x_m,
            y_t: 0.0,
            vx_t: 0.0,
            vy_t: 0.0,
        },
        bineq_bounds: (),
        path_bounds: Path {
            y: Bounds1D {
                lower: Some(-params.corridor_half_width_m),
                upper: Some(params.corridor_half_width_m),
            },
            ax: Bounds1D {
                lower: Some(-params.accel_limit_mps2),
                upper: Some(params.accel_limit_mps2),
            },
            ay: Bounds1D {
                lower: Some(-params.accel_limit_mps2),
                upper: Some(params.accel_limit_mps2),
            },
            jx: Bounds1D {
                lower: Some(-params.jerk_limit_mps3),
                upper: Some(params.jerk_limit_mps3),
            },
            jy: Bounds1D {
                lower: Some(-params.jerk_limit_mps3),
                upper: Some(params.jerk_limit_mps3),
            },
        },
        tf_bounds: Bounds1D {
            lower: Some(params.tf_s),
            upper: Some(params.tf_s),
        },
        initial_guess: DirectCollocationInitialGuess::Interpolated(guess::<N>(params)),
    }
}
fn solve_multiple_shooting(params: &Params) -> Result<SolveArtifact> {
    let compiled = cached_multiple_shooting()?;
    solve_cached_multiple_shooting_problem(
        &compiled.compiled,
        &ms_runtime::<DEFAULT_INTERVALS>(params),
        params.solver_method,
        &params.solver,
        |trajectories, x_arcs, u_arcs| {
            artifact_from_ms_trajectories(params, trajectories, x_arcs, u_arcs)
        },
    )
}
fn solve_direct_collocation(params: &Params) -> Result<SolveArtifact> {
    let compiled = cached_direct_collocation(params.transcription.collocation_family)?;
    solve_cached_direct_collocation_problem(
        &compiled.compiled,
        &dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(params),
        params.solver_method,
        &params.solver,
        |trajectories, time_grid| artifact_from_dc_trajectories(params, trajectories, time_grid),
    )
}
fn solve_multiple_shooting_with_progress<F>(params: &Params, emit: F) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    let mut lifecycle = crate::common::SolveLifecycleReporter::new(emit, params.solver_method);
    let (compiled, running_solver) =
        lifecycle.compile_with_progress(compile_multiple_shooting_with_progress)?;
    solve_cached_multiple_shooting_problem_with_progress(
        &compiled,
        &ms_runtime::<DEFAULT_INTERVALS>(params),
        params.solver_method,
        &params.solver,
        lifecycle.into_emit(),
        running_solver,
        |trajectories, x_arcs, u_arcs| {
            artifact_from_ms_trajectories(params, trajectories, x_arcs, u_arcs)
        },
    )
}
fn solve_direct_collocation_with_progress<F>(params: &Params, emit: F) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    let mut lifecycle = crate::common::SolveLifecycleReporter::new(emit, params.solver_method);
    let (compiled, running_solver) = lifecycle.compile_with_progress(|callback| {
        compile_direct_collocation_with_progress(params.transcription.collocation_family, callback)
    })?;
    solve_cached_direct_collocation_problem_with_progress(
        &compiled,
        &dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(params),
        params.solver_method,
        &params.solver,
        lifecycle.into_emit(),
        running_solver,
        |trajectories, time_grid| artifact_from_dc_trajectories(params, trajectories, time_grid),
    )
}
fn artifact_summary(
    params: &Params,
    final_x: f64,
    final_y: f64,
    max_y: f64,
    min_y: f64,
    peak_jerk: f64,
    tf: f64,
) -> Vec<crate::common::Metric> {
    let mut summary = transcription_metrics(&params.transcription).to_vec();
    summary.extend([
        numeric_metric_with_key(
            MetricKey::FinalX,
            "Final X",
            final_x,
            format!("{final_x:.2} m"),
        ),
        numeric_metric_with_key(
            MetricKey::FinalY,
            "Final Y",
            final_y,
            format!("{final_y:.2} m"),
        ),
        numeric_metric_with_key(MetricKey::MaxY, "Max Y", max_y, format!("{max_y:.2} m")),
        numeric_metric_with_key(MetricKey::MinY, "Min Y", min_y, format!("{min_y:.2} m")),
        metric_with_key(
            MetricKey::PeakJerk,
            "Peak Jerk",
            format!("{peak_jerk:.2} m/s³"),
        ),
        metric_with_key(
            MetricKey::TransferTime,
            "Transfer Time",
            format!("{tf:.2} s"),
        ),
    ]);
    summary
}
fn artifact_from_ms_trajectories<const N: usize>(
    params: &Params,
    trajectories: &MultipleShootingTrajectories<State<f64>, Control<f64>, N>,
    x_arcs: &[IntervalArc<State<f64>>],
    u_arcs: &[IntervalArc<Control<f64>>],
) -> SolveArtifact {
    let mut x = Vec::with_capacity(N + 1);
    let mut y = Vec::with_capacity(N + 1);
    let mut vx = Vec::with_capacity(N + 1);
    let mut vy = Vec::with_capacity(N + 1);
    let mut ax = Vec::with_capacity(N + 1);
    let mut ay = Vec::with_capacity(N + 1);
    let mut jx = Vec::with_capacity(N + 1);
    let mut jy = Vec::with_capacity(N + 1);
    let mut y_ref = Vec::with_capacity(N + 1);
    let mut corridor_upper = Vec::with_capacity(N + 1);
    let mut corridor_lower = Vec::with_capacity(N + 1);
    for index in 0..N {
        let state = &trajectories.x.nodes[index];
        let control = &trajectories.u.nodes[index];
        let jerk = &trajectories.dudt[index];
        x.push(state.x);
        y.push(state.y);
        vx.push(state.vx);
        vy.push(state.vy);
        ax.push(control.ax);
        ay.push(control.ay);
        jx.push(jerk.ax);
        jy.push(jerk.ay);
        y_ref.push(s_reference(state.x, params));
        corridor_upper.push(params.corridor_half_width_m);
        corridor_lower.push(-params.corridor_half_width_m);
    }
    let terminal_state = &trajectories.x.terminal;
    let terminal_accel = &trajectories.u.terminal;
    x.push(terminal_state.x);
    y.push(terminal_state.y);
    vx.push(terminal_state.vx);
    vy.push(terminal_state.vy);
    ax.push(terminal_accel.ax);
    ay.push(terminal_accel.ay);
    jx.push(*jx.last().unwrap_or(&0.0));
    jy.push(*jy.last().unwrap_or(&0.0));
    y_ref.push(s_reference(terminal_state.x, params));
    corridor_upper.push(params.corridor_half_width_m);
    corridor_lower.push(-params.corridor_half_width_m);
    let max_y = y
        .iter()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));
    let min_y = y.iter().fold(f64::INFINITY, |acc, value| acc.min(*value));
    let peak_jerk = jx
        .iter()
        .chain(jy.iter())
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    SolveArtifact::new(
        "Linear S Maneuver",
        artifact_summary(
            params,
            terminal_state.x,
            terminal_state.y,
            max_y,
            min_y,
            peak_jerk,
            trajectories.tf,
        ),
        SolverReport::placeholder(),
        vec![
            chart(
                "x Position",
                "x (m)",
                interval_arc_series("x (m)", x_arcs, PlotMode::LinesMarkers, |state| state.x),
            ),
            chart(
                "y Position",
                "y (m)",
                {
                    let mut series_out = interval_arc_series(
                        "y (m)",
                        x_arcs,
                        PlotMode::LinesMarkers,
                        |state| state.y,
                    );
                    series_out.extend(interval_arc_bound_series(
                        x_arcs,
                        Some(-params.corridor_half_width_m),
                        Some(params.corridor_half_width_m),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
            chart(
                "x Velocity",
                "vx (m/s)",
                interval_arc_series("vx (m/s)", x_arcs, PlotMode::LinesMarkers, |state| state.vx),
            ),
            chart(
                "y Velocity",
                "vy (m/s)",
                interval_arc_series("vy (m/s)", x_arcs, PlotMode::LinesMarkers, |state| state.vy),
            ),
            chart(
                "x Acceleration",
                "ax (m/s²)",
                {
                    let mut series_out = interval_arc_series(
                        "ax (m/s²)",
                        u_arcs,
                        PlotMode::LinesMarkers,
                        |control| control.ax,
                    );
                    series_out.extend(interval_arc_bound_series(
                        u_arcs,
                        Some(-params.accel_limit_mps2),
                        Some(params.accel_limit_mps2),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
            chart(
                "y Acceleration",
                "ay (m/s²)",
                {
                    let mut series_out = interval_arc_series(
                        "ay (m/s²)",
                        u_arcs,
                        PlotMode::LinesMarkers,
                        |control| control.ay,
                    );
                    series_out.extend(interval_arc_bound_series(
                        u_arcs,
                        Some(-params.accel_limit_mps2),
                        Some(params.accel_limit_mps2),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
            chart(
                "x Jerk",
                "jx (m/s³)",
                {
                    let mut series_out = segmented_series(
                        "jx (m/s³)",
                        x_arcs.iter().enumerate().map(|(interval, arc)| {
                            (arc.times.clone(), vec![trajectories.dudt[interval].ax; arc.times.len()])
                        }),
                        PlotMode::LinesMarkers,
                    );
                    series_out.extend(segmented_bound_series(
                        x_arcs.iter().map(|arc| arc.times.clone()),
                        Some(-params.jerk_limit_mps3),
                        Some(params.jerk_limit_mps3),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
            chart(
                "y Jerk",
                "jy (m/s³)",
                {
                    let mut series_out = segmented_series(
                        "jy (m/s³)",
                        x_arcs.iter().enumerate().map(|(interval, arc)| {
                            (arc.times.clone(), vec![trajectories.dudt[interval].ay; arc.times.len()])
                        }),
                        PlotMode::LinesMarkers,
                    );
                    series_out.extend(segmented_bound_series(
                        x_arcs.iter().map(|arc| arc.times.clone()),
                        Some(-params.jerk_limit_mps3),
                        Some(params.jerk_limit_mps3),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
        ],
        Scene2D {
            title: "Planar Path".to_string(),
            x_label: "x (m)".to_string(),
            y_label: "y (m)".to_string(),
            paths: vec![
                ScenePath {
                    name: "Trajectory".to_string(),
                    x: x.clone(),
                    y: y.clone(),
                },
                ScenePath {
                    name: "Reference".to_string(),
                    x: x.clone(),
                    y: y_ref,
                },
            ],
            circles: Vec::new(),
            arrows: Vec::new(),
            animation: None,
        },
        vec![
            "Acceleration is treated as the control-state and jerk is the decision variable.".to_string(),
            "When direct collocation is selected, the chart traces are interval-local arcs so polynomial endpoints and continuity defects stay visible.".to_string(),
        ],
    )
}
fn artifact_from_dc_trajectories<const N: usize, const K: usize>(
    params: &Params,
    trajectories: &DirectCollocationTrajectories<State<f64>, Control<f64>, N, K>,
    time_grid: &DirectCollocationTimeGrid<N, K>,
) -> SolveArtifact {
    let x_arcs =
        direct_collocation_state_like_arcs(&trajectories.x, &trajectories.root_x, time_grid)
            .expect("collocation state arcs should match trajectory layout");
    let u_arcs =
        direct_collocation_state_like_arcs(&trajectories.u, &trajectories.root_u, time_grid)
            .expect("collocation control-state arcs should match trajectory layout");
    let dudt_arcs = direct_collocation_root_arcs(&trajectories.root_dudt, time_grid);
    let mut x_mesh = Vec::with_capacity(N + 1);
    let mut y_mesh = Vec::with_capacity(N + 1);
    let mut y_ref_mesh = Vec::with_capacity(N + 1);
    let mut max_y = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut peak_jerk = 0.0_f64;
    for index in 0..N {
        let state = &trajectories.x.nodes[index];
        x_mesh.push(state.x);
        y_mesh.push(state.y);
        y_ref_mesh.push(s_reference(state.x, params));
        max_y = max_y.max(state.y);
        min_y = min_y.min(state.y);
    }
    x_mesh.push(trajectories.x.terminal.x);
    y_mesh.push(trajectories.x.terminal.y);
    y_ref_mesh.push(s_reference(trajectories.x.terminal.x, params));
    max_y = max_y.max(trajectories.x.terminal.y);
    min_y = min_y.min(trajectories.x.terminal.y);
    for interval in 0..N {
        for root in 0..K {
            max_y = max_y.max(trajectories.root_x.intervals[interval][root].y);
            min_y = min_y.min(trajectories.root_x.intervals[interval][root].y);
            peak_jerk = peak_jerk
                .max(trajectories.root_dudt.intervals[interval][root].ax.abs())
                .max(trajectories.root_dudt.intervals[interval][root].ay.abs());
        }
    }
    SolveArtifact::new(
        "Linear S Maneuver",
        artifact_summary(
            params,
            trajectories.x.terminal.x,
            trajectories.x.terminal.y,
            max_y,
            min_y,
            peak_jerk,
            trajectories.tf,
        ),
        SolverReport::placeholder(),
        vec![
            chart(
                "x Position",
                "x (m)",
                interval_arc_series("x (m)", &x_arcs, PlotMode::LinesMarkers, |state| state.x),
            ),
            chart(
                "y Position",
                "y (m)",
                {
                    let mut series_out =
                        interval_arc_series("y (m)", &x_arcs, PlotMode::LinesMarkers, |state| state.y);
                    series_out.extend(interval_arc_bound_series(
                        &x_arcs,
                        Some(-params.corridor_half_width_m),
                        Some(params.corridor_half_width_m),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
            chart(
                "x Velocity",
                "vx (m/s)",
                interval_arc_series("vx (m/s)", &x_arcs, PlotMode::LinesMarkers, |state| state.vx),
            ),
            chart(
                "y Velocity",
                "vy (m/s)",
                interval_arc_series("vy (m/s)", &x_arcs, PlotMode::LinesMarkers, |state| state.vy),
            ),
            chart(
                "x Acceleration",
                "ax (m/s²)",
                {
                    let mut series_out = interval_arc_series(
                        "ax (m/s²)",
                        &u_arcs,
                        PlotMode::LinesMarkers,
                        |control| control.ax,
                    );
                    series_out.extend(interval_arc_bound_series(
                        &u_arcs,
                        Some(-params.accel_limit_mps2),
                        Some(params.accel_limit_mps2),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
            chart(
                "y Acceleration",
                "ay (m/s²)",
                {
                    let mut series_out = interval_arc_series(
                        "ay (m/s²)",
                        &u_arcs,
                        PlotMode::LinesMarkers,
                        |control| control.ay,
                    );
                    series_out.extend(interval_arc_bound_series(
                        &u_arcs,
                        Some(-params.accel_limit_mps2),
                        Some(params.accel_limit_mps2),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
            chart(
                "x Jerk",
                "jx (m/s³)",
                {
                    let mut series_out = interval_arc_series(
                        "jx (m/s³)",
                        &dudt_arcs,
                        PlotMode::LinesMarkers,
                        |jerk| jerk.ax,
                    );
                    series_out.extend(interval_arc_bound_series(
                        &dudt_arcs,
                        Some(-params.jerk_limit_mps3),
                        Some(params.jerk_limit_mps3),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
            chart(
                "y Jerk",
                "jy (m/s³)",
                {
                    let mut series_out = interval_arc_series(
                        "jy (m/s³)",
                        &dudt_arcs,
                        PlotMode::LinesMarkers,
                        |jerk| jerk.ay,
                    );
                    series_out.extend(interval_arc_bound_series(
                        &dudt_arcs,
                        Some(-params.jerk_limit_mps3),
                        Some(params.jerk_limit_mps3),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
        ],
        Scene2D {
            title: "Planar Path".to_string(),
            x_label: "x (m)".to_string(),
            y_label: "y (m)".to_string(),
            paths: vec![
                ScenePath {
                    name: "Trajectory".to_string(),
                    x: x_mesh.clone(),
                    y: y_mesh.clone(),
                },
                ScenePath {
                    name: "Reference".to_string(),
                    x: x_mesh,
                    y: y_ref_mesh,
                },
            ],
            circles: Vec::new(),
            arrows: Vec::new(),
            animation: None,
        },
        vec![
            "Acceleration is treated as the control-state and jerk is the decision variable.".to_string(),
            "Each direct-collocation interval is drawn separately so the mesh nodes, collocation nodes, and extrapolated interval endpoints are all visible.".to_string(),
        ],
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn s_maneuver_reaches_target_and_weaves() {
        let artifact = solve(&Params::default()).expect("linear s solve should succeed");
        let max_y = crate::find_metric(&artifact.summary, crate::MetricKey::MaxY)
            .and_then(|metric| metric.numeric_value)
            .expect("max y metric should exist");
        let min_y = crate::find_metric(&artifact.summary, crate::MetricKey::MinY)
            .and_then(|metric| metric.numeric_value)
            .expect("min y metric should exist");
        assert!(max_y > 0.5);
        assert!(min_y < -0.5);
    }
}
