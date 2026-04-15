use crate::common::{
    CompileCacheStatus, CompileProgressInfo, CompileProgressUpdate, ContinuousInitialGuess,
    FromMap, LatexSection, MetricKey, OcpRuntimeSpec, OcpSxFunctionConfig, PlotMode, ProblemId,
    ProblemSpec, Scene2D, SceneAnimation, SceneFrame, ScenePath, SolveArtifact, SolveStreamEvent,
    SolverMethod, SolverReport, SqpConfig, StandardOcpParams, TranscriptionConfig, chart,
    default_solver_method, default_sqp_config, default_transcription, deg_to_rad,
    direct_collocation_runtime_from_spec, expect_finite, interval_arc_bound_series,
    interval_arc_series, metric_with_key, multiple_shooting_runtime_from_spec, node_times,
    numeric_metric_with_key, ocp_sx_function_config_from_map, problem_controls,
    problem_scientific_slider_control, problem_slider_control, problem_spec, rad_to_deg,
    sample_or_default, segmented_bound_series, segmented_series, solver_config_from_map,
    solver_method_from_map, transcription_from_map, transcription_metrics,
};
use anyhow::Result;
use optimal_control::{
    Bounds1D, CompiledDirectCollocationOcp, CompiledMultipleShootingOcp, DirectCollocation,
    DirectCollocationRuntimeValues, DirectCollocationTimeGrid, DirectCollocationTrajectories,
    InterpolatedTrajectory, IntervalArc, MultipleShooting, MultipleShootingRuntimeValues,
    MultipleShootingTrajectories, Ocp, direct_collocation_root_arcs,
    direct_collocation_state_like_arcs,
};
use serde::Serialize;
use std::collections::BTreeMap;
use sx_core::SX;
const DEFAULT_INTERVALS: usize = 30;
const DEFAULT_COLLOCATION_DEGREE: usize = 3;
const SUPPORTED_INTERVALS: [usize; 1] = [DEFAULT_INTERVALS];
const SUPPORTED_DEGREES: [usize; 1] = [DEFAULT_COLLOCATION_DEGREE];
const G: f64 = 9.81;
const CART_MASS_KG: f64 = 8.0;
const LOAD_MASS_KG: f64 = 1.6;
const CART_FRICTION_NSPM: f64 = 3.0;
const PIVOT_FRICTION_NMS: f64 = 0.9;
const FORCE_WEIGHT: f64 = 2.0e-4;
#[derive(Clone, Debug, PartialEq, Serialize, optimization::Vectorize)]
pub struct State<T> {
    pub x: T,
    pub v: T,
    pub theta: T,
    pub omega: T,
}
#[derive(Clone, Debug, PartialEq, Serialize, optimization::Vectorize)]
pub struct Control<T> {
    pub force: T,
}
#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct Path<T> {
    x: T,
    theta: T,
    force: T,
    force_rate: T,
}
#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct Boundary<T> {
    x0: T,
    v0: T,
    theta0: T,
    omega0: T,
    force0: T,
    x_t: T,
    v_t: T,
    theta_t: T,
    omega_t: T,
    force_t: T,
}
#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct ModelParams<T> {
    rope_length: T,
    force_rate_weight: T,
}

type MsCompiled<const N: usize> = CompiledMultipleShootingOcp<
    State<SX>,
    Control<SX>,
    ModelParams<SX>,
    Path<SX>,
    Boundary<SX>,
    (),
    N,
    2,
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

thread_local! {
    static MULTIPLE_SHOOTING_CACHE: std::cell::RefCell<
        crate::common::SharedCompileCache<crate::common::MultipleShootingCompileKey, MsCompiled<DEFAULT_INTERVALS>>
    > = std::cell::RefCell::new(crate::common::SharedCompileCache::new());
    static DIRECT_COLLOCATION_CACHE: std::cell::RefCell<
        crate::common::SharedCompileCache<
            crate::common::DirectCollocationCompileVariantKey,
            DcCompiled<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>,
        >
    > = std::cell::RefCell::new(crate::common::SharedCompileCache::new());
}

const PROBLEM_NAME: &str = "Crane Load Transfer";
#[derive(Clone, Debug)]
pub struct Params {
    pub target_x_m: f64,
    pub tf_s: f64,
    pub rope_length_m: f64,
    pub force_limit_n: f64,
    pub force_rate_limit_nps: f64,
    pub force_rate_regularization: f64,
    pub swing_limit_deg: f64,
    pub solver_method: SolverMethod,
    pub solver: SqpConfig,
    pub transcription: TranscriptionConfig,
    pub sx_functions: OcpSxFunctionConfig,
}
impl Default for Params {
    fn default() -> Self {
        Self {
            target_x_m: 10.0,
            tf_s: 7.5,
            rope_length_m: 3.0,
            force_limit_n: 28.0,
            force_rate_limit_nps: 75.0,
            force_rate_regularization: 5.0e-4,
            swing_limit_deg: 15.0,
            solver_method: default_solver_method(),
            solver: default_sqp_config(),
            transcription: default_transcription(DEFAULT_INTERVALS),
            sx_functions: OcpSxFunctionConfig::default(),
        }
    }
}

impl StandardOcpParams for Params {
    fn transcription(&self) -> &TranscriptionConfig {
        &self.transcription
    }

    fn transcription_mut(&mut self) -> &mut TranscriptionConfig {
        &mut self.transcription
    }
}
impl FromMap for Params {
    fn from_map(values: &BTreeMap<String, f64>) -> Result<Self> {
        let defaults = Self::default();
        Ok(Self {
            target_x_m: expect_finite(
                sample_or_default(values, "target_x_m", defaults.target_x_m),
                "target_x_m",
            )?,
            tf_s: expect_finite(sample_or_default(values, "tf_s", defaults.tf_s), "tf_s")?,
            rope_length_m: expect_finite(
                sample_or_default(values, "rope_length_m", defaults.rope_length_m),
                "rope_length_m",
            )?,
            force_limit_n: expect_finite(
                sample_or_default(values, "force_limit_n", defaults.force_limit_n),
                "force_limit_n",
            )?,
            force_rate_limit_nps: expect_finite(
                sample_or_default(
                    values,
                    "force_rate_limit_nps",
                    defaults.force_rate_limit_nps,
                ),
                "force_rate_limit_nps",
            )?,
            force_rate_regularization: expect_finite(
                sample_or_default(
                    values,
                    "force_rate_regularization",
                    defaults.force_rate_regularization,
                ),
                "force_rate_regularization",
            )?,
            swing_limit_deg: expect_finite(
                sample_or_default(values, "swing_limit_deg", defaults.swing_limit_deg),
                "swing_limit_deg",
            )?,
            solver_method: solver_method_from_map(values, defaults.solver_method)?,
            solver: solver_config_from_map(values, defaults.solver)?,
            transcription: transcription_from_map(
                values,
                defaults.transcription,
                &SUPPORTED_INTERVALS,
                &SUPPORTED_DEGREES,
            )?,
            sx_functions: ocp_sx_function_config_from_map(values, defaults.sx_functions)?,
        })
    }
}
pub fn spec() -> ProblemSpec {
    let defaults = Params::default();
    problem_spec(
        ProblemId::CraneTransfer,
        "Crane Load Transfer",
        "A trolley drives a suspended load with a bounded cart force and bounded force-rate while respecting the fully coupled cart-pole dynamics.",
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
                    4.0,
                    16.0,
                    0.25,
                    defaults.target_x_m,
                    "m",
                    "Trolley travel distance.",
                ),
                problem_slider_control(
                    "tf_s",
                    "Transfer Time",
                    3.0,
                    10.0,
                    0.25,
                    defaults.tf_s,
                    "s",
                    "Fixed time allocated for the move.",
                ),
                problem_slider_control(
                    "rope_length_m",
                    "Rope Length",
                    1.0,
                    6.0,
                    0.1,
                    defaults.rope_length_m,
                    "m",
                    "Distance from trolley hook to the payload.",
                ),
                problem_slider_control(
                    "force_limit_n",
                    "Force Limit",
                    5.0,
                    80.0,
                    1.0,
                    defaults.force_limit_n,
                    "N",
                    "Absolute bound on commanded cart force.",
                ),
                problem_slider_control(
                    "force_rate_limit_nps",
                    "Force-Rate Limit",
                    5.0,
                    200.0,
                    1.0,
                    defaults.force_rate_limit_nps,
                    "N/s",
                    "Absolute bound on the rate of change of cart force.",
                ),
                problem_scientific_slider_control(
                    "force_rate_regularization",
                    "Force-Rate Weight",
                    0.0,
                    5.0e-3,
                    5.0e-5,
                    defaults.force_rate_regularization,
                    "",
                    "Quadratic stage-cost weight on force-rate effort.",
                ),
                problem_slider_control(
                    "swing_limit_deg",
                    "Swing Limit",
                    3.0,
                    25.0,
                    0.5,
                    defaults.swing_limit_deg,
                    "deg",
                    "Maximum payload swing angle.",
                ),
            ],
        ),
        vec![
            LatexSection {
                title: "Physical State".to_string(),
                entries: vec![r"\mathbf{x} = \begin{bmatrix} x & v & \theta & \omega \end{bmatrix}^{\mathsf T}".to_string()],
            },
            LatexSection {
                title: "Control-State".to_string(),
                entries: vec![
                    r"\mathbf{u} = \begin{bmatrix} F \end{bmatrix}".to_string(),
                    r"\dot{\mathbf{u}} = \begin{bmatrix} \dot{F} \end{bmatrix}".to_string(),
                ],
            },
            LatexSection {
                title: "Objective".to_string(),
                entries: vec![
                    r"J = \int_0^T \left(1.6\,\theta^2 + w_{\dot{F}}\,\dot{F}^2 + 2\cdot 10^{-4} F^2\right) \, dt".to_string(),
                ],
            },
            LatexSection {
                title: "Differential Equations".to_string(),
                entries: vec![
                    r"\dot{x} = v".to_string(),
                    r"\dot{\theta} = \omega".to_string(),
                    r"\dot{v} = \frac{F - b\,v + m\ell\,\omega^2\sin\theta + m g \sin\theta \cos\theta + \frac{c}{\ell}\,\omega\cos\theta}{M + m\sin^2\theta}".to_string(),
                    r"\dot{\omega} = \frac{-F\cos\theta + b\,v\cos\theta - m\ell\,\omega^2\sin\theta\cos\theta - (M+m)g\sin\theta - \frac{(M+m)c}{m\ell}\,\omega}{\ell\,(M + m\sin^2\theta)}".to_string(),
                    r"\dot{F} = \dot{F}_{\mathrm{cmd}}".to_string(),
                ],
            },
        ],
        vec![
            format!(
                "The cart-pole model uses M = {:.1} kg, m = {:.1} kg, cart friction b = {:.1} N·s/m, and pivot damping c = {:.1} N·m·s.",
                CART_MASS_KG, LOAD_MASS_KG, CART_FRICTION_NSPM, PIVOT_FRICTION_NMS
            ),
            "This example supports both multiple shooting and direct collocation with the same coupled cart-pole dynamics.".to_string(),
        ],
    )
}

fn model<Scheme>(
    scheme: Scheme,
) -> Ocp<State<SX>, Control<SX>, ModelParams<SX>, Path<SX>, Boundary<SX>, (), Scheme> {
    Ocp::new("crane_transfer", scheme)
        .objective_lagrange(
            |state: &State<SX>,
             control: &Control<SX>,
             force_rate: &Control<SX>,
             runtime: &ModelParams<SX>| {
                1.6 * state.theta.sqr()
                    + runtime.force_rate_weight * force_rate.force.sqr()
                    + FORCE_WEIGHT * control.force.sqr()
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
            |state: &State<SX>, control: &Control<SX>, runtime: &ModelParams<SX>| {
                let (x_ddot, theta_ddot) =
                    cart_pole_accelerations_sx(state, control, runtime.rope_length);
                State {
                    x: state.v,
                    v: x_ddot,
                    theta: state.omega,
                    omega: theta_ddot,
                }
            },
        )
        .path_constraints(
            |state: &State<SX>,
             control: &Control<SX>,
             force_rate: &Control<SX>,
             _: &ModelParams<SX>| Path {
                x: state.x,
                theta: state.theta,
                force: control.force,
                force_rate: force_rate.force,
            },
        )
        .boundary_equalities(
            |initial: &State<SX>,
             initial_control: &Control<SX>,
             terminal: &State<SX>,
             terminal_control: &Control<SX>,
             _: &ModelParams<SX>,
             _: &SX| Boundary {
                x0: initial.x,
                v0: initial.v,
                theta0: initial.theta,
                omega0: initial.omega,
                force0: initial_control.force,
                x_t: terminal.x,
                v_t: terminal.v,
                theta_t: terminal.theta,
                omega_t: terminal.omega,
                force_t: terminal_control.force,
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
        .expect("crane model should build")
}

fn cached_multiple_shooting(
    params: &Params,
) -> Result<crate::common::CachedCompile<MsCompiled<DEFAULT_INTERVALS>>> {
    MULTIPLE_SHOOTING_CACHE.with(|cache| {
        crate::common::cached_multiple_shooting_ocp_compile(
            &mut cache.borrow_mut(),
            DEFAULT_INTERVALS,
            params.sx_functions,
            |options| {
                model(MultipleShooting::<DEFAULT_INTERVALS, 2>)
                    .compile_jit_with_ocp_options(options)
            },
        )
    })
}

fn cached_direct_collocation(
    params: &Params,
    family: optimal_control::CollocationFamily,
) -> Result<crate::common::CachedCompile<DcCompiled<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>>>
{
    DIRECT_COLLOCATION_CACHE.with(|cache| {
        crate::common::cached_direct_collocation_ocp_compile(
            &mut cache.borrow_mut(),
            family,
            params.sx_functions,
            |options| {
                model(DirectCollocation::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE> { family })
                    .compile_jit_with_ocp_options(options)
            },
        )
    })
}

fn compile_multiple_shooting_with_progress(
    params: &Params,
    callback: &mut dyn FnMut(CompileProgressUpdate),
) -> Result<(
    std::rc::Rc<std::cell::RefCell<MsCompiled<DEFAULT_INTERVALS>>>,
    CompileProgressInfo,
)> {
    MULTIPLE_SHOOTING_CACHE.with(|cache| {
        crate::common::cached_multiple_shooting_ocp_compile_with_progress(
            &mut cache.borrow_mut(),
            DEFAULT_INTERVALS,
            params.sx_functions,
            callback,
            |options, on_progress| {
                model(MultipleShooting::<DEFAULT_INTERVALS, 2>)
                    .compile_jit_with_ocp_options_and_progress_callback(options, on_progress)
            },
            crate::common::compile_progress_info_from_compiled,
        )
    })
}

fn compile_direct_collocation_with_progress(
    params: &Params,
    family: optimal_control::CollocationFamily,
    callback: &mut dyn FnMut(CompileProgressUpdate),
) -> Result<(
    std::rc::Rc<std::cell::RefCell<DcCompiled<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>>>,
    CompileProgressInfo,
)> {
    DIRECT_COLLOCATION_CACHE.with(|cache| {
        crate::common::cached_direct_collocation_ocp_compile_with_progress(
            &mut cache.borrow_mut(),
            family,
            params.sx_functions,
            callback,
            |options, on_progress| {
                model(DirectCollocation::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE> { family })
                    .compile_jit_with_ocp_options_and_progress_callback(options, on_progress)
            },
            crate::common::compile_progress_info_from_compiled,
        )
    })
}

pub fn prewarm(params: &Params) -> Result<()> {
    crate::common::prewarm_standard_ocp(
        params,
        params.transcription.method,
        params.transcription.collocation_family,
        cached_multiple_shooting,
        cached_direct_collocation,
    )
}

pub fn prewarm_with_progress<F>(params: &Params, emit: F) -> Result<()>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    crate::common::prewarm_standard_ocp_with_progress(
        params,
        params.transcription.method,
        params.transcription.collocation_family,
        params.solver_method,
        emit,
        compile_multiple_shooting_with_progress,
        compile_direct_collocation_with_progress,
    )
}

pub fn compile_cache_statuses() -> Vec<CompileCacheStatus> {
    MULTIPLE_SHOOTING_CACHE.with(|cache| {
        DIRECT_COLLOCATION_CACHE.with(|dc_cache| {
            crate::common::standard_ocp_compile_cache_statuses(
                ProblemId::CraneTransfer,
                PROBLEM_NAME,
                &cache.borrow(),
                &dc_cache.borrow(),
            )
        })
    })
}

pub(crate) fn benchmark_default_case_with_progress(
    transcription: crate::common::TranscriptionMethod,
    preset: crate::benchmark_report::OcpBenchmarkPreset,
    eval_options: optimization::NlpEvaluationBenchmarkOptions,
    on_progress: &mut dyn FnMut(crate::benchmark_report::BenchmarkCaseProgress),
) -> Result<crate::benchmark_report::OcpBenchmarkRecord> {
    crate::common::benchmark_standard_ocp_case_with_progress(
        ProblemId::CraneTransfer,
        PROBLEM_NAME,
        transcription,
        preset,
        eval_options,
        on_progress,
        |options, on_progress| {
            model(MultipleShooting::<DEFAULT_INTERVALS, 2>)
                .compile_jit_with_ocp_options_and_progress_callback(options, on_progress)
        },
        |family, options, on_progress| {
            model(DirectCollocation::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE> { family })
                .compile_jit_with_ocp_options_and_progress_callback(options, on_progress)
        },
        ms_runtime::<DEFAULT_INTERVALS>,
        dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>,
    )
}

pub fn solve(params: &Params) -> Result<SolveArtifact> {
    crate::common::solve_standard_ocp(
        params,
        params.transcription.method,
        params.transcription.collocation_family,
        params.solver_method,
        &params.solver,
        cached_multiple_shooting,
        cached_direct_collocation,
        ms_runtime::<DEFAULT_INTERVALS>,
        dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>,
        |trajectories, x_arcs, u_arcs| {
            artifact_from_ms_trajectories(params, trajectories, x_arcs, u_arcs)
        },
        |trajectories, time_grid| artifact_from_dc_trajectories(params, trajectories, time_grid),
    )
}

pub fn solve_with_progress<F>(params: &Params, emit: F) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    crate::common::solve_standard_ocp_with_progress(
        params,
        params.transcription.method,
        params.transcription.collocation_family,
        params.solver_method,
        &params.solver,
        emit,
        compile_multiple_shooting_with_progress,
        compile_direct_collocation_with_progress,
        ms_runtime::<DEFAULT_INTERVALS>,
        dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>,
        |trajectories, x_arcs, u_arcs| {
            artifact_from_ms_trajectories(params, trajectories, x_arcs, u_arcs)
        },
        |trajectories, time_grid| artifact_from_dc_trajectories(params, trajectories, time_grid),
    )
}

pub(crate) fn solve_from_map(
    values: &std::collections::BTreeMap<String, f64>,
) -> Result<SolveArtifact> {
    crate::common::solve_from_value_map::<Params, _>(values, solve)
}

pub(crate) fn prewarm_from_map(values: &std::collections::BTreeMap<String, f64>) -> Result<()> {
    crate::common::prewarm_from_value_map::<Params, _>(values, prewarm)
}

pub(crate) fn solve_with_progress_boxed(
    values: &std::collections::BTreeMap<String, f64>,
    emit: Box<dyn FnMut(SolveStreamEvent) + Send>,
) -> Result<SolveArtifact> {
    crate::common::solve_with_progress_from_value_map::<Params, _>(values, emit, |params, emit| {
        solve_with_progress(params, emit)
    })
}

pub(crate) fn prewarm_with_progress_boxed(
    values: &std::collections::BTreeMap<String, f64>,
    emit: Box<dyn FnMut(SolveStreamEvent) + Send>,
) -> Result<()> {
    crate::common::prewarm_with_progress_from_value_map::<Params, _>(
        values,
        emit,
        |params, emit| prewarm_with_progress(params, emit),
    )
}

pub fn validate_derivatives(
    params: &Params,
    request: &crate::common::DerivativeCheckRequest,
) -> Result<crate::common::ProblemDerivativeCheck> {
    crate::common::validate_standard_ocp_derivatives(
        ProblemId::CraneTransfer,
        PROBLEM_NAME,
        params,
        params.transcription.method,
        params.transcription.collocation_family,
        params.sx_functions,
        request,
        cached_multiple_shooting,
        cached_direct_collocation,
        ms_runtime::<DEFAULT_INTERVALS>,
        dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>,
    )
}

pub(crate) fn validate_derivatives_from_request(
    request: &crate::common::DerivativeCheckRequest,
) -> Result<crate::common::ProblemDerivativeCheck> {
    let params = Params::from_map(&request.values)?;
    validate_derivatives(&params, request)
}

pub(crate) fn problem_entry() -> crate::ProblemEntry {
    crate::ProblemEntry {
        id: ProblemId::CraneTransfer,
        spec,
        solve_from_map,
        prewarm_from_map,
        validate_derivatives_from_request,
        solve_with_progress_boxed,
        prewarm_with_progress_boxed,
        compile_cache_statuses,
        benchmark_default_case_with_progress,
    }
}
fn smoothstep(s: f64) -> f64 {
    let clipped = s.clamp(0.0, 1.0);
    clipped * clipped * (3.0 - 2.0 * clipped)
}

fn cart_pole_accelerations_sx(
    state: &State<SX>,
    control: &Control<SX>,
    rope_length: SX,
) -> (SX, SX) {
    let sin_theta = state.theta.sin();
    let cos_theta = state.theta.cos();
    let denominator = CART_MASS_KG + LOAD_MASS_KG * sin_theta.sqr();
    let x_ddot = (control.force - CART_FRICTION_NSPM * state.v
        + (LOAD_MASS_KG * rope_length) * state.omega.sqr() * sin_theta
        + (LOAD_MASS_KG * G) * sin_theta * cos_theta
        + (PIVOT_FRICTION_NMS / rope_length) * state.omega * cos_theta)
        / denominator;
    let theta_ddot = (-control.force * cos_theta + CART_FRICTION_NSPM * state.v * cos_theta
        - (LOAD_MASS_KG * rope_length) * state.omega.sqr() * sin_theta * cos_theta
        - ((CART_MASS_KG + LOAD_MASS_KG) * G) * sin_theta
        - ((CART_MASS_KG + LOAD_MASS_KG) * PIVOT_FRICTION_NMS / (LOAD_MASS_KG * rope_length))
            * state.omega)
        / (rope_length * denominator);
    (x_ddot, theta_ddot)
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
fn continuous_guess(
    params: &Params,
) -> ContinuousInitialGuess<State<f64>, Control<f64>, ModelParams<f64>> {
    let sample_count = 2 * params.transcription.intervals + 1;
    let dt = params.tf_s / (sample_count as f64 - 1.0);
    let times = (0..sample_count)
        .map(|index| index as f64 * dt)
        .collect::<Vec<_>>();
    let x = times
        .iter()
        .map(|time| params.target_x_m * smoothstep(time / params.tf_s))
        .collect::<Vec<_>>();
    let v = finite_difference(&x, dt);
    let accel = finite_difference(&v, dt);
    let force = accel
        .iter()
        .zip(v.iter())
        .map(|(accel, v)| CART_MASS_KG * accel + CART_FRICTION_NSPM * v)
        .collect::<Vec<_>>();
    let force_rate = finite_difference(&force, dt);
    ContinuousInitialGuess::Interpolated(InterpolatedTrajectory {
        sample_times: times,
        x_samples: x
            .iter()
            .zip(v.iter())
            .map(|(x, v)| State {
                x: *x,
                v: *v,
                theta: 0.0,
                omega: 0.0,
            })
            .collect(),
        u_samples: force
            .iter()
            .map(|force| Control { force: *force })
            .collect(),
        dudt_samples: force_rate
            .iter()
            .map(|force_rate| Control { force: *force_rate })
            .collect(),
        tf: params.tf_s,
    })
}
fn load_position(state: &State<f64>, rope_length: f64) -> (f64, f64) {
    (
        state.x + rope_length * state.theta.sin(),
        -rope_length * state.theta.cos(),
    )
}
fn frame_for_state(state: &State<f64>, rope_length: f64) -> SceneFrame {
    let trolley = [state.x, 0.0];
    let (load_x, load_y) = load_position(state, rope_length);
    let left_wheel = [state.x - 0.45, 0.0];
    let right_wheel = [state.x + 0.45, 0.0];
    let mut points = BTreeMap::new();
    points.insert("trolley".to_string(), trolley);
    points.insert("load".to_string(), [load_x, load_y]);
    SceneFrame {
        points,
        segments: vec![
            (left_wheel, right_wheel),
            (trolley, [load_x, load_y]),
            ([load_x - 0.25, load_y], [load_x + 0.25, load_y]),
        ],
    }
}
fn runtime_spec(
    params: &Params,
) -> OcpRuntimeSpec<ModelParams<f64>, Path<Bounds1D>, Boundary<f64>, (), State<f64>, Control<f64>> {
    OcpRuntimeSpec {
        parameters: ModelParams {
            rope_length: params.rope_length_m,
            force_rate_weight: params.force_rate_regularization,
        },
        beq: Boundary {
            x0: 0.0,
            v0: 0.0,
            theta0: 0.0,
            omega0: 0.0,
            force0: 0.0,
            x_t: params.target_x_m,
            v_t: 0.0,
            theta_t: 0.0,
            omega_t: 0.0,
            force_t: 0.0,
        },
        bineq_bounds: (),
        path_bounds: Path {
            x: Bounds1D {
                lower: Some(0.0),
                upper: Some(params.target_x_m),
            },
            theta: Bounds1D {
                lower: Some(-deg_to_rad(params.swing_limit_deg)),
                upper: Some(deg_to_rad(params.swing_limit_deg)),
            },
            force: Bounds1D {
                lower: Some(-params.force_limit_n),
                upper: Some(params.force_limit_n),
            },
            force_rate: Bounds1D {
                lower: Some(-params.force_rate_limit_nps),
                upper: Some(params.force_rate_limit_nps),
            },
        },
        tf_bounds: Bounds1D {
            lower: Some(params.tf_s),
            upper: Some(params.tf_s),
        },
        initial_guess: continuous_guess(params),
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
    multiple_shooting_runtime_from_spec(runtime_spec(params))
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
    direct_collocation_runtime_from_spec(runtime_spec(params))
}
fn artifact_summary(
    params: &Params,
    final_x: f64,
    max_swing: f64,
    max_force: f64,
    max_force_rate: f64,
    tf: f64,
) -> Vec<crate::common::Metric> {
    let mut summary = transcription_metrics(&params.transcription).to_vec();
    summary.extend([
        metric_with_key(
            MetricKey::TargetX,
            "Target",
            format!("{:.2} m", params.target_x_m),
        ),
        numeric_metric_with_key(
            MetricKey::FinalX,
            "Final X",
            final_x,
            format!("{final_x:.2} m"),
        ),
        numeric_metric_with_key(
            MetricKey::MaxSwing,
            "Max Swing",
            max_swing,
            format!("{max_swing:.2} deg"),
        ),
        numeric_metric_with_key(
            MetricKey::MaxAccel,
            "Max Force",
            max_force,
            format!("{max_force:.2} N"),
        ),
        numeric_metric_with_key(
            MetricKey::MaxJerk,
            "Max Force Rate",
            max_force_rate,
            format!("{max_force_rate:.2} N/s"),
        ),
        numeric_metric_with_key(MetricKey::Duration, "Duration", tf, format!("{tf:.2} s")),
    ]);
    summary
}
fn artifact_from_interval_data(
    params: &Params,
    final_x: f64,
    max_swing: f64,
    max_force: f64,
    max_force_rate: f64,
    tf: f64,
    animation_times: Vec<f64>,
    frames: Vec<SceneFrame>,
    trolley_x: Vec<f64>,
    load_path_x: Vec<f64>,
    load_path_y: Vec<f64>,
    x_arcs: &[IntervalArc<State<f64>>],
    u_arcs: &[IntervalArc<Control<f64>>],
    force_rate_series: Vec<crate::common::TimeSeries>,
    notes: Vec<String>,
) -> SolveArtifact {
    SolveArtifact::new(
        "Crane Load Transfer",
        artifact_summary(params, final_x, max_swing, max_force, max_force_rate, tf),
        SolverReport::placeholder(),
        vec![
            chart("Trolley Position", "x (m)", {
                let mut series_out =
                    interval_arc_series("Trolley x (m)", x_arcs, PlotMode::LinesMarkers, |state| {
                        state.x
                    });
                series_out.extend(interval_arc_bound_series(
                    x_arcs,
                    Some(0.0),
                    Some(params.target_x_m),
                    PlotMode::Lines,
                ));
                series_out
            }),
            chart(
                "Trolley Velocity",
                "v (m/s)",
                interval_arc_series("Trolley v (m/s)", x_arcs, PlotMode::LinesMarkers, |state| {
                    state.v
                }),
            ),
            chart("Swing Angle", "theta (deg)", {
                let mut series_out =
                    interval_arc_series("Swing (deg)", x_arcs, PlotMode::LinesMarkers, |state| {
                        rad_to_deg(state.theta)
                    });
                series_out.extend(interval_arc_bound_series(
                    x_arcs,
                    Some(-params.swing_limit_deg),
                    Some(params.swing_limit_deg),
                    PlotMode::Lines,
                ));
                series_out
            }),
            chart(
                "Swing Rate",
                "omega (deg/s)",
                interval_arc_series(
                    "Swing Rate (deg/s)",
                    x_arcs,
                    PlotMode::LinesMarkers,
                    |state| rad_to_deg(state.omega),
                ),
            ),
            chart("Cart Force", "Force (N)", {
                let mut series_out =
                    interval_arc_series("Force (N)", u_arcs, PlotMode::LinesMarkers, |control| {
                        control.force
                    });
                series_out.extend(interval_arc_bound_series(
                    u_arcs,
                    Some(-params.force_limit_n),
                    Some(params.force_limit_n),
                    PlotMode::Lines,
                ));
                series_out
            }),
            chart("Force Rate", "Force Rate (N/s)", force_rate_series),
        ],
        Scene2D {
            title: "Crane Geometry".to_string(),
            x_label: "x (m)".to_string(),
            y_label: "height (m)".to_string(),
            paths: vec![
                ScenePath {
                    name: "Trolley".to_string(),
                    x: trolley_x,
                    y: vec![0.0; animation_times.len()],
                },
                ScenePath {
                    name: "Load".to_string(),
                    x: load_path_x,
                    y: load_path_y,
                },
            ],
            circles: Vec::new(),
            arrows: Vec::new(),
            animation: Some(SceneAnimation {
                times: animation_times,
                frames,
            }),
        },
        notes,
    )
}

fn artifact_from_ms_trajectories<const N: usize>(
    params: &Params,
    trajectories: &MultipleShootingTrajectories<State<f64>, Control<f64>, N>,
    x_arcs: &[IntervalArc<State<f64>>],
    u_arcs: &[IntervalArc<Control<f64>>],
) -> SolveArtifact {
    let animation_times = node_times::<N>(trajectories.tf);
    let mut trolley_x = Vec::with_capacity(N + 1);
    let mut swing_deg = Vec::with_capacity(N + 1);
    let mut force = Vec::with_capacity(N + 1);
    let mut force_rate = Vec::with_capacity(N + 1);
    let mut load_path_x = Vec::with_capacity(N + 1);
    let mut load_path_y = Vec::with_capacity(N + 1);
    let mut frames = Vec::with_capacity(N + 1);
    for index in 0..N {
        let state = &trajectories.x.nodes[index];
        let control = &trajectories.u.nodes[index];
        let rate = &trajectories.dudt[index];
        let (load_px, load_py) = load_position(state, params.rope_length_m);
        trolley_x.push(state.x);
        swing_deg.push(rad_to_deg(state.theta));
        force.push(control.force);
        force_rate.push(rate.force);
        load_path_x.push(load_px);
        load_path_y.push(load_py);
        frames.push(frame_for_state(state, params.rope_length_m));
    }
    let terminal = &trajectories.x.terminal;
    let terminal_force = trajectories.u.terminal.force;
    let (load_px, load_py) = load_position(terminal, params.rope_length_m);
    trolley_x.push(terminal.x);
    swing_deg.push(rad_to_deg(terminal.theta));
    force.push(terminal_force);
    force_rate.push(*force_rate.last().unwrap_or(&0.0));
    load_path_x.push(load_px);
    load_path_y.push(load_py);
    frames.push(frame_for_state(terminal, params.rope_length_m));
    let max_swing = swing_deg
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let max_force = force
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let max_force_rate = force_rate
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let mut force_rate_series = segmented_series(
        "Force Rate (N/s)",
        x_arcs.iter().enumerate().map(|(interval, arc)| {
            (
                arc.times.clone(),
                vec![trajectories.dudt[interval].force; arc.times.len()],
            )
        }),
        PlotMode::LinesMarkers,
    );
    force_rate_series.extend(segmented_bound_series(
        x_arcs.iter().map(|arc| arc.times.clone()),
        Some(-params.force_rate_limit_nps),
        Some(params.force_rate_limit_nps),
        PlotMode::Lines,
    ));
    artifact_from_interval_data(
        params,
        terminal.x,
        max_swing,
        max_force,
        max_force_rate,
        trajectories.tf,
        animation_times,
        frames,
        trolley_x,
        load_path_x,
        load_path_y,
        x_arcs,
        u_arcs,
        force_rate_series,
        vec![
            "Cart force is the control-state and force rate is the true decision variable.".to_string(),
            "The multiple-shooting charts show RK4 interval arcs reconstructed from each mesh node with the same substepping used by the transcription.".to_string(),
        ],
    )
}

fn artifact_from_dc_trajectories<const N: usize, const K: usize>(
    params: &Params,
    trajectories: &DirectCollocationTrajectories<State<f64>, Control<f64>, N, K>,
    time_grid: &DirectCollocationTimeGrid<N, K>,
) -> SolveArtifact {
    let animation_times = (0..N)
        .map(|index| time_grid.nodes.nodes[index])
        .chain(std::iter::once(time_grid.nodes.terminal))
        .collect::<Vec<_>>();
    let x_arcs =
        direct_collocation_state_like_arcs(&trajectories.x, &trajectories.root_x, time_grid)
            .expect("collocation state arcs should match trajectory layout");
    let u_arcs =
        direct_collocation_state_like_arcs(&trajectories.u, &trajectories.root_u, time_grid)
            .expect("collocation control-state arcs should match trajectory layout");
    let dudt_arcs = direct_collocation_root_arcs(&trajectories.root_dudt, time_grid);
    let mut trolley_x = Vec::with_capacity(N + 1);
    let mut load_path_x = Vec::with_capacity(N + 1);
    let mut load_path_y = Vec::with_capacity(N + 1);
    let mut frames = Vec::with_capacity(N + 1);
    let mut max_swing = 0.0_f64;
    let mut max_force = 0.0_f64;
    let mut max_force_rate = 0.0_f64;
    for index in 0..N {
        let state = &trajectories.x.nodes[index];
        let (load_px, load_py) = load_position(state, params.rope_length_m);
        trolley_x.push(state.x);
        load_path_x.push(load_px);
        load_path_y.push(load_py);
        frames.push(frame_for_state(state, params.rope_length_m));
        max_swing = max_swing.max(rad_to_deg(state.theta).abs());
    }
    let terminal = &trajectories.x.terminal;
    let (load_px, load_py) = load_position(terminal, params.rope_length_m);
    trolley_x.push(terminal.x);
    load_path_x.push(load_px);
    load_path_y.push(load_py);
    frames.push(frame_for_state(terminal, params.rope_length_m));
    max_swing = max_swing.max(rad_to_deg(terminal.theta).abs());
    for interval in 0..N {
        for root in 0..K {
            max_swing = max_swing
                .max(rad_to_deg(trajectories.root_x.intervals[interval][root].theta).abs());
            max_force = max_force.max(trajectories.root_u.intervals[interval][root].force.abs());
            max_force_rate =
                max_force_rate.max(trajectories.root_dudt.intervals[interval][root].force.abs());
        }
    }
    let mut force_rate_series = interval_arc_series(
        "Force Rate (N/s)",
        &dudt_arcs,
        PlotMode::LinesMarkers,
        |force_rate| force_rate.force,
    );
    force_rate_series.extend(interval_arc_bound_series(
        &dudt_arcs,
        Some(-params.force_rate_limit_nps),
        Some(params.force_rate_limit_nps),
        PlotMode::Lines,
    ));
    artifact_from_interval_data(
        params,
        terminal.x,
        max_swing,
        max_force,
        max_force_rate,
        trajectories.tf,
        animation_times,
        frames,
        trolley_x,
        load_path_x,
        load_path_y,
        &x_arcs,
        &u_arcs,
        force_rate_series,
        vec![
            "Cart force is the control-state and force rate is the true decision variable.".to_string(),
            "The collocation charts are interval-local: state-like quantities include the extrapolated interval endpoint, while force rate is only shown at collocation nodes.".to_string(),
        ],
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crane_reaches_target_with_bounded_swing() {
        let artifact = solve(&Params::default()).expect("crane solve should succeed");
        let final_x = crate::find_metric(&artifact.summary, crate::MetricKey::FinalX)
            .and_then(|metric| metric.numeric_value)
            .expect("final x should exist");
        let max_swing = crate::find_metric(&artifact.summary, crate::MetricKey::MaxSwing)
            .and_then(|metric| metric.numeric_value)
            .expect("max swing should exist");
        assert!((final_x - Params::default().target_x_m).abs() < 0.2);
        assert!(max_swing < 10.0, "swing should remain controlled");
    }
}
