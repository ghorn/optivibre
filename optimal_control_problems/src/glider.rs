use crate::common::{
    CompileCacheStatus, CompileProgressInfo, CompileProgressUpdate, ContinuousInitialGuess,
    FromMap, LatexSection, MetricKey, OcpRuntimeSpec, OcpSxFunctionConfig, PlotMode, ProblemId,
    ProblemSpec, Scene2D, ScenePath, SolveArtifact, SolveStreamEvent, SolverMethod, SolverReport,
    SqpConfig, StandardOcpParams, TranscriptionConfig, chart, default_solver_method,
    default_sqp_config, default_transcription, deg_to_rad, direct_collocation_runtime_from_spec,
    expect_finite, interval_arc_bound_series, interval_arc_series, metric_with_key,
    multiple_shooting_runtime_from_spec, node_times, numeric_metric_with_key,
    ocp_sx_function_config_from_map, problem_controls, problem_scientific_slider_control,
    problem_slider_control, problem_spec, rad_to_deg, sample_or_default, segmented_bound_series,
    segmented_series, solver_config_from_map, solver_method_from_map, transcription_from_map,
    transcription_metrics, trapezoid_integral,
};
use anyhow::{Result, anyhow};
use optimal_control::{
    Bounds1D, CompiledDirectCollocationOcp, CompiledMultipleShootingOcp, DirectCollocation,
    DirectCollocationRuntimeValues, DirectCollocationTimeGrid, DirectCollocationTrajectories,
    IntervalArc, MultipleShooting, MultipleShootingRuntimeValues, MultipleShootingTrajectories,
    Ocp, direct_collocation_root_arcs, direct_collocation_state_like_arcs,
};
use serde::Serialize;
use std::f64::consts::PI;
use sx_core::SX;
const RK4_SUBSTEPS: usize = 2;
const DEFAULT_INTERVALS: usize = 30;
const DEFAULT_COLLOCATION_DEGREE: usize = 3;
const SUPPORTED_INTERVALS: [usize; 1] = [DEFAULT_INTERVALS];
const SUPPORTED_DEGREES: [usize; 1] = [DEFAULT_COLLOCATION_DEGREE];
const ASPECT_RATIO: f64 = 10.0;
const EFFICIENCY: f64 = 0.95;
const CL_SLOPE: f64 = 2.0 * PI * ASPECT_RATIO / 12.0;
const CL_LOWER_BOUND: f64 = -0.5;
const CL_UPPER_BOUND: f64 = 1.5;
const GRAVITY: f64 = 9.81;
const AIR_DENSITY: f64 = 1.15;
const REFERENCE_AREA: f64 = 4.0;
const GLIDER_MASS: f64 = 30.0;
const SPEED_EPS: f64 = 1.0e-3;
const INITIAL_ALTITUDE_M: f64 = 1.0;
const LAUNCH_PATH_SEED_DEG: f64 = 12.0;
const MIN_FLIGHT_TIME_S: f64 = 1.0;
const MAX_FLIGHT_TIME_S: f64 = 500.0;
#[derive(Clone, Debug, PartialEq, Serialize, optimization::Vectorize)]
pub struct State<T> {
    pub x: T,
    pub altitude: T,
    pub vx: T,
    pub vy: T,
}
#[derive(Clone, Debug, PartialEq, Serialize, optimization::Vectorize)]
pub struct Control<T> {
    pub alpha: T,
}
#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct Path<T> {
    altitude: T,
    vx: T,
    cl: T,
    alpha_rate: T,
}
#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct Boundary<T> {
    x0: T,
    altitude0: T,
    speed_sq0: T,
}
#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct ModelParams<T> {
    alpha_rate_weight: T,
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

const PROBLEM_NAME: &str = "Optimal Distance Glider";
#[derive(Clone, Debug)]
pub struct Params {
    pub launch_speed_mps: f64,
    pub initial_alpha_deg: f64,
    pub initial_time_guess_s: f64,
    pub min_time_bound_s: f64,
    pub max_time_bound_s: f64,
    pub max_alpha_rate_deg_s: f64,
    pub alpha_rate_regularization: f64,
    pub solver_method: SolverMethod,
    pub solver: SqpConfig,
    pub transcription: TranscriptionConfig,
    pub sx_functions: OcpSxFunctionConfig,
}
impl Default for Params {
    fn default() -> Self {
        Self {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 7.0,
            initial_time_guess_s: 5.0,
            min_time_bound_s: MIN_FLIGHT_TIME_S,
            max_time_bound_s: MAX_FLIGHT_TIME_S,
            max_alpha_rate_deg_s: 25.0,
            alpha_rate_regularization: 5.0e-3,
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
    fn from_map(values: &std::collections::BTreeMap<String, f64>) -> Result<Self> {
        let defaults = Self::default();
        let min_time_bound_s = expect_finite(
            sample_or_default(values, "min_time_bound_s", defaults.min_time_bound_s),
            "min_time_bound_s",
        )?;
        let max_time_bound_s = expect_finite(
            sample_or_default(values, "max_time_bound_s", defaults.max_time_bound_s),
            "max_time_bound_s",
        )?;
        if min_time_bound_s > max_time_bound_s {
            return Err(anyhow!(
                "min_time_bound_s must be less than or equal to max_time_bound_s"
            ));
        }
        Ok(Self {
            launch_speed_mps: expect_finite(
                sample_or_default(values, "launch_speed_mps", defaults.launch_speed_mps),
                "launch_speed_mps",
            )?,
            initial_alpha_deg: expect_finite(
                sample_or_default(values, "initial_alpha_deg", defaults.initial_alpha_deg),
                "initial_alpha_deg",
            )?,
            initial_time_guess_s: expect_finite(
                sample_or_default(
                    values,
                    "initial_time_guess_s",
                    defaults.initial_time_guess_s,
                ),
                "initial_time_guess_s",
            )?,
            min_time_bound_s,
            max_time_bound_s,
            max_alpha_rate_deg_s: expect_finite(
                sample_or_default(
                    values,
                    "max_alpha_rate_deg_s",
                    defaults.max_alpha_rate_deg_s,
                ),
                "max_alpha_rate_deg_s",
            )?,
            alpha_rate_regularization: expect_finite(
                sample_or_default(
                    values,
                    "alpha_rate_regularization",
                    defaults.alpha_rate_regularization,
                ),
                "alpha_rate_regularization",
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
        ProblemId::OptimalDistanceGlider,
        "Optimal Distance Glider",
        "A planar particle glider with free final time, a launch-angle choice enforced through an initial speed-magnitude constraint, and downrange-distance maximization under altitude, forward-speed, lift-coefficient, and control-rate limits.",
        problem_controls(
            defaults.transcription,
            &SUPPORTED_INTERVALS,
            &SUPPORTED_DEGREES,
            defaults.solver_method,
            defaults.solver,
            vec![
                problem_slider_control(
                    "launch_speed_mps",
                    "Launch Speed",
                    10.0,
                    50.0,
                    0.5,
                    defaults.launch_speed_mps,
                    "m/s",
                    "Initial speed magnitude. The optimizer chooses the launch angle subject to this speed norm.",
                ),
                problem_slider_control(
                    "initial_alpha_deg",
                    "Initial AoA Guess",
                    0.0,
                    15.0,
                    0.25,
                    defaults.initial_alpha_deg,
                    "deg",
                    "Initial guess for the launch angle of attack. This is not a hard boundary condition.",
                ),
                problem_slider_control(
                    "initial_time_guess_s",
                    "Initial Time Guess",
                    0.0,
                    60.0,
                    0.25,
                    defaults.initial_time_guess_s,
                    "s",
                    "Initial guess for the free final time.",
                ),
                problem_slider_control(
                    "min_time_bound_s",
                    "Min Final Time",
                    0.0,
                    500.0,
                    0.5,
                    defaults.min_time_bound_s,
                    "s",
                    "Lower bound on the free final time.",
                ),
                problem_slider_control(
                    "max_time_bound_s",
                    "Max Final Time",
                    1.0,
                    500.0,
                    1.0,
                    defaults.max_time_bound_s,
                    "s",
                    "Upper bound on the free final time.",
                ),
                problem_slider_control(
                    "max_alpha_rate_deg_s",
                    "AoA Rate Limit",
                    5.0,
                    60.0,
                    0.5,
                    defaults.max_alpha_rate_deg_s,
                    "deg/s",
                    "How quickly the glider may retune its angle of attack.",
                ),
                problem_scientific_slider_control(
                    "alpha_rate_regularization",
                    "AoA Rate Weight",
                    0.0,
                    5.0e-2,
                    5.0e-4,
                    defaults.alpha_rate_regularization,
                    "",
                    "Quadratic stage-cost weight on angle-of-attack rate.",
                ),
            ],
        ),
        vec![
            LatexSection {
                title: "Physical State".to_string(),
                entries: vec![r"\mathbf{x} = \begin{bmatrix} x & h & v_x & v_y \end{bmatrix}^{\mathsf T}".to_string()],
            },
            LatexSection {
                title: "Control-State".to_string(),
                entries: vec![r"\mathbf{u} = \begin{bmatrix} \alpha \end{bmatrix}, \qquad \dot{\mathbf{u}} = \begin{bmatrix} \nu_\alpha \end{bmatrix}".to_string()],
            },
            LatexSection {
                title: "Aerodynamic Relations".to_string(),
                entries: vec![
                    r"C_L = 2 \pi \alpha \frac{10}{12}, \qquad -0.5 \le C_L \le 1.5".to_string(),
                    r"C_D = 0.001 + \frac{C_L^2}{\pi \cdot 10 \cdot 0.95}".to_string(),
                    r"L = \tfrac{1}{2} \rho V^2 C_L S_{\mathrm{ref}}, \qquad D = \tfrac{1}{2} \rho V^2 C_D S_{\mathrm{ref}}".to_string(),
                    r"\rho = 1.15, \qquad S_{\mathrm{ref}} = 4.0, \qquad m = 30.0, \qquad \hat{\mathbf{e}}_D = -\frac{1}{V}\begin{bmatrix} v_x \\ v_y \end{bmatrix}, \qquad \hat{\mathbf{e}}_L = \frac{1}{V}\begin{bmatrix} -v_y \\ v_x \end{bmatrix}".to_string(),
                    r"\left(\frac{L}{D}\right)(\alpha) = \frac{C_L}{C_D}".to_string(),
                ],
            },
            LatexSection {
                title: "Objective".to_string(),
                entries: vec![
                    r"J = \int_0^T w_{\dot{\alpha}} \, \nu_\alpha^2 \, dt - x(T)".to_string(),
                ],
            },
            LatexSection {
                title: "Boundary and Path Constraints".to_string(),
                entries: vec![
                    r"x(0) = 0, \qquad h(0) = 1, \qquad v_x(0)^2 + v_y(0)^2 = V_{\mathrm{launch}}^2".to_string(),
                    r"h(t) \ge 0, \qquad v_x(t) \ge 0, \qquad -0.5 \le C_L \le 1.5, \qquad |\nu_\alpha| \le \nu_{\alpha,\max}".to_string(),
                    r"T_{\min} \le T \le T_{\max}".to_string(),
                ],
            },
            LatexSection {
                title: "Differential Equations".to_string(),
                entries: vec![
                    r"V = \sqrt{v_x^2 + v_y^2}, \qquad q_m = \frac{1}{2m} \rho V^2 S_{\mathrm{ref}}".to_string(),
                    r"\dot{x} = v_x".to_string(),
                    r"\dot{h} = v_y".to_string(),
                    r"\dot{v}_x = q_m \left(C_D \hat{e}_{D,x} + C_L \hat{e}_{L,x}\right)".to_string(),
                    r"\dot{v}_y = q_m \left(C_D \hat{e}_{D,y} + C_L \hat{e}_{L,y}\right) - g".to_string(),
                    r"\dot{\alpha} = \nu_\alpha".to_string(),
                ],
            },
        ],
        vec![
            "The lift curve uses CL = 2*pi*alpha*10/12, with the path constraint applied directly as -0.5 <= CL <= 1.5.".to_string(),
            "The particle model carries translational velocity directly, with drag opposing the velocity vector and lift acting orthogonally as (-v_y, v_x)/V.".to_string(),
            "Aerodynamic forces now use L = 0.5*rho*V^2*C_L*S_ref and D = 0.5*rho*V^2*C_D*S_ref with rho = 1.15, S_ref = 4.0 m^2, and mass = 30.0 kg.".to_string(),
            "The launch angle and initial AoA are free: the boundary conditions fix x(0)=0, h(0)=1 m, and ||v(0)|| = V_launch while letting the optimizer choose the vx/vy split and α(0).".to_string(),
            "Final time is free with configurable bounds T_min <= T <= T_max, so the solver can use a terminal flare if that increases downrange distance while respecting h(t) >= 0 and v_x(t) >= 0.".to_string(),
        ],
    )
}
fn cl(alpha: SX) -> SX {
    CL_SLOPE * alpha
}
fn cd(alpha: SX) -> SX {
    let cl_value = cl(alpha);
    0.001 + cl_value.sqr() / (PI * ASPECT_RATIO * EFFICIENCY)
}
fn cl_numeric(alpha: f64) -> f64 {
    CL_SLOPE * alpha
}
fn cd_numeric(alpha: f64) -> f64 {
    let cl_value = cl_numeric(alpha);
    0.001 + cl_value.powi(2) / (PI * ASPECT_RATIO * EFFICIENCY)
}
fn best_glide_alpha() -> f64 {
    (0.001 * PI * ASPECT_RATIO * EFFICIENCY).sqrt() / CL_SLOPE
}
fn glide_ratio_numeric(alpha: f64) -> f64 {
    let cl_value = cl_numeric(alpha);
    let cd_value = cd_numeric(alpha);
    cl_value / cd_value.max(1.0e-6)
}
fn seed_launch_velocity(params: &Params) -> (f64, f64) {
    let path = deg_to_rad(LAUNCH_PATH_SEED_DEG);
    (
        params.launch_speed_mps * path.cos(),
        params.launch_speed_mps * path.sin(),
    )
}

fn glide_slope_deg(vx: f64, vy: f64) -> f64 {
    rad_to_deg(vy.atan2(vx))
}
fn aerodynamic_acceleration_sx(state: &State<SX>, control: &Control<SX>) -> (SX, SX) {
    let speed = (state.vx.sqr() + state.vy.sqr() + SPEED_EPS * SPEED_EPS).sqrt();
    let inv_speed = SX::one() / speed;
    let drag_x = -state.vx * inv_speed;
    let drag_y = -state.vy * inv_speed;
    let lift_x = -state.vy * inv_speed;
    let lift_y = state.vx * inv_speed;
    let force_per_mass = 0.5 * AIR_DENSITY * REFERENCE_AREA * speed.sqr() / GLIDER_MASS;
    let cl_value = cl(control.alpha);
    let cd_value = cd(control.alpha);
    (
        force_per_mass * (cd_value * drag_x + cl_value * lift_x),
        force_per_mass * (cd_value * drag_y + cl_value * lift_y) - GRAVITY,
    )
}
fn model<Scheme>(
    scheme: Scheme,
) -> Ocp<State<SX>, Control<SX>, ModelParams<SX>, Path<SX>, Boundary<SX>, (), Scheme> {
    Ocp::new("glider_problem", scheme)
        .objective_lagrange(
            |_: &State<SX>,
             _: &Control<SX>,
             alpha_rate: &Control<SX>,
             runtime: &ModelParams<SX>| {
                runtime.alpha_rate_weight * alpha_rate.alpha.sqr()
            },
        )
        .objective_mayer(
            |_: &State<SX>,
             _: &Control<SX>,
             terminal: &State<SX>,
             _: &Control<SX>,
             _: &ModelParams<SX>,
             _: &SX| { -terminal.x },
        )
        .ode(
            |state: &State<SX>, control: &Control<SX>, _: &ModelParams<SX>| {
                let (ax, ay) = aerodynamic_acceleration_sx(state, control);
                State {
                    x: state.vx,
                    altitude: state.vy,
                    vx: ax,
                    vy: ay,
                }
            },
        )
        .path_constraints(
            |state: &State<SX>,
             control: &Control<SX>,
             alpha_rate: &Control<SX>,
             _: &ModelParams<SX>| Path {
                altitude: state.altitude,
                vx: state.vx,
                cl: cl(control.alpha),
                alpha_rate: alpha_rate.alpha,
            },
        )
        .boundary_equalities(
            |initial: &State<SX>,
             _: &Control<SX>,
             _: &State<SX>,
             _: &Control<SX>,
             _: &ModelParams<SX>,
             _: &SX| Boundary {
                x0: initial.x,
                altitude0: initial.altitude,
                speed_sq0: initial.vx.sqr() + initial.vy.sqr(),
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
        .expect("glider model should build")
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
                model(MultipleShooting::<DEFAULT_INTERVALS, RK4_SUBSTEPS>)
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
                model(MultipleShooting::<DEFAULT_INTERVALS, RK4_SUBSTEPS>)
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
                ProblemId::OptimalDistanceGlider,
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
        ProblemId::OptimalDistanceGlider,
        PROBLEM_NAME,
        transcription,
        preset,
        eval_options,
        on_progress,
        |options, on_progress| {
            model(MultipleShooting::<DEFAULT_INTERVALS, RK4_SUBSTEPS>)
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

pub(crate) fn problem_entry() -> crate::ProblemEntry {
    crate::ProblemEntry {
        id: ProblemId::OptimalDistanceGlider,
        spec,
        solve_from_map,
        prewarm_from_map,
        solve_with_progress_boxed,
        prewarm_with_progress_boxed,
        compile_cache_statuses,
        benchmark_default_case_with_progress,
    }
}

fn continuous_guess(
    params: &Params,
) -> ContinuousInitialGuess<State<f64>, Control<f64>, ModelParams<f64>> {
    let trim_alpha = best_glide_alpha().max(0.01);
    let (vx0, vy0) = seed_launch_velocity(params);
    ContinuousInitialGuess::Rollout {
        x0: State {
            x: 0.0,
            altitude: INITIAL_ALTITUDE_M,
            vx: vx0,
            vy: vy0,
        },
        u0: Control {
            alpha: deg_to_rad(params.initial_alpha_deg),
        },
        tf: params
            .initial_time_guess_s
            .clamp(params.min_time_bound_s, params.max_time_bound_s),
        controller: Box::new(move |_, _, u, _| Control {
            alpha: 1.5 * (trim_alpha - u.alpha),
        }),
    }
}

fn runtime_spec(
    params: &Params,
) -> OcpRuntimeSpec<ModelParams<f64>, Path<Bounds1D>, Boundary<f64>, (), State<f64>, Control<f64>> {
    OcpRuntimeSpec {
        parameters: ModelParams {
            alpha_rate_weight: params.alpha_rate_regularization,
        },
        beq: Boundary {
            x0: 0.0,
            altitude0: INITIAL_ALTITUDE_M,
            speed_sq0: params.launch_speed_mps.powi(2),
        },
        bineq_bounds: (),
        path_bounds: Path {
            altitude: Bounds1D {
                lower: Some(0.0),
                upper: None,
            },
            vx: Bounds1D {
                lower: Some(0.0),
                upper: None,
            },
            cl: Bounds1D {
                lower: Some(CL_LOWER_BOUND),
                upper: Some(CL_UPPER_BOUND),
            },
            alpha_rate: Bounds1D {
                lower: Some(-deg_to_rad(params.max_alpha_rate_deg_s)),
                upper: Some(deg_to_rad(params.max_alpha_rate_deg_s)),
            },
        },
        tf_bounds: Bounds1D {
            lower: Some(params.min_time_bound_s),
            upper: Some(params.max_time_bound_s),
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
    tf: f64,
    distance: f64,
    terminal_ld: f64,
    peak_altitude: f64,
    trim_cost: f64,
) -> Vec<crate::common::Metric> {
    let mut summary = transcription_metrics(&params.transcription).to_vec();
    summary.extend([
        numeric_metric_with_key(
            MetricKey::Distance,
            "Distance",
            distance,
            format!("{distance:.1} m"),
        ),
        numeric_metric_with_key(MetricKey::FinalTime, "Final Time", tf, format!("{tf:.2} s")),
        metric_with_key(
            MetricKey::BestGlideAlpha,
            "Best-L/D Alpha",
            format!("{:.2} deg", rad_to_deg(best_glide_alpha())),
        ),
        metric_with_key(
            MetricKey::TerminalLiftToDrag,
            "Terminal L/D",
            format!("{terminal_ld:.2}"),
        ),
        metric_with_key(
            MetricKey::PeakAltitude,
            "Peak Altitude",
            format!("{peak_altitude:.2} m"),
        ),
        metric_with_key(MetricKey::TrimCost, "Rate Cost", format!("{trim_cost:.3}")),
    ]);
    summary
}
fn artifact_from_interval_data(
    params: &Params,
    tf: f64,
    distance: f64,
    terminal_ld: f64,
    peak_altitude: f64,
    trim_cost: f64,
    x_arcs: &[IntervalArc<State<f64>>],
    u_arcs: &[IntervalArc<Control<f64>>],
    alpha_rate_series: Vec<crate::common::TimeSeries>,
    x: Vec<f64>,
    y: Vec<f64>,
    notes: Vec<String>,
) -> SolveArtifact {
    SolveArtifact::new(
        "Optimal Distance Glider",
        artifact_summary(params, tf, distance, terminal_ld, peak_altitude, trim_cost),
        SolverReport::placeholder(),
        vec![
            chart(
                "Downrange x",
                "Downrange x (m)",
                interval_arc_series("x (m)", x_arcs, PlotMode::LinesMarkers, |state| state.x),
            ),
            chart("Altitude", "Altitude (m)", {
                let mut series_out =
                    interval_arc_series("Altitude (m)", x_arcs, PlotMode::LinesMarkers, |state| {
                        state.altitude
                    });
                series_out.extend(interval_arc_bound_series(
                    x_arcs,
                    Some(0.0),
                    None,
                    PlotMode::Lines,
                ));
                series_out
            }),
            chart("Horizontal Velocity", "v_x (m/s)", {
                let mut series_out =
                    interval_arc_series("v_x (m/s)", x_arcs, PlotMode::LinesMarkers, |state| {
                        state.vx
                    });
                series_out.extend(interval_arc_bound_series(
                    x_arcs,
                    Some(0.0),
                    None,
                    PlotMode::Lines,
                ));
                series_out
            }),
            chart(
                "Vertical Velocity",
                "v_y (m/s)",
                interval_arc_series("v_y (m/s)", x_arcs, PlotMode::LinesMarkers, |state| {
                    state.vy
                }),
            ),
            chart(
                "Velocity Norm",
                "‖v‖ (m/s)",
                interval_arc_series("‖v‖ (m/s)", x_arcs, PlotMode::LinesMarkers, |state| {
                    (state.vx * state.vx + state.vy * state.vy).sqrt()
                }),
            ),
            chart(
                "Glide Slope",
                "Glide Slope (deg)",
                interval_arc_series(
                    "Glide Slope (deg)",
                    x_arcs,
                    PlotMode::LinesMarkers,
                    |state| glide_slope_deg(state.vx, state.vy),
                ),
            ),
            chart("Lift Coefficient", "C_L (-)", {
                let mut series_out =
                    interval_arc_series("C_L (-)", u_arcs, PlotMode::LinesMarkers, |control| {
                        cl_numeric(control.alpha)
                    });
                series_out.extend(interval_arc_bound_series(
                    u_arcs,
                    Some(CL_LOWER_BOUND),
                    Some(CL_UPPER_BOUND),
                    PlotMode::Lines,
                ));
                series_out
            }),
            chart(
                "Angle of Attack",
                "Angle of Attack (deg)",
                interval_arc_series("AoA (deg)", u_arcs, PlotMode::LinesMarkers, |control| {
                    rad_to_deg(control.alpha)
                }),
            ),
            chart(
                "Angle of Attack Rate",
                "AoA Rate (deg/s)",
                alpha_rate_series,
            ),
        ],
        Scene2D {
            title: "Glide Path".to_string(),
            x_label: "Downrange x (m)".to_string(),
            y_label: "Altitude (m)".to_string(),
            paths: vec![ScenePath {
                name: "Trajectory".to_string(),
                x,
                y,
            }],
            circles: Vec::new(),
            arrows: Vec::new(),
            animation: None,
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
    let tf = trajectories.tf;
    let times = node_times::<N>(tf);
    let mut altitude = Vec::with_capacity(N + 1);
    let mut alpha_rate = Vec::with_capacity(N + 1);
    let mut x = Vec::with_capacity(N + 1);
    let mut y = Vec::with_capacity(N + 1);
    for index in 0..N {
        let state = &trajectories.x.nodes[index];
        let control = &trajectories.u.nodes[index];
        let rate = &trajectories.dudt[index];
        altitude.push(state.altitude);
        alpha_rate.push(rad_to_deg(rate.alpha));
        x.push(state.x);
        y.push(state.altitude);
        let _ = control;
    }
    let terminal = &trajectories.x.terminal;
    let terminal_control = &trajectories.u.terminal;
    altitude.push(terminal.altitude);
    alpha_rate.push(*alpha_rate.last().unwrap_or(&0.0));
    x.push(terminal.x);
    y.push(terminal.altitude);
    let trim_cost = trapezoid_integral(
        &times,
        &alpha_rate
            .iter()
            .map(|value| value * value)
            .collect::<Vec<_>>(),
    );
    let mut alpha_rate_series = segmented_series(
        "AoA Rate (deg/s)",
        x_arcs.iter().enumerate().map(|(interval, arc)| {
            (
                arc.times.clone(),
                vec![rad_to_deg(trajectories.dudt[interval].alpha); arc.times.len()],
            )
        }),
        PlotMode::LinesMarkers,
    );
    alpha_rate_series.extend(segmented_bound_series(
        x_arcs.iter().map(|arc| arc.times.clone()),
        Some(-params.max_alpha_rate_deg_s),
        Some(params.max_alpha_rate_deg_s),
        PlotMode::Lines,
    ));
    artifact_from_interval_data(
        params,
        tf,
        terminal.x,
        glide_ratio_numeric(terminal_control.alpha),
        altitude.iter().fold(f64::NEG_INFINITY, |acc, value| acc.max(*value)),
        trim_cost,
        x_arcs,
        u_arcs,
        alpha_rate_series,
        x,
        y,
        vec![
            "The particle model uses velocity states directly, with drag aligned opposite motion and lift using the orthogonal direction (-v_y, v_x)/V.".to_string(),
            "Lift and drag are computed from dynamic pressure using rho = 1.15 kg/m^3, S_ref = 4.0 m^2, and mass = 30.0 kg.".to_string(),
            "The bounded aerodynamic path constraint is applied directly to lift coefficient, with -0.5 <= C_L <= 1.5 rendered as red overlays.".to_string(),
            "Altitude and forward speed are constrained as h(t) >= 0 and v_x(t) >= 0 throughout the trajectory, with free final time 1 <= T <= 500 s.".to_string(),
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
    let mut x = Vec::with_capacity(N + 1);
    let mut y = Vec::with_capacity(N + 1);
    let mut peak_altitude = f64::NEG_INFINITY;
    for index in 0..N {
        let state = &trajectories.x.nodes[index];
        x.push(state.x);
        y.push(state.altitude);
        peak_altitude = peak_altitude.max(state.altitude);
    }
    x.push(trajectories.x.terminal.x);
    y.push(trajectories.x.terminal.altitude);
    peak_altitude = peak_altitude.max(trajectories.x.terminal.altitude);
    let root_times = dudt_arcs
        .iter()
        .flat_map(|arc| arc.times.iter().copied())
        .collect::<Vec<_>>();
    let root_rates = (0..N)
        .flat_map(|interval| {
            (0..K)
                .map(move |root| rad_to_deg(trajectories.root_dudt.intervals[interval][root].alpha))
        })
        .collect::<Vec<_>>();
    let trim_cost = if root_times.len() >= 2 {
        trapezoid_integral(
            &root_times,
            &root_rates
                .iter()
                .map(|value| value * value)
                .collect::<Vec<_>>(),
        )
    } else {
        0.0
    };
    let mut alpha_rate_series = interval_arc_series(
        "AoA Rate (deg/s)",
        &dudt_arcs,
        PlotMode::LinesMarkers,
        |rate| rad_to_deg(rate.alpha),
    );
    alpha_rate_series.extend(interval_arc_bound_series(
        &dudt_arcs,
        Some(-params.max_alpha_rate_deg_s),
        Some(params.max_alpha_rate_deg_s),
        PlotMode::Lines,
    ));
    artifact_from_interval_data(
        params,
        trajectories.tf,
        trajectories.x.terminal.x,
        glide_ratio_numeric(trajectories.u.terminal.alpha),
        peak_altitude,
        trim_cost,
        &x_arcs,
        &u_arcs,
        alpha_rate_series,
        x,
        y,
        vec![
            "The particle model uses velocity states directly, with drag aligned opposite motion and lift using the orthogonal direction (-v_y, v_x)/V.".to_string(),
            "Lift and drag are computed from dynamic pressure using rho = 1.15 kg/m^3, S_ref = 4.0 m^2, and mass = 30.0 kg.".to_string(),
            "Each collocation interval is rendered as its own start-root-end arc, while the AoA-rate decision variable is only shown at collocation nodes.".to_string(),
            "The bounded aerodynamic path constraint is applied directly to lift coefficient, with -0.5 <= C_L <= 1.5 rendered as red overlays.".to_string(),
            "Altitude and forward speed are constrained as h(t) >= 0 and v_x(t) >= 0 throughout the trajectory, with free final time 1 <= T <= 500 s.".to_string(),
        ],
    )
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glider_converges_to_a_reasonable_glide() {
        let artifact = solve(&Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            ..Params::default()
        })
        .expect("glider solve should succeed");
        let distance_metric = crate::find_metric(&artifact.summary, crate::MetricKey::Distance)
            .expect("distance metric should exist");
        let distance = distance_metric
            .numeric_value
            .expect("distance metric should carry its numeric value");
        let final_time = crate::find_metric(&artifact.summary, crate::MetricKey::FinalTime)
            .expect("final time metric should exist")
            .numeric_value
            .expect("final time metric should carry its numeric value");
        assert!(distance > 25.0, "glider should travel forward");
        assert!(
            (Params::default().min_time_bound_s..=Params::default().max_time_bound_s)
                .contains(&final_time),
            "free final time should stay within the configured bounds"
        );
    }

    #[test]
    #[ignore = "manual profiling helper"]
    fn profile_reduced_direct_collocation_symbolic_setup() {
        let family = optimal_control::CollocationFamily::RadauIIA;
        let ocp = model(optimal_control::DirectCollocation::<6, 2> { family });
        for (label, symbolic_functions) in [
            (
                "inline_all",
                optimal_control::OcpSymbolicFunctionOptions::inline_all(),
            ),
            (
                "default",
                optimal_control::OcpSymbolicFunctionOptions::default(),
            ),
            (
                "function_inline_at_call",
                optimal_control::OcpSymbolicFunctionOptions::function_all_with_call_policy(
                    optimization::CallPolicy::InlineAtCall,
                ),
            ),
        ] {
            let started = std::time::Instant::now();
            let mut symbolic_metadata = None;
            let compiled = ocp
                .compile_jit_with_ocp_options_and_progress_callback(
                    optimal_control::OcpCompileOptions {
                        function_options: optimization::FunctionCompileOptions::from(
                            optimization::LlvmOptimizationLevel::O0,
                        ),
                        symbolic_functions,
                    },
                    |progress| {
                        if let optimal_control::OcpCompileProgress::SymbolicReady(metadata) =
                            progress
                        {
                            symbolic_metadata = Some(metadata);
                        }
                    },
                )
                .expect("compile should succeed");
            println!(
                "{label}: total={:?} symbolic_ready={:?} setup_profile={:?} stats={:?}",
                started.elapsed(),
                symbolic_metadata,
                compiled.backend_compile_report().setup_profile,
                compiled.backend_compile_report().stats,
            );
        }
    }

    #[test]
    #[ignore = "manual profiling helper"]
    fn profile_direct_collocation_symbolic_setup() {
        let family = Params::default().transcription.collocation_family;
        let ocp = model(
            optimal_control::DirectCollocation::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE> {
                family,
            },
        );
        let started = std::time::Instant::now();
        let compiled = ocp
            .compile_jit_with_ocp_options(optimal_control::OcpCompileOptions {
                function_options: optimization::FunctionCompileOptions::from(
                    optimization::LlvmOptimizationLevel::O0,
                ),
                symbolic_functions: optimal_control::OcpSymbolicFunctionOptions::default(),
            })
            .expect("compile should succeed");
        println!(
            "glider dc: total={:?} setup_profile={:?} stats={:?}",
            started.elapsed(),
            compiled.backend_compile_report().setup_profile,
            compiled.backend_compile_report().stats,
        );
    }
}
