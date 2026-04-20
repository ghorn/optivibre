use crate::common::{
    CompileCacheStatus, CompileProgressInfo, CompileProgressUpdate, ContinuousInitialGuess,
    FromMap, LatexSection, MetricKey, OcpRuntimeSpec, OcpSxFunctionConfig, PlotMode, ProblemId,
    ProblemSpec, Scene2D, ScenePath, SolveArtifact, SolveStreamEvent, SolverConfig, SolverMethod,
    SolverReport, StandardOcpParams, TranscriptionConfig, chart, default_solver_config,
    default_solver_method, default_transcription, deg_to_rad, direct_collocation_runtime_from_spec,
    expect_finite, interval_arc_bound_series, interval_arc_series, metric_with_key,
    multiple_shooting_runtime_from_spec, node_times, numeric_metric_with_key,
    ocp_sx_function_config_from_map, problem_controls, problem_scientific_slider_control,
    problem_slider_control, problem_spec, rad_to_deg, sample_or_default, segmented_bound_series,
    segmented_series, select_control, solver_config_from_map, solver_method_from_map,
    transcription_from_map, transcription_metrics, trapezoid_integral,
};
use anyhow::{Result, anyhow};
use optimal_control::{
    Bounds1D, CompiledDirectCollocationOcp, CompiledMultipleShootingOcp, DirectCollocation,
    DirectCollocationRuntimeValues, DirectCollocationTimeGrid, DirectCollocationTrajectories,
    InterpolatedTrajectory, IntervalArc, MultipleShooting, MultipleShootingRuntimeValues,
    MultipleShootingTrajectories, Ocp, OcpScaling, direct_collocation_root_arcs,
    direct_collocation_state_like_arcs,
};
use serde::Serialize;
use std::f64::consts::PI;
use sx_core::SX;
const RK4_SUBSTEPS: usize = 2;
const DEFAULT_INTERVALS: usize = 50;
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
const MIN_FLIGHT_TIME_S: f64 = 1.0;
const MAX_FLIGHT_TIME_S: f64 = 500.0;
const GLIDER_OBJECTIVE_SCALE: f64 = 100.0;
const GLIDER_X_SCALE_M: f64 = 300.0;
const GLIDER_ALTITUDE_SCALE_M: f64 = 50.0;
const GLIDER_ALPHA_SCALE_DEG: f64 = 5.0;
const GLIDER_ALPHA_RATE_SCALE_DEG_S: f64 = 4.0;
const GLIDER_FINAL_TIME_SCALE_S: f64 = 100.0;
const GLIDER_CL_SCALE: f64 = 1.0;
const GLIDER_X0_SCALE_M: f64 = 1.0;
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
    pub scaling_enabled: bool,
    pub solver_method: SolverMethod,
    pub solver: SolverConfig,
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
            alpha_rate_regularization: 5.0e-1,
            scaling_enabled: true,
            solver_method: default_solver_method(),
            solver: default_solver_config(),
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

impl crate::common::HasOcpSxFunctionConfig for Params {
    fn sx_functions_mut(&mut self) -> &mut OcpSxFunctionConfig {
        &mut self.sx_functions
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
        let scaling_enabled = match sample_or_default(
            values,
            "scaling_enabled",
            if defaults.scaling_enabled { 1.0 } else { 0.0 },
        ) {
            value if (value - 1.0).abs() <= 1.0e-9 => true,
            value if value.abs() <= 1.0e-9 => false,
            value => {
                return Err(anyhow!(
                    "scaling_enabled must be either 0 (Off) or 1 (On), got {value}"
                ));
            }
        };
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
            scaling_enabled,
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
                    1.0,
                    5.0e-4,
                    defaults.alpha_rate_regularization,
                    "",
                    "Quadratic stage-cost weight on angle-of-attack rate.",
                ),
                select_control(
                    "scaling_enabled",
                    "Scaling",
                    if defaults.scaling_enabled { 1.0 } else { 0.0 },
                    "",
                    "Enable glider-specific NLP/OCP scaling in the solver while keeping the UI in physical units.",
                    &[(1.0, "On"), (0.0, "Off")],
                    crate::common::ControlSection::Problem,
                    crate::common::ControlVisibility::Always,
                    crate::common::ControlSemantic::ProblemParameter,
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
                    r"h(t) \ge 0, \qquad v_x(t) \ge 1, \qquad -0.5 \le C_L \le 1.5, \qquad |\nu_\alpha| \le \nu_{\alpha,\max}".to_string(),
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
fn glide_slope_deg(vx: f64, vy: f64) -> f64 {
    rad_to_deg(vy.atan2(vx))
}

fn smoothstep(s: f64) -> f64 {
    let t = s.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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

fn level_flight_alpha(speed: f64) -> f64 {
    let dynamic_pressure = 0.5 * AIR_DENSITY * REFERENCE_AREA * speed.max(1.0).powi(2);
    let required_cl = (GLIDER_MASS * GRAVITY) / dynamic_pressure.max(1.0);
    (required_cl / CL_SLOPE).clamp(CL_LOWER_BOUND / CL_SLOPE, CL_UPPER_BOUND / CL_SLOPE)
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

pub fn validate_derivatives(
    params: &Params,
    request: &crate::common::DerivativeCheckRequest,
) -> Result<crate::common::ProblemDerivativeCheck> {
    crate::common::validate_standard_ocp_derivatives(
        ProblemId::OptimalDistanceGlider,
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
    let mut params = Params::from_map(&request.values)?;
    crate::common::apply_derivative_request_overrides(&mut params, request);
    validate_derivatives(&params, request)
}

pub(crate) fn problem_entry() -> crate::ProblemEntry {
    crate::ProblemEntry {
        id: ProblemId::OptimalDistanceGlider,
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

fn continuous_guess(
    params: &Params,
) -> ContinuousInitialGuess<State<f64>, Control<f64>, ModelParams<f64>> {
    let tf = params
        .initial_time_guess_s
        .clamp(params.min_time_bound_s, params.max_time_bound_s);
    let sample_count = 2 * params.transcription.intervals + 1;
    let dt = tf / (sample_count as f64 - 1.0);
    let times = (0..sample_count)
        .map(|index| index as f64 * dt)
        .collect::<Vec<_>>();
    let launch_angle = deg_to_rad(2.5);
    let vx0 = params.launch_speed_mps * launch_angle.cos();
    let vy0 = params.launch_speed_mps * launch_angle.sin();
    let touchdown_time = (0.25 * tf).clamp(20.0, 40.0).min(tf);
    let touchdown_vx = 24.0_f64.min(vx0.max(12.0));
    let terminal_vx = 8.0;
    let altitude = times
        .iter()
        .map(|time| {
            if *time >= touchdown_time {
                0.0
            } else {
                let tau = *time / touchdown_time.max(f64::MIN_POSITIVE);
                let h00 = 2.0 * tau.powi(3) - 3.0 * tau.powi(2) + 1.0;
                let h10 = tau.powi(3) - 2.0 * tau.powi(2) + tau;
                let h01 = -2.0 * tau.powi(3) + 3.0 * tau.powi(2);
                let h11 = tau.powi(3) - tau.powi(2);
                let height =
                    h00 * INITIAL_ALTITUDE_M + h10 * touchdown_time * vy0 + h01 * 0.0 + h11 * 0.0;
                height.max(0.0)
            }
        })
        .collect::<Vec<_>>();
    let vy = finite_difference(&altitude, dt);
    let vx = times
        .iter()
        .map(|time| {
            if *time <= touchdown_time {
                let tau = *time / touchdown_time.max(f64::MIN_POSITIVE);
                vx0 + (touchdown_vx - vx0) * smoothstep(tau)
            } else {
                let tau = (*time - touchdown_time) / (tf - touchdown_time).max(f64::MIN_POSITIVE);
                touchdown_vx + (terminal_vx - touchdown_vx) * smoothstep(tau)
            }
        })
        .collect::<Vec<_>>();
    let mut x = vec![0.0; sample_count];
    for index in 1..sample_count {
        x[index] = x[index - 1] + 0.5 * dt * (vx[index - 1] + vx[index]);
    }
    let alpha_values = times
        .iter()
        .zip(vx.iter())
        .map(|(time, vx)| {
            if *time <= touchdown_time {
                let tau = *time / touchdown_time.max(f64::MIN_POSITIVE);
                deg_to_rad(params.initial_alpha_deg)
                    + (best_glide_alpha() - deg_to_rad(params.initial_alpha_deg)) * smoothstep(tau)
            } else {
                level_flight_alpha(*vx)
            }
        })
        .collect::<Vec<_>>();
    let alpha_rate = finite_difference(&alpha_values, dt);
    ContinuousInitialGuess::Interpolated(InterpolatedTrajectory {
        sample_times: times,
        x_samples: x
            .iter()
            .zip(altitude.iter())
            .zip(vx.iter())
            .zip(vy.iter())
            .map(|(((x, altitude), vx), vy)| State {
                x: *x,
                altitude: *altitude,
                vx: *vx,
                vy: *vy,
            })
            .collect(),
        u_samples: alpha_values
            .iter()
            .map(|alpha| Control { alpha: *alpha })
            .collect(),
        dudt_samples: alpha_rate
            .iter()
            .map(|alpha| Control { alpha: *alpha })
            .collect(),
        tf,
    })
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
                lower: Some(1.0),
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

fn glider_scaling(params: &Params) -> OcpScaling<ModelParams<f64>, State<f64>, Control<f64>> {
    OcpScaling {
        objective: GLIDER_OBJECTIVE_SCALE,
        state: State {
            x: GLIDER_X_SCALE_M,
            altitude: GLIDER_ALTITUDE_SCALE_M,
            vx: params.launch_speed_mps,
            vy: params.launch_speed_mps,
        },
        control: Control {
            alpha: deg_to_rad(GLIDER_ALPHA_SCALE_DEG),
        },
        control_rate: Control {
            alpha: deg_to_rad(GLIDER_ALPHA_RATE_SCALE_DEG_S),
        },
        final_time: GLIDER_FINAL_TIME_SCALE_S,
        parameters: ModelParams {
            alpha_rate_weight: 1.0,
        },
        path: vec![
            GLIDER_ALTITUDE_SCALE_M,
            params.launch_speed_mps,
            GLIDER_CL_SCALE,
            deg_to_rad(GLIDER_ALPHA_RATE_SCALE_DEG_S),
        ],
        boundary_equalities: vec![
            GLIDER_X0_SCALE_M,
            INITIAL_ALTITUDE_M,
            params.launch_speed_mps.powi(2),
        ],
        boundary_inequalities: Vec::new(),
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
    let mut runtime = multiple_shooting_runtime_from_spec(runtime_spec(params));
    runtime.scaling = params.scaling_enabled.then(|| glider_scaling(params));
    runtime
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
    let mut runtime = direct_collocation_runtime_from_spec(runtime_spec(params));
    runtime.scaling = params.scaling_enabled.then(|| glider_scaling(params));
    runtime
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
    let path_bounds = runtime_spec(params).path_bounds;
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
                    path_bounds.altitude.lower,
                    path_bounds.altitude.upper,
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
                    path_bounds.vx.lower,
                    path_bounds.vx.upper,
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
            "Altitude and forward speed are constrained as h(t) >= 0 and v_x(t) >= 1 throughout the trajectory, with free final time 1 <= T <= 500 s.".to_string(),
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
            "Altitude and forward speed are constrained as h(t) >= 0 and v_x(t) >= 1 throughout the trajectory, with free final time 1 <= T <= 500 s.".to_string(),
        ],
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use serde::de::DeserializeOwned;
    use spral_ssids::{
        NativeSpral, NumericFactorOptions, OrderingStrategy, SsidsOptions, SymmetricCscMatrix,
        analyse as spral_analyse, factorize as spral_factorize,
    };
    use std::fs;
    use std::path::Path;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    #[derive(Debug)]
    struct GliderLinearDebugDump {
        matrix_dimension: usize,
        x_dimension: usize,
        inequality_dimension: usize,
        equality_dimension: usize,
        p_offset: usize,
        lambda_offset: usize,
        z_offset: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f64>,
        rhs: Vec<f64>,
        slack: Vec<f64>,
        multipliers: Vec<f64>,
    }

    #[derive(Debug)]
    struct ExactAugmentedReplay {
        factor_time: Duration,
        solve_time: Duration,
        residual_inf: f64,
        solution_inf: f64,
        solution: Vec<f64>,
        inertia: String,
    }

    #[derive(Debug)]
    struct AugmentedStepBlocks {
        dx: Vec<f64>,
        ds: Vec<f64>,
        d_lambda: Vec<f64>,
        dz: Vec<f64>,
        p: Vec<f64>,
    }

    fn parse_dump_value<T>(text: &str, prefix: &str) -> T
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Debug,
    {
        text.lines()
            .find_map(|line| line.strip_prefix(prefix))
            .expect("expected dump key to be present")
            .parse::<T>()
            .expect("expected dump scalar to parse")
    }

    fn parse_dump_vec<T>(text: &str, prefix: &str) -> Vec<T>
    where
        T: DeserializeOwned,
    {
        let value = text
            .lines()
            .find_map(|line| line.strip_prefix(prefix))
            .expect("expected dump vector to be present");
        serde_json::from_str(value).expect("expected dump vector to parse")
    }

    fn load_glider_linear_debug_dump(path: &Path) -> GliderLinearDebugDump {
        let text = fs::read_to_string(path).expect("expected linear debug dump to exist");
        GliderLinearDebugDump {
            matrix_dimension: parse_dump_value(&text, "matrix_dimension="),
            x_dimension: parse_dump_value(&text, "x_dimension="),
            inequality_dimension: parse_dump_value(&text, "inequality_dimension="),
            equality_dimension: parse_dump_value(&text, "equality_dimension="),
            p_offset: parse_dump_value(&text, "p_offset="),
            lambda_offset: parse_dump_value(&text, "lambda_offset="),
            z_offset: parse_dump_value(&text, "z_offset="),
            col_ptrs: parse_dump_vec(&text, "col_ptrs="),
            row_indices: parse_dump_vec(&text, "row_indices="),
            values: parse_dump_vec(&text, "values="),
            rhs: parse_dump_vec(&text, "rhs="),
            slack: parse_dump_vec(&text, "slack="),
            multipliers: parse_dump_vec(&text, "multipliers="),
        }
    }

    fn symmetric_lower_csc_mat_vec(dump: &GliderLinearDebugDump, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; dump.matrix_dimension];
        for col in 0..dump.matrix_dimension {
            for idx in dump.col_ptrs[col]..dump.col_ptrs[col + 1] {
                let row = dump.row_indices[idx];
                let value = dump.values[idx];
                y[row] += value * x[col];
                if row != col {
                    y[col] += value * x[row];
                }
            }
        }
        y
    }

    fn residual_inf(dump: &GliderLinearDebugDump, solution: &[f64]) -> f64 {
        symmetric_lower_csc_mat_vec(dump, solution)
            .into_iter()
            .zip(dump.rhs.iter().copied())
            .map(|(lhs, rhs)| (rhs - lhs).abs())
            .fold(0.0_f64, f64::max)
    }

    fn residual_vector(dump: &GliderLinearDebugDump, solution: &[f64]) -> Vec<f64> {
        symmetric_lower_csc_mat_vec(dump, solution)
            .into_iter()
            .zip(dump.rhs.iter().copied())
            .map(|(lhs, rhs)| rhs - lhs)
            .collect::<Vec<_>>()
    }

    fn solution_inf(solution: &[f64]) -> f64 {
        solution
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()))
    }

    fn delta_inf(lhs: &[f64], rhs: &[f64]) -> f64 {
        lhs.iter()
            .zip(rhs.iter())
            .map(|(lhs_i, rhs_i)| (lhs_i - rhs_i).abs())
            .fold(0.0_f64, f64::max)
    }

    fn sorted_glider_dump_paths(dir: &Path) -> Vec<std::path::PathBuf> {
        let mut paths = fs::read_dir(dir)
            .expect("dump directory should be readable")
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with("nlip_kkt_iter_") && name.ends_with(".txt")
                    })
            })
            .collect::<Vec<_>>();
        paths.sort();
        paths
    }

    fn augmented_step_blocks(
        dump: &GliderLinearDebugDump,
        solution: &[f64],
    ) -> AugmentedStepBlocks {
        let dx = solution[..dump.x_dimension].to_vec();
        let p = solution[dump.p_offset..dump.p_offset + dump.inequality_dimension].to_vec();
        let d_lambda =
            solution[dump.lambda_offset..dump.lambda_offset + dump.equality_dimension].to_vec();
        let dz = solution[dump.z_offset..dump.z_offset + dump.inequality_dimension].to_vec();
        let ds = p
            .iter()
            .zip(dump.slack.iter())
            .zip(dump.multipliers.iter())
            .map(|((p_i, slack_i), multiplier_i)| {
                let scaling = (slack_i.max(1e-16) / multiplier_i.max(1e-16)).sqrt();
                p_i * scaling
            })
            .collect::<Vec<_>>();
        AugmentedStepBlocks {
            dx,
            ds,
            d_lambda,
            dz,
            p,
        }
    }

    fn replay_rust_augmented_spral(dump: &GliderLinearDebugDump) -> ExactAugmentedReplay {
        let structure = SymmetricCscMatrix::new(
            dump.matrix_dimension,
            &dump.col_ptrs,
            &dump.row_indices,
            None,
        )
        .expect("dumped augmented CSC should validate");
        let numeric = SymmetricCscMatrix::new(
            dump.matrix_dimension,
            &dump.col_ptrs,
            &dump.row_indices,
            Some(&dump.values),
        )
        .expect("dumped augmented CSC values should validate");
        let (symbolic, _) = spral_analyse(
            structure,
            &SsidsOptions {
                ordering: OrderingStrategy::ApproximateMinimumDegree,
            },
        )
        .expect("rust spral analyse should succeed on dumped KKT");
        let factor_started = Instant::now();
        let (mut factor, _) = spral_factorize(numeric, &symbolic, &NumericFactorOptions::default())
            .expect("rust spral factorization should succeed on dumped KKT");
        let factor_time = factor_started.elapsed();
        let solve_started = Instant::now();
        let mut solution = factor
            .solve(&dump.rhs)
            .expect("rust spral solve should succeed on dumped KKT");
        let mut solve_time = solve_started.elapsed();
        for _ in 0..10 {
            let residual = residual_vector(dump, &solution);
            if residual.iter().all(|value| value.abs() <= f64::EPSILON) {
                break;
            }
            let correction_started = Instant::now();
            let correction = factor
                .solve(&residual)
                .expect("rust spral iterative refinement should succeed on dumped KKT");
            solve_time += correction_started.elapsed();
            for (solution_i, correction_i) in solution.iter_mut().zip(correction.iter()) {
                *solution_i += correction_i;
            }
        }
        ExactAugmentedReplay {
            factor_time,
            solve_time,
            residual_inf: residual_inf(dump, &solution),
            solution_inf: solution_inf(&solution),
            solution,
            inertia: format!("{:?}", factor.inertia()),
        }
    }

    fn replay_native_augmented_spral(dump: &GliderLinearDebugDump) -> ExactAugmentedReplay {
        let structure = SymmetricCscMatrix::new(
            dump.matrix_dimension,
            &dump.col_ptrs,
            &dump.row_indices,
            None,
        )
        .expect("dumped augmented CSC should validate");
        let numeric = SymmetricCscMatrix::new(
            dump.matrix_dimension,
            &dump.col_ptrs,
            &dump.row_indices,
            Some(&dump.values),
        )
        .expect("dumped augmented CSC values should validate");
        let native = NativeSpral::load().expect("native SPRAL should be available locally");
        let mut session = native
            .analyse(structure)
            .expect("native spral analyse should succeed on dumped KKT");
        let factor_started = Instant::now();
        let factor_info = session
            .factorize(numeric)
            .expect("native spral factorization should succeed on dumped KKT");
        let factor_time = factor_started.elapsed();
        let solve_started = Instant::now();
        let mut solution = session
            .solve(&dump.rhs)
            .expect("native spral solve should succeed on dumped KKT");
        let mut solve_time = solve_started.elapsed();
        for _ in 0..10 {
            let residual = residual_vector(dump, &solution);
            if residual.iter().all(|value| value.abs() <= f64::EPSILON) {
                break;
            }
            let correction_started = Instant::now();
            let correction = session
                .solve(&residual)
                .expect("native spral iterative refinement should succeed on dumped KKT");
            solve_time += correction_started.elapsed();
            for (solution_i, correction_i) in solution.iter_mut().zip(correction.iter()) {
                *solution_i += correction_i;
            }
        }
        ExactAugmentedReplay {
            factor_time,
            solve_time,
            residual_inf: residual_inf(dump, &solution),
            solution_inf: solution_inf(&solution),
            solution,
            inertia: format!("{:?}", factor_info.inertia),
        }
    }

    fn native_spral_debug_result(
        report: &optimization::InteriorPointLinearDebugReport,
    ) -> &optimization::InteriorPointLinearDebugBackendResult {
        report
            .results
            .iter()
            .find(|result| {
                result.solver == optimization::InteriorPointLinearSolver::NativeSpralSsids
            })
            .expect("expected native SPRAL comparison result")
    }

    #[test]
    fn glider_runtime_scaling_defaults_match_problem_tuning() {
        let params = Params::default();
        let runtime = ms_runtime::<2>(&params);
        let scaling = runtime
            .scaling
            .expect("glider scaling should be enabled by default");

        assert_abs_diff_eq!(scaling.objective, GLIDER_OBJECTIVE_SCALE, epsilon = 1e-12);
        assert_abs_diff_eq!(scaling.state.x, GLIDER_X_SCALE_M, epsilon = 1e-12);
        assert_abs_diff_eq!(
            scaling.state.altitude,
            GLIDER_ALTITUDE_SCALE_M,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(scaling.state.vx, params.launch_speed_mps, epsilon = 1e-12);
        assert_abs_diff_eq!(scaling.state.vy, params.launch_speed_mps, epsilon = 1e-12);
        assert_abs_diff_eq!(
            scaling.control.alpha,
            deg_to_rad(GLIDER_ALPHA_SCALE_DEG),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            scaling.control_rate.alpha,
            deg_to_rad(GLIDER_ALPHA_RATE_SCALE_DEG_S),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            scaling.final_time,
            GLIDER_FINAL_TIME_SCALE_S,
            epsilon = 1e-12
        );
        assert_eq!(
            scaling.path,
            vec![
                GLIDER_ALTITUDE_SCALE_M,
                params.launch_speed_mps,
                GLIDER_CL_SCALE,
                deg_to_rad(GLIDER_ALPHA_RATE_SCALE_DEG_S),
            ]
        );
        assert_eq!(
            scaling.boundary_equalities,
            vec![
                GLIDER_X0_SCALE_M,
                INITIAL_ALTITUDE_M,
                params.launch_speed_mps.powi(2),
            ]
        );
        assert!(scaling.boundary_inequalities.is_empty());
    }

    #[test]
    fn glider_runtime_scaling_can_be_disabled() {
        let runtime = ms_runtime::<2>(&Params {
            scaling_enabled: false,
            ..Params::default()
        });
        assert!(runtime.scaling.is_none());
    }

    #[test]
    fn reduced_scaled_glider_direct_collocation_line_search_merit_progresses() {
        const N: usize = 8;
        const K: usize = DEFAULT_COLLOCATION_DEGREE;
        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            scaling_enabled: true,
            ..Params::default()
        };
        let family = params.transcription.collocation_family;
        let runtime = dc_runtime::<N, K>(&params);
        let compiled = model(optimal_control::DirectCollocation::<N, K> { family })
            .compile_jit_with_ocp_options(crate::common::ocp_compile_options(
                crate::common::interactive_direct_collocation_opt_level(),
                params.sx_functions,
            ))
            .expect("reduced glider direct collocation should compile");

        let mut sqp_options = crate::common::sqp_options(&params.solver);
        sqp_options.globalization = optimization::SqpGlobalization::LineSearchMerit(
            optimization::LineSearchMeritOptions::default(),
        );
        sqp_options.verbose = false;
        sqp_options.max_iters = 6;

        let mut saw_iterate = false;
        let result = compiled.solve_sqp_with_callback(&runtime, &sqp_options, |snapshot| {
            saw_iterate = true;
            let _ = snapshot;
        });
        match result {
            Ok(_) => {}
            Err(optimization::ClarabelSqpError::MaxIterations { .. }) => {}
            Err(optimization::ClarabelSqpError::QpSolve {
                status: optimization::SqpQpRawStatus::InsufficientProgress,
                ..
            }) if saw_iterate => {}
            Err(other) => panic!(
                "scaled direct-collocation SQP with line-search merit should at least progress, got {other:?}"
            ),
        }

        assert!(saw_iterate, "glider SQP should emit at least one iterate");
    }

    #[test]
    fn glider_converges_to_a_reasonable_glide() {
        const N: usize = 8;
        const K: usize = DEFAULT_COLLOCATION_DEGREE;

        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            scaling_enabled: true,
            ..Params::default()
        };
        let family = params.transcription.collocation_family;
        let runtime = dc_runtime::<N, K>(&params);
        let compiled = model(optimal_control::DirectCollocation::<N, K> { family })
            .compile_jit_with_ocp_options(crate::common::ocp_compile_options(
                crate::common::interactive_direct_collocation_opt_level(),
                params.sx_functions,
            ))
            .expect("reduced glider direct collocation should compile");

        let mut sqp_options = crate::common::sqp_options(&params.solver);
        sqp_options.globalization = optimization::SqpGlobalization::LineSearchMerit(
            optimization::LineSearchMeritOptions::default(),
        );
        sqp_options.verbose = false;
        sqp_options.max_iters = 20;

        let mut last_snapshot = None;
        let result = compiled.solve_sqp_with_callback(&runtime, &sqp_options, |snapshot| {
            last_snapshot = Some(snapshot.clone());
        });

        match result {
            Ok(_) => {}
            Err(optimization::ClarabelSqpError::MaxIterations { .. }) => {}
            Err(optimization::ClarabelSqpError::QpSolve {
                status: optimization::SqpQpRawStatus::NumericalError,
                ..
            }) => {}
            Err(optimization::ClarabelSqpError::QpSolve {
                status: optimization::SqpQpRawStatus::InsufficientProgress,
                ..
            }) => {}
            Err(other) => panic!("reduced glider solve should progress, got {other:?}"),
        }

        let snapshot = last_snapshot.expect("glider SQP should emit at least one iterate");
        let distance = snapshot.trajectories.x.terminal.x;
        let final_time = snapshot.trajectories.tf;
        assert!(distance > 25.0, "glider should travel forward");
        assert!(
            (Params::default().min_time_bound_s..=Params::default().max_time_bound_s)
                .contains(&final_time),
            "free final time should stay within the configured bounds"
        );
    }

    #[test]
    fn glider_sqp_filter_restoration_avoids_catastrophic_crash_paths() {
        const N: usize = 8;
        const K: usize = DEFAULT_COLLOCATION_DEGREE;
        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            scaling_enabled: false,
            ..Params::default()
        };
        let family = params.transcription.collocation_family;
        let runtime = dc_runtime::<N, K>(&params);
        let compiled = model(optimal_control::DirectCollocation::<N, K> { family })
            .compile_jit_with_ocp_options(crate::common::ocp_compile_options(
                crate::common::interactive_direct_collocation_opt_level(),
                params.sx_functions,
            ))
            .expect("reduced glider direct collocation should compile");

        let mut sqp_options = crate::common::sqp_options(&params.solver);
        sqp_options.globalization = optimization::SqpGlobalization::LineSearchFilter(
            optimization::LineSearchFilterOptions::default(),
        );
        sqp_options.verbose = false;
        sqp_options.max_iters = 6;

        let mut snapshots = Vec::new();
        let result = compiled.solve_sqp_with_callback(&runtime, &sqp_options, |snapshot| {
            snapshots.push(snapshot.solver.clone());
        });

        let restoration_failed = matches!(
            result,
            Err(optimization::ClarabelSqpError::RestorationFailed { .. })
        );
        match result {
            Ok(_) => {}
            Err(optimization::ClarabelSqpError::MaxIterations { .. }) => {}
            Err(optimization::ClarabelSqpError::QpSolve {
                status: optimization::SqpQpRawStatus::InsufficientProgress,
                ..
            }) => {}
            Err(optimization::ClarabelSqpError::QpSolve {
                status: optimization::SqpQpRawStatus::NumericalError,
                ..
            }) => {}
            Err(optimization::ClarabelSqpError::RestorationFailed { .. }) => {}
            Err(other) => {
                panic!("glider SQP should progress past the old early failure, got {other:?}")
            }
        }

        if restoration_failed {
            return;
        }

        let accepted_steps = snapshots
            .iter()
            .filter(|snapshot| snapshot.phase == optimization::SqpIterationPhase::AcceptedStep)
            .filter(|snapshot| snapshot.line_search.is_some())
            .collect::<Vec<_>>();
        assert!(
            accepted_steps.len() >= 2,
            "expected at least 2 accepted SQP iterates, got {}",
            accepted_steps.len()
        );
        for snapshot in accepted_steps.iter().take(2) {
            let primal_inf = snapshot
                .eq_inf
                .into_iter()
                .chain(snapshot.ineq_inf)
                .fold(0.0_f64, f64::max);
            assert!(
                primal_inf < 1.0,
                "early SQP iterate left the sane feasibility neighborhood: {snapshot:?}"
            );
            assert!(
                snapshot.line_search.as_ref().is_some_and(
                    |info| info.step_kind != Some(optimization::SqpStepKind::Objective)
                ),
                "early SQP iterate should not be accepted as an objective step far from feasibility: {snapshot:?}"
            );
        }
    }

    #[test]
    #[ignore = "manual failure diagnostics helper"]
    fn print_current_glider_solver_failures() {
        for solver_method in [SolverMethod::Sqp, SolverMethod::Nlip] {
            let mut log_lines = Vec::new();
            let result = solve_with_progress(
                &Params {
                    launch_speed_mps: 30.0,
                    initial_alpha_deg: 6.0,
                    max_alpha_rate_deg_s: 12.0,
                    solver_method,
                    ..Params::default()
                },
                |event| match event {
                    crate::common::SolveStreamEvent::Status { status } => {
                        log_lines.push(format!(
                            "[status {:?}] {}",
                            status.solver_method, status.solver.status_label
                        ));
                    }
                    crate::common::SolveStreamEvent::Log { line, .. } => log_lines.push(line),
                    crate::common::SolveStreamEvent::Error { message } => {
                        log_lines.push(format!("error: {message}"));
                    }
                    crate::common::SolveStreamEvent::Iteration { progress, .. } => {
                        log_lines.push(format!(
                            "[iteration {} {:?}] obj={:.3e} eq_inf={:?} ineq_inf={:?} dual_inf={:.3e}",
                            progress.iteration,
                            progress.phase,
                            progress.objective,
                            progress.eq_inf,
                            progress.ineq_inf,
                            progress.dual_inf
                        ));
                    }
                    crate::common::SolveStreamEvent::Final { artifact } => {
                        log_lines.push(format!("[final] {}", artifact.solver.status_label));
                    }
                },
            );
            println!("\n=== glider {solver_method:?} ===");
            for line in &log_lines {
                println!("{line}");
            }
            println!(
                "result: {:?}",
                result.as_ref().map(|_| ()).map_err(|err| err.to_string())
            );
        }

        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            ..Params::default()
        };
        let runtime = dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(&params);
        let compiled = cached_direct_collocation(&params, params.transcription.collocation_family)
            .expect("glider direct collocation should compile");
        let compiled = compiled.compiled.borrow();

        let mut sqp_options = crate::common::sqp_options(&params.solver);
        sqp_options.globalization = optimization::SqpGlobalization::LineSearchMerit(
            optimization::LineSearchMeritOptions::default(),
        );
        sqp_options.verbose = false;
        sqp_options.max_iters = 4;
        let mut sqp_iterations = Vec::new();
        let sqp_result = compiled.solve_sqp_with_callback(&runtime, &sqp_options, |snapshot| {
            sqp_iterations.push(snapshot.solver.clone());
        });
        println!("\n=== glider SQP filter=off ===");
        for snapshot in &sqp_iterations {
            println!(
                "iter={} phase={:?} obj={:.3e} eq_inf={:?} ineq_inf={:?} dual_inf={:.3e} step_inf={:?} alpha={:?} ls_it={:?}",
                snapshot.iteration,
                snapshot.phase,
                snapshot.objective,
                snapshot.eq_inf,
                snapshot.ineq_inf,
                snapshot.dual_inf,
                snapshot.step_inf,
                snapshot
                    .line_search
                    .as_ref()
                    .map(|info| info.accepted_alpha),
                snapshot
                    .line_search
                    .as_ref()
                    .map(|info| info.backtrack_count),
            );
        }
        println!(
            "result: {:?}",
            sqp_result
                .as_ref()
                .map(|_| ())
                .map_err(|err| err.to_string())
        );

        let mut nlip_options = crate::common::nlip_options(&params.solver);
        nlip_options.verbose = false;
        nlip_options.max_iters = 4;
        let mut nlip_iterations = Vec::new();
        let nlip_result =
            compiled.solve_interior_point_with_callback(&runtime, &nlip_options, |snapshot| {
                nlip_iterations.push(snapshot.solver.clone());
            });
        println!("\n=== glider NLIP filter ===");
        for snapshot in &nlip_iterations {
            println!(
                "iter={} phase={:?} obj={:.3e} eq_inf={:?} ineq_inf={:?} dual_inf={:.3e} step_inf={:?} alpha={:?} ls_it={:?}",
                snapshot.iteration,
                snapshot.phase,
                snapshot.objective,
                snapshot.eq_inf,
                snapshot.ineq_inf,
                snapshot.dual_inf,
                snapshot.step_inf,
                snapshot.alpha,
                snapshot.line_search_iterations,
            );
        }
        println!(
            "result: {:?}",
            nlip_result
                .as_ref()
                .map(|_| ())
                .map_err(|err| err.to_string())
        );
    }

    #[test]
    #[ignore = "manual NLIP-only diagnostics helper"]
    fn print_current_glider_nlip_failure() {
        let mut log_lines = Vec::new();
        let mut params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            initial_time_guess_s: 140.0,
            max_alpha_rate_deg_s: 12.0,
            solver_method: SolverMethod::Nlip,
            ..Params::default()
        };
        params.solver.max_iters = 400;
        let result = solve_with_progress(&params, |event| match event {
            crate::common::SolveStreamEvent::Status { status } => {
                log_lines.push(format!(
                    "[status {:?}] {}",
                    status.solver_method, status.solver.status_label
                ));
            }
            crate::common::SolveStreamEvent::Log { line, .. } => log_lines.push(line),
            crate::common::SolveStreamEvent::Error { message } => {
                log_lines.push(format!("error: {message}"));
            }
            crate::common::SolveStreamEvent::Iteration { progress, .. } => {
                log_lines.push(format!(
                    "[iteration {} {:?}] obj={:.3e} eq_inf={:?} ineq_inf={:?} dual_inf={:.3e}",
                    progress.iteration,
                    progress.phase,
                    progress.objective,
                    progress.eq_inf,
                    progress.ineq_inf,
                    progress.dual_inf
                ));
            }
            crate::common::SolveStreamEvent::Final { artifact } => {
                log_lines.push(format!("[final] {}", artifact.solver.status_label));
            }
        });
        println!("\n=== glider Nlip only ===");
        for line in &log_lines {
            println!("{line}");
        }
        match &result {
            Ok(_) => println!("result: Ok(())"),
            Err(err) => {
                println!("result: Err({err})");
                if let Some(optimization::InteriorPointSolveError::LineSearchFailed {
                    context, ..
                }) = err.downcast_ref::<optimization::InteriorPointSolveError>()
                    && let Some(line_search) = &context.failed_line_search
                {
                    println!(
                        "line_search: merit={:.3e} current_barrier_obj={:.3e} current_primal={:.3e} alpha_min={:.3e} rejected={} backtracks={} soc_attempted={} watchdog_active={}",
                        line_search.current_merit,
                        line_search.current_barrier_objective,
                        line_search.current_primal_inf,
                        line_search.alpha_min,
                        line_search.rejected_trials.len(),
                        line_search.backtrack_count,
                        line_search.second_order_correction_attempted,
                        line_search.watchdog_active,
                    );
                    for (idx, trial) in line_search.rejected_trials.iter().take(8).enumerate() {
                        println!(
                            "  reject[{idx}] alpha={:.3e} alpha_du={:?} merit={:?} barrier_obj={:?} primal={:?} dual={:?} comp={:?} filter={:?} dominated={:?} obj_red={:?} viol_red={:?} switch={:?}",
                            trial.alpha,
                            trial.alpha_du,
                            trial.merit,
                            trial.barrier_objective,
                            trial.primal_inf,
                            trial.dual_inf,
                            trial.comp_inf,
                            trial.filter_acceptable,
                            trial.filter_dominated,
                            trial.filter_sufficient_objective_reduction,
                            trial.filter_sufficient_violation_reduction,
                            trial.switching_condition_satisfied,
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "manual NLIP KKT compare helper"]
    fn print_current_glider_nlip_linear_debug_compare() {
        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            solver_method: SolverMethod::Nlip,
            ..Params::default()
        };
        let compiled = cached_direct_collocation(&params, params.transcription.collocation_family)
            .expect("glider direct collocation should compile");
        let compiled = compiled.compiled.borrow();
        let runtime = dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(&params);
        let mut options = crate::common::nlip_options(&params.solver);
        options.max_iters = 120;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![
                optimization::InteriorPointLinearSolver::NativeSpralSsids,
                optimization::InteriorPointLinearSolver::SparseQdldl,
            ],
            schedule: optimization::InteriorPointLinearDebugSchedule::EveryIteration,
            dump_dir: None,
        });

        let mut first_divergence = None;
        let result = compiled.solve_interior_point_with_callback(&runtime, &options, |snapshot| {
            let solver = &snapshot.solver;
            if let Some(report) = solver.linear_debug.as_ref() {
                println!(
                    "iter={} phase={:?} verdict={:?} primary={} obj={:.3e} eq_inf={:?} ineq_inf={:?} dual_inf={:.3e} comp_inf={:?}",
                    solver.iteration,
                    solver.phase,
                    report.verdict,
                    report.primary_solver.label(),
                    solver.objective,
                    solver.eq_inf,
                    solver.ineq_inf,
                    solver.dual_inf,
                    solver.comp_inf,
                );
                for result in &report.results {
                    println!(
                        "  solver={} success={} inertia={:?} residual={:?} step_inf={:?} factor_time={:?} solve_time={:?} step_delta={:?} dx_delta={:?} dlambda_delta={:?} ds_delta={:?} dz_delta={:?} detail={:?}",
                        result.solver.label(),
                        result.success,
                        result.inertia,
                        result.residual_inf,
                        result.step_inf,
                        result.factorization_time,
                        result.solve_time,
                        result.step_delta_inf,
                        result.dx_delta_inf,
                        result.d_lambda_delta_inf,
                        result.ds_delta_inf,
                        result.dz_delta_inf,
                        result.detail,
                    );
                }
                for note in &report.notes {
                    println!("  note={note}");
                }
                if first_divergence.is_none()
                    && !matches!(
                        report.verdict,
                        optimization::InteriorPointLinearDebugVerdict::Consistent
                    )
                {
                    first_divergence = Some((
                        solver.iteration,
                        solver.phase,
                        report.verdict,
                        report.results.clone(),
                        report.notes.clone(),
                    ));
                }
            }
        });

        println!("\n=== glider NLIP linear-debug compare ===");
        match &result {
            Ok(summary) => {
                println!(
                    "result: Ok(iterations={} termination={:?} objective={:.6e})",
                    summary.solver.iterations, summary.solver.termination, summary.solver.objective
                );
            }
            Err(err) => println!("result: Err({err})"),
        }
        if let Some((iteration, phase, verdict, results, notes)) = first_divergence {
            println!(
                "first divergence: iter={} phase={:?} verdict={:?}",
                iteration, phase, verdict
            );
            for result in results {
                println!(
                    "  {} success={} inertia={:?} residual={:?} step_delta={:?} detail={:?}",
                    result.solver.label(),
                    result.success,
                    result.inertia,
                    result.residual_inf,
                    result.step_delta_inf,
                    result.detail,
                );
            }
            for note in notes {
                println!("  note={note}");
            }
        } else {
            println!("no linear-solver divergence detected before termination");
        }
    }

    #[test]
    #[ignore = "manual NLIP native-vs-QDLDL compare helper"]
    fn print_current_glider_nlip_native_vs_qdldl_compare() {
        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            solver_method: SolverMethod::Nlip,
            ..Params::default()
        };
        let compiled = cached_direct_collocation(&params, params.transcription.collocation_family)
            .expect("glider direct collocation should compile");
        let compiled = compiled.compiled.borrow();
        let runtime = dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(&params);
        let mut options = crate::common::nlip_options(&params.solver);
        options.linear_solver = optimization::InteriorPointLinearSolver::NativeSpralSsids;
        options.max_iters = 120;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![optimization::InteriorPointLinearSolver::SparseQdldl],
            schedule: optimization::InteriorPointLinearDebugSchedule::EveryIteration,
            dump_dir: None,
        });

        let mut first_divergence = None;
        let result = compiled.solve_interior_point_with_callback(&runtime, &options, |snapshot| {
            let solver = &snapshot.solver;
            if let Some(report) = solver.linear_debug.as_ref() {
                println!(
                    "iter={} phase={:?} verdict={:?} primary={} obj={:.3e} eq_inf={:?} ineq_inf={:?} dual_inf={:.3e} comp_inf={:?}",
                    solver.iteration,
                    solver.phase,
                    report.verdict,
                    report.primary_solver.label(),
                    solver.objective,
                    solver.eq_inf,
                    solver.ineq_inf,
                    solver.dual_inf,
                    solver.comp_inf,
                );
                for result in &report.results {
                    println!(
                        "  solver={} success={} inertia={:?} residual={:?} step_inf={:?} factor_time={:?} solve_time={:?} step_delta={:?} dx_delta={:?} dlambda_delta={:?} ds_delta={:?} dz_delta={:?} detail={:?}",
                        result.solver.label(),
                        result.success,
                        result.inertia,
                        result.residual_inf,
                        result.step_inf,
                        result.factorization_time,
                        result.solve_time,
                        result.step_delta_inf,
                        result.dx_delta_inf,
                        result.d_lambda_delta_inf,
                        result.ds_delta_inf,
                        result.dz_delta_inf,
                        result.detail,
                    );
                }
                for note in &report.notes {
                    println!("  note={note}");
                }
                if first_divergence.is_none()
                    && !matches!(
                        report.verdict,
                        optimization::InteriorPointLinearDebugVerdict::Consistent
                    )
                {
                    first_divergence = Some((
                        solver.iteration,
                        solver.phase,
                        report.verdict,
                        report.results.clone(),
                        report.notes.clone(),
                    ));
                }
            }
        });

        println!("\n=== glider NLIP native-vs-QDLDL compare ===");
        match &result {
            Ok(summary) => {
                println!(
                    "result: Ok(iterations={} termination={:?} objective={:.6e})",
                    summary.solver.iterations, summary.solver.termination, summary.solver.objective
                );
            }
            Err(err) => println!("result: Err({err})"),
        }
        if let Some((iteration, phase, verdict, results, notes)) = first_divergence {
            println!(
                "first divergence: iter={} phase={:?} verdict={:?}",
                iteration, phase, verdict
            );
            for result in results {
                println!(
                    "  {} success={} inertia={:?} residual={:?} step_delta={:?} detail={:?}",
                    result.solver.label(),
                    result.success,
                    result.inertia,
                    result.residual_inf,
                    result.step_delta_inf,
                    result.detail,
                );
            }
            for note in notes {
                println!("  note={note}");
            }
        } else {
            println!("no linear-solver divergence detected before termination");
        }
    }

    #[test]
    #[ignore = "manual NLIP iteration-0 augmented replay helper"]
    fn print_current_glider_nlip_iteration0_augmented_compare() {
        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            solver_method: SolverMethod::Nlip,
            ..Params::default()
        };
        let compiled = cached_direct_collocation(&params, params.transcription.collocation_family)
            .expect("glider direct collocation should compile");
        let compiled = compiled.compiled.borrow();
        let runtime = dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(&params);
        let dump_dir = TempDir::new().expect("temp dump dir should create");
        let mut options = crate::common::nlip_options(&params.solver);
        options.linear_solver = optimization::InteriorPointLinearSolver::SpralSsids;
        options.max_iters = 1;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![
                optimization::InteriorPointLinearSolver::NativeSpralSsids,
                optimization::InteriorPointLinearSolver::SparseQdldl,
            ],
            schedule: optimization::InteriorPointLinearDebugSchedule::FirstIteration,
            dump_dir: Some(dump_dir.path().to_path_buf()),
        });

        let mut first_report = None;
        let result = compiled.solve_interior_point_with_callback(&runtime, &options, |snapshot| {
            if first_report.is_none() {
                first_report = snapshot.solver.linear_debug.clone();
            }
        });

        let dump = load_glider_linear_debug_dump(&dump_dir.path().join("nlip_kkt_iter_0000.txt"));
        let rust_exact = replay_rust_augmented_spral(&dump);
        let native_exact = replay_native_augmented_spral(&dump);
        let rust_blocks = augmented_step_blocks(&dump, &rust_exact.solution);
        let native_blocks = augmented_step_blocks(&dump, &native_exact.solution);
        let report = first_report.expect("expected first iteration linear debug report");

        println!("\n=== glider NLIP iteration-0 augmented replay ===");
        match &result {
            Ok(summary) => {
                println!(
                    "result: Ok(iterations={} termination={:?} objective={:.6e})",
                    summary.solver.iterations, summary.solver.termination, summary.solver.objective
                );
            }
            Err(err) => println!("result: Err({err})"),
        }

        println!("live compare results:");
        for result in &report.results {
            println!(
                "  solver={} success={} inertia={:?} residual={:?} step_inf={:?} factor_time={:?} solve_time={:?} step_delta={:?} dx_delta={:?} dlambda_delta={:?} ds_delta={:?} dz_delta={:?}",
                result.solver.label(),
                result.success,
                result.inertia,
                result.residual_inf,
                result.step_inf,
                result.factorization_time,
                result.solve_time,
                result.step_delta_inf,
                result.dx_delta_inf,
                result.d_lambda_delta_inf,
                result.ds_delta_inf,
                result.dz_delta_inf,
            );
        }

        println!("exact augmented replay:");
        println!(
            "  rust_spral factor={:?} solve={:?} residual={:.3e} solution_inf={:.3e} inertia={}",
            rust_exact.factor_time,
            rust_exact.solve_time,
            rust_exact.residual_inf,
            rust_exact.solution_inf,
            rust_exact.inertia,
        );
        println!(
            "  native_spral factor={:?} solve={:?} residual={:.3e} solution_inf={:.3e} inertia={}",
            native_exact.factor_time,
            native_exact.solve_time,
            native_exact.residual_inf,
            native_exact.solution_inf,
            native_exact.inertia,
        );
        println!(
            "  augmented deltas: solution={:.6e} dx={:.6e} dp={:.6e} ds={:.6e} dlambda={:.6e} dz={:.6e}",
            delta_inf(&rust_exact.solution, &native_exact.solution),
            delta_inf(&rust_blocks.dx, &native_blocks.dx),
            delta_inf(&rust_blocks.p, &native_blocks.p),
            delta_inf(&rust_blocks.ds, &native_blocks.ds),
            delta_inf(&rust_blocks.d_lambda, &native_blocks.d_lambda),
            delta_inf(&rust_blocks.dz, &native_blocks.dz),
        );
    }

    #[test]
    #[ignore = "manual release-oriented native-vs-rust SPRAL parity check"]
    fn glider_native_spral_matches_rust_spral_extremely_closely() {
        if NativeSpral::load().is_err() {
            eprintln!("skipping glider native SPRAL parity test: library unavailable");
            return;
        }

        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            solver_method: SolverMethod::Nlip,
            ..Params::default()
        };
        let compiled = cached_direct_collocation(&params, params.transcription.collocation_family)
            .expect("glider direct collocation should compile");
        let compiled = compiled.compiled.borrow();
        let runtime = dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(&params);
        let mut options = crate::common::nlip_options(&params.solver);
        options.max_iters = 10;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![optimization::InteriorPointLinearSolver::NativeSpralSsids],
            schedule: optimization::InteriorPointLinearDebugSchedule::EveryIteration,
            dump_dir: None,
        });

        let mut reports = Vec::new();
        let _ = compiled.solve_interior_point_with_callback(&runtime, &options, |snapshot| {
            if let Some(report) = snapshot.solver.linear_debug.clone() {
                reports.push(report);
            }
        });

        assert!(
            !reports.is_empty(),
            "expected glider run to emit native SPRAL comparison reports"
        );
        for report in &reports {
            assert_eq!(
                report.primary_solver,
                optimization::InteriorPointLinearSolver::SpralSsids
            );
            assert_eq!(
                report.schedule,
                optimization::InteriorPointLinearDebugSchedule::EveryIteration
            );
            assert_eq!(report.results.len(), 2);
            let native = native_spral_debug_result(report);
            assert!(native.success, "native SPRAL comparison failed: {native:?}");
            assert!(
                native
                    .residual_inf
                    .expect("native residual should be present")
                    <= 1e-9,
                "native residual too large: {native:?}"
            );
            assert!(
                native.step_delta_inf.expect("step delta should be present") <= 5e-9,
                "native-vs-rust step delta too large: {native:?}"
            );
            assert!(
                native.dx_delta_inf.expect("dx delta should be present") <= 1e-9,
                "native-vs-rust dx delta too large: {native:?}"
            );
            assert!(
                native
                    .d_lambda_delta_inf
                    .expect("dlambda delta should be present")
                    <= 5e-9,
                "native-vs-rust dlambda delta too large: {native:?}"
            );
            assert!(
                native.ds_delta_inf.expect("ds delta should be present") <= 1e-9,
                "native-vs-rust ds delta too large: {native:?}"
            );
            assert!(
                native.dz_delta_inf.expect("dz delta should be present") <= 1e-9,
                "native-vs-rust dz delta too large: {native:?}"
            );
        }
    }

    #[test]
    #[ignore = "manual release-oriented exact augmented parity check"]
    fn glider_native_spral_exact_augmented_replay_matches_rust_to_machine_precision() {
        if NativeSpral::load().is_err() {
            eprintln!("skipping glider augmented SPRAL parity test: library unavailable");
            return;
        }

        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            solver_method: SolverMethod::Nlip,
            ..Params::default()
        };
        let compiled = cached_direct_collocation(&params, params.transcription.collocation_family)
            .expect("glider direct collocation should compile");
        let compiled = compiled.compiled.borrow();
        let runtime = dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(&params);
        let dump_dir = TempDir::new().expect("temp dump dir should create");
        let mut options = crate::common::nlip_options(&params.solver);
        options.linear_solver = optimization::InteriorPointLinearSolver::SpralSsids;
        options.max_iters = 1;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![optimization::InteriorPointLinearSolver::NativeSpralSsids],
            schedule: optimization::InteriorPointLinearDebugSchedule::FirstIteration,
            dump_dir: Some(dump_dir.path().to_path_buf()),
        });

        let _ = compiled.solve_interior_point_with_callback(&runtime, &options, |_snapshot| {});

        let dump = load_glider_linear_debug_dump(&dump_dir.path().join("nlip_kkt_iter_0000.txt"));
        let rust_exact = replay_rust_augmented_spral(&dump);
        let native_exact = replay_native_augmented_spral(&dump);
        let rust_blocks = augmented_step_blocks(&dump, &rust_exact.solution);
        let native_blocks = augmented_step_blocks(&dump, &native_exact.solution);

        assert!(
            rust_exact.residual_inf <= 1e-12,
            "rust residual too large: {rust_exact:?}"
        );
        assert!(
            native_exact.residual_inf <= 1e-12,
            "native residual too large: {native_exact:?}"
        );
        assert_eq!(rust_exact.inertia, native_exact.inertia);
        assert!(
            delta_inf(&rust_exact.solution, &native_exact.solution) <= 1e-12,
            "augmented solution delta too large"
        );
        assert!(
            delta_inf(&rust_blocks.dx, &native_blocks.dx) <= 1e-12,
            "dx delta too large"
        );
        assert!(
            delta_inf(&rust_blocks.p, &native_blocks.p) <= 1e-12,
            "dp delta too large"
        );
        assert!(
            delta_inf(&rust_blocks.ds, &native_blocks.ds) <= 1e-12,
            "ds delta too large"
        );
        assert!(
            delta_inf(&rust_blocks.d_lambda, &native_blocks.d_lambda) <= 1e-12,
            "dlambda delta too large"
        );
        assert!(
            delta_inf(&rust_blocks.dz, &native_blocks.dz) <= 1e-12,
            "dz delta too large"
        );
    }

    #[test]
    #[ignore = "manual release-oriented exact augmented parity sweep"]
    fn glider_native_spral_exact_augmented_replay_matches_each_dump_extremely_closely() {
        if NativeSpral::load().is_err() {
            eprintln!("skipping glider augmented SPRAL parity sweep: library unavailable");
            return;
        }

        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            solver_method: SolverMethod::Nlip,
            ..Params::default()
        };
        let compiled = cached_direct_collocation(&params, params.transcription.collocation_family)
            .expect("glider direct collocation should compile");
        let compiled = compiled.compiled.borrow();
        let runtime = dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(&params);
        let dump_dir = TempDir::new().expect("temp dump dir should create");
        let mut options = crate::common::nlip_options(&params.solver);
        options.linear_solver = optimization::InteriorPointLinearSolver::SpralSsids;
        options.max_iters = 10;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![optimization::InteriorPointLinearSolver::NativeSpralSsids],
            schedule: optimization::InteriorPointLinearDebugSchedule::EveryIteration,
            dump_dir: Some(dump_dir.path().to_path_buf()),
        });

        let mut reports = Vec::new();
        let _ = compiled.solve_interior_point_with_callback(&runtime, &options, |snapshot| {
            if let Some(report) = snapshot.solver.linear_debug.clone() {
                reports.push(report);
            }
        });

        assert!(
            !reports.is_empty(),
            "expected glider run to emit native SPRAL comparison reports"
        );
        let dump_paths = sorted_glider_dump_paths(dump_dir.path());
        assert_eq!(
            dump_paths.len(),
            reports.len(),
            "expected one dumped KKT snapshot per glider linear-debug report"
        );

        for dump_path in dump_paths {
            let dump = load_glider_linear_debug_dump(&dump_path);
            let rust_exact = replay_rust_augmented_spral(&dump);
            let native_exact = replay_native_augmented_spral(&dump);
            let rust_blocks = augmented_step_blocks(&dump, &rust_exact.solution);
            let native_blocks = augmented_step_blocks(&dump, &native_exact.solution);
            let solution_delta = delta_inf(&rust_exact.solution, &native_exact.solution);
            let dx_delta = delta_inf(&rust_blocks.dx, &native_blocks.dx);
            let p_delta = delta_inf(&rust_blocks.p, &native_blocks.p);
            let ds_delta = delta_inf(&rust_blocks.ds, &native_blocks.ds);
            let dlambda_delta = delta_inf(&rust_blocks.d_lambda, &native_blocks.d_lambda);
            let dz_delta = delta_inf(&rust_blocks.dz, &native_blocks.dz);

            assert!(
                rust_exact.residual_inf <= 5e-10,
                "rust residual too large for {}: {:?}",
                dump_path.display(),
                rust_exact
            );
            assert!(
                native_exact.residual_inf <= 5e-10,
                "native residual too large for {}: {:?}",
                dump_path.display(),
                native_exact
            );
            assert_eq!(rust_exact.inertia, native_exact.inertia);
            assert!(
                solution_delta <= 1e-10,
                "augmented solution delta too large for {}",
                dump_path.display()
            );
            assert!(
                dx_delta <= 1e-10,
                "dx delta too large for {}",
                dump_path.display()
            );
            assert!(
                p_delta <= 1e-10,
                "dp delta too large for {}",
                dump_path.display()
            );
            assert!(
                ds_delta <= 1e-10,
                "ds delta too large for {}",
                dump_path.display()
            );
            assert!(
                dlambda_delta <= 1e-10,
                "dlambda delta too large for {}",
                dump_path.display()
            );
            assert!(
                dz_delta <= 1e-10,
                "dz delta too large for {}",
                dump_path.display()
            );
        }
    }

    #[test]
    #[ignore = "manual release-oriented exact augmented refactorization sweep"]
    fn glider_native_spral_exact_augmented_refactorize_sequence_matches_each_dump_extremely_closely()
     {
        if NativeSpral::load().is_err() {
            eprintln!("skipping glider augmented SPRAL refactorization sweep: library unavailable");
            return;
        }

        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            solver_method: SolverMethod::Nlip,
            ..Params::default()
        };
        let compiled = cached_direct_collocation(&params, params.transcription.collocation_family)
            .expect("glider direct collocation should compile");
        let compiled = compiled.compiled.borrow();
        let runtime = dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(&params);
        let dump_dir = TempDir::new().expect("temp dump dir should create");
        let mut options = crate::common::nlip_options(&params.solver);
        options.linear_solver = optimization::InteriorPointLinearSolver::SpralSsids;
        options.max_iters = 10;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![optimization::InteriorPointLinearSolver::NativeSpralSsids],
            schedule: optimization::InteriorPointLinearDebugSchedule::EveryIteration,
            dump_dir: Some(dump_dir.path().to_path_buf()),
        });

        let mut reports = Vec::new();
        let _ = compiled.solve_interior_point_with_callback(&runtime, &options, |snapshot| {
            if let Some(report) = snapshot.solver.linear_debug.clone() {
                reports.push(report);
            }
        });

        assert!(
            !reports.is_empty(),
            "expected glider run to emit native SPRAL comparison reports"
        );
        let dump_paths = sorted_glider_dump_paths(dump_dir.path());
        assert_eq!(
            dump_paths.len(),
            reports.len(),
            "expected one dumped KKT snapshot per glider linear-debug report"
        );

        let first_dump = load_glider_linear_debug_dump(&dump_paths[0]);
        let first_structure = SymmetricCscMatrix::new(
            first_dump.matrix_dimension,
            &first_dump.col_ptrs,
            &first_dump.row_indices,
            None,
        )
        .expect("first dumped augmented CSC should validate");
        let first_numeric = SymmetricCscMatrix::new(
            first_dump.matrix_dimension,
            &first_dump.col_ptrs,
            &first_dump.row_indices,
            Some(&first_dump.values),
        )
        .expect("first dumped augmented CSC values should validate");
        let (symbolic, _) = spral_analyse(
            first_structure,
            &SsidsOptions {
                ordering: OrderingStrategy::ApproximateMinimumDegree,
            },
        )
        .expect("rust spral analyse should succeed on first dumped KKT");
        let (mut rust_factor, _) =
            spral_factorize(first_numeric, &symbolic, &NumericFactorOptions::default())
                .expect("rust spral factorization should succeed on first dumped KKT");

        let native = NativeSpral::load().expect("native SPRAL should be available locally");
        let mut native_session = native
            .analyse(
                SymmetricCscMatrix::new(
                    first_dump.matrix_dimension,
                    &first_dump.col_ptrs,
                    &first_dump.row_indices,
                    None,
                )
                .expect("first dumped native structure should validate"),
            )
            .expect("native spral analyse should succeed on first dumped KKT");
        native_session
            .factorize(
                SymmetricCscMatrix::new(
                    first_dump.matrix_dimension,
                    &first_dump.col_ptrs,
                    &first_dump.row_indices,
                    Some(&first_dump.values),
                )
                .expect("first dumped native numeric matrix should validate"),
            )
            .expect("native spral factorization should succeed on first dumped KKT");

        for (index, dump_path) in dump_paths.iter().enumerate() {
            let dump = load_glider_linear_debug_dump(dump_path);
            let numeric = SymmetricCscMatrix::new(
                dump.matrix_dimension,
                &dump.col_ptrs,
                &dump.row_indices,
                Some(&dump.values),
            )
            .expect("dumped augmented CSC values should validate");

            if index > 0 {
                rust_factor
                    .refactorize(numeric)
                    .expect("rust spral refactorization should succeed on dumped KKT");
                native_session
                    .refactorize(numeric)
                    .expect("native spral refactorization should succeed on dumped KKT");
            }

            let mut rust_solution = rust_factor
                .solve(&dump.rhs)
                .expect("rust spral solve should succeed on dumped KKT");
            for _ in 0..10 {
                let residual = residual_vector(&dump, &rust_solution);
                if residual.iter().all(|value| value.abs() <= f64::EPSILON) {
                    break;
                }
                let correction = rust_factor
                    .solve(&residual)
                    .expect("rust spral iterative refinement should succeed on dumped KKT");
                for (solution_i, correction_i) in rust_solution.iter_mut().zip(correction.iter()) {
                    *solution_i += correction_i;
                }
            }

            let mut native_solution = native_session
                .solve(&dump.rhs)
                .expect("native spral solve should succeed on dumped KKT");
            for _ in 0..10 {
                let residual = residual_vector(&dump, &native_solution);
                if residual.iter().all(|value| value.abs() <= f64::EPSILON) {
                    break;
                }
                let correction = native_session
                    .solve(&residual)
                    .expect("native spral iterative refinement should succeed on dumped KKT");
                for (solution_i, correction_i) in native_solution.iter_mut().zip(correction.iter())
                {
                    *solution_i += correction_i;
                }
            }

            let rust_blocks = augmented_step_blocks(&dump, &rust_solution);
            let native_blocks = augmented_step_blocks(&dump, &native_solution);
            let solution_delta = delta_inf(&rust_solution, &native_solution);
            let dx_delta = delta_inf(&rust_blocks.dx, &native_blocks.dx);
            let p_delta = delta_inf(&rust_blocks.p, &native_blocks.p);
            let ds_delta = delta_inf(&rust_blocks.ds, &native_blocks.ds);
            let dlambda_delta = delta_inf(&rust_blocks.d_lambda, &native_blocks.d_lambda);
            let dz_delta = delta_inf(&rust_blocks.dz, &native_blocks.dz);

            assert!(
                residual_inf(&dump, &rust_solution) <= 1e-8,
                "rust residual too large for {}",
                dump_path.display()
            );
            assert!(
                residual_inf(&dump, &native_solution) <= 1e-8,
                "native residual too large for {}",
                dump_path.display()
            );
            assert_eq!(
                rust_factor.inertia(),
                native_session.factor_info().expect("factor info").inertia
            );
            assert!(
                solution_delta <= 1e-10,
                "augmented solution delta too large for {}",
                dump_path.display()
            );
            assert!(
                dx_delta <= 1e-10,
                "dx delta too large for {}",
                dump_path.display()
            );
            assert!(
                p_delta <= 1e-10,
                "dp delta too large for {}",
                dump_path.display()
            );
            assert!(
                ds_delta <= 1e-10,
                "ds delta too large for {}",
                dump_path.display()
            );
            assert!(
                dlambda_delta <= 1e-10,
                "dlambda delta too large for {}",
                dump_path.display()
            );
            assert!(
                dz_delta <= 1e-10,
                "dz delta too large for {}",
                dump_path.display()
            );
        }
    }

    #[test]
    #[ignore = "manual NLIP strict-convergence diagnostics helper"]
    fn print_current_glider_nlip_strict_repro() {
        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            solver_method: SolverMethod::Nlip,
            ..Params::default()
        };
        let compiled = cached_direct_collocation(&params, params.transcription.collocation_family)
            .expect("glider direct collocation should compile");
        let compiled = compiled.compiled.borrow();
        let strict_runtime = dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(&params);
        let mut options = crate::common::nlip_options(&params.solver);
        options.max_iters = 400;
        options.acceptable_iter = 0;
        options.verbose = false;

        let mut iterations = Vec::new();
        let mut last_tf = None;
        let mut last_x = None;
        let result =
            compiled.solve_interior_point_with_callback(&strict_runtime, &options, |snapshot| {
                let solver = snapshot.solver.clone();
                last_tf = Some(snapshot.trajectories.tf);
                last_x = Some(snapshot.trajectories.x.terminal.x);
                if solver.iteration % 10 == 0 {
                    println!(
                        "iter={} phase={:?} obj={:.3e} tf={:.3e} x_T={:.3e} eq_inf={:?} ineq_inf={:?} dual_inf={:.3e} comp_inf={:?} overall={:.3e} alpha={:?} ls_it={:?} step_kind={:?} step_tag={:?}",
                        solver.iteration,
                        solver.phase,
                        solver.objective,
                        snapshot.trajectories.tf,
                        snapshot.trajectories.x.terminal.x,
                        solver.eq_inf,
                        solver.ineq_inf,
                        solver.dual_inf,
                        solver.comp_inf,
                        solver.overall_inf,
                        solver.alpha,
                        solver.line_search_iterations,
                        solver.step_kind,
                        solver.step_tag,
                    );
                    if let Some(line_search) = &solver.line_search {
                        println!(
                            "  line_search alpha_pr={:.3e} alpha_du={:?} accepted={:?} sigma={:.3e} current_merit={:.3e} current_barrier_obj={:.3e} current_primal_inf={:.3e} alpha_min={:.3e} rejected={} soc_attempted={} soc_used={} watchdog_active={} watchdog_accepted={}",
                            line_search.initial_alpha_pr,
                            line_search.initial_alpha_du,
                            line_search.accepted_alpha,
                            line_search.sigma,
                            line_search.current_merit,
                            line_search.current_barrier_objective,
                            line_search.current_primal_inf,
                            line_search.alpha_min,
                            line_search.rejected_trials.len(),
                            line_search.second_order_correction_attempted,
                            line_search.second_order_correction_used,
                            line_search.watchdog_active,
                            line_search.watchdog_accepted,
                        );
                        for (index, trial) in line_search.rejected_trials.iter().enumerate().take(8)
                        {
                            println!(
                                "    reject[{index}] alpha={:.3e} alpha_du={:?} merit={:?} barrier_obj={:?} primal={:?} dual={:?} comp={:?} local_filter={:?} filter={:?} dominated={:?} obj_red={:?} viol_red={:?} switch={:?}",
                                trial.alpha,
                                trial.alpha_du,
                                trial.merit,
                                trial.barrier_objective,
                                trial.primal_inf,
                                trial.dual_inf,
                                trial.comp_inf,
                                trial.local_filter_acceptable,
                                trial.filter_acceptable,
                                trial.filter_dominated,
                                trial.filter_sufficient_objective_reduction,
                                trial.filter_sufficient_violation_reduction,
                                trial.switching_condition_satisfied,
                            );
                        }
                    }
                }
                iterations.push(solver);
            });

        println!("\n=== glider NLIP strict repro ===");
        if let Some(last) = iterations.last() {
            println!(
                "last iter={} phase={:?} obj={:.3e} tf={:?} x_T={:?} eq_inf={:?} ineq_inf={:?} dual_inf={:.3e} comp_inf={:?} overall={:.3e}",
                last.iteration,
                last.phase,
                last.objective,
                last_tf,
                last_x,
                last.eq_inf,
                last.ineq_inf,
                last.dual_inf,
                last.comp_inf,
                last.overall_inf,
            );
        }
        if let Err(err) = &result {
            match err {
                optimization::InteriorPointSolveError::LineSearchFailed {
                    merit,
                    mu,
                    step_inf_norm,
                    context,
                } => {
                    println!(
                        "line_search_failure merit={merit:.6e} mu={mu:.6e} step_inf={step_inf_norm:.6e}"
                    );
                    if let Some(info) = &context.failed_line_search {
                        println!(
                            "  ls current_merit={:.6e} current_barrier_obj={:.6e} current_primal_inf={:.6e} alpha_min={:.6e} rejected={}",
                            info.current_merit,
                            info.current_barrier_objective,
                            info.current_primal_inf,
                            info.alpha_min,
                            info.rejected_trials.len(),
                        );
                        for (index, trial) in info.rejected_trials.iter().enumerate().take(8) {
                            println!(
                                "    reject[{index}] alpha={:.6e} alpha_du={:?} merit={:?} barrier_obj={:?} primal={:?} dual={:?} comp={:?} local_filter={:?} filter={:?} dominated={:?} obj_red={:?} viol_red={:?} switch={:?}",
                                trial.alpha,
                                trial.alpha_du,
                                trial.merit,
                                trial.barrier_objective,
                                trial.primal_inf,
                                trial.dual_inf,
                                trial.comp_inf,
                                trial.local_filter_acceptable,
                                trial.filter_acceptable,
                                trial.filter_dominated,
                                trial.filter_sufficient_objective_reduction,
                                trial.filter_sufficient_violation_reduction,
                                trial.switching_condition_satisfied,
                            );
                        }
                    }
                }
                optimization::InteriorPointSolveError::LinearSolve { context, .. } => {
                    if let Some(info) = &context.failed_linear_solve {
                        println!(
                            "linear_solve_failure preferred={:?} dim={} attempts={}",
                            info.preferred_solver,
                            info.matrix_dimension,
                            info.attempts.len(),
                        );
                        for (index, attempt) in info.attempts.iter().enumerate() {
                            println!(
                                "  attempt[{index}] solver={:?} reg={:.6e} kind={:?} sol_inf={:?} sol_lim={:?} res_inf={:?} res_lim={:?}",
                                attempt.solver,
                                attempt.regularization,
                                attempt.failure_kind,
                                attempt.solution_inf,
                                attempt.solution_inf_limit,
                                attempt.residual_inf,
                                attempt.residual_inf_limit,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        println!(
            "result: {:?}",
            result.as_ref().map(|_| ()).map_err(|err| err.to_string())
        );
    }

    #[test]
    #[ignore = "manual NLIP strict-convergence diagnostics helper without watchdog"]
    fn print_current_glider_nlip_strict_repro_no_watchdog() {
        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            solver_method: SolverMethod::Nlip,
            ..Params::default()
        };
        let compiled = cached_direct_collocation(&params, params.transcription.collocation_family)
            .expect("glider direct collocation should compile");
        let compiled = compiled.compiled.borrow();
        let strict_runtime = dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(&params);
        let mut options = crate::common::nlip_options(&params.solver);
        options.max_iters = 400;
        options.acceptable_iter = 0;
        options.watchdog_shortened_iter_trigger = 0;
        options.watchdog_trial_iter_max = 0;
        options.verbose = false;

        let mut iterations = Vec::new();
        let mut last_tf = None;
        let mut last_x = None;
        let result =
            compiled.solve_interior_point_with_callback(&strict_runtime, &options, |snapshot| {
                let solver = snapshot.solver.clone();
                last_tf = Some(snapshot.trajectories.tf);
                last_x = Some(snapshot.trajectories.x.terminal.x);
                if solver.iteration % 10 == 0 {
                    println!(
                        "iter={} phase={:?} obj={:.3e} tf={:.3e} x_T={:.3e} eq_inf={:?} ineq_inf={:?} dual_inf={:.3e} comp_inf={:?} overall={:.3e} alpha={:?} ls_it={:?} step_kind={:?} step_tag={:?}",
                        solver.iteration,
                        solver.phase,
                        solver.objective,
                        snapshot.trajectories.tf,
                        snapshot.trajectories.x.terminal.x,
                        solver.eq_inf,
                        solver.ineq_inf,
                        solver.dual_inf,
                        solver.comp_inf,
                        solver.overall_inf,
                        solver.alpha,
                        solver.line_search_iterations,
                        solver.step_kind,
                        solver.step_tag,
                    );
                    if let Some(line_search) = &solver.line_search {
                        println!(
                            "  line_search alpha_pr={:.3e} alpha_du={:?} accepted={:?} sigma={:.3e} current_merit={:.3e} current_barrier_obj={:.3e} current_primal_inf={:.3e} alpha_min={:.3e} rejected={} soc_attempted={} soc_used={} watchdog_active={} watchdog_accepted={}",
                            line_search.initial_alpha_pr,
                            line_search.initial_alpha_du,
                            line_search.accepted_alpha,
                            line_search.sigma,
                            line_search.current_merit,
                            line_search.current_barrier_objective,
                            line_search.current_primal_inf,
                            line_search.alpha_min,
                            line_search.rejected_trials.len(),
                            line_search.second_order_correction_attempted,
                            line_search.second_order_correction_used,
                            line_search.watchdog_active,
                            line_search.watchdog_accepted,
                        );
                    }
                }
                iterations.push(solver);
            });

        println!("\n=== glider NLIP strict repro (no watchdog) ===");
        if let Some(last) = iterations.last() {
            println!(
                "last iter={} phase={:?} obj={:.3e} tf={:?} x_T={:?} eq_inf={:?} ineq_inf={:?} dual_inf={:.3e} comp_inf={:?} overall={:.3e}",
                last.iteration,
                last.phase,
                last.objective,
                last_tf,
                last_x,
                last.eq_inf,
                last.ineq_inf,
                last.dual_inf,
                last.comp_inf,
                last.overall_inf,
            );
        }
        if let Err(optimization::InteriorPointSolveError::LinearSolve { context, .. }) = &result
            && let Some(info) = &context.failed_linear_solve
        {
            println!(
                "linear_solve_failure preferred={:?} dim={} attempts={}",
                info.preferred_solver,
                info.matrix_dimension,
                info.attempts.len(),
            );
            for (index, attempt) in info.attempts.iter().enumerate() {
                println!(
                    "  attempt[{index}] solver={:?} reg={:.6e} kind={:?} sol_inf={:?} sol_lim={:?} res_inf={:?} res_lim={:?}",
                    attempt.solver,
                    attempt.regularization,
                    attempt.failure_kind,
                    attempt.solution_inf,
                    attempt.solution_inf_limit,
                    attempt.residual_inf,
                    attempt.residual_inf_limit,
                );
            }
        }
        println!(
            "result: {:?}",
            result.as_ref().map(|_| ()).map_err(|err| err.to_string())
        );
    }

    #[test]
    #[ignore = "manual IPOPT diagnostics helper"]
    fn print_current_glider_ipopt_repro() {
        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            solver_method: SolverMethod::Ipopt,
            ..Params::default()
        };
        let compiled = cached_direct_collocation(&params, params.transcription.collocation_family)
            .expect("glider direct collocation should compile");
        let compiled = compiled.compiled.borrow();
        let runtime = dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(&params);
        let options = crate::common::ipopt_options(&params.solver);
        let mut last = None;
        let result = compiled.solve_ipopt_with_callback(&runtime, &options, |snapshot| {
            if snapshot.solver.iteration % 10 == 0 {
                println!(
                    "iter={} phase={:?} obj={:.3e} inf_pr={:.3e} inf_du={:.3e} mu={:.3e} alpha_pr={:.3e} alpha_du={:.3e}",
                    snapshot.solver.iteration,
                    snapshot.solver.phase,
                    snapshot.solver.objective,
                    snapshot.solver.primal_inf,
                    snapshot.solver.dual_inf,
                    snapshot.solver.barrier_parameter,
                    snapshot.solver.alpha_pr,
                    snapshot.solver.alpha_du,
                );
            }
            last = Some(snapshot.solver.clone());
        });

        println!("\n=== glider IPOPT repro ===");
        if let Some(last) = last {
            println!(
                "last iter={} phase={:?} obj={:.3e} inf_pr={:.3e} inf_du={:.3e} mu={:.3e} alpha_pr={:.3e} alpha_du={:.3e}",
                last.iteration,
                last.phase,
                last.objective,
                last.primal_inf,
                last.dual_inf,
                last.barrier_parameter,
                last.alpha_pr,
                last.alpha_du,
            );
        }
        println!(
            "result: {:?}",
            result.as_ref().map(|_| ()).map_err(|err| err.to_string())
        );
    }

    #[test]
    #[ignore = "manual NLIP strict summary helper without per-iteration callbacks"]
    fn print_current_glider_nlip_strict_fast() {
        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            solver_method: SolverMethod::Nlip,
            ..Params::default()
        };
        let compiled = cached_direct_collocation(&params, params.transcription.collocation_family)
            .expect("glider direct collocation should compile");
        let compiled = compiled.compiled.borrow();
        let runtime = dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(&params);
        let mut options = crate::common::nlip_options(&params.solver);
        options.max_iters = 400;
        options.acceptable_iter = 0;
        options.verbose = false;

        let started = std::time::Instant::now();
        let result = compiled.solve_interior_point(&runtime, &options);
        let elapsed = started.elapsed();

        println!("\n=== glider NLIP strict fast ===");
        match result {
            Ok(summary) => {
                println!(
                    "ok iter={} term={:?} obj={:.6e} eq_inf={:.6e} ineq_inf={:.6e} dual_inf={:.6e} comp_inf={:.6e} overall={:.6e} tf={:.6e} x_T={:.6e} elapsed={:.3?}",
                    summary.solver.iterations,
                    summary.solver.termination,
                    summary.solver.objective,
                    summary.solver.equality_inf_norm,
                    summary.solver.inequality_inf_norm,
                    summary.solver.dual_inf_norm,
                    summary.solver.complementarity_inf_norm,
                    summary.solver.overall_inf_norm,
                    summary.trajectories.tf,
                    summary.trajectories.x.terminal.x,
                    elapsed,
                );
            }
            Err(err) => {
                if let optimization::InteriorPointSolveError::MaxIterations { context, .. } = &err {
                    if let Some(state) = context.final_state.as_ref() {
                        println!(
                            "max_iters iter={} phase={:?} obj={:.6e} eq_inf={:?} ineq_inf={:?} dual_inf={:.6e} comp_inf={:?} overall={:.6e} step_kind={:?} step_tag={:?}",
                            state.iteration,
                            state.phase,
                            state.objective,
                            state.eq_inf,
                            state.ineq_inf,
                            state.dual_inf,
                            state.comp_inf,
                            state.overall_inf,
                            state.step_kind,
                            state.step_tag,
                        );
                    }
                    if let Some(last) = context.last_accepted_state.as_ref() {
                        println!(
                            "last_accepted iter={} phase={:?} obj={:.6e} eq_inf={:?} ineq_inf={:?} dual_inf={:.6e} comp_inf={:?} overall={:.6e} step_kind={:?} step_tag={:?}",
                            last.iteration,
                            last.phase,
                            last.objective,
                            last.eq_inf,
                            last.ineq_inf,
                            last.dual_inf,
                            last.comp_inf,
                            last.overall_inf,
                            last.step_kind,
                            last.step_tag,
                        );
                    }
                    if let Some(line_search) = context.failed_line_search.as_ref() {
                        println!(
                            "failed_line_search merit={:.6e} barrier_obj={:.6e} primal={:.6e} alpha_min={:.6e} rejected={} watchdog_active={} watchdog_accepted={} tiny_step={}",
                            line_search.current_merit,
                            line_search.current_barrier_objective,
                            line_search.current_primal_inf,
                            line_search.alpha_min,
                            line_search.rejected_trials.len(),
                            line_search.watchdog_active,
                            line_search.watchdog_accepted,
                            line_search.tiny_step,
                        );
                    }
                    if let Some(direction) = context.failed_direction_diagnostics.as_ref() {
                        println!(
                            "failed_direction dx_inf={:.6e} dlambda_inf={:.6e} ds_inf={:.6e} dz_inf={:.6e} alpha_pr_limiter={:?} alpha_du_limiter={:?}",
                            direction.dx_inf,
                            direction.d_lambda_inf,
                            direction.ds_inf,
                            direction.dz_inf,
                            direction.alpha_pr_limiter,
                            direction.alpha_du_limiter,
                        );
                    }
                }
                println!("err={err}");
                println!("elapsed={elapsed:.3?}");
            }
        }
    }

    #[test]
    #[ignore = "manual NLIP strict summary helper without watchdog or per-iteration callbacks"]
    fn print_current_glider_nlip_strict_fast_no_watchdog() {
        let params = Params {
            launch_speed_mps: 30.0,
            initial_alpha_deg: 6.0,
            max_alpha_rate_deg_s: 12.0,
            solver_method: SolverMethod::Nlip,
            ..Params::default()
        };
        let compiled = cached_direct_collocation(&params, params.transcription.collocation_family)
            .expect("glider direct collocation should compile");
        let compiled = compiled.compiled.borrow();
        let runtime = dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(&params);
        let mut options = crate::common::nlip_options(&params.solver);
        options.max_iters = 400;
        options.acceptable_iter = 0;
        options.watchdog_shortened_iter_trigger = 0;
        options.watchdog_trial_iter_max = 0;
        options.verbose = false;

        let started = std::time::Instant::now();
        let result = compiled.solve_interior_point(&runtime, &options);
        let elapsed = started.elapsed();

        println!("\n=== glider NLIP strict fast (no watchdog) ===");
        match result {
            Ok(summary) => {
                println!(
                    "ok iter={} term={:?} obj={:.6e} eq_inf={:.6e} ineq_inf={:.6e} dual_inf={:.6e} comp_inf={:.6e} overall={:.6e} tf={:.6e} x_T={:.6e} elapsed={:.3?}",
                    summary.solver.iterations,
                    summary.solver.termination,
                    summary.solver.objective,
                    summary.solver.equality_inf_norm,
                    summary.solver.inequality_inf_norm,
                    summary.solver.dual_inf_norm,
                    summary.solver.complementarity_inf_norm,
                    summary.solver.overall_inf_norm,
                    summary.trajectories.tf,
                    summary.trajectories.x.terminal.x,
                    elapsed,
                );
            }
            Err(err) => {
                if let optimization::InteriorPointSolveError::MaxIterations { context, .. } = &err {
                    if let Some(state) = context.final_state.as_ref() {
                        println!(
                            "max_iters iter={} phase={:?} obj={:.6e} eq_inf={:?} ineq_inf={:?} dual_inf={:.6e} comp_inf={:?} overall={:.6e} step_kind={:?} step_tag={:?}",
                            state.iteration,
                            state.phase,
                            state.objective,
                            state.eq_inf,
                            state.ineq_inf,
                            state.dual_inf,
                            state.comp_inf,
                            state.overall_inf,
                            state.step_kind,
                            state.step_tag,
                        );
                    }
                }
                println!("err={err}");
                println!("elapsed={elapsed:.3?}");
            }
        }
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
                        hessian_strategy: sx_core::HessianStrategy::LowerTriangleByColumn,
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
        let ocp = model(optimal_control::DirectCollocation::<
            DEFAULT_INTERVALS,
            DEFAULT_COLLOCATION_DEGREE,
        > {
            family,
        });
        let started = std::time::Instant::now();
        let compiled = ocp
            .compile_jit_with_ocp_options(optimal_control::OcpCompileOptions {
                function_options: optimization::FunctionCompileOptions::from(
                    optimization::LlvmOptimizationLevel::O0,
                ),
                symbolic_functions: optimal_control::OcpSymbolicFunctionOptions::default(),
                hessian_strategy: sx_core::HessianStrategy::LowerTriangleByColumn,
            })
            .expect("compile should succeed");
        println!(
            "glider dc: total={:?} setup_profile={:?} stats={:?}",
            started.elapsed(),
            compiled.backend_compile_report().setup_profile,
            compiled.backend_compile_report().stats,
        );
    }

    fn jacobian_is_clean(summary: &optimization::ValidationSummary) -> bool {
        summary.max_abs_error <= 5.0e-5
            && summary.max_rel_error <= 5.0e-4
            && summary.sparsity.missing_from_analytic == 0
            && summary.sparsity.extra_in_analytic == 0
    }

    fn assert_jacobian_clean(label: &str, kind: &str, summary: &optimization::ValidationSummary) {
        assert!(
            jacobian_is_clean(summary),
            "{label} expected clean {kind} Jacobian, got {summary:?}"
        );
    }

    fn require_release_mode_for_manual_policy_checks() {
        assert!(
            !cfg!(debug_assertions),
            "manual reduced glider Jacobian policy checks must be run in release mode\n\ntry:\n  cargo test -p optimal_control_problems --release reduced_direct_collocation_jacobian_policies_stay_clean -- --ignored"
        );
    }

    fn require_release_mode_for_manual_hessian_checks() {
        assert!(
            !cfg!(debug_assertions),
            "manual glider Hessian policy checks must be run in release mode\n\ntry:\n  cargo test -p optimal_control_problems --release direct_collocation_hessian_policies_stay_clean -- --ignored --nocapture"
        );
    }

    type DcDecisionLayout = (
        optimal_control::Mesh<State<SX>, DEFAULT_INTERVALS>,
        optimal_control::Mesh<Control<SX>, DEFAULT_INTERVALS>,
        optimal_control::IntervalGrid<State<SX>, DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>,
        optimal_control::IntervalGrid<Control<SX>, DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>,
        optimal_control::IntervalGrid<Control<SX>, DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>,
        SX,
    );

    fn dc_decision_layout_names() -> Vec<String> {
        let mut names = Vec::new();
        <DcDecisionLayout as optimization::Vectorize<SX>>::flat_layout_names("w", &mut names);
        names
    }

    fn named_hessian_entry(names: &[String], entry: &optimization::ValidationWorstEntry) -> String {
        let row_name = names
            .get(entry.row)
            .map_or("<row-oob>", std::string::String::as_str);
        let col_name = names
            .get(entry.col)
            .map_or("<col-oob>", std::string::String::as_str);
        format!(
            "({}, {}) {row_name} vs {col_name}: analytic={:.6e} fd={:.6e} abs={:.6e} rel={:.6e}",
            entry.row,
            entry.col,
            entry.analytic,
            entry.finite_difference,
            entry.abs_error,
            entry.rel_error
        )
    }

    #[test]
    #[ignore = "manual reduced multiple-shooting Jacobian policy regression check"]
    fn reduced_multiple_shooting_jacobian_policies_stay_clean() {
        require_release_mode_for_manual_policy_checks();
        const N: usize = 8;
        let params = Params::default();
        let runtime = ms_runtime::<N>(&params);

        for (label, symbolic_functions) in [
            (
                "baseline",
                optimal_control::OcpSymbolicFunctionOptions::default(),
            ),
            (
                "inline_all",
                optimal_control::OcpSymbolicFunctionOptions::inline_all(),
            ),
            (
                "at_call",
                optimal_control::OcpSymbolicFunctionOptions::function_all_with_call_policy(
                    optimization::CallPolicy::InlineAtCall,
                ),
            ),
            (
                "at_lowering",
                optimal_control::OcpSymbolicFunctionOptions::function_all_with_call_policy(
                    optimization::CallPolicy::InlineAtLowering,
                ),
            ),
            (
                "in_llvm",
                optimal_control::OcpSymbolicFunctionOptions::function_all_with_call_policy(
                    optimization::CallPolicy::InlineInLLVM,
                ),
            ),
            (
                "noinline_llvm",
                optimal_control::OcpSymbolicFunctionOptions::function_all_with_call_policy(
                    optimization::CallPolicy::NoInlineLLVM,
                ),
            ),
        ] {
            let compiled = model(MultipleShooting::<N, RK4_SUBSTEPS>)
                .compile_jit_with_ocp_options(optimal_control::OcpCompileOptions {
                    function_options: optimization::FunctionCompileOptions::from(
                        optimization::LlvmOptimizationLevel::O0,
                    ),
                    symbolic_functions,
                    hessian_strategy: sx_core::HessianStrategy::LowerTriangleByColumn,
                })
                .expect("compile should succeed");

            let stats = compiled.nlp_compile_stats();
            let validation = compiled
                .validate_nlp_derivatives(
                    &runtime,
                    &vec![1.0; stats.equality_count],
                    &vec![1.0; stats.inequality_count],
                    optimization::FiniteDifferenceValidationOptions {
                        first_order_step: 1.0e-6,
                        second_order_step: 1.0e-4,
                        zero_tolerance: 1.0e-7,
                    },
                )
                .expect("finite-difference validation should succeed");
            let equality_jacobian = validation
                .equality_jacobian
                .as_ref()
                .expect("multiple-shooting equality Jacobian summary should exist");
            assert_jacobian_clean(label, "equality", equality_jacobian);
        }
    }

    #[test]
    #[ignore = "manual reduced direct-collocation Jacobian policy regression check"]
    fn reduced_direct_collocation_jacobian_policies_stay_clean() {
        require_release_mode_for_manual_policy_checks();
        const N: usize = 8;
        const K: usize = DEFAULT_COLLOCATION_DEGREE;
        let params = Params::default();
        let runtime = dc_runtime::<N, K>(&params);

        for (label, symbolic_functions) in [
            (
                "baseline",
                optimal_control::OcpSymbolicFunctionOptions::direct_collocation_default(),
            ),
            (
                "inline_all",
                optimal_control::OcpSymbolicFunctionOptions::inline_all(),
            ),
            (
                "at_call",
                optimal_control::OcpSymbolicFunctionOptions::function_all_with_call_policy(
                    optimization::CallPolicy::InlineAtCall,
                ),
            ),
            (
                "at_lowering",
                optimal_control::OcpSymbolicFunctionOptions::function_all_with_call_policy(
                    optimization::CallPolicy::InlineAtLowering,
                ),
            ),
            (
                "in_llvm",
                optimal_control::OcpSymbolicFunctionOptions::function_all_with_call_policy(
                    optimization::CallPolicy::InlineInLLVM,
                ),
            ),
            (
                "noinline_llvm",
                optimal_control::OcpSymbolicFunctionOptions::function_all_with_call_policy(
                    optimization::CallPolicy::NoInlineLLVM,
                ),
            ),
        ] {
            let compiled = model(DirectCollocation::<N, K>::default())
                .compile_jit_with_ocp_options(optimal_control::OcpCompileOptions {
                    function_options: optimization::FunctionCompileOptions::from(
                        optimization::LlvmOptimizationLevel::O0,
                    ),
                    symbolic_functions,
                    hessian_strategy: sx_core::HessianStrategy::LowerTriangleByColumn,
                })
                .expect("compile should succeed");

            let stats = compiled.nlp_compile_stats();
            let validation = compiled
                .validate_nlp_derivatives(
                    &runtime,
                    &vec![1.0; stats.equality_count],
                    &vec![1.0; stats.inequality_count],
                    optimization::FiniteDifferenceValidationOptions {
                        first_order_step: 1.0e-6,
                        second_order_step: 1.0e-4,
                        zero_tolerance: 1.0e-7,
                    },
                )
                .expect("finite-difference validation should succeed");
            let equality_jacobian = validation
                .equality_jacobian
                .as_ref()
                .expect("direct-collocation equality Jacobian summary should exist");
            let inequality_jacobian = validation
                .inequality_jacobian
                .as_ref()
                .expect("direct-collocation inequality Jacobian summary should exist");
            assert_jacobian_clean(label, "equality", equality_jacobian);
            assert_jacobian_clean(label, "inequality", inequality_jacobian);
        }
    }

    #[test]
    #[ignore = "manual direct-collocation Hessian policy regression check"]
    fn direct_collocation_hessian_policies_stay_clean() {
        require_release_mode_for_manual_hessian_checks();
        let names = dc_decision_layout_names();
        for (label, symbolic_functions) in [
            (
                "baseline",
                crate::benchmark_report::OcpBenchmarkPreset::Baseline.sx_function_config(),
            ),
            (
                "inline_all",
                crate::benchmark_report::OcpBenchmarkPreset::InlineAll.sx_function_config(),
            ),
            (
                "at_call",
                crate::benchmark_report::OcpBenchmarkPreset::FunctionInlineAtCall
                    .sx_function_config(),
            ),
            (
                "at_lowering",
                crate::benchmark_report::OcpBenchmarkPreset::FunctionInlineAtLowering
                    .sx_function_config(),
            ),
            (
                "in_llvm",
                crate::benchmark_report::OcpBenchmarkPreset::FunctionInlineInLlvm
                    .sx_function_config(),
            ),
            (
                "noinline_llvm",
                crate::benchmark_report::OcpBenchmarkPreset::FunctionNoInlineLlvm
                    .sx_function_config(),
            ),
        ] {
            let mut values = std::collections::BTreeMap::new();
            values.insert("transcription_method".to_string(), 1.0);
            let request = crate::common::DerivativeCheckRequest {
                values,
                finite_difference: optimization::FiniteDifferenceValidationOptions {
                    first_order_step: 1.0e-6,
                    second_order_step: 1.0e-4,
                    zero_tolerance: 1.0e-7,
                },
                sx_functions_override: Some(symbolic_functions),
                ..crate::common::DerivativeCheckRequest::default()
            };
            let check =
                validate_derivatives_from_request(&request).expect("derivative check should run");
            let summary = &check.report.lagrangian_hessian;
            let worst = summary
                .worst_entry
                .as_ref()
                .map(|entry| named_hessian_entry(&names, entry))
                .unwrap_or_else(|| "worst=none".to_string());
            let worst_missing = summary
                .sparsity
                .worst_missing_from_analytic
                .as_ref()
                .map(|entry| named_hessian_entry(&names, entry))
                .unwrap_or_else(|| "worst_missing=none".to_string());
            let worst_extra = summary
                .sparsity
                .worst_extra_in_analytic
                .as_ref()
                .map(|entry| named_hessian_entry(&names, entry))
                .unwrap_or_else(|| "worst_extra=none".to_string());
            assert!(
                summary
                    .is_within_tolerances(optimization::ValidationTolerances::new(1.0e-4, 1.0e-3)),
                "{label} expected clean glider direct-collocation Hessian\nworst: {worst}\nworst_missing: {worst_missing}\nworst_extra: {worst_extra}\nsummary: {summary:?}"
            );
        }
    }
}
