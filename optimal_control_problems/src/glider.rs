use crate::common::{
    CompileCacheStatus, CompileProgressInfo, CompileProgressUpdate, CompiledDirectCollocationOcp,
    CompiledMultipleShootingOcp, ContinuousInitialGuess, DirectCollocationRuntimeValues,
    DirectCollocationTimeGrid, DirectCollocationTrajectories, FromMap, LatexSection, MetricKey,
    MultipleShootingRuntimeValues, MultipleShootingTrajectories, OcpRuntimeSpec,
    OcpSxFunctionConfig, PlotMode, ProblemId, ProblemSpec, Scene2D, ScenePath, SolveArtifact,
    SolveStreamEvent, SolverConfig, SolverMethod, SolverReport, StandardOcpParams,
    TranscriptionConfig, chart, default_solver_config, default_solver_method,
    default_transcription, deg_to_rad, direct_collocation_runtime_from_spec, expect_finite,
    interval_arc_bound_series, interval_arc_series, metric_with_key,
    multiple_shooting_runtime_from_spec, node_times, numeric_metric_with_key,
    ocp_sx_function_config_from_map, problem_controls, problem_scientific_slider_control,
    problem_slider_control, problem_spec, rad_to_deg, sample_or_default, segmented_bound_series,
    segmented_series, select_control, solver_config_from_map, solver_method_from_map,
    transcription_from_map, transcription_metrics, trapezoid_integral,
};
use anyhow::{Result, anyhow};
use optimal_control::runtime::{
    DirectCollocation, MultipleShooting, direct_collocation_root_arcs,
    direct_collocation_state_like_arcs,
};
use optimal_control::{Bounds1D, FinalTime, InterpolatedTrajectory, IntervalArc, Ocp, OcpScaling};
use serde::Serialize;
use std::f64::consts::PI;
use sx_core::SX;
const RK4_SUBSTEPS: usize = 2;
const DEFAULT_INTERVALS: usize = 50;
const DEFAULT_COLLOCATION_DEGREE: usize = 3;
const SUPPORTED_INTERVALS: [usize; 7] = [10, 20, 30, 40, DEFAULT_INTERVALS, 60, 80];
const SUPPORTED_DEGREES: [usize; 5] = [1, 2, DEFAULT_COLLOCATION_DEGREE, 4, 5];
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

type MsCompiled = CompiledMultipleShootingOcp<
    State<SX>,
    Control<SX>,
    ModelParams<SX>,
    Path<SX>,
    Boundary<SX>,
    (),
>;

type DcCompiled = CompiledDirectCollocationOcp<
    State<SX>,
    Control<SX>,
    ModelParams<SX>,
    Path<SX>,
    Boundary<SX>,
    (),
>;

thread_local! {
    static MULTIPLE_SHOOTING_CACHE: std::cell::RefCell<
        crate::common::SharedCompileCache<crate::common::MultipleShootingCompileKey, MsCompiled>
    > = std::cell::RefCell::new(crate::common::SharedCompileCache::new());
    static DIRECT_COLLOCATION_CACHE: std::cell::RefCell<
        crate::common::SharedCompileCache<
            crate::common::DirectCollocationCompileVariantKey,
            DcCompiled,
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

fn cached_multiple_shooting(params: &Params) -> Result<crate::common::CachedCompile<MsCompiled>> {
    let intervals = params.transcription.intervals;
    MULTIPLE_SHOOTING_CACHE.with(|cache| {
        crate::common::cached_multiple_shooting_ocp_compile(
            &mut cache.borrow_mut(),
            intervals,
            params.sx_functions,
            |options| {
                model(MultipleShooting {
                    intervals,
                    rk4_substeps: RK4_SUBSTEPS,
                })
                .compile_jit_with_ocp_options(options)
            },
        )
    })
}

fn cached_direct_collocation(
    params: &Params,
    family: optimal_control::CollocationFamily,
) -> Result<crate::common::CachedCompile<DcCompiled>> {
    let intervals = params.transcription.intervals;
    let order = params.transcription.collocation_degree;
    DIRECT_COLLOCATION_CACHE.with(|cache| {
        crate::common::cached_direct_collocation_ocp_compile(
            &mut cache.borrow_mut(),
            intervals,
            order,
            family,
            params.transcription.time_grid,
            params.sx_functions,
            |options| {
                model(DirectCollocation {
                    intervals,
                    order,
                    family,
                    time_grid: params.transcription.time_grid,
                })
                .compile_jit_with_ocp_options(options)
            },
        )
    })
}

fn compile_multiple_shooting_with_progress(
    params: &Params,
    callback: &mut dyn FnMut(CompileProgressUpdate),
) -> Result<(
    std::rc::Rc<std::cell::RefCell<MsCompiled>>,
    CompileProgressInfo,
)> {
    let intervals = params.transcription.intervals;
    MULTIPLE_SHOOTING_CACHE.with(|cache| {
        crate::common::cached_multiple_shooting_ocp_compile_with_progress(
            &mut cache.borrow_mut(),
            intervals,
            params.sx_functions,
            callback,
            |options, on_progress| {
                model(MultipleShooting {
                    intervals,
                    rk4_substeps: RK4_SUBSTEPS,
                })
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
    std::rc::Rc<std::cell::RefCell<DcCompiled>>,
    CompileProgressInfo,
)> {
    let intervals = params.transcription.intervals;
    let order = params.transcription.collocation_degree;
    DIRECT_COLLOCATION_CACHE.with(|cache| {
        crate::common::cached_direct_collocation_ocp_compile_with_progress(
            &mut cache.borrow_mut(),
            intervals,
            order,
            family,
            params.transcription.time_grid,
            params.sx_functions,
            callback,
            |options, on_progress| {
                model(DirectCollocation {
                    intervals,
                    order,
                    family,
                    time_grid: params.transcription.time_grid,
                })
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
    crate::common::benchmark_standard_ocp_case_with_progress::<_, _, _, _, _, _, _, _>(
        ProblemId::OptimalDistanceGlider,
        PROBLEM_NAME,
        transcription,
        preset,
        eval_options,
        on_progress,
        |options, on_progress| {
            model(MultipleShooting {
                intervals: DEFAULT_INTERVALS,
                rk4_substeps: RK4_SUBSTEPS,
            })
            .compile_jit_with_ocp_options_and_progress_callback(options, on_progress)
        },
        |family, options, on_progress| {
            model(DirectCollocation {
                intervals: DEFAULT_INTERVALS,
                order: DEFAULT_COLLOCATION_DEGREE,
                family,
                time_grid: Default::default(),
            })
            .compile_jit_with_ocp_options_and_progress_callback(options, on_progress)
        },
        ms_runtime,
        dc_runtime,
    )
}

pub fn solve(params: &Params) -> Result<SolveArtifact> {
    crate::common::solve_standard_ocp::<_, _, _, _, _, _, _, _, _>(
        params,
        params.transcription.method,
        params.transcription.collocation_family,
        params.solver_method,
        &params.solver,
        cached_multiple_shooting,
        cached_direct_collocation,
        ms_runtime,
        dc_runtime,
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
    crate::common::solve_standard_ocp_with_progress::<_, _, _, _, _, _, _, _, _, _>(
        params,
        params.transcription.method,
        params.transcription.collocation_family,
        params.solver_method,
        &params.solver,
        emit,
        compile_multiple_shooting_with_progress,
        compile_direct_collocation_with_progress,
        ms_runtime,
        dc_runtime,
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
    crate::common::validate_standard_ocp_derivatives::<_, _, _, _, _, _, _>(
        ProblemId::OptimalDistanceGlider,
        PROBLEM_NAME,
        params,
        params.transcription.method,
        params.transcription.collocation_family,
        params.sx_functions,
        request,
        cached_multiple_shooting,
        cached_direct_collocation,
        ms_runtime,
        dc_runtime,
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
        global: FinalTime { tf },
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
        global: FinalTime {
            tf: GLIDER_FINAL_TIME_SCALE_S,
        },
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

fn ms_runtime(
    params: &Params,
) -> MultipleShootingRuntimeValues<
    ModelParams<f64>,
    Path<Bounds1D>,
    Boundary<f64>,
    (),
    State<f64>,
    Control<f64>,
> {
    let mut runtime = multiple_shooting_runtime_from_spec(runtime_spec(params));
    runtime.scaling = params.scaling_enabled.then(|| glider_scaling(params));
    runtime
}

fn dc_runtime(
    params: &Params,
) -> DirectCollocationRuntimeValues<
    ModelParams<f64>,
    Path<Bounds1D>,
    Boundary<f64>,
    (),
    State<f64>,
    Control<f64>,
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

fn artifact_from_ms_trajectories(
    params: &Params,
    trajectories: &MultipleShootingTrajectories<State<f64>, Control<f64>>,
    x_arcs: &[IntervalArc<State<f64>>],
    u_arcs: &[IntervalArc<Control<f64>>],
) -> SolveArtifact {
    let tf = trajectories.tf;
    let intervals = trajectories.interval_count();
    let times = node_times(tf, intervals);
    let mut altitude = Vec::with_capacity(intervals + 1);
    let mut alpha_rate = Vec::with_capacity(intervals + 1);
    let mut x = Vec::with_capacity(intervals + 1);
    let mut y = Vec::with_capacity(intervals + 1);
    for index in 0..intervals {
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

fn artifact_from_dc_trajectories(
    params: &Params,
    trajectories: &DirectCollocationTrajectories<State<f64>, Control<f64>>,
    time_grid: &DirectCollocationTimeGrid,
) -> SolveArtifact {
    let x_arcs =
        direct_collocation_state_like_arcs(&trajectories.x, &trajectories.root_x, time_grid)
            .expect("collocation state arcs should match trajectory layout");
    let u_arcs =
        direct_collocation_state_like_arcs(&trajectories.u, &trajectories.root_u, time_grid)
            .expect("collocation control-state arcs should match trajectory layout");
    let dudt_arcs = direct_collocation_root_arcs(&trajectories.root_dudt, time_grid);
    let intervals = trajectories.x.nodes.len();
    let mut x = Vec::with_capacity(intervals + 1);
    let mut y = Vec::with_capacity(intervals + 1);
    let mut peak_altitude = f64::NEG_INFINITY;
    for index in 0..intervals {
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
    let root_rates = trajectories
        .root_dudt
        .intervals
        .iter()
        .flat_map(|interval| interval.iter().map(|rate| rad_to_deg(rate.alpha)))
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
    use ssids_rs::{
        FactorProfile, NativeSpral, NativeSpralSession, NumericFactorOptions, OrderingStrategy,
        SolveProfile, SsidsOptions, SymbolicFactor, SymmetricCscMatrix, analyse as spral_analyse,
        approximate_minimum_degree_permutation as spral_amd_permutation,
        factorize as spral_factorize, factorize_with_profile as spral_factorize_with_profile,
    };
    use std::collections::BTreeMap;
    #[cfg(feature = "ipopt")]
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::Path;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    #[allow(dead_code)]
    #[derive(Debug)]
    struct GliderLinearDebugDump {
        matrix_dimension: usize,
        x_dimension: usize,
        inequality_dimension: usize,
        equality_dimension: usize,
        p_offset: usize,
        lambda_offset: usize,
        z_offset: usize,
        barrier_parameter: f64,
        x_full_indices: Vec<usize>,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f64>,
        rhs: Vec<f64>,
        r_dual: Vec<f64>,
        bound_diagonal: Vec<f64>,
        bound_rhs: Vec<f64>,
        slack: Vec<f64>,
        multipliers: Vec<f64>,
        linear_solution_final: Option<Vec<f64>>,
        linear_solution_prefinal: Option<Vec<f64>>,
        linear_trace_rhs_prefinal: Option<Vec<f64>>,
        linear_trace_solution_prefinal_unrefined: Option<Vec<f64>>,
        linear_trace_refinement_rhs: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_solution: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_accumulated_solution: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_ratios: Option<Vec<[f64; 2]>>,
        linear_trace_refinement_residual_x: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_x_after_w: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_x_after_jc: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_x_after_jd: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_x_after_pxl: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_x_after_pxu: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_x_after_add_two_vectors: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_s: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_s_after_pdu: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_s_after_pdl: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_s_after_add_two_vectors: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_s_after_delta: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_c: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_d: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_z_lower: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_z_upper: Option<Vec<Vec<f64>>>,
        linear_trace_refinement_residual_v_upper: Option<Vec<Vec<f64>>>,
    }

    #[cfg(feature = "ipopt")]
    #[derive(Debug)]
    struct IpoptSpralInterfaceDump {
        call_index: usize,
        ndim: usize,
        nonzeros: usize,
        nrhs: usize,
        control_ordering: i32,
        control_scaling: i32,
        info_num_neg: Option<i32>,
        info_num_delay: Option<i32>,
        info_num_two: Option<i32>,
        ia: Vec<usize>,
        ja: Vec<usize>,
        values: Vec<f64>,
        rhs: Vec<f64>,
        scaling: Option<Vec<f64>>,
    }

    #[cfg(feature = "ipopt")]
    #[derive(Debug)]
    struct IpoptFullSpaceResidualDump {
        call_index: usize,
        resid_x: Vec<f64>,
        resid_x_after_w: Vec<f64>,
        resid_x_after_jc: Vec<f64>,
        resid_x_after_jd: Vec<f64>,
        resid_x_after_pxl: Vec<f64>,
        resid_x_after_pxu: Vec<f64>,
        resid_x_after_add_two_vectors: Vec<f64>,
        resid_s: Vec<f64>,
        resid_s_after_pdu: Vec<f64>,
        resid_s_after_pdl: Vec<f64>,
        resid_s_after_add_two_vectors: Vec<f64>,
        resid_s_after_delta: Vec<f64>,
    }

    #[derive(Debug)]
    struct ExactAugmentedReplay {
        factor_time: Duration,
        solve_time: Duration,
        residual_inf: f64,
        solution_inf: f64,
        solution: Vec<f64>,
        inertia: String,
        factor_profile: Option<FactorProfile>,
        solve_profile: Option<SolveProfile>,
    }

    #[derive(Clone, Debug)]
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
            .unwrap_or_else(|| panic!("expected dump key {prefix:?} to be present"))
            .parse::<T>()
            .unwrap_or_else(|error| panic!("expected dump scalar {prefix:?} to parse: {error:?}"))
    }

    fn parse_dump_vec<T>(text: &str, prefix: &str) -> Vec<T>
    where
        T: DeserializeOwned,
    {
        let value = text
            .lines()
            .find_map(|line| line.strip_prefix(prefix))
            .unwrap_or_else(|| panic!("expected dump vector {prefix:?} to be present"));
        serde_json::from_str(value)
            .unwrap_or_else(|error| panic!("expected dump vector {prefix:?} to parse: {error:?}"))
    }

    fn parse_optional_dump_vec<T>(text: &str, prefix: &str) -> Option<Vec<T>>
    where
        T: DeserializeOwned,
    {
        let value = text.lines().find_map(|line| line.strip_prefix(prefix))?;
        Some(serde_json::from_str(value).expect("expected optional dump vector to parse"))
    }

    #[cfg(feature = "ipopt")]
    fn parse_optional_dump_value<T>(text: &str, prefix: &str) -> Option<T>
    where
        T: std::str::FromStr,
        T::Err: std::fmt::Debug,
    {
        text.lines()
            .find_map(|line| line.strip_prefix(prefix))
            .map(|value| {
                value
                    .parse::<T>()
                    .expect("expected optional dump scalar to parse")
            })
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
            barrier_parameter: parse_dump_value(&text, "barrier_parameter="),
            x_full_indices: parse_optional_dump_vec(&text, "x_full_indices=")
                .unwrap_or_else(|| (0..parse_dump_value(&text, "x_dimension=")).collect()),
            col_ptrs: parse_dump_vec(&text, "col_ptrs="),
            row_indices: parse_dump_vec(&text, "row_indices="),
            values: parse_dump_vec(&text, "values="),
            rhs: parse_dump_vec(&text, "rhs="),
            r_dual: parse_dump_vec(&text, "r_dual="),
            bound_diagonal: parse_optional_dump_vec(&text, "bound_diagonal=")
                .unwrap_or_else(|| vec![0.0; parse_dump_value(&text, "x_dimension=")]),
            bound_rhs: parse_dump_vec(&text, "bound_rhs="),
            slack: parse_dump_vec(&text, "slack="),
            multipliers: parse_dump_vec(&text, "multipliers="),
            linear_solution_final: parse_optional_dump_vec(&text, "linear_solution_final="),
            linear_solution_prefinal: parse_optional_dump_vec(&text, "linear_solution_prefinal="),
            linear_trace_rhs_prefinal: parse_optional_dump_vec(&text, "linear_trace_rhs_prefinal="),
            linear_trace_solution_prefinal_unrefined: parse_optional_dump_vec(
                &text,
                "linear_trace_solution_prefinal_unrefined=",
            ),
            linear_trace_refinement_rhs: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_rhs=",
            ),
            linear_trace_refinement_solution: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_solution=",
            ),
            linear_trace_refinement_accumulated_solution: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_accumulated_solution=",
            ),
            linear_trace_refinement_residual_ratios: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_ratios=",
            ),
            linear_trace_refinement_residual_x: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_x=",
            ),
            linear_trace_refinement_residual_x_after_w: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_x_after_w=",
            ),
            linear_trace_refinement_residual_x_after_jc: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_x_after_jc=",
            ),
            linear_trace_refinement_residual_x_after_jd: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_x_after_jd=",
            ),
            linear_trace_refinement_residual_x_after_pxl: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_x_after_pxl=",
            ),
            linear_trace_refinement_residual_x_after_pxu: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_x_after_pxu=",
            ),
            linear_trace_refinement_residual_x_after_add_two_vectors: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_x_after_add_two_vectors=",
            ),
            linear_trace_refinement_residual_s: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_s=",
            ),
            linear_trace_refinement_residual_s_after_pdu: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_s_after_pdu=",
            ),
            linear_trace_refinement_residual_s_after_pdl: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_s_after_pdl=",
            ),
            linear_trace_refinement_residual_s_after_add_two_vectors: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_s_after_add_two_vectors=",
            ),
            linear_trace_refinement_residual_s_after_delta: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_s_after_delta=",
            ),
            linear_trace_refinement_residual_c: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_c=",
            ),
            linear_trace_refinement_residual_d: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_d=",
            ),
            linear_trace_refinement_residual_z_lower: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_z_lower=",
            ),
            linear_trace_refinement_residual_z_upper: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_z_upper=",
            ),
            linear_trace_refinement_residual_v_upper: parse_optional_dump_vec(
                &text,
                "linear_trace_refinement_residual_v_upper=",
            ),
        }
    }

    #[cfg(feature = "ipopt")]
    fn load_ipopt_spral_interface_dump(path: &Path) -> IpoptSpralInterfaceDump {
        let text = fs::read_to_string(path).expect("expected IPOPT SPRAL dump to exist");
        IpoptSpralInterfaceDump {
            call_index: parse_dump_value(&text, "call_index="),
            ndim: parse_dump_value(&text, "ndim="),
            nonzeros: parse_dump_value(&text, "nonzeros="),
            nrhs: parse_dump_value(&text, "nrhs="),
            control_ordering: parse_dump_value(&text, "control_ordering="),
            control_scaling: parse_dump_value(&text, "control_scaling="),
            info_num_neg: parse_optional_dump_value(&text, "info_num_neg="),
            info_num_delay: parse_optional_dump_value(&text, "info_num_delay="),
            info_num_two: parse_optional_dump_value(&text, "info_num_two="),
            ia: parse_dump_vec(&text, "ia="),
            ja: parse_dump_vec(&text, "ja="),
            values: parse_dump_vec(&text, "values="),
            rhs: parse_dump_vec(&text, "rhs="),
            scaling: parse_optional_dump_vec(&text, "scaling="),
        }
    }

    #[cfg(feature = "ipopt")]
    fn try_load_ipopt_full_space_residual_dump(path: &Path) -> Option<IpoptFullSpaceResidualDump> {
        let text = fs::read_to_string(path).ok()?;
        Some(IpoptFullSpaceResidualDump {
            call_index: parse_optional_dump_value(&text, "call_index=")?,
            resid_x: parse_optional_dump_vec(&text, "resid_x=")?,
            resid_x_after_w: parse_optional_dump_vec(&text, "resid_x_after_W=")?,
            resid_x_after_jc: parse_optional_dump_vec(&text, "resid_x_after_Jc=")?,
            resid_x_after_jd: parse_optional_dump_vec(&text, "resid_x_after_Jd=")?,
            resid_x_after_pxl: parse_optional_dump_vec(&text, "resid_x_after_PxL=")?,
            resid_x_after_pxu: parse_optional_dump_vec(&text, "resid_x_after_PxU=")?,
            resid_x_after_add_two_vectors: parse_optional_dump_vec(
                &text,
                "resid_x_after_AddTwoVectors=",
            )?,
            resid_s: parse_optional_dump_vec(&text, "resid_s=")?,
            resid_s_after_pdu: parse_optional_dump_vec(&text, "resid_s_after_PdU=")?,
            resid_s_after_pdl: parse_optional_dump_vec(&text, "resid_s_after_PdL=")?,
            resid_s_after_add_two_vectors: parse_optional_dump_vec(
                &text,
                "resid_s_after_AddTwoVectors=",
            )?,
            resid_s_after_delta: parse_optional_dump_vec(&text, "resid_s_after_delta=")?,
        })
    }

    fn failure_linear_debug_report<T>(
        result: &std::result::Result<T, optimization::InteriorPointSolveError>,
    ) -> Option<optimization::InteriorPointLinearDebugReport> {
        let Err(error) = result else {
            return None;
        };
        let context = match error {
            optimization::InteriorPointSolveError::LinearSolve { context, .. }
            | optimization::InteriorPointSolveError::LineSearchFailed { context, .. }
            | optimization::InteriorPointSolveError::RestorationFailed { context, .. }
            | optimization::InteriorPointSolveError::LocalInfeasibility { context }
            | optimization::InteriorPointSolveError::DivergingIterates { context, .. }
            | optimization::InteriorPointSolveError::CpuTimeExceeded { context, .. }
            | optimization::InteriorPointSolveError::WallTimeExceeded { context, .. }
            | optimization::InteriorPointSolveError::UserRequestedStop { context }
            | optimization::InteriorPointSolveError::SearchDirectionTooSmall { context }
            | optimization::InteriorPointSolveError::MaxIterations { context, .. } => context,
            optimization::InteriorPointSolveError::InvalidInput(_) => return None,
        };
        context
            .failed_linear_solve
            .as_ref()
            .and_then(|diagnostics| diagnostics.debug_report.clone())
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
        let (symbolic, _) = spral_analyse(
            structure,
            &SsidsOptions {
                ordering: OrderingStrategy::ApproximateMinimumDegree,
            },
        )
        .expect("rust spral analyse should succeed on dumped KKT");
        replay_rust_augmented_spral_with_symbolic(dump, &symbolic)
    }

    fn replay_rust_augmented_spral_with_symbolic(
        dump: &GliderLinearDebugDump,
        symbolic: &SymbolicFactor,
    ) -> ExactAugmentedReplay {
        let factor_started = Instant::now();
        let numeric = SymmetricCscMatrix::new(
            dump.matrix_dimension,
            &dump.col_ptrs,
            &dump.row_indices,
            Some(&dump.values),
        )
        .expect("dumped augmented CSC values should validate");
        let (mut factor, _, factor_profile) =
            spral_factorize_with_profile(numeric, symbolic, &NumericFactorOptions::default())
                .expect("rust spral factorization should succeed on dumped KKT");
        let factor_time = factor_started.elapsed();
        let solve_started = Instant::now();
        let (mut solution, mut solve_profile) = factor
            .solve_with_profile(&dump.rhs)
            .expect("rust spral solve should succeed on dumped KKT");
        let mut solve_time = solve_started.elapsed();
        for _ in 0..10 {
            let residual = residual_vector(dump, &solution);
            if residual.iter().all(|value| value.abs() <= f64::EPSILON) {
                break;
            }
            let correction_started = Instant::now();
            let (correction, correction_profile) = factor
                .solve_with_profile(&residual)
                .expect("rust spral iterative refinement should succeed on dumped KKT");
            solve_time += correction_started.elapsed();
            solve_profile.accumulate(&correction_profile);
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
            factor_profile: Some(factor_profile),
            solve_profile: Some(solve_profile),
        }
    }

    fn replay_rust_augmented_spral_unprofiled_with_symbolic(
        dump: &GliderLinearDebugDump,
        symbolic: &SymbolicFactor,
    ) -> ExactAugmentedReplay {
        let factor_started = Instant::now();
        let numeric = SymmetricCscMatrix::new(
            dump.matrix_dimension,
            &dump.col_ptrs,
            &dump.row_indices,
            Some(&dump.values),
        )
        .expect("dumped augmented CSC values should validate");
        let (mut factor, _) = spral_factorize(numeric, symbolic, &NumericFactorOptions::default())
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
            factor_profile: None,
            solve_profile: None,
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
        let native = NativeSpral::load().expect("native SPRAL should be available locally");
        let rust_initial_order =
            spral_amd_permutation(structure).expect("rust AMD order should succeed on dumped KKT");
        let mut session = native
            .analyse_with_options_and_user_ordering(
                structure,
                &NumericFactorOptions::default(),
                rust_initial_order.inverse(),
            )
            .expect("native spral analyse should succeed on dumped KKT");
        replay_native_augmented_spral_with_session(dump, &mut session)
    }

    fn replay_native_augmented_spral_with_session(
        dump: &GliderLinearDebugDump,
        session: &mut NativeSpralSession,
    ) -> ExactAugmentedReplay {
        let factor_started = Instant::now();
        let numeric = SymmetricCscMatrix::new(
            dump.matrix_dimension,
            &dump.col_ptrs,
            &dump.row_indices,
            Some(&dump.values),
        )
        .expect("dumped augmented CSC values should validate");
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
            factor_profile: None,
            solve_profile: None,
        }
    }

    fn load_current_glider_iteration0_augmented_dump() -> GliderLinearDebugDump {
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
        let runtime = dc_runtime(&params);
        let dump_dir = TempDir::new().expect("temp dump dir should create");
        let mut options = crate::common::nlip_options(&params.solver);
        options.linear_solver = optimization::InteriorPointLinearSolver::SsidsRs;
        options.max_iters = 1;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![optimization::InteriorPointLinearSolver::SpralSrc],
            schedule: optimization::InteriorPointLinearDebugSchedule::FirstIteration,
            dump_dir: Some(dump_dir.path().to_path_buf()),
        });

        let _ = compiled.solve_interior_point_with_callback(&runtime, &options, |_snapshot| {});
        load_glider_linear_debug_dump(&dump_dir.path().join("nlip_kkt_iter_0000.txt"))
    }

    fn glider_iteration0_rust_symbolic(dump: &GliderLinearDebugDump) -> SymbolicFactor {
        let structure = SymmetricCscMatrix::new(
            dump.matrix_dimension,
            &dump.col_ptrs,
            &dump.row_indices,
            None,
        )
        .expect("dumped augmented CSC should validate");
        spral_analyse(
            structure,
            &SsidsOptions {
                ordering: OrderingStrategy::ApproximateMinimumDegree,
            },
        )
        .expect("rust spral analyse should succeed on dumped KKT")
        .0
    }

    fn median_duration(values: &[Duration]) -> Duration {
        assert!(!values.is_empty());
        let mut values = values.to_vec();
        values.sort_unstable();
        values[values.len() / 2]
    }

    fn median_profile_duration(
        profiles: &[FactorProfile],
        select: impl Fn(&FactorProfile) -> Duration,
    ) -> Duration {
        let values = profiles.iter().map(select).collect::<Vec<_>>();
        median_duration(&values)
    }

    fn median_profile_usize(
        profiles: &[FactorProfile],
        select: impl Fn(&FactorProfile) -> usize,
    ) -> usize {
        assert!(!profiles.is_empty());
        let mut values = profiles.iter().map(select).collect::<Vec<_>>();
        values.sort_unstable();
        values[values.len() / 2]
    }

    fn median_solve_profile_duration(
        profiles: &[SolveProfile],
        select: impl Fn(&SolveProfile) -> Duration,
    ) -> Duration {
        let values = profiles.iter().map(select).collect::<Vec<_>>();
        median_duration(&values)
    }

    fn native_spral_debug_result(
        report: &optimization::InteriorPointLinearDebugReport,
    ) -> &optimization::InteriorPointLinearDebugBackendResult {
        report
            .results
            .iter()
            .find(|result| result.solver == optimization::InteriorPointLinearSolver::SpralSrc)
            .expect("expected native SPRAL comparison result")
    }

    #[test]
    fn glider_runtime_scaling_defaults_match_problem_tuning() {
        let params = Params::default();
        let runtime = ms_runtime(&params);
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
            scaling.global.tf,
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
        let runtime = ms_runtime(&Params {
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
        let runtime = dc_runtime(&params);
        let compiled = model(DirectCollocation {
            intervals: N,
            order: K,
            family,
            time_grid: Default::default(),
        })
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
                status:
                    optimization::SqpQpRawStatus::InsufficientProgress
                    | optimization::SqpQpRawStatus::NumericalError,
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
        let runtime = dc_runtime(&params);
        let compiled = model(DirectCollocation {
            intervals: N,
            order: K,
            family,
            time_grid: Default::default(),
        })
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
        let runtime = dc_runtime(&params);
        let compiled = model(DirectCollocation {
            intervals: N,
            order: K,
            family,
            time_grid: Default::default(),
        })
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
        let runtime = dc_runtime(&params);
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
        let runtime = dc_runtime(&params);
        let mut options = crate::common::nlip_options(&params.solver);
        options.max_iters = 120;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![
                optimization::InteriorPointLinearSolver::SpralSrc,
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
        let runtime = dc_runtime(&params);
        let mut options = crate::common::nlip_options(&params.solver);
        options.linear_solver = optimization::InteriorPointLinearSolver::SpralSrc;
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
        let runtime = dc_runtime(&params);
        let dump_dir = TempDir::new().expect("temp dump dir should create");
        let mut options = crate::common::nlip_options(&params.solver);
        options.linear_solver = optimization::InteriorPointLinearSolver::SsidsRs;
        options.max_iters = 1;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![
                optimization::InteriorPointLinearSolver::SpralSrc,
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
        if first_report.is_none() {
            first_report = failure_linear_debug_report(&result);
        }

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
        if let Some(profile) = &rust_exact.factor_profile {
            println!(
                "  rust_spral factor_profile symbolic_tree={:?} pattern={:?} values={:?} fronts={:?} root_delayed={:?} inverse={:?} lower_storage={:?} solve_panel_storage={:?} diagonal_storage={:?} bytes={:?} recorded={:?}",
                profile.symbolic_front_tree_time,
                profile.permuted_pattern_time,
                profile.permuted_values_time,
                profile.front_factorization_time,
                profile.root_delayed_factorization_time,
                profile.factor_inverse_time,
                profile.lower_storage_time,
                profile.solve_panel_storage_time,
                profile.diagonal_storage_time,
                profile.factor_bytes_time,
                profile.total_recorded_time(),
            );
            println!(
                "  rust_spral front_profile assembly={:?} dense_factor={:?} fronts={} local_dense_entries={} root_delayed_blocks={}",
                profile.front_assembly_time,
                profile.dense_front_factorization_time,
                profile.front_count,
                profile.local_dense_entries,
                profile.root_delayed_blocks,
            );
            println!(
                "  rust_spral front_detail_profile child_merge={:?} row_setup={:?} matrix_entries={:?} contribution={:?} local_merge={:?} zeroed_bytes={} child_results={} child_contrib_entries={} child_factor_cols={} child_solve_panels={} local_factor_cols={} local_solve_panels={}",
                profile.front_child_result_merge_time,
                profile.front_row_setup_time,
                profile.front_matrix_entry_assembly_time,
                profile.front_contribution_assembly_time,
                profile.front_local_result_merge_time,
                profile.local_dense_bytes_zeroed,
                profile.front_child_result_count,
                profile.front_child_contribution_entries,
                profile.front_child_factor_columns,
                profile.front_child_solve_panels,
                profile.front_local_factor_columns,
                profile.front_local_solve_panels,
            );
            println!(
                "  rust_spral small_leaf_profile subtrees={} fronts={} app_fronts={} columns={} dense_entries={} small_leaf_tpp={:?} small_leaf_pivot_search={:?} small_leaf_pivot_factor={:?} small_leaf_pivot_scale={:?} small_leaf_pivot_update={:?} small_leaf_calc_ld={:?} small_leaf_gemm={:?} small_leaf_pack={:?} small_leaf_solve_panel={:?} small_leaf_output={:?}",
                profile.spral_small_leaf_subtrees,
                profile.spral_small_leaf_fronts,
                profile.spral_small_leaf_app_fronts,
                profile.spral_small_leaf_columns,
                profile.spral_small_leaf_dense_entries,
                profile.spral_small_leaf_tpp_time,
                profile.spral_small_leaf_pivot_search_time,
                profile.spral_small_leaf_pivot_factor_time,
                profile.spral_small_leaf_pivot_scale_time,
                profile.spral_small_leaf_pivot_update_time,
                profile.spral_small_leaf_contribution_calc_ld_time,
                profile.spral_small_leaf_contribution_gemm_time,
                profile.spral_small_leaf_contribution_pack_time,
                profile.spral_small_leaf_solve_panel_build_time,
                profile.spral_small_leaf_output_append_time,
            );
            println!(
                "  rust_spral dense_front_profile tpp={:?} tpp_pivot_search={:?} tpp_pivot_factor={:?} tpp_column_storage={:?} tpp_contribution_pack={:?} app_pivot_factor={:?} app_maxloc={:?} app_swap={:?} app_pivot_update={:?} app_block_apply={:?} app_block_trsm={:?} app_block_diag={:?} app_failed_scan={:?} app_backup={:?} app_restore={:?} app_accepted_update={:?} app_accepted_ld={:?} app_accepted_gemm={:?} app_column_storage={:?} solve_panel_build={:?}",
                profile.tpp_factorization_time,
                profile.tpp_pivot_search_time,
                profile.tpp_pivot_factor_time,
                profile.tpp_column_storage_time,
                profile.tpp_contribution_pack_time,
                profile.app_pivot_factor_time,
                profile.app_maxloc_time,
                profile.app_symmetric_swap_time,
                profile.app_pivot_update_time,
                profile.app_block_pivot_apply_time,
                profile.app_block_triangular_solve_time,
                profile.app_block_diagonal_apply_time,
                profile.app_failed_pivot_scan_time,
                profile.app_backup_time,
                profile.app_restore_time,
                profile.app_accepted_update_time,
                profile.app_accepted_ld_time,
                profile.app_accepted_gemm_time,
                profile.app_column_storage_time,
                profile.solve_panel_build_time,
            );
            println!(
                "  rust_spral dense_front_counters tpp_fronts={} tpp_pivots={} tpp_factor_entries={} app_fronts={} app_panels={} app_maxloc_calls={} app_swaps={} app_1x1={} app_2x2={} app_zero={} app_diag_1x1={} app_offdiag_1x1={} app_front_le32={} app_front_33_64={} app_front_65_96={} app_front_97_128={} app_front_129_160={} app_front_161_256={} app_front_257_512={} app_front_gt512={}",
                profile.tpp_front_count,
                profile.tpp_pivots,
                profile.tpp_factor_column_entries,
                profile.app_front_count,
                profile.app_panel_count,
                profile.app_maxloc_calls,
                profile.app_symmetric_swaps,
                profile.app_one_by_one_pivots,
                profile.app_two_by_two_pivots,
                profile.app_zero_pivots,
                profile.app_diagonal_one_by_one_pivots,
                profile.app_offdiag_one_by_one_pivots,
                profile.app_front_size_histogram[0],
                profile.app_front_size_histogram[1],
                profile.app_front_size_histogram[2],
                profile.app_front_size_histogram[3],
                profile.app_front_size_histogram[4],
                profile.app_front_size_histogram[5],
                profile.app_front_size_histogram[6],
                profile.app_front_size_histogram[7],
            );
        }
        if let Some(profile) = &rust_exact.solve_profile {
            println!(
                "  rust_spral solve_profile input_perm={:?} forward={:?} diagonal={:?} backward={:?} output_perm={:?} recorded={:?}",
                profile.input_permutation_time,
                profile.forward_substitution_time,
                profile.diagonal_solve_time,
                profile.backward_substitution_time,
                profile.output_permutation_time,
                profile.total_recorded_time(),
            );
            println!(
                "  rust_spral backward_profile trailing_update={:?} triangular={:?} trailing_columns={} trailing_dense_entries={} triangular_columns={} triangular_dense_entries={}",
                profile.backward_trailing_update_time,
                profile.backward_triangular_solve_time,
                profile.backward_trailing_update_columns,
                profile.backward_trailing_update_dense_entries,
                profile.backward_triangular_columns,
                profile.backward_triangular_dense_entries,
            );
        }
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
    #[ignore = "manual in-process glider SSIDS native-vs-rust timing helper"]
    fn print_current_glider_nlip_iteration0_augmented_inprocess_profile() {
        if NativeSpral::load().is_err() {
            eprintln!("skipping glider in-process SPRAL profile: native library unavailable");
            return;
        }
        let repeats = std::env::var("SSIDS_GLIDER_INPROCESS_REPEATS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(15)
            .max(1);
        let profile_side =
            std::env::var("SSIDS_GLIDER_INPROCESS_SIDE").unwrap_or_else(|_| "paired".to_string());
        let print_samples = std::env::var("SSIDS_GLIDER_INPROCESS_PRINT_SAMPLES")
            .ok()
            .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
        let rotate_order = std::env::var("SSIDS_GLIDER_INPROCESS_ROTATE_ORDER")
            .ok()
            .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
        let (run_rust, run_native) = match profile_side.as_str() {
            "paired" => (true, true),
            "rust" => (true, false),
            "native" => (false, true),
            other => {
                panic!("SSIDS_GLIDER_INPROCESS_SIDE must be paired, rust, or native; got {other:?}")
            }
        };
        let dump = load_current_glider_iteration0_augmented_dump();
        let symbolic = glider_iteration0_rust_symbolic(&dump);
        let structure = SymmetricCscMatrix::new(
            dump.matrix_dimension,
            &dump.col_ptrs,
            &dump.row_indices,
            None,
        )
        .expect("dumped augmented CSC should validate");
        let native = NativeSpral::load().expect("native SPRAL should be available locally");
        let rust_initial_order =
            spral_amd_permutation(structure).expect("rust AMD order should succeed on dumped KKT");
        let mut native_session = native
            .analyse_with_options_and_user_ordering(
                structure,
                &NumericFactorOptions::default(),
                rust_initial_order.inverse(),
            )
            .expect("native spral analyse should succeed on dumped KKT");

        let warmup_rust = replay_rust_augmented_spral_with_symbolic(&dump, &symbolic);
        let warmup_native = replay_native_augmented_spral_with_session(&dump, &mut native_session);
        assert_eq!(warmup_rust.inertia, warmup_native.inertia);
        assert!(
            delta_inf(&warmup_rust.solution, &warmup_native.solution) <= 1e-12,
            "warmup augmented solution delta too large"
        );
        let warmup_rust_blocks = augmented_step_blocks(&dump, &warmup_rust.solution);
        let warmup_native_blocks = augmented_step_blocks(&dump, &warmup_native.solution);
        eprintln!(
            "ssids_glider_side_profile_begin side={profile_side} repeats={repeats} print_samples={print_samples} rotate_order={rotate_order}"
        );

        let mut rust_factor_times = Vec::with_capacity(repeats);
        let mut rust_solve_times = Vec::with_capacity(repeats);
        let mut rust_unprofiled_factor_times = Vec::with_capacity(repeats);
        let mut rust_unprofiled_solve_times = Vec::with_capacity(repeats);
        let mut native_factor_times = Vec::with_capacity(repeats);
        let mut native_solve_times = Vec::with_capacity(repeats);
        let mut rust_factor_profiles = Vec::with_capacity(repeats);
        let mut rust_solve_profiles = Vec::with_capacity(repeats);
        let mut final_solution_delta = 0.0;
        let mut final_dx_delta = 0.0;
        let mut final_dp_delta = 0.0;
        let mut final_ds_delta = 0.0;
        let mut final_dlambda_delta = 0.0;
        let mut final_dz_delta = 0.0;

        for sample_index in 0..repeats {
            let mut rust_exact = None;
            let mut rust_unprofiled_exact = None;
            let mut native_exact = None;
            let order_offset = if rotate_order { sample_index % 3 } else { 0 };
            for order_slot in 0..3 {
                match (order_slot + order_offset) % 3 {
                    0 if run_rust => {
                        rust_exact =
                            Some(replay_rust_augmented_spral_with_symbolic(&dump, &symbolic));
                    }
                    1 if run_rust => {
                        rust_unprofiled_exact = Some(
                            replay_rust_augmented_spral_unprofiled_with_symbolic(&dump, &symbolic),
                        );
                    }
                    2 if run_native => {
                        native_exact = Some(replay_native_augmented_spral_with_session(
                            &dump,
                            &mut native_session,
                        ));
                    }
                    _ => {}
                }
            }
            let rust_reference = rust_exact.as_ref().unwrap_or(&warmup_rust);
            let native_reference = native_exact.as_ref().unwrap_or(&warmup_native);
            assert_eq!(rust_reference.inertia, native_reference.inertia);
            assert!(
                rust_reference.residual_inf <= 1e-10,
                "rust residual too large: {rust_reference:?}"
            );
            assert!(
                native_reference.residual_inf <= 1e-10,
                "native residual too large: {native_reference:?}"
            );
            let rust_blocks = rust_exact
                .as_ref()
                .map(|exact| augmented_step_blocks(&dump, &exact.solution))
                .unwrap_or_else(|| warmup_rust_blocks.clone());
            let native_blocks = native_exact
                .as_ref()
                .map(|exact| augmented_step_blocks(&dump, &exact.solution))
                .unwrap_or_else(|| warmup_native_blocks.clone());
            final_solution_delta = delta_inf(&rust_reference.solution, &native_reference.solution);
            final_dx_delta = delta_inf(&rust_blocks.dx, &native_blocks.dx);
            final_dp_delta = delta_inf(&rust_blocks.p, &native_blocks.p);
            final_ds_delta = delta_inf(&rust_blocks.ds, &native_blocks.ds);
            final_dlambda_delta = delta_inf(&rust_blocks.d_lambda, &native_blocks.d_lambda);
            final_dz_delta = delta_inf(&rust_blocks.dz, &native_blocks.dz);
            assert!(
                final_solution_delta <= 1e-12,
                "augmented solution delta too large"
            );
            if let Some(rust_unprofiled_exact) = &rust_unprofiled_exact {
                assert_eq!(rust_unprofiled_exact.inertia, native_reference.inertia);
                assert!(
                    rust_unprofiled_exact.residual_inf <= 1e-10,
                    "unprofiled rust residual too large: {rust_unprofiled_exact:?}"
                );
                assert!(
                    delta_inf(&rust_unprofiled_exact.solution, &native_reference.solution) <= 1e-12,
                    "unprofiled augmented solution delta too large"
                );
            }

            if let Some(rust_exact) = rust_exact {
                if print_samples {
                    println!(
                        "  ssids_glider_sample index={} impl=rust profile=profiled factor={:?} solve={:?} residual={:.3e} solution_inf={:.3e} inertia={}",
                        sample_index,
                        rust_exact.factor_time,
                        rust_exact.solve_time,
                        rust_exact.residual_inf,
                        rust_exact.solution_inf,
                        rust_exact.inertia,
                    );
                }
                rust_factor_times.push(rust_exact.factor_time);
                rust_solve_times.push(rust_exact.solve_time);
                rust_factor_profiles.push(rust_exact.factor_profile.expect("rust factor profile"));
                rust_solve_profiles.push(rust_exact.solve_profile.expect("rust solve profile"));
            }
            if let Some(rust_unprofiled_exact) = rust_unprofiled_exact {
                if print_samples {
                    println!(
                        "  ssids_glider_sample index={} impl=rust profile=unprofiled factor={:?} solve={:?} residual={:.3e} solution_inf={:.3e} inertia={}",
                        sample_index,
                        rust_unprofiled_exact.factor_time,
                        rust_unprofiled_exact.solve_time,
                        rust_unprofiled_exact.residual_inf,
                        rust_unprofiled_exact.solution_inf,
                        rust_unprofiled_exact.inertia,
                    );
                }
                rust_unprofiled_factor_times.push(rust_unprofiled_exact.factor_time);
                rust_unprofiled_solve_times.push(rust_unprofiled_exact.solve_time);
            }
            if let Some(native_exact) = native_exact {
                if print_samples {
                    println!(
                        "  ssids_glider_sample index={} impl=native profile=native factor={:?} solve={:?} residual={:.3e} solution_inf={:.3e} inertia={}",
                        sample_index,
                        native_exact.factor_time,
                        native_exact.solve_time,
                        native_exact.residual_inf,
                        native_exact.solution_inf,
                        native_exact.inertia,
                    );
                }
                native_factor_times.push(native_exact.factor_time);
                native_solve_times.push(native_exact.solve_time);
            }
        }

        println!("\n=== glider NLIP iteration-0 in-process augmented profile ===");
        println!(
            "  repeats={} side={} dimension={} nnz={}",
            repeats,
            profile_side,
            dump.matrix_dimension,
            dump.values.len()
        );
        if !rust_factor_times.is_empty() {
            let rust_factor = median_duration(&rust_factor_times);
            let rust_solve = median_duration(&rust_solve_times);
            println!(
                "  rust_spral factor={:?} solve={:?} residual={:.3e} solution_inf={:.3e} inertia={}",
                rust_factor,
                rust_solve,
                warmup_rust.residual_inf,
                warmup_rust.solution_inf,
                warmup_rust.inertia,
            );
            let rust_unprofiled_factor = median_duration(&rust_unprofiled_factor_times);
            let rust_unprofiled_solve = median_duration(&rust_unprofiled_solve_times);
            println!(
                "  rust_spral_unprofiled factor={:?} solve={:?} residual={:.3e} solution_inf={:.3e} inertia={}",
                rust_unprofiled_factor,
                rust_unprofiled_solve,
                warmup_rust.residual_inf,
                warmup_rust.solution_inf,
                warmup_rust.inertia,
            );
            println!(
                "  rust_spral factor_profile symbolic_tree={:?} pattern={:?} values={:?} fronts={:?} root_delayed={:?} inverse={:?} lower_storage={:?} solve_panel_storage={:?} diagonal_storage={:?} bytes={:?} recorded={:?}",
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .symbolic_front_tree_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .permuted_pattern_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .permuted_values_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .front_factorization_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .root_delayed_factorization_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .factor_inverse_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .lower_storage_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .solve_panel_storage_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .diagonal_storage_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile.factor_bytes_time),
                median_duration(
                    &rust_factor_profiles
                        .iter()
                        .map(FactorProfile::total_recorded_time)
                        .collect::<Vec<_>>()
                ),
            );
            println!(
                "  rust_spral front_profile assembly={:?} dense_factor={:?} fronts={} local_dense_entries={} root_delayed_blocks={}",
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .front_assembly_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .dense_front_factorization_time),
                rust_factor_profiles[0].front_count,
                rust_factor_profiles[0].local_dense_entries,
                rust_factor_profiles[0].root_delayed_blocks,
            );
            println!(
                "  rust_spral front_detail_profile child_merge={:?} row_setup={:?} matrix_entries={:?} contribution={:?} local_merge={:?} zeroed_bytes={} child_results={} child_contrib_entries={} child_factor_cols={} child_solve_panels={} local_factor_cols={} local_solve_panels={}",
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .front_child_result_merge_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .front_row_setup_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .front_matrix_entry_assembly_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .front_contribution_assembly_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .front_local_result_merge_time),
                rust_factor_profiles[0].local_dense_bytes_zeroed,
                rust_factor_profiles[0].front_child_result_count,
                rust_factor_profiles[0].front_child_contribution_entries,
                rust_factor_profiles[0].front_child_factor_columns,
                rust_factor_profiles[0].front_child_solve_panels,
                rust_factor_profiles[0].front_local_factor_columns,
                rust_factor_profiles[0].front_local_solve_panels,
            );
            println!(
                "  rust_spral small_leaf_profile subtrees={} fronts={} app_fronts={} columns={} dense_entries={} small_leaf_tpp={:?} small_leaf_pivot_search={:?} small_leaf_pivot_factor={:?} small_leaf_pivot_scale={:?} small_leaf_pivot_update={:?} small_leaf_calc_ld={:?} small_leaf_gemm={:?} small_leaf_pack={:?} small_leaf_solve_panel={:?} small_leaf_output={:?}",
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_subtrees),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_fronts),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_app_fronts),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_columns),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_dense_entries),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_tpp_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_pivot_search_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_pivot_factor_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_pivot_scale_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_pivot_update_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_contribution_calc_ld_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_contribution_gemm_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_contribution_pack_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_solve_panel_build_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .spral_small_leaf_output_append_time),
            );
            println!(
                "  rust_spral dense_front_profile tpp={:?} tpp_pivot_search={:?} tpp_pivot_factor={:?} tpp_column_storage={:?} tpp_contribution_pack={:?} app_pivot_factor={:?} app_maxloc={:?} app_swap={:?} app_pivot_update={:?} app_block_apply={:?} app_block_trsm={:?} app_block_diag={:?} app_failed_scan={:?} app_backup={:?} app_restore={:?} app_accepted_update={:?} app_accepted_ld={:?} app_accepted_gemm={:?} app_column_storage={:?} solve_panel_build={:?}",
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .tpp_factorization_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .tpp_pivot_search_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .tpp_pivot_factor_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .tpp_column_storage_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .tpp_contribution_pack_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .app_pivot_factor_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile.app_maxloc_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .app_symmetric_swap_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .app_pivot_update_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .app_block_pivot_apply_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .app_block_triangular_solve_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .app_block_diagonal_apply_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .app_failed_pivot_scan_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile.app_backup_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile.app_restore_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .app_accepted_update_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .app_accepted_ld_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .app_accepted_gemm_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .app_column_storage_time),
                median_profile_duration(&rust_factor_profiles, |profile| profile
                    .solve_panel_build_time),
            );
            println!(
                "  rust_spral dense_front_counters tpp_fronts={} tpp_pivots={} tpp_factor_entries={} app_fronts={} app_panels={} app_maxloc_calls={} app_swaps={} app_1x1={} app_2x2={} app_zero={} app_diag_1x1={} app_offdiag_1x1={} app_front_le32={} app_front_33_64={} app_front_65_96={} app_front_97_128={} app_front_129_160={} app_front_161_256={} app_front_257_512={} app_front_gt512={}",
                median_profile_usize(&rust_factor_profiles, |profile| profile.tpp_front_count),
                median_profile_usize(&rust_factor_profiles, |profile| profile.tpp_pivots),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .tpp_factor_column_entries),
                median_profile_usize(&rust_factor_profiles, |profile| profile.app_front_count),
                median_profile_usize(&rust_factor_profiles, |profile| profile.app_panel_count),
                median_profile_usize(&rust_factor_profiles, |profile| profile.app_maxloc_calls),
                median_profile_usize(&rust_factor_profiles, |profile| profile.app_symmetric_swaps),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .app_one_by_one_pivots),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .app_two_by_two_pivots),
                median_profile_usize(&rust_factor_profiles, |profile| profile.app_zero_pivots),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .app_diagonal_one_by_one_pivots),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .app_offdiag_one_by_one_pivots),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .app_front_size_histogram[0]),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .app_front_size_histogram[1]),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .app_front_size_histogram[2]),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .app_front_size_histogram[3]),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .app_front_size_histogram[4]),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .app_front_size_histogram[5]),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .app_front_size_histogram[6]),
                median_profile_usize(&rust_factor_profiles, |profile| profile
                    .app_front_size_histogram[7]),
            );
            println!(
                "  rust_spral solve_profile input_perm={:?} forward={:?} diagonal={:?} backward={:?} output_perm={:?} recorded={:?}",
                median_solve_profile_duration(&rust_solve_profiles, |profile| profile
                    .input_permutation_time),
                median_solve_profile_duration(&rust_solve_profiles, |profile| profile
                    .forward_substitution_time),
                median_solve_profile_duration(&rust_solve_profiles, |profile| profile
                    .diagonal_solve_time),
                median_solve_profile_duration(&rust_solve_profiles, |profile| profile
                    .backward_substitution_time),
                median_solve_profile_duration(&rust_solve_profiles, |profile| profile
                    .output_permutation_time),
                median_duration(
                    &rust_solve_profiles
                        .iter()
                        .map(SolveProfile::total_recorded_time)
                        .collect::<Vec<_>>()
                ),
            );
            println!(
                "  rust_spral backward_profile trailing_update={:?} triangular={:?} trailing_columns={} trailing_dense_entries={} triangular_columns={} triangular_dense_entries={}",
                median_solve_profile_duration(&rust_solve_profiles, |profile| profile
                    .backward_trailing_update_time),
                median_solve_profile_duration(&rust_solve_profiles, |profile| profile
                    .backward_triangular_solve_time),
                rust_solve_profiles[0].backward_trailing_update_columns,
                rust_solve_profiles[0].backward_trailing_update_dense_entries,
                rust_solve_profiles[0].backward_triangular_columns,
                rust_solve_profiles[0].backward_triangular_dense_entries,
            );
        }
        if !native_factor_times.is_empty() {
            let native_factor = median_duration(&native_factor_times);
            let native_solve = median_duration(&native_solve_times);
            println!(
                "  native_spral factor={:?} solve={:?} residual={:.3e} solution_inf={:.3e} inertia={}",
                native_factor,
                native_solve,
                warmup_native.residual_inf,
                warmup_native.solution_inf,
                warmup_native.inertia,
            );
        }
        println!(
            "  augmented deltas: solution={:.6e} dx={:.6e} dp={:.6e} ds={:.6e} dlambda={:.6e} dz={:.6e}",
            final_solution_delta,
            final_dx_delta,
            final_dp_delta,
            final_ds_delta,
            final_dlambda_delta,
            final_dz_delta,
        );
    }

    #[test]
    #[ignore = "manual glider SSIDS small-leaf TPP exact-panel capture helper"]
    fn print_current_glider_small_leaf_tpp_panel_capture_summary() {
        let dump = load_current_glider_iteration0_augmented_dump();
        let symbolic = glider_iteration0_rust_symbolic(&dump);

        ssids_rs::debug_disable_spral_small_leaf_tpp_panel_captures();
        ssids_rs::debug_clear_spral_small_leaf_tpp_panel_captures();
        let rust_exact = replay_rust_augmented_spral_unprofiled_with_symbolic(&dump, &symbolic);
        let captures = ssids_rs::debug_take_spral_small_leaf_tpp_panel_captures();

        assert!(
            rust_exact.residual_inf <= 1e-10,
            "rust residual too large: {rust_exact:?}"
        );
        assert!(
            !captures.is_empty(),
            "glider replay did not capture any small-leaf TPP panels"
        );

        let mut ordered = (0..captures.len()).collect::<Vec<_>>();
        ordered.sort_by_key(|&index| {
            let capture = &captures[index];
            std::cmp::Reverse((
                capture.m * capture.n,
                capture.m,
                capture.n,
                capture.front_id,
            ))
        });
        let total_lower_entries = captures
            .iter()
            .map(|capture| capture.lcol.len())
            .sum::<usize>();
        let total_dense_area = captures
            .iter()
            .map(|capture| capture.m * capture.n)
            .sum::<usize>();
        let limit = std::env::var("SSIDS_GLIDER_TPP_PANEL_CAPTURE_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(16)
            .min(captures.len());

        println!("\n=== glider small-leaf TPP exact-panel capture summary ===");
        println!(
            "  panels={} total_dense_area={} total_lower_entries={} residual={:.3e}",
            captures.len(),
            total_dense_area,
            total_lower_entries,
            rust_exact.residual_inf
        );
        println!("  top panels by m*n:");
        for &index in ordered.iter().take(limit) {
            let capture = &captures[index];
            println!(
                "    index={} front={} m={} n={} ldl={} lcol_len={} hash={:#018x} rows_prefix={:?} perm_prefix={:?}",
                index,
                capture.front_id,
                capture.m,
                capture.n,
                capture.ldl,
                capture.lcol.len(),
                capture.stable_hash,
                &capture.rows[..capture.rows.len().min(8)],
                &capture.perm[..capture.perm.len().min(8)],
            );
        }

        if std::env::var("SSIDS_GLIDER_TPP_PANEL_CAPTURE_PRINT_LITERAL").is_ok() {
            let capture = &captures[ordered[0]];
            println!("  largest_panel_debug={capture:#?}");
        }
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
        let runtime = dc_runtime(&params);
        let mut options = crate::common::nlip_options(&params.solver);
        options.max_iters = 10;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![optimization::InteriorPointLinearSolver::SpralSrc],
            schedule: optimization::InteriorPointLinearDebugSchedule::EveryIteration,
            dump_dir: None,
        });

        let mut reports = Vec::new();
        let result = compiled.solve_interior_point_with_callback(&runtime, &options, |snapshot| {
            if let Some(report) = snapshot.solver.linear_debug.clone() {
                reports.push(report);
            }
        });
        if let Some(report) = failure_linear_debug_report(&result) {
            reports.push(report);
        }

        assert!(
            !reports.is_empty(),
            "expected glider run to emit native SPRAL comparison reports"
        );
        for report in &reports {
            assert_eq!(
                report.primary_solver,
                optimization::InteriorPointLinearSolver::SsidsRs
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
        let runtime = dc_runtime(&params);
        let dump_dir = TempDir::new().expect("temp dump dir should create");
        let mut options = crate::common::nlip_options(&params.solver);
        options.linear_solver = optimization::InteriorPointLinearSolver::SsidsRs;
        options.max_iters = 1;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![optimization::InteriorPointLinearSolver::SpralSrc],
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
        let runtime = dc_runtime(&params);
        let dump_dir = TempDir::new().expect("temp dump dir should create");
        let mut options = crate::common::nlip_options(&params.solver);
        options.linear_solver = optimization::InteriorPointLinearSolver::SsidsRs;
        options.max_iters = 10;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![optimization::InteriorPointLinearSolver::SpralSrc],
            schedule: optimization::InteriorPointLinearDebugSchedule::EveryIteration,
            dump_dir: Some(dump_dir.path().to_path_buf()),
        });

        let mut reports = Vec::new();
        let result = compiled.solve_interior_point_with_callback(&runtime, &options, |snapshot| {
            if let Some(report) = snapshot.solver.linear_debug.clone() {
                reports.push(report);
            }
        });
        if let Some(report) = failure_linear_debug_report(&result) {
            reports.push(report);
        }

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
        let runtime = dc_runtime(&params);
        let dump_dir = TempDir::new().expect("temp dump dir should create");
        let mut options = crate::common::nlip_options(&params.solver);
        options.linear_solver = optimization::InteriorPointLinearSolver::SsidsRs;
        options.max_iters = 10;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![optimization::InteriorPointLinearSolver::SpralSrc],
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
        let strict_runtime = dc_runtime(&params);
        let dump_dir = TempDir::new().expect("temp dump dir should create");
        let mut options = crate::common::nlip_options(&params.solver);
        optimization::apply_native_spral_parity_to_nlip_options(&mut options);
        options.max_iters = 400;
        options.acceptable_iter = 0;
        options.verbose = false;
        options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: vec![optimization::InteriorPointLinearSolver::SsidsRs],
            schedule: optimization::InteriorPointLinearDebugSchedule::FailuresOnly,
            dump_dir: Some(dump_dir.path().to_path_buf()),
        });

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
                            "  line_search alpha_pr={:.3e} alpha_du={:?} alpha_y={:?} accepted={:?} accepted_du={:?} accepted_y={:?} sigma={:.3e} current_merit={:.3e} current_barrier_obj={:.3e} current_primal_inf={:.3e} alpha_min={:.3e} rejected={} soc_attempted={} soc_used={} watchdog_active={} watchdog_accepted={}",
                            line_search.initial_alpha_pr,
                            line_search.initial_alpha_du,
                            line_search.initial_alpha_y,
                            line_search.accepted_alpha,
                            line_search.accepted_alpha_du,
                            line_search.accepted_alpha_y,
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
                                "  attempt[{index}] solver={:?} reg={:.6e} inertia={:?} kind={:?} sol_inf={:?} sol_lim={:?} res_inf={:?} res_lim={:?} detail={:?}",
                                attempt.solver,
                                attempt.regularization,
                                attempt.inertia,
                                attempt.failure_kind,
                                attempt.solution_inf,
                                attempt.solution_inf_limit,
                                attempt.residual_inf,
                                attempt.residual_inf_limit,
                                attempt.detail,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        if let Some(report) = failure_linear_debug_report(&result) {
            println!(
                "linear_debug_failure verdict={:?} primary={}",
                report.verdict,
                report.primary_solver.label(),
            );
            for result in &report.results {
                println!(
                    "  debug solver={} success={} reg={:.6e} inertia={:?} residual={:?} step_inf={:?} detail={:?}",
                    result.solver.label(),
                    result.success,
                    result.regularization,
                    result.inertia,
                    result.residual_inf,
                    result.step_inf,
                    result.detail,
                );
            }
            for note in &report.notes {
                println!("  debug note={note}");
            }
            let dump_paths = sorted_glider_dump_paths(dump_dir.path());
            if let Some(dump_path) = dump_paths.last() {
                let dump = load_glider_linear_debug_dump(dump_path);
                let rust_exact = replay_rust_augmented_spral(&dump);
                let native_exact = replay_native_augmented_spral(&dump);
                println!(
                    "  failure_dump={} rust_exact inertia={} residual={:.3e} native_exact inertia={} residual={:.3e}",
                    dump_path.display(),
                    rust_exact.inertia,
                    rust_exact.residual_inf,
                    native_exact.inertia,
                    native_exact.residual_inf,
                );
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
        let strict_runtime = dc_runtime(&params);
        let mut options = crate::common::nlip_options(&params.solver);
        optimization::apply_native_spral_parity_to_nlip_options(&mut options);
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
                            "  line_search alpha_pr={:.3e} alpha_du={:?} alpha_y={:?} accepted={:?} accepted_du={:?} accepted_y={:?} sigma={:.3e} current_merit={:.3e} current_barrier_obj={:.3e} current_primal_inf={:.3e} alpha_min={:.3e} rejected={} soc_attempted={} soc_used={} watchdog_active={} watchdog_accepted={}",
                            line_search.initial_alpha_pr,
                            line_search.initial_alpha_du,
                            line_search.initial_alpha_y,
                            line_search.accepted_alpha,
                            line_search.accepted_alpha_du,
                            line_search.accepted_alpha_y,
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
                    "  attempt[{index}] solver={:?} reg={:.6e} inertia={:?} kind={:?} sol_inf={:?} sol_lim={:?} res_inf={:?} res_lim={:?} detail={:?}",
                    attempt.solver,
                    attempt.regularization,
                    attempt.inertia,
                    attempt.failure_kind,
                    attempt.solution_inf,
                    attempt.solution_inf_limit,
                    attempt.residual_inf,
                    attempt.residual_inf_limit,
                    attempt.detail,
                );
            }
        }
        println!(
            "result: {:?}",
            result.as_ref().map(|_| ()).map_err(|err| err.to_string())
        );
    }

    #[test]
    #[cfg(feature = "ipopt")]
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
        let runtime = dc_runtime(&params);
        let mut options = crate::common::ipopt_options(&params.solver);
        optimization::apply_native_spral_parity_to_ipopt_options(&mut options);
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
    #[cfg(feature = "ipopt")]
    #[ignore = "manual local-SPRAL IPOPT strategy comparison for the glider repro"]
    fn print_current_glider_ipopt_mu_strategy_compare() {
        fn local_spral_ipopt_options(config: &SolverConfig) -> optimization::IpoptOptions {
            let mut options = crate::common::ipopt_options(config);
            optimization::apply_native_spral_parity_to_ipopt_options(&mut options);
            options
        }

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
        let runtime = dc_runtime(&params);

        for (label, mut options) in [
            ("default-local-spral", {
                let mut options = crate::common::ipopt_options(&params.solver);
                options.capture_provenance = true;
                options
            }),
            ("explicit-spral-adaptive", {
                let mut options = local_spral_ipopt_options(&params.solver);
                options.mu_strategy = optimization::IpoptMuStrategy::Adaptive;
                options
            }),
            ("explicit-spral-monotone", {
                let mut options = local_spral_ipopt_options(&params.solver);
                options.mu_strategy = optimization::IpoptMuStrategy::Monotone;
                options
            }),
        ] {
            options.print_level = 0;
            options.suppress_banner = true;
            let mut last_solver = None;
            let mut last_tf = None;
            let mut last_x = None;
            let result = compiled.solve_ipopt_with_callback(&runtime, &options, |snapshot| {
                last_solver = Some(snapshot.solver.clone());
                last_tf = Some(snapshot.trajectories.tf);
                last_x = Some(snapshot.trajectories.x.terminal.x);
            });
            if last_solver.is_none()
                && let Err(optimization::IpoptSolveError::Solve { snapshots, .. }) = &result
            {
                last_solver = snapshots.last().cloned();
            }
            println!("\n=== glider IPOPT local SPRAL {label} ===");
            if let Some(last) = last_solver {
                println!(
                    "last iter={} phase={:?} obj={:.6e} tf={:?} x_T={:?} inf_pr={:.6e} inf_du={:.6e} mu={:.6e} alpha_pr={:.6e} alpha_du={:.6e}",
                    last.iteration,
                    last.phase,
                    last.objective,
                    last_tf,
                    last_x,
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
        let runtime = dc_runtime(&params);
        let mut options = crate::common::nlip_options(&params.solver);
        optimization::apply_native_spral_parity_to_nlip_options(&mut options);
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
        let runtime = dc_runtime(&params);
        let mut options = crate::common::nlip_options(&params.solver);
        optimization::apply_native_spral_parity_to_nlip_options(&mut options);
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

    #[cfg(feature = "ipopt")]
    #[test]
    #[ignore = "manual native-SPRAL NLIP/IPOPT parity divergence helper"]
    fn print_current_glider_native_spral_ipopt_first_divergence() {
        #[derive(Clone)]
        struct TracePoint {
            iteration: usize,
            x: Vec<f64>,
            objective: f64,
            primal_inf: f64,
            dual_inf: f64,
            mu: f64,
            tf: f64,
            terminal_x: f64,
            regularization: Option<f64>,
            primal_shift: f64,
            dual_regularization: f64,
            step_inf: f64,
            dx_inf: f64,
            ds_inf: f64,
            dz_inf: f64,
            alpha_pr: f64,
            alpha_du: f64,
            alpha_y: f64,
            step_tag: String,
            trial_count: usize,
            events: String,
            inertia: String,
            linear_stats: String,
            linear_detail: String,
            alpha_pr_limiter: String,
            alpha_du_limiter: String,
            alpha_du_limiters: String,
        }

        #[derive(Clone)]
        struct VariableBoundView {
            lower: Vec<Option<f64>>,
            upper: Vec<Option<f64>>,
        }

        #[derive(Clone)]
        struct IpoptOcpTracePoint {
            solver: optimization::IpoptIterationSnapshot,
            tf: f64,
            terminal_x: f64,
        }

        fn log_gap(lhs: f64, rhs: f64, floor: f64) -> f64 {
            let lhs = lhs.abs().max(floor).log10();
            let rhs = rhs.abs().max(floor).log10();
            (lhs - rhs).abs()
        }

        fn positive_regularization(value: f64) -> Option<f64> {
            (value.is_finite() && value > 0.0).then_some(value)
        }

        fn regularization_text(value: Option<f64>) -> String {
            value.map_or_else(|| "--".to_string(), |value| format!("{value:.6e}"))
        }

        fn regularization_log_gap(lhs: Option<f64>, rhs: Option<f64>) -> f64 {
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => log_gap(lhs, rhs, 1.0e-20),
                (Some(lhs), None) => log_gap(lhs, 0.0, 1.0e-20),
                (None, Some(rhs)) => log_gap(0.0, rhs, 1.0e-20),
                (None, None) => 0.0,
            }
        }

        fn restoration_bridge_state_matches(nlip: &TracePoint, ipopt: &TracePoint) -> bool {
            nlip.step_tag == "r"
                && (nlip.objective - ipopt.objective).abs() <= 1.0e-5
                && (nlip.primal_inf - ipopt.primal_inf).abs() <= 1.0e-6
                && (nlip.dual_inf - ipopt.dual_inf).abs() <= 1.0e-6
                && (nlip.mu - ipopt.mu).abs() <= 1.0e-12
                && (nlip.tf - ipopt.tf).abs() <= 1.0e-6
                && (nlip.terminal_x - ipopt.terminal_x).abs() <= 1.0e-5
                && nlip.trial_count.abs_diff(ipopt.trial_count) <= 1
        }

        fn is_restoration_bridge_trace_index(
            nlip_trace: &[TracePoint],
            ipopt_trace: &[TracePoint],
            index: usize,
        ) -> bool {
            nlip_trace
                .get(index)
                .zip(ipopt_trace.get(index))
                .is_some_and(|(nlip, ipopt)| restoration_bridge_state_matches(nlip, ipopt))
        }

        fn parse_ipopt_step_tags(journal_output: Option<&str>) -> BTreeMap<usize, String> {
            let mut tags = BTreeMap::new();
            let Some(journal) = journal_output else {
                return tags;
            };
            for line in journal.lines() {
                let trimmed = line.trim_start();
                if trimmed.is_empty() || !trimmed.as_bytes()[0].is_ascii_digit() {
                    continue;
                }
                let tokens = trimmed.split_whitespace().collect::<Vec<_>>();
                let Some(iteration) = tokens.first().and_then(|token| token.parse::<usize>().ok())
                else {
                    continue;
                };
                let Some(alpha_pr_token) = tokens.iter().rev().nth(1).copied() else {
                    continue;
                };
                let Some(step_char) = alpha_pr_token
                    .chars()
                    .last()
                    .filter(|value| value.is_ascii_alphabetic())
                else {
                    continue;
                };
                tags.insert(iteration, step_char.to_string());
            }
            tags
        }

        fn print_ipopt_linear_journal_excerpt(journal_output: Option<&str>) {
            if std::env::var_os("GLIDER_PARITY_PRINT_IPOPT_LINEAR_JOURNAL").is_none() {
                return;
            }
            let Some(journal) = journal_output else {
                println!("ipopt linear journal excerpt unavailable");
                return;
            };
            println!("ipopt linear journal excerpt:");
            for line in journal.lines() {
                let trimmed = line.trim_start();
                if trimmed.contains("residual_ratio =")
                    || trimmed.contains("max-norm resid_")
                    || trimmed.contains("nrm_rhs =")
                    || trimmed.contains("Perturbation parameters:")
                    || trimmed.contains("Solving system with delta_x=")
                    || trimmed.contains("Number of trial factorizations performed:")
                    || trimmed.contains("Number of negative eigenvalues")
                    || trimmed.contains("Asking augmented system solver")
                {
                    println!("  {trimmed}");
                }
            }
        }

        #[derive(Clone, Copy)]
        struct JournalVectorFingerprint {
            len: usize,
            inf: f64,
            sum: f64,
            hash: u64,
            negative_zero_count: usize,
            positive_zero_count: usize,
        }

        fn journal_vector_fingerprint(values: &[f64]) -> JournalVectorFingerprint {
            const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
            const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
            let mut inf = 0.0_f64;
            let mut sum = 0.0_f64;
            let mut hash = FNV_OFFSET ^ values.len() as u64;
            let mut negative_zero_count = 0;
            let mut positive_zero_count = 0;
            for value in values {
                inf = inf.max(value.abs());
                sum += *value;
                hash = (hash ^ value.to_bits()).wrapping_mul(FNV_PRIME);
                if *value == 0.0 {
                    if value.is_sign_negative() {
                        negative_zero_count += 1;
                    } else {
                        positive_zero_count += 1;
                    }
                }
            }
            JournalVectorFingerprint {
                len: values.len(),
                inf,
                sum,
                hash,
                negative_zero_count,
                positive_zero_count,
            }
        }

        fn journal_vector_fingerprint_text(fingerprint: JournalVectorFingerprint) -> String {
            format!(
                "len={} inf={:.12e} sum={:.12e} hash={:016x} z[-/+]={}/{}",
                fingerprint.len,
                fingerprint.inf,
                fingerprint.sum,
                fingerprint.hash,
                fingerprint.negative_zero_count,
                fingerprint.positive_zero_count,
            )
        }

        fn journal_augmented_block_fingerprint_text(values: &[f64], dims: &[usize; 4]) -> String {
            let x_end = dims[0];
            let p_end = x_end + dims[1];
            let lambda_end = p_end + dims[2];
            let z_end = lambda_end + dims[3];
            if values.len() != z_end {
                return "--".to_string();
            }
            format!(
                "x[{}],p[{}],lambda[{}],z[{}]",
                journal_vector_fingerprint_text(journal_vector_fingerprint(&values[..x_end])),
                journal_vector_fingerprint_text(journal_vector_fingerprint(&values[x_end..p_end])),
                journal_vector_fingerprint_text(journal_vector_fingerprint(
                    &values[p_end..lambda_end],
                )),
                journal_vector_fingerprint_text(journal_vector_fingerprint(
                    &values[lambda_end..z_end],
                )),
            )
        }

        fn journal_value_after_equals(line: &str) -> Option<f64> {
            line.split_once('=')
                .and_then(|(_, value)| value.trim().parse::<f64>().ok())
        }

        fn journal_first_value_after_equals(line: &str) -> Option<f64> {
            let after_equals = line.split_once('=')?.1;
            after_equals.split_whitespace().next()?.parse::<f64>().ok()
        }

        fn journal_kkt_component(line: &str) -> Option<(usize, usize)> {
            let start = line.find("KKT[")? + "KKT[".len();
            let first_end = line[start..].find(']')? + start;
            let row_block = line[start..first_end].parse::<usize>().ok()?;
            let second_start = line[first_end + 1..].find('[')? + first_end + 2;
            let second_end = line[second_start..].find(']')? + second_start;
            let col_block = line[second_start..second_end].parse::<usize>().ok()?;
            Some((row_block, col_block))
        }

        fn journal_kkt_entry(line: &str) -> Option<(usize, usize, usize, usize, f64)> {
            let (block_row, block_col) = journal_kkt_component(line)?;
            let (local_row, local_col, value) = journal_kkt_local_entry(line)?;
            Some((block_row, block_col, local_row, local_col, value))
        }

        fn journal_kkt_local_entry(line: &str) -> Option<(usize, usize, f64)> {
            let equals = line.find('=')?;
            let index_start = line[..equals].rfind('[')? + 1;
            let index_end = line[index_start..equals].find(']')? + index_start;
            let indices = &line[index_start..index_end];
            let (row, col) = indices
                .split_once(',')
                .map_or((indices, indices), |(row, col)| (row, col));
            let value = journal_first_value_after_equals(line)?;
            Some((
                row.trim().parse::<usize>().ok()?,
                col.trim().parse::<usize>().ok()?,
                value,
            ))
        }

        fn journal_kkt_global_entry(line: &str, dims: &[usize; 4]) -> Option<(usize, usize, f64)> {
            let (block_row, block_col, local_row, local_col, value) = journal_kkt_entry(line)?;
            let row_offset = dims.iter().take(block_row).sum::<usize>();
            let col_offset = dims.iter().take(block_col).sum::<usize>();
            let row = row_offset + local_row.checked_sub(1)?;
            let col = col_offset + local_col.checked_sub(1)?;
            Some((row.min(col), row.max(col), value))
        }

        fn append_journal_kkt_identity_entries(
            line: &str,
            dims: &[usize; 4],
            triplets: &mut Vec<(usize, usize, f64)>,
        ) {
            if !line.contains("IdentityMatrix") || !line.contains("the factor") {
                return;
            }
            let Some((block_row, block_col)) = journal_kkt_component(line) else {
                return;
            };
            let Some((_, factor_text)) = line.rsplit_once("the factor") else {
                return;
            };
            let Some(factor) = factor_text.trim_end_matches('.').trim().parse::<f64>().ok() else {
                return;
            };
            let row_offset = dims.iter().take(block_row).sum::<usize>();
            let col_offset = dims.iter().take(block_col).sum::<usize>();
            for index in 0..dims[block_row].min(dims[block_col]) {
                let row = row_offset + index;
                let col = col_offset + index;
                triplets.push((row.min(col), row.max(col), factor));
            }
        }

        fn append_journal_kkt_zero_diagonal_entries(
            line: &str,
            dims: &[usize; 4],
            triplets: &mut Vec<(usize, usize, f64)>,
        ) {
            if !line.contains("DiagMatrix")
                || !(line.contains("\"KKT[2][2]\"") || line.contains("\"KKT[3][3]\""))
            {
                return;
            }
            let Some((block_row, block_col)) = journal_kkt_component(line) else {
                return;
            };
            let offset = dims.iter().take(block_row).sum::<usize>();
            for index in 0..dims[block_row].min(dims[block_col]) {
                let diag = offset + index;
                triplets.push((diag, diag, 0.0));
            }
        }

        fn compressed_journal_kkt_values(
            mut triplets: Vec<(usize, usize, f64)>,
        ) -> Vec<((usize, usize), f64)> {
            // Mirrors IpTripletToCSRConverter.cpp: sort by triangular key,
            // take the first triplet value, then add duplicate triplets.
            triplets.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0).then(lhs.1.cmp(&rhs.1)));
            let mut compressed = Vec::new();
            for (row, col, value) in triplets {
                if let Some((last_key, last_value)) = compressed.last_mut()
                    && *last_key == (row, col)
                {
                    *last_value += value;
                    continue;
                }
                compressed.push(((row, col), value));
            }
            compressed
        }

        fn ipopt_augmented_journal_kkt_values(
            journal: &str,
            dims: &[usize; 4],
        ) -> Vec<Vec<((usize, usize), f64)>> {
            let mut matrices = Vec::new();
            let mut current_triplets = Vec::new();
            let mut in_kkt = false;
            let mut current_component: Option<(usize, usize)> = None;
            for line in journal.lines() {
                let trimmed = line.trim_start();
                if trimmed.starts_with("CompoundSymMatrix \"KKT\"") {
                    if !current_triplets.is_empty() {
                        matrices.push(compressed_journal_kkt_values(std::mem::take(
                            &mut current_triplets,
                        )));
                    }
                    in_kkt = true;
                    current_component = None;
                    continue;
                }
                if !in_kkt {
                    continue;
                }
                if trimmed.contains("\"KKT[")
                    && let Some(component) = journal_kkt_component(trimmed)
                {
                    current_component = Some(component);
                }
                append_journal_kkt_identity_entries(trimmed, dims, &mut current_triplets);
                append_journal_kkt_zero_diagonal_entries(trimmed, dims, &mut current_triplets);
                if let Some(entry) = journal_kkt_global_entry(trimmed, dims) {
                    current_triplets.push(entry);
                } else if trimmed.contains("Term:")
                    && let (Some((block_row, block_col)), Some((local_row, local_col, value))) =
                        (current_component, journal_kkt_local_entry(trimmed))
                {
                    let row_offset = dims.iter().take(block_row).sum::<usize>();
                    let col_offset = dims.iter().take(block_col).sum::<usize>();
                    let row = row_offset + local_row.saturating_sub(1);
                    let col = col_offset + local_col.saturating_sub(1);
                    current_triplets.push((row.min(col), row.max(col), value));
                }
            }
            if !current_triplets.is_empty() {
                matrices.push(compressed_journal_kkt_values(current_triplets));
            }
            matrices
        }

        fn dump_kkt_keys(dump: &GliderLinearDebugDump) -> Vec<(usize, usize)> {
            let mut keys = Vec::with_capacity(dump.row_indices.len());
            for col in 0..dump.matrix_dimension {
                for index in dump.col_ptrs[col]..dump.col_ptrs[col + 1] {
                    keys.push((col, dump.row_indices[index]));
                }
            }
            keys
        }

        fn journal_kkt_diff_summary_text(
            dump: &GliderLinearDebugDump,
            ipopt_matrix: &[((usize, usize), f64)],
        ) -> String {
            let dump_keys = dump_kkt_keys(dump);
            if dump_keys.len() != ipopt_matrix.len() {
                return format!(
                    "len_mismatch nlip={} ipopt={}",
                    dump_keys.len(),
                    ipopt_matrix.len()
                );
            }
            let mut first_structure_diff = None;
            let mut first_bits_diff = None;
            let mut max_abs_diff = 0.0_f64;
            let mut max_abs_index = 0;
            let mut max_abs_key = (0, 0);
            let mut max_abs_lhs = 0.0_f64;
            let mut max_abs_rhs = 0.0_f64;
            for (index, ((dump_key, &nlip_value), &(ipopt_key, ipopt_value))) in dump_keys
                .iter()
                .zip(dump.values.iter())
                .zip(ipopt_matrix.iter())
                .enumerate()
            {
                if *dump_key != ipopt_key && first_structure_diff.is_none() {
                    first_structure_diff = Some((index, *dump_key, ipopt_key));
                }
                if nlip_value.to_bits() != ipopt_value.to_bits() && first_bits_diff.is_none() {
                    first_bits_diff = Some((index, nlip_value, ipopt_value));
                }
                let abs_diff = (nlip_value - ipopt_value).abs();
                if abs_diff > max_abs_diff {
                    max_abs_diff = abs_diff;
                    max_abs_index = index;
                    max_abs_key = *dump_key;
                    max_abs_lhs = nlip_value;
                    max_abs_rhs = ipopt_value;
                }
            }
            let structure = first_structure_diff.map_or_else(
                || "first_structure_diff=none".to_string(),
                |(index, nlip_key, ipopt_key)| {
                    format!(
                        "first_structure_diff=index={index} nlip=({},{}) ipopt=({},{})",
                        nlip_key.0, nlip_key.1, ipopt_key.0, ipopt_key.1
                    )
                },
            );
            let first = first_bits_diff.map_or_else(
                || "first_bits_diff=none".to_string(),
                |(index, lhs_i, rhs_i)| {
                    format!(
                        "first_bits_diff=index={} lhs={:.17e}/{:016x} rhs={:.17e}/{:016x}",
                        index,
                        lhs_i,
                        lhs_i.to_bits(),
                        rhs_i,
                        rhs_i.to_bits()
                    )
                },
            );
            format!(
                "{structure} {first} max_abs_diff={max_abs_diff:.17e} max_abs_index={max_abs_index} max_abs_key=({},{}) max_abs_values=({max_abs_lhs:.17e},{max_abs_rhs:.17e})",
                max_abs_key.0, max_abs_key.1
            )
        }

        fn journal_kkt_max_abs_diff(
            dump: &GliderLinearDebugDump,
            ipopt_matrix: &[((usize, usize), f64)],
        ) -> f64 {
            let dump_keys = dump_kkt_keys(dump);
            if dump_keys.len() != ipopt_matrix.len() {
                return f64::INFINITY;
            }
            dump_keys
                .iter()
                .zip(dump.values.iter())
                .zip(ipopt_matrix.iter())
                .try_fold(
                    0.0_f64,
                    |acc, ((dump_key, &nlip_value), &(ipopt_key, ipopt_value))| {
                        if *dump_key != ipopt_key {
                            None
                        } else {
                            Some(acc.max((nlip_value - ipopt_value).abs()))
                        }
                    },
                )
                .unwrap_or(f64::INFINITY)
        }

        fn journal_sol_component(line: &str) -> Option<usize> {
            let start = line.find("SOL[ 0][")? + "SOL[ 0][".len();
            let rest = &line[start..];
            let end = rest.find(']')?;
            rest[..end].trim().parse::<usize>().ok()
        }

        fn ipopt_augmented_journal_vectors(journal: &str) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
            let mut rhs_vectors: Vec<Vec<f64>> = Vec::new();
            let mut current_rhs = Vec::new();
            let mut sol_vectors: Vec<Vec<f64>> = Vec::new();
            let mut current_sol_components: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());

            for line in journal.lines() {
                let trimmed = line.trim_start();
                if trimmed.starts_with("Trhs[") {
                    if let Some(value) = journal_value_after_equals(trimmed) {
                        current_rhs.push(value);
                    }
                    continue;
                }
                if !current_rhs.is_empty() {
                    rhs_vectors.push(std::mem::take(&mut current_rhs));
                }

                if trimmed.contains("SOL[ 0][") {
                    if let (Some(component), Some(value)) = (
                        journal_sol_component(trimmed),
                        journal_value_after_equals(trimmed),
                    ) && let Some(values) = current_sol_components.get_mut(component)
                    {
                        values.push(value);
                    }
                    continue;
                }
                if current_sol_components
                    .iter()
                    .any(|values| !values.is_empty())
                    && trimmed.starts_with("CompoundVector")
                {
                    let mut solution = Vec::new();
                    for component in current_sol_components.iter_mut() {
                        solution.append(component);
                    }
                    sol_vectors.push(solution);
                }
            }
            if !current_rhs.is_empty() {
                rhs_vectors.push(current_rhs);
            }
            if current_sol_components
                .iter()
                .any(|values| !values.is_empty())
            {
                let mut solution = Vec::new();
                for component in current_sol_components.iter_mut() {
                    solution.append(component);
                }
                sol_vectors.push(solution);
            }
            (rhs_vectors, sol_vectors)
        }

        fn ipopt_journal_dense_vectors(journal: &str, name: &str) -> Vec<Vec<f64>> {
            let header = format!("DenseVector \"{name}\" with ");
            let entry_prefix = format!("{name}[");
            let mut vectors = Vec::new();
            let mut current = Vec::new();
            let mut in_vector = false;
            for line in journal.lines() {
                let trimmed = line.trim_start();
                if trimmed.starts_with(&header) {
                    if !current.is_empty() {
                        vectors.push(std::mem::take(&mut current));
                    }
                    in_vector = true;
                    continue;
                }
                if !in_vector {
                    continue;
                }
                if trimmed.starts_with(&entry_prefix) {
                    if let Some(value) = journal_value_after_equals(trimmed) {
                        current.push(value);
                    }
                    continue;
                }
                if !current.is_empty() {
                    vectors.push(std::mem::take(&mut current));
                }
                in_vector = false;
            }
            if !current.is_empty() {
                vectors.push(current);
            }
            vectors
        }

        fn print_ipopt_augmented_journal_fingerprints(journal_output: Option<&str>) {
            if std::env::var_os("GLIDER_PARITY_PRINT_IPOPT_AUGMENTED_FINGERPRINTS").is_none() {
                return;
            }
            let Some(journal) = journal_output else {
                println!("ipopt augmented journal fingerprints unavailable");
                return;
            };

            let (rhs_vectors, sol_vectors) = ipopt_augmented_journal_vectors(journal);

            let dims = sol_vectors
                .first()
                .and_then(|solution| {
                    let rhs_len = rhs_vectors.first().map_or(solution.len(), Vec::len);
                    (solution.len() == rhs_len).then_some([1154, 902, 1000, 902])
                })
                .filter(|dims| {
                    dims.iter().sum::<usize>() == rhs_vectors.first().map_or(0, Vec::len)
                });

            println!(
                "ipopt augmented journal fingerprints rhs_count={} sol_count={}",
                rhs_vectors.len(),
                sol_vectors.len()
            );
            for (index, rhs) in rhs_vectors.iter().enumerate() {
                let blocks = dims.map_or_else(
                    || "--".to_string(),
                    |dims| journal_augmented_block_fingerprint_text(rhs, &dims),
                );
                println!(
                    "  ipopt_rhs[{index}] [{}] blocks[{blocks}]",
                    journal_vector_fingerprint_text(journal_vector_fingerprint(rhs)),
                );
            }
            for (index, solution) in sol_vectors.iter().enumerate() {
                let blocks = dims.map_or_else(
                    || "--".to_string(),
                    |dims| journal_augmented_block_fingerprint_text(solution, &dims),
                );
                println!(
                    "  ipopt_sol[{index}] [{}] blocks[{blocks}]",
                    journal_vector_fingerprint_text(journal_vector_fingerprint(solution)),
                );
            }
        }

        fn journal_vector_diff_summary_text(lhs: &[f64], rhs: &[f64]) -> String {
            if lhs.len() != rhs.len() {
                return format!("len_mismatch lhs={} rhs={}", lhs.len(), rhs.len());
            }
            let mut first_bits_diff: Option<(usize, f64, f64)> = None;
            let mut max_abs_diff = 0.0_f64;
            let mut max_abs_index = 0;
            for (index, (&lhs_i, &rhs_i)) in lhs.iter().zip(rhs.iter()).enumerate() {
                if lhs_i.to_bits() != rhs_i.to_bits() && first_bits_diff.is_none() {
                    first_bits_diff = Some((index, lhs_i, rhs_i));
                }
                let abs_diff = (lhs_i - rhs_i).abs();
                if abs_diff > max_abs_diff {
                    max_abs_diff = abs_diff;
                    max_abs_index = index;
                }
            }
            let first = first_bits_diff.map_or_else(
                || "first_bits_diff=none".to_string(),
                |(index, lhs_i, rhs_i)| {
                    format!(
                        "first_bits_diff=index={} lhs={:.17e}/{:016x} rhs={:.17e}/{:016x}",
                        index,
                        lhs_i,
                        lhs_i.to_bits(),
                        rhs_i,
                        rhs_i.to_bits()
                    )
                },
            );
            format!(
                "{first} max_abs_diff={:.17e} max_abs_index={max_abs_index}",
                max_abs_diff
            )
        }

        fn journal_vector_max_abs_diff(lhs: &[f64], rhs: &[f64]) -> f64 {
            if lhs.len() != rhs.len() {
                return f64::INFINITY;
            }
            lhs.iter()
                .zip(rhs.iter())
                .fold(0.0_f64, |acc, (&lhs_i, &rhs_i)| {
                    acc.max((lhs_i - rhs_i).abs())
                })
        }

        fn journal_vector_max_abs_diff_index(lhs: &[f64], rhs: &[f64]) -> Option<usize> {
            if lhs.len() != rhs.len() || lhs.is_empty() {
                return None;
            }
            let mut max_abs_diff = 0.0_f64;
            let mut max_abs_index = 0;
            for (index, (&lhs_i, &rhs_i)) in lhs.iter().zip(rhs.iter()).enumerate() {
                let abs_diff = (lhs_i - rhs_i).abs();
                if abs_diff > max_abs_diff {
                    max_abs_diff = abs_diff;
                    max_abs_index = index;
                }
            }
            Some(max_abs_index)
        }

        fn augmented_vector_label(
            index: usize,
            dump: &GliderLinearDebugDump,
            intervals: usize,
            order: usize,
        ) -> String {
            if index < dump.x_dimension {
                dump.x_full_indices.get(index).map_or_else(
                    || format!("x[{index}]"),
                    |full_index| {
                        format!(
                            "x[{index}->{}]",
                            glider_decision_label(*full_index, intervals, order)
                        )
                    },
                )
            } else if index < dump.lambda_offset {
                format!("p[{}]", index - dump.p_offset)
            } else if index < dump.z_offset {
                format!("lambda[{}]", index - dump.lambda_offset)
            } else if index < dump.matrix_dimension {
                format!("z[{}]", index - dump.z_offset)
            } else {
                format!("out_of_bounds[{index}]")
            }
        }

        fn augmented_matrix_value_location_text(
            dump: &GliderLinearDebugDump,
            value_index: usize,
            intervals: usize,
            order: usize,
        ) -> String {
            if value_index >= dump.values.len() || dump.col_ptrs.len() != dump.matrix_dimension + 1
            {
                return format!("max_entry_index={value_index}");
            }
            let column = (0..dump.matrix_dimension)
                .find(|&column| {
                    dump.col_ptrs[column] <= value_index && value_index < dump.col_ptrs[column + 1]
                })
                .unwrap_or(dump.matrix_dimension);
            let row = dump
                .row_indices
                .get(value_index)
                .copied()
                .unwrap_or(dump.matrix_dimension);
            format!(
                "max_entry_index={value_index} row={row}({}) col={column}({})",
                augmented_vector_label(row, dump, intervals, order),
                augmented_vector_label(column, dump, intervals, order),
            )
        }

        fn augmented_vector_diff_location_text(
            dump: &GliderLinearDebugDump,
            lhs: &[f64],
            rhs: &[f64],
            intervals: usize,
            order: usize,
        ) -> String {
            journal_vector_max_abs_diff_index(lhs, rhs).map_or_else(
                || "max_abs_index=none".to_string(),
                |index| {
                    format!(
                        "max_abs_block={}",
                        augmented_vector_label(index, dump, intervals, order)
                    )
                },
            )
        }

        fn augmented_vector_diff_component_text(
            dump: &GliderLinearDebugDump,
            lhs: &[f64],
            rhs: &[f64],
            intervals: usize,
            order: usize,
        ) -> String {
            journal_vector_max_abs_diff_index(lhs, rhs).map_or_else(
                || "max_abs_index=none".to_string(),
                |index| {
                    let lhs_value = lhs.get(index).copied().unwrap_or(f64::NAN);
                    let rhs_value = rhs.get(index).copied().unwrap_or(f64::NAN);
                    let diff = lhs_value - rhs_value;
                    let label = augmented_vector_label(index, dump, intervals, order);
                    if index < dump.x_dimension {
                        let final_rhs = dump.rhs.get(index).copied().unwrap_or(f64::NAN);
                        let prefinal_rhs = -final_rhs;
                        let r_dual = dump.r_dual.get(index).copied().unwrap_or(f64::NAN);
                        let bound_rhs = dump.bound_rhs.get(index).copied().unwrap_or(f64::NAN);
                        let bound_diagonal =
                            dump.bound_diagonal.get(index).copied().unwrap_or(f64::NAN);
                        return format!(
                            "max_abs_index={index} {label} nlip={lhs_value:.17e} ipopt={rhs_value:.17e} diff={diff:.17e} assembled_prefinal={prefinal_rhs:.17e} assembled_final={final_rhs:.17e} r_dual={r_dual:.17e} bound_rhs={bound_rhs:.17e} r_dual_minus_bound_rhs={:.17e} bound_diagonal={bound_diagonal:.17e}",
                            r_dual - bound_rhs,
                        );
                    }
                    if index >= dump.p_offset && index < dump.lambda_offset {
                        let row = index - dump.p_offset;
                        let slack = dump.slack.get(row).copied().unwrap_or(f64::NAN);
                        let multiplier = dump.multipliers.get(row).copied().unwrap_or(f64::NAN);
                        let complementarity = slack * multiplier - dump.barrier_parameter;
                        return format!(
                            "max_abs_index={index} {label} nlip={lhs_value:.17e} ipopt={rhs_value:.17e} diff={diff:.17e} slack={slack:.17e} multiplier={multiplier:.17e} complementarity={complementarity:.17e}",
                        );
                    }
                    format!(
                        "max_abs_index={index} {label} nlip={lhs_value:.17e} ipopt={rhs_value:.17e} diff={diff:.17e}",
                    )
                },
            )
        }

        fn reconstructed_prefinal_upper_slack_multiplier_steps(
            dump: &GliderLinearDebugDump,
            solution: &[f64],
        ) -> Vec<f64> {
            let count = dump
                .inequality_dimension
                .min(dump.slack.len())
                .min(dump.multipliers.len())
                .min(solution.len().saturating_sub(dump.p_offset));
            (0..count)
                .map(|index| {
                    let slack = dump.slack[index];
                    let multiplier = dump.multipliers[index];
                    let r_cent = slack * multiplier - dump.barrier_parameter;
                    let ds = solution[dump.p_offset + index];
                    // Mirrors ExpansionMatrix::SinvBlrmZMTdBrImpl for
                    // Pd_U.SinvBlrmZMTdBr(1., slack_s_U, rhs.v_U, v_U, sol.s, sol.v_U).
                    (r_cent + multiplier * ds) / slack
                })
                .collect()
        }

        fn upper_slack_prefinal_channel_diff_text(
            dump: &GliderLinearDebugDump,
            nlip_solution: &[f64],
            ipopt_solution: &[f64],
            intervals: usize,
            order: usize,
        ) -> String {
            let nlip_v = reconstructed_prefinal_upper_slack_multiplier_steps(dump, nlip_solution);
            let ipopt_v = reconstructed_prefinal_upper_slack_multiplier_steps(dump, ipopt_solution);
            let count = nlip_v
                .len()
                .min(ipopt_v.len())
                .min(nlip_solution.len().saturating_sub(dump.p_offset))
                .min(ipopt_solution.len().saturating_sub(dump.p_offset));
            let mut diffs = (0..count)
                .map(|index| {
                    let nlip_ds = nlip_solution[dump.p_offset + index];
                    let ipopt_ds = ipopt_solution[dump.p_offset + index];
                    let ds_gap = (nlip_ds - ipopt_ds).abs();
                    let vu_gap = (nlip_v[index] - ipopt_v[index]).abs();
                    let gain = if ds_gap > 0.0 {
                        vu_gap / ds_gap
                    } else {
                        f64::INFINITY
                    };
                    (
                        index,
                        vu_gap.max(ds_gap),
                        ds_gap,
                        vu_gap,
                        gain,
                        nlip_ds,
                        ipopt_ds,
                        nlip_v[index],
                        ipopt_v[index],
                        dump.slack[index],
                        dump.multipliers[index],
                    )
                })
                .collect::<Vec<_>>();
            diffs.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1).then(lhs.0.cmp(&rhs.0)));
            if diffs.is_empty() {
                return "--".to_string();
            }
            diffs
                .into_iter()
                .take(4)
                .map(
                    |(
                        index,
                        _score,
                        ds_gap,
                        vu_gap,
                        gain,
                        nlip_ds,
                        ipopt_ds,
                        nlip_vu,
                        ipopt_vu,
                        slack,
                        multiplier,
                    )| {
                        format!(
                            "#{} {} ds[n={nlip_ds:.12e},i={ipopt_ds:.12e},d={ds_gap:.3e}] prefinal_vU[n={nlip_vu:.12e},i={ipopt_vu:.12e},d={vu_gap:.3e}] gain={gain:.3e} slack={slack:.12e} vU={multiplier:.12e}",
                            index,
                            glider_inequality_label(index, intervals, order),
                        )
                    },
                )
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn journal_augmented_block_diff_summary_text(
            lhs: &[f64],
            rhs: &[f64],
            dims: &[usize; 4],
        ) -> String {
            let labels = ["x", "p", "lambda", "z"];
            let mut start = 0;
            let mut parts = Vec::new();
            for (label, len) in labels.iter().zip(dims.iter()) {
                let end = start + len;
                if lhs.len() >= end && rhs.len() >= end {
                    parts.push(format!(
                        "{label}[{}]",
                        journal_vector_diff_summary_text(&lhs[start..end], &rhs[start..end])
                    ));
                } else {
                    parts.push(format!("{label}[len_mismatch]"));
                }
                start = end;
            }
            parts.join(" ")
        }

        fn print_ranked_journal_vector_matches(
            label: &str,
            nlip_vector: &[f64],
            ipopt_label: &str,
            ipopt_vectors: &[Vec<f64>],
            dims: &[usize; 4],
        ) -> Option<(usize, f64)> {
            let mut ranked = ipopt_vectors
                .iter()
                .enumerate()
                .map(|(index, vector)| (index, journal_vector_max_abs_diff(nlip_vector, vector)))
                .collect::<Vec<_>>();
            ranked.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)));
            for (rank, (index, max_abs_diff)) in ranked.iter().take(4).enumerate() {
                let ipopt_vector = &ipopt_vectors[*index];
                println!(
                    "  {label}_best_{ipopt_label}_match[{rank}] {ipopt_label}[{index}] max_abs_diff={max_abs_diff:.17e} {} blocks[{}]",
                    journal_vector_diff_summary_text(nlip_vector, ipopt_vector),
                    journal_augmented_block_diff_summary_text(nlip_vector, ipopt_vector, dims),
                );
            }
            ranked.first().copied()
        }

        fn solve_boundary_match_text(label: &str, best: Option<(usize, f64)>) -> String {
            best.map_or_else(
                || format!("{label}=none"),
                |(index, max_abs_diff)| format!("{label}[{index}]={max_abs_diff:.3e}"),
            )
        }

        fn solve_boundary_cumulative_match_text(
            label: &str,
            best: Option<(usize, usize, f64)>,
        ) -> String {
            best.map_or_else(
                || format!("{label}=none"),
                |(start, terms, max_abs_diff)| {
                    format!("{label}[start={start},terms={terms}]={max_abs_diff:.3e}")
                },
            )
        }

        fn ipopt_cumulative_solution_windows(
            solutions: &[Vec<f64>],
            max_terms: usize,
        ) -> Vec<(usize, usize, Vec<f64>)> {
            let mut windows = Vec::new();
            for start in 0..solutions.len() {
                let mut cumulative = Vec::new();
                for terms in 1..=max_terms {
                    let index = start + terms - 1;
                    let Some(solution) = solutions.get(index) else {
                        break;
                    };
                    if terms == 1 {
                        cumulative = solution.clone();
                    } else if cumulative.len() == solution.len() {
                        for (value, correction) in cumulative.iter_mut().zip(solution.iter()) {
                            // IpPDFullSpaceSolver.cpp::SolveOnce applies the
                            // iterative-refinement correction with alpha=-1,
                            // beta=1, so cumulative SOL = first - correction.
                            *value -= correction;
                        }
                    } else {
                        break;
                    }
                    windows.push((start, terms, cumulative.clone()));
                }
            }
            windows
        }

        fn glider_augmented_fingerprint_iteration() -> usize {
            std::env::var("GLIDER_PARITY_NLIP_AUGMENTED_ITER")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0)
        }

        fn print_nlip_refinement_residual_block_fingerprints(
            label: &str,
            blocks: Option<&[Vec<f64>]>,
        ) {
            if let Some(blocks) = blocks {
                for (step, values) in blocks.iter().enumerate() {
                    println!(
                        "  nlip_trace_refinement_residual_{label}[{step}] [{}]",
                        journal_vector_fingerprint_text(journal_vector_fingerprint(values)),
                    );
                }
            }
        }

        fn print_nlip_augmented_dump_fingerprints(
            dump_dir: Option<&Path>,
            journal_output: Option<&str>,
            intervals: usize,
            order: usize,
        ) {
            if std::env::var_os("GLIDER_PARITY_PRINT_NLIP_AUGMENTED_FINGERPRINTS").is_none() {
                return;
            }
            let Some(dump_dir) = dump_dir else {
                println!("nlip augmented dump fingerprints unavailable");
                return;
            };
            let dump_iteration = glider_augmented_fingerprint_iteration();
            let path = dump_dir.join(format!("nlip_kkt_iter_{dump_iteration:04}.txt"));
            if !path.exists() {
                println!(
                    "nlip augmented dump fingerprints missing path={}",
                    path.display()
                );
                return;
            }
            let dump = load_glider_linear_debug_dump(&path);
            let dims = [
                dump.x_dimension,
                dump.inequality_dimension,
                dump.equality_dimension,
                dump.inequality_dimension,
            ];
            println!(
                "nlip augmented dump fingerprints matrix_dim={} rhs_len={} values_len={} p_offset={} lambda_offset={} z_offset={}",
                dump.matrix_dimension,
                dump.rhs.len(),
                dump.values.len(),
                dump.p_offset,
                dump.lambda_offset,
                dump.z_offset,
            );
            println!(
                "  nlip_rhs_final [{}] blocks[{}]",
                journal_vector_fingerprint_text(journal_vector_fingerprint(&dump.rhs)),
                journal_augmented_block_fingerprint_text(&dump.rhs, &dims),
            );
            let prefinal_rhs = dump.rhs.iter().map(|value| -*value).collect::<Vec<_>>();
            println!(
                "  nlip_rhs_prefinal [{}] blocks[{}]",
                journal_vector_fingerprint_text(journal_vector_fingerprint(&prefinal_rhs)),
                journal_augmented_block_fingerprint_text(&prefinal_rhs, &dims),
            );
            println!(
                "  nlip_r_dual [{}]",
                journal_vector_fingerprint_text(journal_vector_fingerprint(&dump.r_dual)),
            );
            println!(
                "  nlip_bound_rhs [{}]",
                journal_vector_fingerprint_text(journal_vector_fingerprint(&dump.bound_rhs)),
            );
            println!(
                "  nlip_bound_diagonal [{}]",
                journal_vector_fingerprint_text(journal_vector_fingerprint(&dump.bound_diagonal)),
            );
            let Some(journal) = journal_output else {
                return;
            };
            let ipopt_grad_lag_x_vectors = ipopt_journal_dense_vectors(journal, "curr_grad_lag_x");
            println!(
                "  ipopt_curr_grad_lag_x_count={}",
                ipopt_grad_lag_x_vectors.len()
            );
            let mut ranked_grad_lag_x = ipopt_grad_lag_x_vectors
                .iter()
                .enumerate()
                .map(|(index, gradient)| {
                    (index, journal_vector_max_abs_diff(&dump.r_dual, gradient))
                })
                .collect::<Vec<_>>();
            ranked_grad_lag_x.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)));
            for (rank, (index, max_abs_diff)) in ranked_grad_lag_x.iter().take(4).enumerate() {
                let gradient = &ipopt_grad_lag_x_vectors[*index];
                println!(
                    "  nlip_best_curr_grad_lag_x_match[{rank}] ipopt_curr_grad_lag_x[{index}] max_abs_diff={max_abs_diff:.17e} {}",
                    journal_vector_diff_summary_text(&dump.r_dual, gradient),
                );
            }
            let ipopt_matrices = ipopt_augmented_journal_kkt_values(journal, &dims);
            println!("  ipopt_kkt_matrix_count={}", ipopt_matrices.len());
            for (rank, matrix) in ipopt_matrices.iter().take(4).enumerate() {
                let values = matrix.iter().map(|(_, value)| *value).collect::<Vec<_>>();
                println!(
                    "  nlip_vs_ipopt_kkt[{rank}] [{}] {}",
                    journal_vector_fingerprint_text(journal_vector_fingerprint(&values)),
                    journal_kkt_diff_summary_text(&dump, matrix),
                );
            }
            let mut ranked_kkt = ipopt_matrices
                .iter()
                .enumerate()
                .map(|(index, matrix)| (index, journal_kkt_max_abs_diff(&dump, matrix)))
                .collect::<Vec<_>>();
            ranked_kkt.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)));
            let best_kkt_match = ranked_kkt.first().copied();
            for (rank, (index, max_abs_diff)) in ranked_kkt.iter().take(4).enumerate() {
                let matrix = &ipopt_matrices[*index];
                let values = matrix.iter().map(|(_, value)| *value).collect::<Vec<_>>();
                println!(
                    "  nlip_best_kkt_match[{rank}] ipopt_kkt[{index}] max_abs_diff={max_abs_diff:.17e} [{}] {}",
                    journal_vector_fingerprint_text(journal_vector_fingerprint(&values)),
                    journal_kkt_diff_summary_text(&dump, matrix),
                );
            }
            let (ipopt_rhs_vectors, ipopt_sol_vectors) = ipopt_augmented_journal_vectors(journal);
            let Some(ipopt_prefinal_rhs) = ipopt_rhs_vectors.get(1) else {
                return;
            };
            println!(
                "  nlip_vs_ipopt_rhs[1] {}",
                journal_vector_diff_summary_text(&prefinal_rhs, ipopt_prefinal_rhs)
            );
            let mut start = 0;
            for (name, len) in [
                ("x", dims[0]),
                ("p", dims[1]),
                ("lambda", dims[2]),
                ("z", dims[3]),
            ] {
                let end = start + len;
                if prefinal_rhs.len() >= end && ipopt_prefinal_rhs.len() >= end {
                    println!(
                        "    block_{name} {}",
                        journal_vector_diff_summary_text(
                            &prefinal_rhs[start..end],
                            &ipopt_prefinal_rhs[start..end],
                        )
                    );
                }
                start = end;
            }
            let mut ranked_rhs = ipopt_rhs_vectors
                .iter()
                .enumerate()
                .map(|(index, rhs)| (index, journal_vector_max_abs_diff(&prefinal_rhs, rhs)))
                .collect::<Vec<_>>();
            ranked_rhs.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)));
            let best_prefinal_rhs_match = ranked_rhs.first().copied();
            for (rank, (index, max_abs_diff)) in ranked_rhs.iter().take(4).enumerate() {
                let ipopt_rhs = &ipopt_rhs_vectors[*index];
                println!(
                    "  nlip_best_rhs_match[{rank}] ipopt_rhs[{index}] max_abs_diff={max_abs_diff:.17e} {}",
                    journal_vector_diff_summary_text(&prefinal_rhs, ipopt_rhs),
                );
            }
            let mut best_prefinal_solution_match = None;
            let mut best_prefinal_cumulative_match: Option<(usize, usize, f64)> = None;
            let mut best_trace_rhs_match = None;
            let mut best_trace_solution_match = None;
            let mut best_trace_accumulated_match: Option<(usize, usize, f64)> = None;
            if let Some(nlip_prefinal_solution) = dump.linear_solution_prefinal.as_ref() {
                println!(
                    "  nlip_solution_prefinal [{}] blocks[{}]",
                    journal_vector_fingerprint_text(journal_vector_fingerprint(
                        nlip_prefinal_solution
                    )),
                    journal_augmented_block_fingerprint_text(nlip_prefinal_solution, &dims),
                );
                let mut ranked_sol = ipopt_sol_vectors
                    .iter()
                    .enumerate()
                    .map(|(index, solution)| {
                        (
                            index,
                            journal_vector_max_abs_diff(nlip_prefinal_solution, solution),
                        )
                    })
                    .collect::<Vec<_>>();
                ranked_sol.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)));
                best_prefinal_solution_match = ranked_sol.first().copied();
                for (rank, (index, max_abs_diff)) in ranked_sol.iter().take(4).enumerate() {
                    let solution = &ipopt_sol_vectors[*index];
                    println!(
                        "  nlip_best_sol_match[{rank}] ipopt_sol[{index}] max_abs_diff={max_abs_diff:.17e} {}",
                        journal_vector_diff_summary_text(nlip_prefinal_solution, solution),
                    );
                }
                let cumulative_windows = ipopt_cumulative_solution_windows(&ipopt_sol_vectors, 4);
                let mut ranked_cumulative_sol = cumulative_windows
                    .iter()
                    .enumerate()
                    .map(|(rank_index, (_, _, solution))| {
                        (
                            rank_index,
                            journal_vector_max_abs_diff(nlip_prefinal_solution, solution),
                        )
                    })
                    .collect::<Vec<_>>();
                ranked_cumulative_sol
                    .sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)));
                best_prefinal_cumulative_match =
                    ranked_cumulative_sol.first().map(|(window_index, diff)| {
                        let (start, terms, _) = &cumulative_windows[*window_index];
                        (*start, *terms, *diff)
                    });
                for (rank, (window_index, max_abs_diff)) in
                    ranked_cumulative_sol.iter().take(4).enumerate()
                {
                    let (start, terms, solution) = &cumulative_windows[*window_index];
                    println!(
                        "  nlip_best_cumulative_sol_match[{rank}] ipopt_sol_start={start} terms={terms} max_abs_diff={max_abs_diff:.17e} {}",
                        journal_vector_diff_summary_text(nlip_prefinal_solution, solution),
                    );
                }
            }
            if let Some(nlip_final_solution) = dump.linear_solution_final.as_ref() {
                println!(
                    "  nlip_solution_final [{}] blocks[{}]",
                    journal_vector_fingerprint_text(journal_vector_fingerprint(
                        nlip_final_solution
                    )),
                    journal_augmented_block_fingerprint_text(nlip_final_solution, &dims),
                );
            }
            if let Some(trace_rhs) = dump.linear_trace_rhs_prefinal.as_ref() {
                println!(
                    "  nlip_trace_rhs_prefinal [{}] blocks[{}]",
                    journal_vector_fingerprint_text(journal_vector_fingerprint(trace_rhs)),
                    journal_augmented_block_fingerprint_text(trace_rhs, &dims),
                );
                best_trace_rhs_match = print_ranked_journal_vector_matches(
                    "nlip_trace_rhs_prefinal",
                    trace_rhs,
                    "ipopt_rhs",
                    &ipopt_rhs_vectors,
                    &dims,
                );
            }
            if let Some(trace_solution) = dump.linear_trace_solution_prefinal_unrefined.as_ref() {
                println!(
                    "  nlip_trace_solution_prefinal_unrefined [{}] blocks[{}]",
                    journal_vector_fingerprint_text(journal_vector_fingerprint(trace_solution)),
                    journal_augmented_block_fingerprint_text(trace_solution, &dims),
                );
                best_trace_solution_match = print_ranked_journal_vector_matches(
                    "nlip_trace_solution_prefinal_unrefined",
                    trace_solution,
                    "ipopt_sol",
                    &ipopt_sol_vectors,
                    &dims,
                );
                if let Some((best_index, _)) = best_trace_solution_match
                    && let Some(ipopt_solution) = ipopt_sol_vectors.get(best_index)
                {
                    println!(
                        "  nlip_trace_solution_prefinal_unrefined_upper_slack_channel ipopt_sol[{best_index}] {}",
                        upper_slack_prefinal_channel_diff_text(
                            &dump,
                            trace_solution,
                            ipopt_solution,
                            intervals,
                            order,
                        )
                    );
                }
            }
            if let Some(ratios) = dump.linear_trace_refinement_residual_ratios.as_ref() {
                for (step, ratio) in ratios.iter().enumerate() {
                    println!(
                        "  nlip_trace_refinement_ratio[{step}] before={:.17e} after={:.17e}",
                        ratio[0], ratio[1],
                    );
                }
            }
            print_nlip_refinement_residual_block_fingerprints(
                "x",
                dump.linear_trace_refinement_residual_x.as_deref(),
            );
            print_nlip_refinement_residual_block_fingerprints(
                "s",
                dump.linear_trace_refinement_residual_s.as_deref(),
            );
            print_nlip_refinement_residual_block_fingerprints(
                "c",
                dump.linear_trace_refinement_residual_c.as_deref(),
            );
            print_nlip_refinement_residual_block_fingerprints(
                "d",
                dump.linear_trace_refinement_residual_d.as_deref(),
            );
            print_nlip_refinement_residual_block_fingerprints(
                "z_lower",
                dump.linear_trace_refinement_residual_z_lower.as_deref(),
            );
            print_nlip_refinement_residual_block_fingerprints(
                "z_upper",
                dump.linear_trace_refinement_residual_z_upper.as_deref(),
            );
            print_nlip_refinement_residual_block_fingerprints(
                "v_upper",
                dump.linear_trace_refinement_residual_v_upper.as_deref(),
            );
            if let Some(refinement_rhs) = dump.linear_trace_refinement_rhs.as_ref() {
                for (step, rhs) in refinement_rhs.iter().enumerate() {
                    println!(
                        "  nlip_trace_refinement_rhs[{step}] [{}] blocks[{}]",
                        journal_vector_fingerprint_text(journal_vector_fingerprint(rhs)),
                        journal_augmented_block_fingerprint_text(rhs, &dims),
                    );
                    print_ranked_journal_vector_matches(
                        &format!("nlip_trace_refinement_rhs[{step}]"),
                        rhs,
                        "ipopt_rhs",
                        &ipopt_rhs_vectors,
                        &dims,
                    );
                }
            }
            if let Some(refinement_solution) = dump.linear_trace_refinement_solution.as_ref() {
                for (step, solution) in refinement_solution.iter().enumerate() {
                    println!(
                        "  nlip_trace_refinement_solution[{step}] [{}] blocks[{}]",
                        journal_vector_fingerprint_text(journal_vector_fingerprint(solution)),
                        journal_augmented_block_fingerprint_text(solution, &dims),
                    );
                    print_ranked_journal_vector_matches(
                        &format!("nlip_trace_refinement_solution[{step}]"),
                        solution,
                        "ipopt_sol",
                        &ipopt_sol_vectors,
                        &dims,
                    );
                }
            }
            if let Some(accumulated) = dump.linear_trace_refinement_accumulated_solution.as_ref() {
                let cumulative_windows = ipopt_cumulative_solution_windows(&ipopt_sol_vectors, 4);
                for (step, solution) in accumulated.iter().enumerate() {
                    let mut ranked = cumulative_windows
                        .iter()
                        .enumerate()
                        .map(|(window_index, (_, _, ipopt_solution))| {
                            (
                                window_index,
                                journal_vector_max_abs_diff(solution, ipopt_solution),
                            )
                        })
                        .collect::<Vec<_>>();
                    ranked.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)));
                    if let Some((window_index, max_abs_diff)) = ranked.first() {
                        let (start, terms, _) = &cumulative_windows[*window_index];
                        let candidate = (*start, *terms, *max_abs_diff);
                        best_trace_accumulated_match =
                            best_trace_accumulated_match.map_or(Some(candidate), |current| {
                                if candidate.2 < current.2 {
                                    Some(candidate)
                                } else {
                                    Some(current)
                                }
                            });
                    }
                    for (rank, (window_index, max_abs_diff)) in ranked.iter().take(4).enumerate() {
                        let (start, terms, ipopt_solution) = &cumulative_windows[*window_index];
                        println!(
                            "  nlip_trace_refinement_accumulated_solution[{step}]_best_ipopt_cumulative_match[{rank}] ipopt_sol_start={start} terms={terms} max_abs_diff={max_abs_diff:.17e} {} blocks[{}]",
                            journal_vector_diff_summary_text(solution, ipopt_solution),
                            journal_augmented_block_diff_summary_text(
                                solution,
                                ipopt_solution,
                                &dims
                            ),
                        );
                        if rank == 0 {
                            println!(
                                "  nlip_trace_refinement_accumulated_solution[{step}]_upper_slack_channel ipopt_sol_start={start} terms={terms} {}",
                                upper_slack_prefinal_channel_diff_text(
                                    &dump,
                                    solution,
                                    ipopt_solution,
                                    intervals,
                                    order,
                                )
                            );
                        }
                    }
                }
            }
            println!(
                "  nlip_solve_boundary_summary iter={} {} {} {} {} {} {} {} {}",
                dump_iteration,
                solve_boundary_match_text("kkt", best_kkt_match),
                solve_boundary_match_text("rhs_prefinal", best_prefinal_rhs_match),
                solve_boundary_match_text("sol_prefinal", best_prefinal_solution_match),
                solve_boundary_cumulative_match_text(
                    "sol_prefinal_cumulative",
                    best_prefinal_cumulative_match,
                ),
                solve_boundary_match_text("trace_rhs_prefinal", best_trace_rhs_match),
                solve_boundary_match_text(
                    "trace_sol_prefinal_unrefined",
                    best_trace_solution_match,
                ),
                solve_boundary_cumulative_match_text(
                    "trace_refinement_accumulated",
                    best_trace_accumulated_match,
                ),
                solve_boundary_match_text(
                    "ipopt_curr_grad_lag_x",
                    ranked_grad_lag_x.first().copied(),
                ),
            );
        }

        fn sorted_ipopt_spral_dump_paths(dump_dir: &Path, phase: &str) -> Vec<std::path::PathBuf> {
            let suffix = format!("_{phase}.txt");
            let mut paths = fs::read_dir(dump_dir)
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to read IPOPT SPRAL dump dir {}: {error}",
                        dump_dir.display()
                    )
                })
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| {
                            name.starts_with("ipopt_spral_") && name.ends_with(&suffix)
                        })
                })
                .collect::<Vec<_>>();
            paths.sort();
            paths
        }

        fn sorted_nlip_linear_debug_dump_paths(dump_dir: &Path) -> Vec<std::path::PathBuf> {
            let mut paths = fs::read_dir(dump_dir)
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to read NLIP linear debug dump dir {}: {error}",
                        dump_dir.display()
                    )
                })
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
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

        fn nlip_linear_debug_dump_iteration(path: &Path) -> Option<usize> {
            let name = path.file_name()?.to_str()?;
            let index = name.strip_prefix("nlip_kkt_iter_")?.strip_suffix(".txt")?;
            index.parse().ok()
        }

        fn sorted_ipopt_full_space_residual_dump_paths(dump_dir: &Path) -> Vec<std::path::PathBuf> {
            let mut paths = fs::read_dir(dump_dir)
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to read IPOPT full-space residual dump dir {}: {error}",
                        dump_dir.display()
                    )
                })
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| {
                            name.starts_with("ipopt_pdfullspace_residual_")
                                && name.ends_with(".txt")
                        })
                })
                .collect::<Vec<_>>();
            paths.sort();
            paths
        }

        fn ipopt_full_space_residual_stage<'a>(
            dump: &'a IpoptFullSpaceResidualDump,
            stage: &str,
        ) -> &'a [f64] {
            match stage {
                "x_after_w" => &dump.resid_x_after_w,
                "x_after_jc" => &dump.resid_x_after_jc,
                "x_after_jd" => &dump.resid_x_after_jd,
                "x_after_pxl" => &dump.resid_x_after_pxl,
                "x_after_pxu" => &dump.resid_x_after_pxu,
                "x_after_add_two_vectors" => &dump.resid_x_after_add_two_vectors,
                "x" => &dump.resid_x,
                "s_after_pdu" => &dump.resid_s_after_pdu,
                "s_after_pdl" => &dump.resid_s_after_pdl,
                "s_after_add_two_vectors" => &dump.resid_s_after_add_two_vectors,
                "s_after_delta" => &dump.resid_s_after_delta,
                "s" => &dump.resid_s,
                _ => panic!("unknown IPOPT residual stage {stage}"),
            }
        }

        fn residual_stage_diff_location_text(
            stage: &str,
            dump: &GliderLinearDebugDump,
            lhs: &[f64],
            rhs: &[f64],
            intervals: usize,
            order: usize,
        ) -> String {
            if stage.starts_with('x') {
                return augmented_vector_diff_location_text(dump, lhs, rhs, intervals, order);
            }
            journal_vector_max_abs_diff_index(lhs, rhs).map_or_else(
                || "max_abs_index=none".to_string(),
                |index| {
                    let lhs_value = lhs.get(index).copied().unwrap_or(f64::NAN);
                    let rhs_value = rhs.get(index).copied().unwrap_or(f64::NAN);
                    let diff = lhs_value - rhs_value;
                    if stage.starts_with('s') {
                        let slack = dump.slack.get(index).copied().unwrap_or(f64::NAN);
                        let multiplier = dump.multipliers.get(index).copied().unwrap_or(f64::NAN);
                        return format!(
                            "max_abs_index={index} s[{index}] nlip={lhs_value:.17e} ipopt={rhs_value:.17e} diff={diff:.17e} slack={slack:.17e} multiplier={multiplier:.17e}",
                        );
                    }
                    format!(
                        "max_abs_index={index} nlip={lhs_value:.17e} ipopt={rhs_value:.17e} diff={diff:.17e}",
                    )
                },
            )
        }

        fn print_ranked_ipopt_residual_stage_matches(
            label: &str,
            nlip_vector: &[f64],
            stage: &str,
            ipopt: &[IpoptFullSpaceResidualDump],
            nlip: &GliderLinearDebugDump,
            intervals: usize,
            order: usize,
        ) -> Option<(usize, f64)> {
            let mut ranked = ipopt
                .iter()
                .map(|dump| {
                    (
                        dump.call_index,
                        journal_vector_max_abs_diff(
                            nlip_vector,
                            ipopt_full_space_residual_stage(dump, stage),
                        ),
                    )
                })
                .collect::<Vec<_>>();
            ranked.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)));
            for (rank, (call_index, max_abs_diff)) in ranked.iter().take(4).enumerate() {
                let dump = ipopt
                    .iter()
                    .find(|dump| dump.call_index == *call_index)
                    .expect("ranked residual call should exist");
                let ipopt_vector = ipopt_full_space_residual_stage(dump, stage);
                println!(
                    "  {label}_best_ipopt_residual_{stage}_match[{rank}] call={call_index} max_abs_diff={max_abs_diff:.17e} [{}] {} {}",
                    journal_vector_fingerprint_text(journal_vector_fingerprint(ipopt_vector)),
                    journal_vector_diff_summary_text(nlip_vector, ipopt_vector),
                    residual_stage_diff_location_text(
                        stage,
                        nlip,
                        nlip_vector,
                        ipopt_vector,
                        intervals,
                        order,
                    ),
                );
            }
            ranked.first().copied()
        }

        fn print_ipopt_full_space_residual_dump_fingerprints(
            ipopt_dump_dir: Option<&Path>,
            nlip_dump_dir: Option<&Path>,
            intervals: usize,
            order: usize,
        ) {
            let Some(ipopt_dump_dir) = ipopt_dump_dir else {
                return;
            };
            let Some(nlip_dump_dir) = nlip_dump_dir else {
                println!("ipopt residual dump comparison unavailable: missing NLIP dump dir");
                return;
            };
            let dump_iteration = glider_augmented_fingerprint_iteration();
            let nlip_path = nlip_dump_dir.join(format!("nlip_kkt_iter_{dump_iteration:04}.txt"));
            if !nlip_path.exists() {
                println!(
                    "ipopt residual dump comparison unavailable: missing {}",
                    nlip_path.display()
                );
                return;
            }
            let nlip = load_glider_linear_debug_dump(&nlip_path);
            let residual_paths = sorted_ipopt_full_space_residual_dump_paths(ipopt_dump_dir);
            let ipopt = residual_paths
                .iter()
                .filter_map(|path| try_load_ipopt_full_space_residual_dump(path))
                .collect::<Vec<_>>();
            let skipped = residual_paths.len().saturating_sub(ipopt.len());
            println!(
                "ipopt residual dump comparison dir={} nlip_iter={} calls={} skipped_incomplete={}",
                ipopt_dump_dir.display(),
                dump_iteration,
                ipopt.len(),
                skipped,
            );
            if ipopt.is_empty() {
                return;
            }
            let stages = [
                (
                    "x_after_w",
                    nlip.linear_trace_refinement_residual_x_after_w.as_deref(),
                ),
                (
                    "x_after_jc",
                    nlip.linear_trace_refinement_residual_x_after_jc.as_deref(),
                ),
                (
                    "x_after_jd",
                    nlip.linear_trace_refinement_residual_x_after_jd.as_deref(),
                ),
                (
                    "x_after_pxl",
                    nlip.linear_trace_refinement_residual_x_after_pxl.as_deref(),
                ),
                (
                    "x_after_pxu",
                    nlip.linear_trace_refinement_residual_x_after_pxu.as_deref(),
                ),
                (
                    "x_after_add_two_vectors",
                    nlip.linear_trace_refinement_residual_x_after_add_two_vectors
                        .as_deref(),
                ),
                ("x", nlip.linear_trace_refinement_residual_x.as_deref()),
                (
                    "s_after_pdu",
                    nlip.linear_trace_refinement_residual_s_after_pdu.as_deref(),
                ),
                (
                    "s_after_pdl",
                    nlip.linear_trace_refinement_residual_s_after_pdl.as_deref(),
                ),
                (
                    "s_after_add_two_vectors",
                    nlip.linear_trace_refinement_residual_s_after_add_two_vectors
                        .as_deref(),
                ),
                (
                    "s_after_delta",
                    nlip.linear_trace_refinement_residual_s_after_delta
                        .as_deref(),
                ),
                ("s", nlip.linear_trace_refinement_residual_s.as_deref()),
            ];
            let mut summary = Vec::new();
            for (stage, values) in stages {
                let Some(values) = values else {
                    continue;
                };
                for (step, nlip_vector) in values.iter().enumerate() {
                    let label = format!("ipopt_residual_stage_{stage}[{step}]");
                    let best = print_ranked_ipopt_residual_stage_matches(
                        &label,
                        nlip_vector,
                        stage,
                        &ipopt,
                        &nlip,
                        intervals,
                        order,
                    );
                    if step == 0 {
                        summary.push(solve_boundary_match_text(stage, best));
                    }
                }
            }
            if !summary.is_empty() {
                println!(
                    "  ipopt_residual_stage_summary iter={} {}",
                    dump_iteration,
                    summary.join(" "),
                );
            }
        }

        fn ipopt_spral_lower_csc_col_ptrs(dump: &IpoptSpralInterfaceDump) -> Vec<usize> {
            dump.ia
                .iter()
                .map(|&value| value.checked_sub(1).expect("IPOPT SPRAL IA is one-based"))
                .collect()
        }

        fn ipopt_spral_lower_csc_row_indices(dump: &IpoptSpralInterfaceDump) -> Vec<usize> {
            dump.ja
                .iter()
                .map(|&value| value.checked_sub(1).expect("IPOPT SPRAL JA is one-based"))
                .collect()
        }

        fn ipopt_spral_structure_diff_summary_text(
            nlip: &GliderLinearDebugDump,
            ipopt: &IpoptSpralInterfaceDump,
        ) -> String {
            let ipopt_col_ptrs = ipopt_spral_lower_csc_col_ptrs(ipopt);
            let ipopt_row_indices = ipopt_spral_lower_csc_row_indices(ipopt);
            let col_len_match = nlip.col_ptrs.len() == ipopt_col_ptrs.len();
            let row_len_match = nlip.row_indices.len() == ipopt_row_indices.len();
            let first_col_diff = nlip
                .col_ptrs
                .iter()
                .zip(ipopt_col_ptrs.iter())
                .enumerate()
                .find_map(|(index, (&lhs, &rhs))| (lhs != rhs).then_some((index, lhs, rhs)));
            let first_row_diff = nlip
                .row_indices
                .iter()
                .zip(ipopt_row_indices.iter())
                .enumerate()
                .find_map(|(index, (&lhs, &rhs))| (lhs != rhs).then_some((index, lhs, rhs)));
            format!(
                "dim={} nnz={} col_len_match={} row_len_match={} first_col_diff={} first_row_diff={}",
                ipopt.ndim,
                ipopt.nonzeros,
                col_len_match,
                row_len_match,
                first_col_diff.map_or_else(
                    || "none".to_string(),
                    |(index, lhs, rhs)| format!("{index}:{lhs}!={rhs}")
                ),
                first_row_diff.map_or_else(
                    || "none".to_string(),
                    |(index, lhs, rhs)| format!("{index}:{lhs}!={rhs}")
                ),
            )
        }

        fn ipopt_spral_matrix_max_abs_diff(
            nlip: &GliderLinearDebugDump,
            ipopt: &IpoptSpralInterfaceDump,
        ) -> f64 {
            let ipopt_col_ptrs = ipopt_spral_lower_csc_col_ptrs(ipopt);
            let ipopt_row_indices = ipopt_spral_lower_csc_row_indices(ipopt);
            if nlip.col_ptrs != ipopt_col_ptrs
                || nlip.row_indices != ipopt_row_indices
                || nlip.values.len() != ipopt.values.len()
            {
                return f64::INFINITY;
            }
            journal_vector_max_abs_diff(&nlip.values, &ipopt.values)
        }

        fn ipopt_spral_factor_info_text(dump: Option<&IpoptSpralInterfaceDump>) -> String {
            dump.map_or_else(
                || "factor_info=missing".to_string(),
                |dump| {
                    format!(
                        "ordering={} scaling={} neg={} delay={} two={} scaling_len={}",
                        dump.control_ordering,
                        dump.control_scaling,
                        dump.info_num_neg
                            .map_or_else(|| "missing".to_string(), |value| value.to_string()),
                        dump.info_num_delay
                            .map_or_else(|| "missing".to_string(), |value| value.to_string()),
                        dump.info_num_two
                            .map_or_else(|| "missing".to_string(), |value| value.to_string()),
                        dump.scaling.as_ref().map_or(0, Vec::len),
                    )
                },
            )
        }

        fn best_ipopt_spral_matrix_match(
            nlip: &GliderLinearDebugDump,
            before: &[IpoptSpralInterfaceDump],
        ) -> Option<(usize, f64)> {
            before
                .iter()
                .map(|dump| (dump.call_index, ipopt_spral_matrix_max_abs_diff(nlip, dump)))
                .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)))
        }

        fn best_ipopt_spral_rhs_match(
            nlip_vector: &[f64],
            before: &[IpoptSpralInterfaceDump],
        ) -> Option<(usize, f64)> {
            before
                .iter()
                .map(|dump| {
                    (
                        dump.call_index,
                        journal_vector_max_abs_diff(nlip_vector, &dump.rhs),
                    )
                })
                .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)))
        }

        fn best_ipopt_spral_solution_match(
            nlip_vector: &[f64],
            after_solve: &BTreeMap<usize, IpoptSpralInterfaceDump>,
        ) -> Option<(usize, f64)> {
            after_solve
                .iter()
                .map(|(call_index, dump)| {
                    (
                        *call_index,
                        journal_vector_max_abs_diff(nlip_vector, &dump.rhs),
                    )
                })
                .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)))
        }

        fn best_ipopt_spral_cumulative_solution_match(
            nlip_vector: &[f64],
            after_solve: &BTreeMap<usize, IpoptSpralInterfaceDump>,
            max_terms: usize,
        ) -> Option<(usize, usize, f64)> {
            ipopt_spral_cumulative_solution_windows(after_solve, max_terms)
                .into_iter()
                .map(|(start_call, terms, solution)| {
                    (
                        start_call,
                        terms,
                        journal_vector_max_abs_diff(nlip_vector, &solution),
                    )
                })
                .min_by(|lhs, rhs| lhs.2.total_cmp(&rhs.2).then(lhs.0.cmp(&rhs.0)))
        }

        fn print_ranked_ipopt_spral_rhs_matches(
            label: &str,
            nlip_vector: &[f64],
            before: &[IpoptSpralInterfaceDump],
            nlip: &GliderLinearDebugDump,
            intervals: usize,
            order: usize,
        ) -> Option<(usize, f64)> {
            let mut ranked = before
                .iter()
                .map(|dump| {
                    (
                        dump.call_index,
                        journal_vector_max_abs_diff(nlip_vector, &dump.rhs),
                    )
                })
                .collect::<Vec<_>>();
            ranked.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)));
            for (rank, (call_index, max_abs_diff)) in ranked.iter().take(4).enumerate() {
                let dump = before
                    .iter()
                    .find(|dump| dump.call_index == *call_index)
                    .expect("ranked RHS call should exist");
                println!(
                    "  {label}_best_ipopt_spral_rhs_match[{rank}] call={call_index} nrhs={} max_abs_diff={max_abs_diff:.17e} [{}] {} {}",
                    dump.nrhs,
                    journal_vector_fingerprint_text(journal_vector_fingerprint(&dump.rhs)),
                    journal_vector_diff_summary_text(nlip_vector, &dump.rhs),
                    augmented_vector_diff_location_text(
                        nlip,
                        nlip_vector,
                        &dump.rhs,
                        intervals,
                        order,
                    ),
                );
                println!(
                    "    rhs_component {}",
                    augmented_vector_diff_component_text(
                        nlip,
                        nlip_vector,
                        &dump.rhs,
                        intervals,
                        order,
                    ),
                );
            }
            ranked.first().copied()
        }

        fn print_ranked_ipopt_spral_solution_matches(
            label: &str,
            nlip_vector: &[f64],
            after_solve: &BTreeMap<usize, IpoptSpralInterfaceDump>,
            nlip: &GliderLinearDebugDump,
            intervals: usize,
            order: usize,
        ) -> Option<(usize, f64)> {
            let mut ranked = after_solve
                .iter()
                .map(|(call_index, dump)| {
                    (
                        *call_index,
                        journal_vector_max_abs_diff(nlip_vector, &dump.rhs),
                    )
                })
                .collect::<Vec<_>>();
            ranked.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)));
            for (rank, (call_index, max_abs_diff)) in ranked.iter().take(4).enumerate() {
                let dump = after_solve
                    .get(call_index)
                    .expect("ranked solution call should exist");
                println!(
                    "  {label}_best_ipopt_spral_solution_match[{rank}] call={call_index} max_abs_diff={max_abs_diff:.17e} [{}] {} {}",
                    journal_vector_fingerprint_text(journal_vector_fingerprint(&dump.rhs)),
                    journal_vector_diff_summary_text(nlip_vector, &dump.rhs),
                    augmented_vector_diff_location_text(
                        nlip,
                        nlip_vector,
                        &dump.rhs,
                        intervals,
                        order,
                    ),
                );
            }
            ranked.first().copied()
        }

        fn ipopt_spral_cumulative_solution_windows(
            after_solve: &BTreeMap<usize, IpoptSpralInterfaceDump>,
            max_terms: usize,
        ) -> Vec<(usize, usize, Vec<f64>)> {
            let solutions = after_solve
                .iter()
                .map(|(call_index, dump)| (*call_index, dump.rhs.clone()))
                .collect::<Vec<_>>();
            let mut windows = Vec::new();
            for start in 0..solutions.len() {
                let (start_call, first_solution) = &solutions[start];
                let mut cumulative = Vec::new();
                for terms in 1..=max_terms {
                    let index = start + terms - 1;
                    let Some((_, solution)) = solutions.get(index) else {
                        break;
                    };
                    if terms == 1 {
                        cumulative = first_solution.clone();
                    } else if cumulative.len() == solution.len() {
                        for (value, correction) in cumulative.iter_mut().zip(solution.iter()) {
                            // IpPDFullSpaceSolver.cpp::SolveOnce applies an
                            // iterative-refinement correction as alpha=-1,
                            // beta=1: the accumulated step is first - corr.
                            *value -= correction;
                        }
                    } else {
                        break;
                    }
                    windows.push((*start_call, terms, cumulative.clone()));
                }
            }
            windows
        }

        fn print_ranked_ipopt_spral_cumulative_solution_matches(
            label: &str,
            nlip_vector: &[f64],
            after_solve: &BTreeMap<usize, IpoptSpralInterfaceDump>,
            nlip: &GliderLinearDebugDump,
            intervals: usize,
            order: usize,
        ) -> Option<(usize, usize, f64)> {
            let windows = ipopt_spral_cumulative_solution_windows(after_solve, 4);
            let mut ranked = windows
                .iter()
                .enumerate()
                .map(|(window_index, (_, _, solution))| {
                    (
                        window_index,
                        journal_vector_max_abs_diff(nlip_vector, solution),
                    )
                })
                .collect::<Vec<_>>();
            ranked.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)));
            for (rank, (window_index, max_abs_diff)) in ranked.iter().take(4).enumerate() {
                let (start_call, terms, solution) = &windows[*window_index];
                println!(
                    "  {label}_best_ipopt_spral_cumulative_solution_match[{rank}] start_call={start_call} terms={terms} max_abs_diff={max_abs_diff:.17e} {} blocks[{}]",
                    journal_vector_diff_summary_text(nlip_vector, solution),
                    journal_augmented_block_diff_summary_text(
                        nlip_vector,
                        solution,
                        &[
                            nlip.x_dimension,
                            nlip.inequality_dimension,
                            nlip.equality_dimension,
                            nlip.inequality_dimension,
                        ],
                    ),
                );
                if rank == 0 {
                    println!(
                        "  {label}_upper_slack_channel start_call={start_call} terms={terms} {}",
                        upper_slack_prefinal_channel_diff_text(
                            nlip,
                            nlip_vector,
                            solution,
                            intervals,
                            order,
                        ),
                    );
                }
            }
            ranked.first().map(|(window_index, max_abs_diff)| {
                let (start_call, terms, _) = &windows[*window_index];
                (*start_call, *terms, *max_abs_diff)
            })
        }

        fn print_ipopt_spral_interface_dump_fingerprints_for_iteration(
            label: &str,
            ipopt_dump_dir: Option<&Path>,
            nlip_dump_dir: Option<&Path>,
            dump_iteration: usize,
            intervals: usize,
            order: usize,
        ) {
            let Some(ipopt_dump_dir) = ipopt_dump_dir else {
                return;
            };
            let Some(nlip_dump_dir) = nlip_dump_dir else {
                println!("ipopt spral dump comparison unavailable: missing NLIP dump dir");
                return;
            };
            let nlip_path = nlip_dump_dir.join(format!("nlip_kkt_iter_{dump_iteration:04}.txt"));
            if !nlip_path.exists() {
                println!(
                    "{label} dump comparison unavailable: missing {}",
                    nlip_path.display()
                );
                return;
            }
            let nlip = load_glider_linear_debug_dump(&nlip_path);
            let prefinal_rhs = nlip.rhs.iter().map(|value| -*value).collect::<Vec<_>>();
            let nlip_trace_rhs = nlip
                .linear_trace_rhs_prefinal
                .as_deref()
                .unwrap_or(prefinal_rhs.as_slice());
            let before_paths = sorted_ipopt_spral_dump_paths(ipopt_dump_dir, "before");
            let after_factor = sorted_ipopt_spral_dump_paths(ipopt_dump_dir, "after_factor")
                .into_iter()
                .map(|path| {
                    let dump = load_ipopt_spral_interface_dump(&path);
                    (dump.call_index, dump)
                })
                .collect::<BTreeMap<_, _>>();
            let after_solve = sorted_ipopt_spral_dump_paths(ipopt_dump_dir, "after_solve")
                .into_iter()
                .map(|path| {
                    let dump = load_ipopt_spral_interface_dump(&path);
                    (dump.call_index, dump)
                })
                .collect::<BTreeMap<_, _>>();
            let before = before_paths
                .iter()
                .map(|path| load_ipopt_spral_interface_dump(path))
                .collect::<Vec<_>>();
            println!(
                "{label} dump comparison dir={} nlip_iter={} before={} after_factor={} after_solve={}",
                ipopt_dump_dir.display(),
                dump_iteration,
                before.len(),
                after_factor.len(),
                after_solve.len(),
            );
            if before.is_empty() {
                return;
            }
            let mut best_spral_solution_match = None;
            let mut best_spral_refinement_rhs_match = None;
            let mut best_spral_refinement_solution_match = None;
            let mut best_spral_refinement_accumulated_match = None;
            let mut ranked_matrix = before
                .iter()
                .map(|dump| {
                    (
                        dump.call_index,
                        ipopt_spral_matrix_max_abs_diff(&nlip, dump),
                    )
                })
                .collect::<Vec<_>>();
            ranked_matrix.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)));
            for (rank, (call_index, max_abs_diff)) in ranked_matrix.iter().take(4).enumerate() {
                let dump = before
                    .iter()
                    .find(|dump| dump.call_index == *call_index)
                    .expect("ranked matrix call should exist");
                println!(
                    "  {label}_best_matrix_match[{rank}] call={call_index} max_abs_diff={max_abs_diff:.17e} [{}] {} {}",
                    journal_vector_fingerprint_text(journal_vector_fingerprint(&dump.values)),
                    ipopt_spral_structure_diff_summary_text(&nlip, dump),
                    ipopt_spral_factor_info_text(after_factor.get(call_index)),
                );
                println!(
                    "    values {} {}",
                    journal_vector_diff_summary_text(&nlip.values, &dump.values),
                    journal_vector_max_abs_diff_index(&nlip.values, &dump.values).map_or_else(
                        || "max_entry_index=none".to_string(),
                        |index| {
                            augmented_matrix_value_location_text(&nlip, index, intervals, order)
                        },
                    )
                );
            }

            let best_spral_rhs_match = print_ranked_ipopt_spral_rhs_matches(
                label,
                nlip_trace_rhs,
                &before,
                &nlip,
                intervals,
                order,
            );

            if let Some(nlip_solution) = nlip.linear_trace_solution_prefinal_unrefined.as_ref() {
                best_spral_solution_match = print_ranked_ipopt_spral_solution_matches(
                    label,
                    nlip_solution,
                    &after_solve,
                    &nlip,
                    intervals,
                    order,
                );
            }
            if let Some(refinement_rhs) = nlip.linear_trace_refinement_rhs.as_ref() {
                for (step, rhs) in refinement_rhs.iter().enumerate() {
                    let candidate = print_ranked_ipopt_spral_rhs_matches(
                        &format!("{label}_refinement_rhs[{step}]"),
                        rhs,
                        &before,
                        &nlip,
                        intervals,
                        order,
                    );
                    best_spral_refinement_rhs_match = best_spral_refinement_rhs_match.map_or(
                        candidate,
                        |current: (usize, f64)| match candidate {
                            Some(candidate) if candidate.1 < current.1 => Some(candidate),
                            _ => Some(current),
                        },
                    );
                }
            }
            if let Some(refinement_solution) = nlip.linear_trace_refinement_solution.as_ref() {
                for (step, solution) in refinement_solution.iter().enumerate() {
                    let candidate = print_ranked_ipopt_spral_solution_matches(
                        &format!("{label}_refinement_solution[{step}]"),
                        solution,
                        &after_solve,
                        &nlip,
                        intervals,
                        order,
                    );
                    best_spral_refinement_solution_match = best_spral_refinement_solution_match
                        .map_or(candidate, |current: (usize, f64)| match candidate {
                            Some(candidate) if candidate.1 < current.1 => Some(candidate),
                            _ => Some(current),
                        });
                }
            }
            if let Some(accumulated) = nlip.linear_trace_refinement_accumulated_solution.as_ref() {
                for (step, solution) in accumulated.iter().enumerate() {
                    let candidate = print_ranked_ipopt_spral_cumulative_solution_matches(
                        &format!("{label}_refinement_accumulated_solution[{step}]"),
                        solution,
                        &after_solve,
                        &nlip,
                        intervals,
                        order,
                    );
                    best_spral_refinement_accumulated_match =
                        best_spral_refinement_accumulated_match.map_or(
                            candidate,
                            |current: (usize, usize, f64)| match candidate {
                                Some(candidate) if candidate.2 < current.2 => Some(candidate),
                                _ => Some(current),
                            },
                        );
                }
            }
            println!(
                "  {label}_solve_boundary_summary iter={} {} {} {} {} {}",
                dump_iteration,
                solve_boundary_match_text("rhs_prefinal", best_spral_rhs_match),
                solve_boundary_match_text("sol_prefinal_unrefined", best_spral_solution_match),
                solve_boundary_match_text("refinement_rhs", best_spral_refinement_rhs_match),
                solve_boundary_match_text(
                    "refinement_solution",
                    best_spral_refinement_solution_match,
                ),
                solve_boundary_cumulative_match_text(
                    "refinement_accumulated",
                    best_spral_refinement_accumulated_match,
                ),
            );
        }

        fn print_ipopt_spral_interface_dump_fingerprints(
            ipopt_dump_dir: Option<&Path>,
            nlip_dump_dir: Option<&Path>,
            intervals: usize,
            order: usize,
        ) {
            print_ipopt_spral_interface_dump_fingerprints_for_iteration(
                "ipopt_spral",
                ipopt_dump_dir,
                nlip_dump_dir,
                glider_augmented_fingerprint_iteration(),
                intervals,
                order,
            );
        }

        fn sorted_nlip_restoration_dump_dirs(base_dir: &Path) -> Vec<std::path::PathBuf> {
            let mut paths = fs::read_dir(base_dir)
                .ok()
                .into_iter()
                .flat_map(|entries| entries.filter_map(Result::ok))
                .map(|entry| entry.path())
                .filter(|path| path.is_dir())
                .collect::<Vec<_>>();
            paths.sort();
            paths
        }

        fn print_ipopt_spral_restoration_dump_fingerprints(
            ipopt_dump_dir: Option<&Path>,
            intervals: usize,
            order: usize,
        ) {
            let Some(restoration_base_dir) =
                std::env::var_os("NLIP_RESTORATION_LINEAR_DEBUG_DUMP_DIR")
                    .map(std::path::PathBuf::from)
            else {
                return;
            };
            let dump_iteration = std::env::var("GLIDER_PARITY_NLIP_RESTORATION_AUGMENTED_ITER")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            let max_dirs = std::env::var("GLIDER_PARITY_NLIP_RESTORATION_MAX_DUMP_DIRS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(4);
            let dirs = sorted_nlip_restoration_dump_dirs(&restoration_base_dir);
            println!(
                "ipopt spral restoration dump comparison base_dir={} dirs={} nlip_iter={} max_dirs={}",
                restoration_base_dir.display(),
                dirs.len(),
                dump_iteration,
                max_dirs,
            );
            for dir in dirs.into_iter().take(max_dirs) {
                let label = format!(
                    "ipopt_spral_restoration_{}",
                    dir.file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("unknown")
                );
                print_ipopt_spral_interface_dump_fingerprints_for_iteration(
                    &label,
                    ipopt_dump_dir,
                    Some(&dir),
                    dump_iteration,
                    intervals,
                    order,
                );
                let inner_dir = dir.join("aug_resto_inner");
                if inner_dir.is_dir() {
                    print_ipopt_spral_interface_dump_fingerprints_for_iteration(
                        &format!("{label}_inner"),
                        ipopt_dump_dir,
                        Some(&inner_dir),
                        dump_iteration,
                        intervals,
                        order,
                    );
                }
            }
        }

        fn print_ipopt_spral_solve_boundary_ladder(
            ipopt_dump_dir: Option<&Path>,
            nlip_dump_dir: Option<&Path>,
        ) {
            if std::env::var_os("GLIDER_PARITY_PRINT_IPOPT_SPRAL_LADDER").is_none() {
                return;
            }
            let Some(ipopt_dump_dir) = ipopt_dump_dir else {
                println!("ipopt spral ladder unavailable: missing IPOPT dump dir");
                return;
            };
            let Some(nlip_dump_dir) = nlip_dump_dir else {
                println!("ipopt spral ladder unavailable: missing NLIP dump dir");
                return;
            };
            let max_iters = std::env::var("GLIDER_PARITY_IPOPT_SPRAL_LADDER_MAX_ITERS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(24);
            let before = sorted_ipopt_spral_dump_paths(ipopt_dump_dir, "before")
                .iter()
                .map(|path| load_ipopt_spral_interface_dump(path))
                .collect::<Vec<_>>();
            let after_factor = sorted_ipopt_spral_dump_paths(ipopt_dump_dir, "after_factor")
                .into_iter()
                .map(|path| {
                    let dump = load_ipopt_spral_interface_dump(&path);
                    (dump.call_index, dump)
                })
                .collect::<BTreeMap<_, _>>();
            let after_solve = sorted_ipopt_spral_dump_paths(ipopt_dump_dir, "after_solve")
                .into_iter()
                .map(|path| {
                    let dump = load_ipopt_spral_interface_dump(&path);
                    (dump.call_index, dump)
                })
                .collect::<BTreeMap<_, _>>();
            let nlip_paths = sorted_nlip_linear_debug_dump_paths(nlip_dump_dir);
            println!(
                "ipopt spral solve-boundary ladder ipopt_dir={} nlip_dir={} nlip_dumps={} before={} after_factor={} after_solve={} max_iters={}",
                ipopt_dump_dir.display(),
                nlip_dump_dir.display(),
                nlip_paths.len(),
                before.len(),
                after_factor.len(),
                after_solve.len(),
                max_iters,
            );
            if before.is_empty() || nlip_paths.is_empty() {
                return;
            }
            for path in nlip_paths.into_iter().take(max_iters) {
                let iteration = nlip_linear_debug_dump_iteration(&path).unwrap_or(usize::MAX);
                let nlip = load_glider_linear_debug_dump(&path);
                let prefinal_rhs = nlip.rhs.iter().map(|value| -*value).collect::<Vec<_>>();
                let nlip_trace_rhs = nlip
                    .linear_trace_rhs_prefinal
                    .as_deref()
                    .unwrap_or(prefinal_rhs.as_slice());
                let matrix_match = best_ipopt_spral_matrix_match(&nlip, &before);
                let rhs_match = best_ipopt_spral_rhs_match(nlip_trace_rhs, &before);
                let solution_match = nlip
                    .linear_trace_solution_prefinal_unrefined
                    .as_deref()
                    .and_then(|solution| best_ipopt_spral_solution_match(solution, &after_solve));
                let refinement_rhs_match =
                    nlip.linear_trace_refinement_rhs.as_ref().and_then(|steps| {
                        steps
                            .iter()
                            .filter_map(|rhs| best_ipopt_spral_rhs_match(rhs, &before))
                            .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)))
                    });
                let refinement_solution_match = nlip
                    .linear_trace_refinement_solution
                    .as_ref()
                    .and_then(|steps| {
                        steps
                            .iter()
                            .filter_map(|solution| {
                                best_ipopt_spral_solution_match(solution, &after_solve)
                            })
                            .min_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then(lhs.0.cmp(&rhs.0)))
                    });
                let refinement_accumulated_match = nlip
                    .linear_trace_refinement_accumulated_solution
                    .as_ref()
                    .and_then(|steps| {
                        steps
                            .iter()
                            .filter_map(|solution| {
                                best_ipopt_spral_cumulative_solution_match(
                                    solution,
                                    &after_solve,
                                    4,
                                )
                            })
                            .min_by(|lhs, rhs| lhs.2.total_cmp(&rhs.2).then(lhs.0.cmp(&rhs.0)))
                    });
                let factor_info = matrix_match
                    .and_then(|(call_index, _)| after_factor.get(&call_index))
                    .map_or_else(
                        || "factor_info=missing".to_string(),
                        |dump| ipopt_spral_factor_info_text(Some(dump)),
                    );
                println!(
                    "  ipopt_spral_ladder iter={} {} {} {} {} {} {} {}",
                    iteration,
                    solve_boundary_match_text("matrix", matrix_match),
                    solve_boundary_match_text("rhs_prefinal", rhs_match),
                    solve_boundary_match_text("sol_prefinal_unrefined", solution_match),
                    solve_boundary_match_text("refinement_rhs", refinement_rhs_match),
                    solve_boundary_match_text("refinement_solution", refinement_solution_match),
                    solve_boundary_cumulative_match_text(
                        "refinement_accumulated",
                        refinement_accumulated_match,
                    ),
                    factor_info,
                );
            }
        }

        #[derive(Clone, Copy, Debug)]
        enum DivergenceKind {
            Objective,
            Primal,
            Dual,
            Barrier,
            Regularization,
            AlphaPr,
            AlphaDu,
            StepTag,
            TrialCount,
        }

        fn nlip_primary_inertia_text(
            snapshot: &optimization::InteriorPointIterationSnapshot,
        ) -> String {
            snapshot
                .linear_debug
                .as_ref()
                .and_then(|report| {
                    report
                        .results
                        .iter()
                        .find(|result| result.solver == report.primary_solver)
                })
                .and_then(|result| result.inertia)
                .map_or_else(
                    || "--".to_string(),
                    |inertia| {
                        format!(
                            "+{}/-{}/0{}",
                            inertia.positive, inertia.negative, inertia.zero
                        )
                    },
                )
        }

        fn nlip_primary_detail_text(
            snapshot: &optimization::InteriorPointIterationSnapshot,
        ) -> String {
            snapshot
                .linear_debug
                .as_ref()
                .and_then(|report| {
                    report
                        .results
                        .iter()
                        .find(|result| result.solver == report.primary_solver)
                })
                .and_then(|result| result.detail.as_deref())
                .map_or_else(|| "--".to_string(), ToString::to_string)
        }

        fn nlip_primary_linear_stats_text(
            snapshot: &optimization::InteriorPointIterationSnapshot,
        ) -> String {
            snapshot
                .linear_debug
                .as_ref()
                .and_then(|report| {
                    report
                        .results
                        .iter()
                        .find(|result| result.solver == report.primary_solver)
                })
                .map(|result| {
                    format!(
                        "residual={} solution={} step={}",
                        result
                            .residual_inf
                            .map_or_else(|| "--".to_string(), |value| format!("{value:.3e}")),
                        result
                            .solution_inf
                            .map_or_else(|| "--".to_string(), |value| format!("{value:.3e}")),
                        result
                            .step_inf
                            .map_or_else(|| "--".to_string(), |value| format!("{value:.3e}")),
                    )
                })
                .unwrap_or_else(|| "--".to_string())
        }

        fn glider_decision_count(intervals: usize, order: usize) -> usize {
            const STATE_LEN: usize = 4;
            const CONTROL_LEN: usize = 1;
            (intervals + 1) * (STATE_LEN + CONTROL_LEN)
                + intervals * order * (STATE_LEN + 2 * CONTROL_LEN)
                + 1
        }

        fn variable_bound_view(params: &Params) -> VariableBoundView {
            const STATE_LEN: usize = 4;
            const CONTROL_LEN: usize = 1;
            const X_FIELD: usize = 0;
            const ALTITUDE_FIELD: usize = 1;
            const VX_FIELD: usize = 2;

            let intervals = params.transcription.intervals;
            let order = params.transcription.collocation_degree;
            let dimension = glider_decision_count(intervals, order);
            let mut lower = vec![None; dimension];
            let mut upper = vec![None; dimension];

            // Keep this in step with optimal_control::build_raw_bounds: affine
            // boundary and path rows are promoted to variable bounds before the
            // NLP reaches either NLIP or IPOPT.
            lower[X_FIELD] = Some(0.0);
            upper[X_FIELD] = Some(0.0);
            lower[ALTITUDE_FIELD] = Some(INITIAL_ALTITUDE_M);
            upper[ALTITUDE_FIELD] = Some(INITIAL_ALTITUDE_M);

            let x_mesh_len = (intervals + 1) * STATE_LEN;
            let u_mesh_len = (intervals + 1) * CONTROL_LEN;
            let root_x_start = x_mesh_len + u_mesh_len;
            let root_u_start = root_x_start + intervals * order * STATE_LEN;
            let root_dudt_start = root_u_start + intervals * order * CONTROL_LEN;
            for point in 0..intervals * order {
                let root_x_index = root_x_start + point * STATE_LEN;
                lower[root_x_index + ALTITUDE_FIELD] = Some(0.0);
                lower[root_x_index + VX_FIELD] = Some(1.0);

                let root_u_index = root_u_start + point * CONTROL_LEN;
                lower[root_u_index] = Some(CL_LOWER_BOUND / CL_SLOPE);
                upper[root_u_index] = Some(CL_UPPER_BOUND / CL_SLOPE);

                let root_dudt_index = root_dudt_start + point * CONTROL_LEN;
                lower[root_dudt_index] = Some(-deg_to_rad(params.max_alpha_rate_deg_s));
                upper[root_dudt_index] = Some(deg_to_rad(params.max_alpha_rate_deg_s));
            }

            let tf_index = dimension - 1;
            lower[tf_index] = Some(params.min_time_bound_s);
            upper[tf_index] = Some(params.max_time_bound_s);
            VariableBoundView { lower, upper }
        }

        fn glider_path_inequality_label(index: usize, intervals: usize, order: usize) -> String {
            let point = index / 6;
            let residual = index % 6;
            let interval = point / order;
            let root = point % order;
            if interval >= intervals {
                return format!("ineq[{index}]");
            }
            let name = match residual {
                0 => "lower(path.altitude)",
                1 => "lower(path.vx)",
                2 => "lower(path.cl)",
                3 => "upper(path.cl)",
                4 => "lower(path.alpha_rate)",
                5 => "upper(path.alpha_rate)",
                _ => unreachable!(),
            };
            format!("{name}[{interval}][{root}]")
        }

        fn glider_inequality_label(index: usize, intervals: usize, order: usize) -> String {
            const BOUNDARY_SPEED_SQ_ROWS: usize = 2;
            if index == 0 {
                return "lower(boundary.speed_sq0)".to_string();
            }
            if index == 1 {
                return "upper(boundary.speed_sq0)".to_string();
            }

            let path_row_count = intervals * order * 6;
            let path_index = index.saturating_sub(BOUNDARY_SPEED_SQ_ROWS);
            if path_index < path_row_count {
                return glider_path_inequality_label(path_index, intervals, order);
            }
            format!("ineq[{index}]")
        }

        fn glider_path_inequality_scaling_value(params: &Params, path_index: usize) -> f64 {
            match path_index % 6 {
                0 => GLIDER_ALTITUDE_SCALE_M,
                1 => params.launch_speed_mps,
                2 | 3 => GLIDER_CL_SCALE,
                4 | 5 => deg_to_rad(GLIDER_ALPHA_RATE_SCALE_DEG_S),
                _ => unreachable!(),
            }
        }

        fn glider_inequality_multiplier_output_factor(
            index: usize,
            params: &Params,
            intervals: usize,
            order: usize,
        ) -> Option<f64> {
            const BOUNDARY_SPEED_SQ_ROWS: usize = 2;
            if !params.scaling_enabled {
                return Some(1.0);
            }
            let scale = if index < BOUNDARY_SPEED_SQ_ROWS {
                params.launch_speed_mps.powi(2)
            } else {
                let path_index = index - BOUNDARY_SPEED_SQ_ROWS;
                if path_index >= intervals * order * 6 {
                    return None;
                }
                glider_path_inequality_scaling_value(params, path_index)
            };
            Some(GLIDER_OBJECTIVE_SCALE / scale)
        }

        fn glider_equality_label(index: usize, intervals: usize, order: usize) -> String {
            const STATE_LEN: usize = 4;
            const CONTROL_LEN: usize = 1;

            let collocation_x_len = intervals * order * STATE_LEN;
            if index < collocation_x_len {
                let point = index / STATE_LEN;
                let interval = point / order;
                let root = point % order;
                let field = state_field_name(index % STATE_LEN);
                return format!("collocation_x[{interval}][{root}].{field}");
            }

            let mut offset = index - collocation_x_len;
            let collocation_u_len = intervals * order * CONTROL_LEN;
            if offset < collocation_u_len {
                let point = offset / CONTROL_LEN;
                let interval = point / order;
                let root = point % order;
                let field = control_field_name(0);
                return format!("collocation_u[{interval}][{root}].{field}");
            }

            offset -= collocation_u_len;
            let continuity_x_len = intervals * STATE_LEN;
            if offset < continuity_x_len {
                let interval = offset / STATE_LEN;
                let field = state_field_name(offset % STATE_LEN);
                return format!("continuity_x[{interval}].{field}");
            }

            offset -= continuity_x_len;
            let continuity_u_len = intervals * CONTROL_LEN;
            if offset < continuity_u_len {
                let interval = offset / CONTROL_LEN;
                let field = control_field_name(0);
                return format!("continuity_u[{interval}].{field}");
            }

            format!("eq[{index}]")
        }

        fn glider_state_scaling_value(params: &Params, field_index: usize) -> f64 {
            match field_index {
                0 => GLIDER_X_SCALE_M,
                1 => GLIDER_ALTITUDE_SCALE_M,
                2 | 3 => params.launch_speed_mps,
                _ => 1.0,
            }
        }

        fn glider_control_scaling_value(field_index: usize) -> f64 {
            match field_index {
                0 => deg_to_rad(GLIDER_ALPHA_SCALE_DEG),
                _ => 1.0,
            }
        }

        fn glider_equality_scaling_value(
            index: usize,
            params: &Params,
            intervals: usize,
            order: usize,
        ) -> Option<f64> {
            const STATE_LEN: usize = 4;
            const CONTROL_LEN: usize = 1;

            if !params.scaling_enabled {
                return Some(1.0);
            }

            let collocation_x_len = intervals * order * STATE_LEN;
            if index < collocation_x_len {
                return Some(glider_state_scaling_value(params, index % STATE_LEN));
            }

            let mut offset = index - collocation_x_len;
            let collocation_u_len = intervals * order * CONTROL_LEN;
            if offset < collocation_u_len {
                return Some(glider_control_scaling_value(0));
            }

            offset -= collocation_u_len;
            let continuity_x_len = intervals * STATE_LEN;
            if offset < continuity_x_len {
                return Some(glider_state_scaling_value(params, offset % STATE_LEN));
            }

            offset -= continuity_x_len;
            let continuity_u_len = intervals * CONTROL_LEN;
            if offset < continuity_u_len {
                return Some(glider_control_scaling_value(0));
            }

            None
        }

        fn glider_equality_multiplier_output_factor(
            index: usize,
            params: &Params,
            intervals: usize,
            order: usize,
        ) -> Option<f64> {
            let scale = glider_equality_scaling_value(index, params, intervals, order)?;
            Some(if params.scaling_enabled {
                GLIDER_OBJECTIVE_SCALE / scale
            } else {
                1.0
            })
        }

        fn primal_limiter_label(
            limiter: &optimization::InteriorPointBoundaryLimiter,
            x: &[f64],
            slack: Option<&[f64]>,
            bounds: &VariableBoundView,
            intervals: usize,
            order: usize,
        ) -> String {
            match limiter.kind {
                optimization::InteriorPointBoundaryLimiterKind::Slack => {
                    return format!(
                        "s_upper({})",
                        glider_inequality_label(limiter.index, intervals, order)
                    );
                }
                optimization::InteriorPointBoundaryLimiterKind::VariableLowerBound => {
                    return format!(
                        "lower({})",
                        glider_decision_label(limiter.index, intervals, order)
                    );
                }
                optimization::InteriorPointBoundaryLimiterKind::VariableUpperBound => {
                    return format!(
                        "upper({})",
                        glider_decision_label(limiter.index, intervals, order)
                    );
                }
                optimization::InteriorPointBoundaryLimiterKind::Multiplier => {
                    return format!("multiplier[{}]", limiter.index);
                }
                optimization::InteriorPointBoundaryLimiterKind::Unknown => {}
            }
            if let Some(slack) = slack
                && let Some(&value) = slack.get(limiter.index)
            {
                let value = value.max(1.0e-16);
                if (value - limiter.value).abs() <= 1.0e-7 * value.abs().max(1.0) {
                    return format!(
                        "s_upper({})",
                        glider_inequality_label(limiter.index, intervals, order)
                    );
                }
            }
            if slack.is_some() && limiter.index < x.len() {
                let lower = bounds.lower.get(limiter.index).copied().flatten();
                let upper = bounds.upper.get(limiter.index).copied().flatten();
                if lower.is_some() && lower == upper {
                    return format!(
                        "fixed({})",
                        glider_decision_label(limiter.index, intervals, order)
                    );
                }
                if let Some(lower) = lower {
                    let slack = (x[limiter.index] - lower).max(1.0e-16);
                    if (slack - limiter.value).abs() <= 1.0e-7 * slack.abs().max(1.0) {
                        return format!(
                            "lower({})",
                            glider_decision_label(limiter.index, intervals, order)
                        );
                    }
                }
                if let Some(upper) = upper {
                    let slack = (upper - x[limiter.index]).max(1.0e-16);
                    if (slack - limiter.value).abs() <= 1.0e-7 * slack.abs().max(1.0) {
                        return format!(
                            "upper({})",
                            glider_decision_label(limiter.index, intervals, order)
                        );
                    }
                }
            }
            format!(
                "slack_or_multiplier({})",
                glider_inequality_label(limiter.index, intervals, order)
            )
        }

        fn limiter_text(
            limiter: Option<&optimization::InteriorPointBoundaryLimiter>,
            x: &[f64],
            slack: Option<&[f64]>,
            bounds: &VariableBoundView,
            intervals: usize,
            order: usize,
        ) -> String {
            limiter.map_or_else(
                || "--".to_string(),
                |limiter| {
                    let label = primal_limiter_label(limiter, x, slack, bounds, intervals, order);
                    format!(
                        "#{} {} val={:.3e} dir={:.3e} a={:.3e}",
                        limiter.index, label, limiter.value, limiter.direction, limiter.alpha
                    )
                },
            )
        }

        fn limiters_text(limiters: &[optimization::InteriorPointBoundaryLimiter]) -> String {
            if limiters.is_empty() {
                return "--".to_string();
            }
            limiters
                .iter()
                .map(|limiter| {
                    format!(
                        "#{} val={:.3e} dir={:.3e} a={:.6e}",
                        limiter.index, limiter.value, limiter.direction, limiter.alpha
                    )
                })
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn state_field_name(index: usize) -> &'static str {
            match index {
                0 => "x",
                1 => "altitude",
                2 => "vx",
                3 => "vy",
                _ => "?",
            }
        }

        fn control_field_name(index: usize) -> &'static str {
            match index {
                0 => "alpha",
                _ => "?",
            }
        }

        fn glider_decision_label(index: usize, intervals: usize, order: usize) -> String {
            const STATE_LEN: usize = 4;
            const CONTROL_LEN: usize = 1;

            let x_len = (intervals + 1) * STATE_LEN;
            if index < x_len {
                let node = index / STATE_LEN;
                let field = state_field_name(index % STATE_LEN);
                return if node == intervals {
                    format!("x[T].{field}")
                } else {
                    format!("x[{node}].{field}")
                };
            }

            let mut offset = index - x_len;
            let u_len = (intervals + 1) * CONTROL_LEN;
            if offset < u_len {
                let node = offset / CONTROL_LEN;
                let field = control_field_name(0);
                return if node == intervals {
                    format!("u[T].{field}")
                } else {
                    format!("u[{node}].{field}")
                };
            }

            offset -= u_len;
            let root_x_len = intervals * order * STATE_LEN;
            if offset < root_x_len {
                let point = offset / STATE_LEN;
                let interval = point / order;
                let root = point % order;
                let field = state_field_name(offset % STATE_LEN);
                return format!("root_x[{interval}][{root}].{field}");
            }

            offset -= root_x_len;
            let root_u_len = intervals * order * CONTROL_LEN;
            if offset < root_u_len {
                let point = offset / CONTROL_LEN;
                let interval = point / order;
                let root = point % order;
                let field = control_field_name(0);
                return format!("root_u[{interval}][{root}].{field}");
            }

            offset -= root_u_len;
            let root_dudt_len = intervals * order * CONTROL_LEN;
            if offset < root_dudt_len {
                let point = offset / CONTROL_LEN;
                let interval = point / order;
                let root = point % order;
                let field = control_field_name(0);
                return format!("root_dudt[{interval}][{root}].{field}");
            }

            offset -= root_dudt_len;
            if offset == 0 {
                "tf".to_string()
            } else {
                format!("unknown[{index}]")
            }
        }

        fn top_x_diffs(
            nlip: &TracePoint,
            ipopt: &TracePoint,
            intervals: usize,
            order: usize,
        ) -> String {
            let mut diffs = nlip
                .x
                .iter()
                .zip(ipopt.x.iter())
                .enumerate()
                .map(|(index, (nlip_value, ipopt_value))| {
                    (
                        index,
                        nlip_value,
                        ipopt_value,
                        (nlip_value - ipopt_value).abs(),
                    )
                })
                .collect::<Vec<_>>();
            diffs.sort_by(|lhs, rhs| rhs.3.total_cmp(&lhs.3));
            if diffs.is_empty() {
                return "--".to_string();
            }
            diffs
                .into_iter()
                .take(8)
                .map(|(index, nlip_value, ipopt_value, diff)| {
                    let label = glider_decision_label(index, intervals, order);
                    format!("#{index} {label} n={nlip_value:.6e} i={ipopt_value:.6e} d={diff:.3e}")
                })
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn top_step_delta_diffs(
            prev_nlip: &TracePoint,
            nlip: &TracePoint,
            prev_ipopt: &TracePoint,
            ipopt: &TracePoint,
            intervals: usize,
            order: usize,
        ) -> String {
            let mut diffs = nlip
                .x
                .iter()
                .zip(prev_nlip.x.iter())
                .zip(ipopt.x.iter().zip(prev_ipopt.x.iter()))
                .enumerate()
                .map(
                    |(index, ((nlip_value, prev_nlip_value), (ipopt_value, prev_ipopt_value)))| {
                        let nlip_delta = nlip_value - prev_nlip_value;
                        let ipopt_delta = ipopt_value - prev_ipopt_value;
                        (
                            index,
                            nlip_delta,
                            ipopt_delta,
                            (nlip_delta - ipopt_delta).abs(),
                        )
                    },
                )
                .collect::<Vec<_>>();
            diffs.sort_by(|lhs, rhs| rhs.3.total_cmp(&lhs.3));
            if diffs.is_empty() {
                return "--".to_string();
            }
            diffs
                .into_iter()
                .take(8)
                .map(|(index, nlip_delta, ipopt_delta, diff)| {
                    let label = glider_decision_label(index, intervals, order);
                    format!(
                        "#{index} {label} dn={nlip_delta:.6e} di={ipopt_delta:.6e} d={diff:.3e}"
                    )
                })
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn top_direction_estimate_diffs(
            prev_nlip: &TracePoint,
            nlip: &TracePoint,
            prev_ipopt: &TracePoint,
            ipopt: &TracePoint,
            intervals: usize,
            order: usize,
        ) -> String {
            let nlip_alpha = nlip.alpha_pr.abs().max(1.0e-16);
            let ipopt_alpha = ipopt.alpha_pr.abs().max(1.0e-16);
            let mut diffs = nlip
                .x
                .iter()
                .zip(prev_nlip.x.iter())
                .zip(ipopt.x.iter().zip(prev_ipopt.x.iter()))
                .enumerate()
                .map(
                    |(index, ((nlip_value, prev_nlip_value), (ipopt_value, prev_ipopt_value)))| {
                        let nlip_direction = (nlip_value - prev_nlip_value) / nlip_alpha;
                        let ipopt_direction = (ipopt_value - prev_ipopt_value) / ipopt_alpha;
                        (
                            index,
                            nlip_direction,
                            ipopt_direction,
                            (nlip_direction - ipopt_direction).abs(),
                        )
                    },
                )
                .collect::<Vec<_>>();
            diffs.sort_by(|lhs, rhs| rhs.3.total_cmp(&lhs.3));
            if diffs.is_empty() {
                return "--".to_string();
            }
            diffs
                .into_iter()
                .take(8)
                .map(|(index, nlip_direction, ipopt_direction, diff)| {
                    let label = glider_decision_label(index, intervals, order);
                    format!(
                        "#{index} {label} dn={nlip_direction:.6e} di={ipopt_direction:.6e} d={diff:.3e}"
                    )
                })
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn top_vector_direction_diffs(
            prev_nlip: &[f64],
            nlip: &[f64],
            prev_ipopt: &[f64],
            ipopt: &[f64],
            nlip_alpha: f64,
            ipopt_alpha: f64,
            label: impl Fn(usize) -> String,
        ) -> String {
            let nlip_alpha = nlip_alpha.abs().max(1.0e-16);
            let ipopt_alpha = ipopt_alpha.abs().max(1.0e-16);
            let mut diffs = nlip
                .iter()
                .zip(prev_nlip.iter())
                .zip(ipopt.iter().zip(prev_ipopt.iter()))
                .enumerate()
                .map(
                    |(index, ((nlip_value, prev_nlip_value), (ipopt_value, prev_ipopt_value)))| {
                        let nlip_direction = (nlip_value - prev_nlip_value) / nlip_alpha;
                        let ipopt_direction = (ipopt_value - prev_ipopt_value) / ipopt_alpha;
                        (
                            index,
                            nlip_direction,
                            ipopt_direction,
                            (nlip_direction - ipopt_direction).abs(),
                        )
                    },
                )
                .collect::<Vec<_>>();
            diffs.sort_by(|lhs, rhs| rhs.3.total_cmp(&lhs.3));
            if diffs.is_empty() {
                return "--".to_string();
            }
            diffs
                .into_iter()
                .take(6)
                .map(|(index, nlip_direction, ipopt_direction, diff)| {
                    format!(
                        "#{index} {} dn={nlip_direction:.6e} di={ipopt_direction:.6e} d={diff:.3e}",
                        label(index)
                    )
                })
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn max_abs(values: &[f64]) -> f64 {
            values
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()))
        }

        fn max_update_reconstruction_gap(
            previous: &[f64],
            current: &[f64],
            direction: &[f64],
            alpha: f64,
        ) -> f64 {
            previous
                .iter()
                .zip(current.iter())
                .zip(direction.iter())
                .fold(0.0_f64, |acc, ((previous, current), direction)| {
                    acc.max((current - previous - alpha * direction).abs())
                })
        }

        fn max_scaled_direction_gap(
            nlip_direction: &[f64],
            nlip_alpha: f64,
            ipopt_direction: &[f64],
            ipopt_alpha: f64,
        ) -> f64 {
            nlip_direction.iter().zip(ipopt_direction.iter()).fold(
                0.0_f64,
                |acc, (nlip_direction, ipopt_direction)| {
                    acc.max((nlip_alpha * nlip_direction - ipopt_alpha * ipopt_direction).abs())
                },
            )
        }

        fn step_application_metric(
            name: &str,
            prev_nlip: &[f64],
            nlip: &[f64],
            nlip_direction: &[f64],
            nlip_alpha: f64,
            prev_ipopt: &[f64],
            ipopt: &[f64],
            ipopt_direction: &[f64],
            ipopt_alpha: f64,
        ) -> String {
            let count = prev_nlip
                .len()
                .min(nlip.len())
                .min(nlip_direction.len())
                .min(prev_ipopt.len())
                .min(ipopt.len())
                .min(ipopt_direction.len());
            if count == 0 {
                return format!("{name}[--]");
            }
            let prev_nlip = &prev_nlip[..count];
            let nlip = &nlip[..count];
            let nlip_direction = &nlip_direction[..count];
            let prev_ipopt = &prev_ipopt[..count];
            let ipopt = &ipopt[..count];
            let ipopt_direction = &ipopt_direction[..count];
            let alpha_gap = (nlip_alpha - ipopt_alpha).abs();
            let nlip_update_gap =
                max_update_reconstruction_gap(prev_nlip, nlip, nlip_direction, nlip_alpha);
            let ipopt_update_gap =
                max_update_reconstruction_gap(prev_ipopt, ipopt, ipopt_direction, ipopt_alpha);
            let state_delta_gap = prev_nlip
                .iter()
                .zip(nlip.iter())
                .zip(prev_ipopt.iter().zip(ipopt.iter()))
                .fold(0.0_f64, |acc, ((prev_nlip, nlip), (prev_ipopt, ipopt))| {
                    acc.max(((nlip - prev_nlip) - (ipopt - prev_ipopt)).abs())
                });
            let direction_gap = max_vector_direction_diff(
                prev_nlip,
                nlip,
                prev_ipopt,
                ipopt,
                nlip_alpha,
                ipopt_alpha,
            );
            let stored_direction_gap = nlip_direction.iter().zip(ipopt_direction.iter()).fold(
                0.0_f64,
                |acc, (nlip_direction, ipopt_direction)| {
                    acc.max((nlip_direction - ipopt_direction).abs())
                },
            );
            let scaled_direction_gap =
                max_scaled_direction_gap(nlip_direction, nlip_alpha, ipopt_direction, ipopt_alpha);
            let max_direction = max_abs(nlip_direction).max(max_abs(ipopt_direction));
            let alpha_scaled_direction = alpha_gap * max_direction;
            format!(
                "{name}[alpha_gap={alpha_gap:.3e},max_dir={max_direction:.3e},alpha*max_dir={alpha_scaled_direction:.3e},delta_gap={state_delta_gap:.3e},fd_dir_gap={direction_gap:.3e},stored_dir_gap={stored_direction_gap:.3e},scaled_dir_gap={scaled_direction_gap:.3e},recon_n={nlip_update_gap:.3e},recon_i={ipopt_update_gap:.3e}]"
            )
        }

        fn step_application_summary(
            prev_nlip: &optimization::InteriorPointIterationSnapshot,
            nlip: &optimization::InteriorPointIterationSnapshot,
            prev_ipopt: &optimization::IpoptIterationSnapshot,
            ipopt: &optimization::IpoptIterationSnapshot,
            nlip_alpha_pr: f64,
            nlip_alpha_du: f64,
            ipopt_alpha_pr: f64,
            ipopt_alpha_du: f64,
        ) -> String {
            let Some(nlip_direction) = nlip.step_direction.as_ref() else {
                return "--".to_string();
            };
            // IpoptData::delta() is the accepted step still exposed to the
            // TNLP intermediate callback after AcceptTrialPoint; compare that
            // stored direction against the actual accepted-state update.
            [
                step_application_metric(
                    "x",
                    &prev_nlip.x,
                    &nlip.x,
                    &nlip_direction.x,
                    nlip_alpha_pr,
                    &prev_ipopt.x,
                    &ipopt.x,
                    &ipopt.direction_x,
                    ipopt_alpha_pr,
                ),
                step_application_metric(
                    "y_c",
                    prev_nlip.equality_multipliers.as_deref().unwrap_or(&[]),
                    nlip.equality_multipliers.as_deref().unwrap_or(&[]),
                    &nlip_direction.equality_multipliers,
                    nlip_alpha_pr,
                    &prev_ipopt.equality_multipliers,
                    &ipopt.equality_multipliers,
                    &ipopt.direction_equality_multipliers,
                    ipopt_alpha_pr,
                ),
                step_application_metric(
                    "y_d",
                    prev_nlip.inequality_multipliers.as_deref().unwrap_or(&[]),
                    nlip.inequality_multipliers.as_deref().unwrap_or(&[]),
                    &nlip_direction.inequality_multipliers,
                    nlip_alpha_pr,
                    &prev_ipopt.inequality_multipliers,
                    &ipopt.inequality_multipliers,
                    &ipopt.direction_inequality_multipliers,
                    ipopt_alpha_pr,
                ),
                step_application_metric(
                    "v_U",
                    prev_nlip.slack_multipliers.as_deref().unwrap_or(&[]),
                    nlip.slack_multipliers.as_deref().unwrap_or(&[]),
                    &nlip_direction.slack_multipliers,
                    nlip_alpha_du,
                    &prev_ipopt.slack_upper_bound_multipliers,
                    &ipopt.slack_upper_bound_multipliers,
                    &ipopt.direction_slack_upper_bound_multipliers,
                    ipopt_alpha_du,
                ),
            ]
            .join("; ")
        }

        fn top_upper_slack_multiplier_direction_diffs(
            prev_nlip: &optimization::InteriorPointIterationSnapshot,
            nlip: &optimization::InteriorPointIterationSnapshot,
            prev_ipopt: &optimization::IpoptIterationSnapshot,
            ipopt: &optimization::IpoptIterationSnapshot,
            nlip_alpha_pr: f64,
            nlip_alpha_du: f64,
            _nlip_mu: f64,
            ipopt_alpha_pr: f64,
            ipopt_alpha_du: f64,
            _ipopt_mu: f64,
            intervals: usize,
            order: usize,
        ) -> String {
            let prev_nlip_slack = prev_nlip.slack_primal.as_deref().unwrap_or(&[]);
            let nlip_slack = nlip.slack_primal.as_deref().unwrap_or(&[]);
            let prev_nlip_v = prev_nlip.slack_multipliers.as_deref().unwrap_or(&[]);
            let nlip_v = nlip.slack_multipliers.as_deref().unwrap_or(&[]);
            let nlip_alpha_pr = nlip_alpha_pr.abs().max(1.0e-16);
            let nlip_alpha_du = nlip_alpha_du.abs().max(1.0e-16);
            let ipopt_alpha_pr = ipopt_alpha_pr.abs().max(1.0e-16);
            let ipopt_alpha_du = ipopt_alpha_du.abs().max(1.0e-16);
            // Accepted-state finite differences include the post-line-search
            // IPOPT/NLIP bound multiplier safeguard, so this is a direction
            // comparison probe rather than a raw PDFullSpaceSolver residual.

            let count = prev_nlip_slack
                .len()
                .min(nlip_slack.len())
                .min(prev_nlip_v.len())
                .min(nlip_v.len())
                .min(prev_ipopt.kkt_slack_distance.len())
                .min(ipopt.kkt_slack_distance.len())
                .min(prev_ipopt.internal_slack.len())
                .min(ipopt.internal_slack.len())
                .min(prev_ipopt.slack_upper_bound_multipliers.len())
                .min(ipopt.slack_upper_bound_multipliers.len());
            let mut diffs = (0..count)
                .map(|index| {
                    let nlip_internal_ds =
                        (nlip_slack[index] - prev_nlip_slack[index]) / nlip_alpha_pr;
                    let nlip_actual_dv = (nlip_v[index] - prev_nlip_v[index]) / nlip_alpha_du;

                    let ipopt_internal_ds = (ipopt.internal_slack[index]
                        - prev_ipopt.internal_slack[index])
                        / ipopt_alpha_pr;
                    let ipopt_actual_dv = (ipopt.slack_upper_bound_multipliers[index]
                        - prev_ipopt.slack_upper_bound_multipliers[index])
                        / ipopt_alpha_du;

                    (
                        index,
                        nlip_actual_dv,
                        ipopt_actual_dv,
                        nlip_internal_ds,
                        ipopt_internal_ds,
                        (nlip_actual_dv - ipopt_actual_dv).abs(),
                        (nlip_internal_ds - ipopt_internal_ds).abs(),
                    )
                })
                .collect::<Vec<_>>();
            diffs.sort_by(|lhs, rhs| rhs.5.total_cmp(&lhs.5));
            if diffs.is_empty() {
                return "--".to_string();
            }
            diffs
                .into_iter()
                .take(4)
                .map(
                    |(
                        index,
                        nlip_actual_dv,
                        ipopt_actual_dv,
                        nlip_internal_ds,
                        ipopt_internal_ds,
                        actual_gap,
                        slack_step_gap,
                    )| {
                        format!(
                            "#{} {} dv[n={nlip_actual_dv:.6e},i={ipopt_actual_dv:.6e},d={actual_gap:.3e}] ds_internal[n={nlip_internal_ds:.6e},i={ipopt_internal_ds:.6e},d={slack_step_gap:.3e}]",
                            index,
                            glider_inequality_label(index, intervals, order)
                        )
                    },
                )
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn top_upper_slack_kkt_rhs_diffs(
            nlip: &optimization::InteriorPointIterationSnapshot,
            ipopt: &optimization::IpoptIterationSnapshot,
            params: &Params,
            intervals: usize,
            order: usize,
        ) -> String {
            let nlip_slack = nlip.slack_primal.as_deref().unwrap_or(&[]);
            let nlip_y = nlip.inequality_multipliers.as_deref().unwrap_or(&[]);
            let nlip_v = nlip.slack_multipliers.as_deref().unwrap_or(&[]);
            let nlip_rhs_s = nlip.kkt_slack_stationarity.as_deref().unwrap_or(&[]);
            let nlip_rhs_v = nlip.kkt_slack_complementarity.as_deref().unwrap_or(&[]);
            let nlip_sigma = nlip.kkt_slack_sigma.as_deref().unwrap_or(&[]);
            let count = nlip_slack
                .len()
                .min(nlip_y.len())
                .min(nlip_v.len())
                .min(nlip_rhs_s.len())
                .min(nlip_rhs_v.len())
                .min(nlip_sigma.len())
                .min(ipopt.internal_slack.len())
                .min(ipopt.inequality_multipliers.len())
                .min(ipopt.slack_upper_bound_multipliers.len())
                .min(ipopt.kkt_slack_stationarity.len())
                .min(ipopt.kkt_slack_complementarity.len())
                .min(ipopt.kkt_slack_sigma.len())
                .min(ipopt.kkt_slack_distance.len());

            let mut diffs = (0..count)
                .map(|index| {
                    let ipopt_distance = ipopt.kkt_slack_distance[index];
                    let ipopt_upper = ipopt.internal_slack[index] + ipopt_distance;
                    let nlip_distance = ipopt_upper - nlip_slack[index];
                    // Mirrors IpPDSearchDirCalc::ComputeSearchDirection's
                    // rhs.s/rhs.v_U construction and
                    // IpPDFullSpaceSolver::SolveOnce's upper-slack
                    // Pd_U.AddMSinvZ(-1.0, slack_s_U, rhs.v_U, augRhs_s).
                    let nlip_comp_over_slack = nlip_rhs_v[index] / nlip_distance;
                    let ipopt_comp_over_slack =
                        ipopt.kkt_slack_complementarity[index] / ipopt_distance;
                    let nlip_solve_once_aug_rhs = nlip_rhs_s[index] - nlip_comp_over_slack;
                    let ipopt_solve_once_aug_rhs =
                        ipopt.kkt_slack_stationarity[index] - ipopt_comp_over_slack;
                    let distance_gap = (nlip_distance - ipopt_distance).abs();
                    let y_gap = (nlip_y[index] - ipopt.inequality_multipliers[index]).abs();
                    let v_gap = (nlip_v[index] - ipopt.slack_upper_bound_multipliers[index]).abs();
                    let rhs_s_gap = (nlip_rhs_s[index] - ipopt.kkt_slack_stationarity[index]).abs();
                    let rhs_v_gap =
                        (nlip_rhs_v[index] - ipopt.kkt_slack_complementarity[index]).abs();
                    let comp_over_slack_gap = (nlip_comp_over_slack - ipopt_comp_over_slack).abs();
                    let aug_rhs_gap = (nlip_solve_once_aug_rhs - ipopt_solve_once_aug_rhs).abs();
                    let score = aug_rhs_gap
                        .max(rhs_s_gap)
                        .max(rhs_v_gap)
                        .max(comp_over_slack_gap)
                        .max(distance_gap);
                    (
                        index,
                        score,
                        distance_gap,
                        y_gap,
                        v_gap,
                        rhs_s_gap,
                        rhs_v_gap,
                        comp_over_slack_gap,
                        aug_rhs_gap,
                        nlip_distance,
                        ipopt_distance,
                        nlip_y[index],
                        ipopt.inequality_multipliers[index],
                        nlip_v[index],
                        ipopt.slack_upper_bound_multipliers[index],
                        nlip_rhs_s[index],
                        ipopt.kkt_slack_stationarity[index],
                        nlip_rhs_v[index],
                        ipopt.kkt_slack_complementarity[index],
                        nlip_comp_over_slack,
                        ipopt_comp_over_slack,
                        nlip_solve_once_aug_rhs,
                        ipopt_solve_once_aug_rhs,
                        nlip_sigma[index],
                        ipopt.kkt_slack_sigma[index],
                    )
                })
                .collect::<Vec<_>>();
            diffs.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
            if diffs.is_empty() {
                return "--".to_string();
            }
            diffs
                .into_iter()
                .take(4)
                .map(
                    |(
                        index,
                        score,
                        distance_gap,
                        y_gap,
                        v_gap,
                        rhs_s_gap,
                        rhs_v_gap,
                        comp_over_slack_gap,
                        aug_rhs_gap,
                        nlip_distance,
                        ipopt_distance,
                        nlip_y,
                        ipopt_y,
                        nlip_v,
                        ipopt_v,
                        nlip_rhs_s,
                        ipopt_rhs_s,
                        nlip_rhs_v,
                        ipopt_rhs_v,
                        nlip_comp_over_slack,
                        ipopt_comp_over_slack,
                        nlip_aug_rhs,
                        ipopt_aug_rhs,
                        nlip_sigma,
                        ipopt_sigma,
                    )| {
                        let output_factor = glider_inequality_multiplier_output_factor(
                            index, params, intervals, order,
                        )
                        .unwrap_or(1.0);
                        let internal_y_gap = y_gap / output_factor.abs();
                        let internal_v_gap = v_gap / output_factor.abs();
                        format!(
                            "#{} {} score={score:.3e} gaps[dist={distance_gap:.3e},y={y_gap:.3e},vU={v_gap:.3e},rhs_s={rhs_s_gap:.3e},rhs_v={rhs_v_gap:.3e},rhs_v/s={comp_over_slack_gap:.3e},aug={aug_rhs_gap:.3e}] output_factor={output_factor:.3e} implied_internal_gaps[y={internal_y_gap:.3e},vU={internal_v_gap:.3e}] dist[n={nlip_distance:.12e},i={ipopt_distance:.12e}] y[n={nlip_y:.12e},i={ipopt_y:.12e}] vU[n={nlip_v:.12e},i={ipopt_v:.12e}] rhs_s[n={nlip_rhs_s:.12e},i={ipopt_rhs_s:.12e}] rhs_v[n={nlip_rhs_v:.12e},i={ipopt_rhs_v:.12e}] rhs_v/s[n={nlip_comp_over_slack:.12e},i={ipopt_comp_over_slack:.12e}] aug_rhs[n={nlip_aug_rhs:.12e},i={ipopt_aug_rhs:.12e}] sigma[n={nlip_sigma:.12e},i={ipopt_sigma:.12e}]",
                            index,
                            glider_inequality_label(index, intervals, order)
                        )
                    },
                )
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn top_upper_slack_next_solve_rhs_diffs(
            previous_nlip: &optimization::InteriorPointIterationSnapshot,
            previous_ipopt: &optimization::IpoptIterationSnapshot,
            next_nlip_mu: f64,
            next_ipopt_mu: f64,
            params: &Params,
            kappa_d: f64,
            intervals: usize,
            order: usize,
        ) -> String {
            let nlip_slack = previous_nlip.slack_primal.as_deref().unwrap_or(&[]);
            let nlip_y = previous_nlip
                .inequality_multipliers
                .as_deref()
                .unwrap_or(&[]);
            let nlip_v = previous_nlip.slack_multipliers.as_deref().unwrap_or(&[]);
            let count = nlip_slack
                .len()
                .min(nlip_y.len())
                .min(nlip_v.len())
                .min(previous_ipopt.internal_slack.len())
                .min(previous_ipopt.kkt_slack_distance.len())
                .min(previous_ipopt.inequality_multipliers.len())
                .min(previous_ipopt.slack_upper_bound_multipliers.len());
            let nlip_damping = next_nlip_mu * kappa_d;
            let ipopt_damping = next_ipopt_mu * kappa_d;

            let mut diffs = (0..count)
                .map(|index| {
                    let ipopt_upper = previous_ipopt.internal_slack[index]
                        + previous_ipopt.kkt_slack_distance[index];
                    let nlip_distance = ipopt_upper - nlip_slack[index];
                    let ipopt_distance = previous_ipopt.kkt_slack_distance[index];
                    let nlip_rhs_s = nlip_v[index] - nlip_y[index] - nlip_damping;
                    let ipopt_rhs_s = previous_ipopt.slack_upper_bound_multipliers[index]
                        - previous_ipopt.inequality_multipliers[index]
                        - ipopt_damping;
                    let nlip_rhs_v = nlip_distance * nlip_v[index] - next_nlip_mu;
                    let ipopt_rhs_v = ipopt_distance
                        * previous_ipopt.slack_upper_bound_multipliers[index]
                        - next_ipopt_mu;
                    let nlip_rhs_v_over_s = nlip_rhs_v / nlip_distance;
                    let ipopt_rhs_v_over_s = ipopt_rhs_v / ipopt_distance;
                    // Mirrors IpPDSearchDirCalc::ComputeSearchDirection rhs.s/rhs.v_U
                    // plus IpPDFullSpaceSolver::SolveOnce Pd_U.AddMSinvZ(-1.0, ...):
                    // prefinal augmented upper-slack RHS is rhs_s - rhs_v_U / slack_s_U.
                    let nlip_aug_rhs = nlip_rhs_s - nlip_rhs_v_over_s;
                    let ipopt_aug_rhs = ipopt_rhs_s - ipopt_rhs_v_over_s;
                    let distance_gap = (nlip_distance - ipopt_distance).abs();
                    let y_gap =
                        (nlip_y[index] - previous_ipopt.inequality_multipliers[index]).abs();
                    let v_gap =
                        (nlip_v[index] - previous_ipopt.slack_upper_bound_multipliers[index]).abs();
                    let rhs_s_gap = (nlip_rhs_s - ipopt_rhs_s).abs();
                    let rhs_v_gap = (nlip_rhs_v - ipopt_rhs_v).abs();
                    let rhs_v_over_s_gap = (nlip_rhs_v_over_s - ipopt_rhs_v_over_s).abs();
                    let aug_rhs_gap = (nlip_aug_rhs - ipopt_aug_rhs).abs();
                    let score = aug_rhs_gap
                        .max(rhs_s_gap)
                        .max(rhs_v_gap)
                        .max(rhs_v_over_s_gap)
                        .max(distance_gap);
                    (
                        index,
                        score,
                        distance_gap,
                        y_gap,
                        v_gap,
                        rhs_s_gap,
                        rhs_v_gap,
                        rhs_v_over_s_gap,
                        aug_rhs_gap,
                        nlip_distance,
                        ipopt_distance,
                        nlip_y[index],
                        previous_ipopt.inequality_multipliers[index],
                        nlip_v[index],
                        previous_ipopt.slack_upper_bound_multipliers[index],
                        nlip_rhs_s,
                        ipopt_rhs_s,
                        nlip_rhs_v,
                        ipopt_rhs_v,
                        nlip_rhs_v_over_s,
                        ipopt_rhs_v_over_s,
                        nlip_aug_rhs,
                        ipopt_aug_rhs,
                    )
                })
                .collect::<Vec<_>>();
            diffs.sort_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1));
            if diffs.is_empty() {
                return "--".to_string();
            }
            diffs
                .into_iter()
                .take(4)
                .map(
                    |(
                        index,
                        score,
                        distance_gap,
                        y_gap,
                        v_gap,
                        rhs_s_gap,
                        rhs_v_gap,
                        rhs_v_over_s_gap,
                        aug_rhs_gap,
                        nlip_distance,
                        ipopt_distance,
                        nlip_y,
                        ipopt_y,
                        nlip_v,
                        ipopt_v,
                        nlip_rhs_s,
                        ipopt_rhs_s,
                        nlip_rhs_v,
                        ipopt_rhs_v,
                        nlip_rhs_v_over_s,
                        ipopt_rhs_v_over_s,
                        nlip_aug_rhs,
                        ipopt_aug_rhs,
                    )| {
                        let output_factor = glider_inequality_multiplier_output_factor(
                            index, params, intervals, order,
                        )
                        .unwrap_or(1.0);
                        let internal_y_gap = y_gap / output_factor.abs();
                        let internal_v_gap = v_gap / output_factor.abs();
                        format!(
                            "#{} {} score={score:.3e} gaps[dist={distance_gap:.3e},y={y_gap:.3e},vU={v_gap:.3e},rhs_s={rhs_s_gap:.3e},rhs_v={rhs_v_gap:.3e},rhs_v/s={rhs_v_over_s_gap:.3e},aug={aug_rhs_gap:.3e}] output_factor={output_factor:.3e} implied_internal_gaps[y={internal_y_gap:.3e},vU={internal_v_gap:.3e}] dist[n={nlip_distance:.12e},i={ipopt_distance:.12e}] y[n={nlip_y:.12e},i={ipopt_y:.12e}] vU[n={nlip_v:.12e},i={ipopt_v:.12e}] rhs_s[n={nlip_rhs_s:.12e},i={ipopt_rhs_s:.12e}] rhs_v[n={nlip_rhs_v:.12e},i={ipopt_rhs_v:.12e}] rhs_v/s[n={nlip_rhs_v_over_s:.12e},i={ipopt_rhs_v_over_s:.12e}] aug_rhs[n={nlip_aug_rhs:.12e},i={ipopt_aug_rhs:.12e}]",
                            index,
                            glider_inequality_label(index, intervals, order)
                        )
                    },
                )
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn max_direction_estimate_diff(
            prev_nlip: &TracePoint,
            nlip: &TracePoint,
            prev_ipopt: &TracePoint,
            ipopt: &TracePoint,
        ) -> f64 {
            let nlip_alpha = nlip.alpha_pr.abs().max(1.0e-16);
            let ipopt_alpha = ipopt.alpha_pr.abs().max(1.0e-16);
            nlip.x
                .iter()
                .zip(prev_nlip.x.iter())
                .zip(ipopt.x.iter().zip(prev_ipopt.x.iter()))
                .fold(
                    0.0_f64,
                    |acc, ((nlip_value, prev_nlip_value), (ipopt_value, prev_ipopt_value))| {
                        let nlip_direction = (nlip_value - prev_nlip_value) / nlip_alpha;
                        let ipopt_direction = (ipopt_value - prev_ipopt_value) / ipopt_alpha;
                        acc.max((nlip_direction - ipopt_direction).abs())
                    },
                )
        }

        fn max_vector_direction_diff(
            prev_nlip: &[f64],
            nlip: &[f64],
            prev_ipopt: &[f64],
            ipopt: &[f64],
            nlip_alpha: f64,
            ipopt_alpha: f64,
        ) -> f64 {
            let nlip_alpha = nlip_alpha.abs().max(1.0e-16);
            let ipopt_alpha = ipopt_alpha.abs().max(1.0e-16);
            nlip.iter()
                .zip(prev_nlip.iter())
                .zip(ipopt.iter().zip(prev_ipopt.iter()))
                .fold(
                    0.0_f64,
                    |acc, ((nlip_value, prev_nlip_value), (ipopt_value, prev_ipopt_value))| {
                        let nlip_direction = (nlip_value - prev_nlip_value) / nlip_alpha;
                        let ipopt_direction = (ipopt_value - prev_ipopt_value) / ipopt_alpha;
                        acc.max((nlip_direction - ipopt_direction).abs())
                    },
                )
        }

        fn direction_gap_ladder(nlip_trace: &[TracePoint], ipopt_trace: &[TracePoint]) -> String {
            [1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1]
                .iter()
                .map(|&threshold| {
                    let marker = (1..nlip_trace.len().min(ipopt_trace.len())).find_map(|index| {
                        if is_restoration_bridge_trace_index(nlip_trace, ipopt_trace, index) {
                            return None;
                        }
                        let gap = max_direction_estimate_diff(
                            &nlip_trace[index - 1],
                            &nlip_trace[index],
                            &ipopt_trace[index - 1],
                            &ipopt_trace[index],
                        );
                        (gap > threshold).then_some((index, gap))
                    });
                    match marker {
                        Some((index, gap)) => format!(
                            ">{threshold:.0e}:index={index},nlip_iter={},ipopt_iter={},gap={gap:.3e}",
                            nlip_trace[index].iteration, ipopt_trace[index].iteration
                        ),
                        None => format!(">{threshold:.0e}:none"),
                    }
                })
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn alpha_gap_ladder(nlip_trace: &[TracePoint], ipopt_trace: &[TracePoint]) -> String {
            fn metric_gap(
                metric: &str,
                thresholds: &[f64],
                nlip_trace: &[TracePoint],
                ipopt_trace: &[TracePoint],
                gap: impl Fn(&TracePoint, &TracePoint) -> f64,
            ) -> String {
                let compared = nlip_trace.len().min(ipopt_trace.len());
                let threshold_text = thresholds
                    .iter()
                    .map(|&threshold| {
                        let marker = (0..compared).find_map(|index| {
                            let nlip = &nlip_trace[index];
                            let ipopt = &ipopt_trace[index];
                            if is_restoration_bridge_trace_index(nlip_trace, ipopt_trace, index) {
                                return None;
                            }
                            let gap = gap(nlip, ipopt);
                            (gap.is_finite() && gap > threshold).then_some((index, gap))
                        });
                        match marker {
                            Some((index, gap)) => format!(
                                ">{threshold:.0e}:index={index},nlip_iter={},ipopt_iter={},gap={gap:.3e}",
                                nlip_trace[index].iteration, ipopt_trace[index].iteration
                            ),
                            None => format!(">{threshold:.0e}:none"),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{metric}[{threshold_text}]")
            }

            let thresholds = [1.0e-16, 1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4];
            [
                metric_gap(
                    "alpha_pr",
                    &thresholds,
                    nlip_trace,
                    ipopt_trace,
                    |nlip, ipopt| (nlip.alpha_pr - ipopt.alpha_pr).abs(),
                ),
                metric_gap(
                    "alpha_du",
                    &thresholds,
                    nlip_trace,
                    ipopt_trace,
                    |nlip, ipopt| (nlip.alpha_du - ipopt.alpha_du).abs(),
                ),
                metric_gap(
                    "alpha_y",
                    &thresholds,
                    nlip_trace,
                    ipopt_trace,
                    |nlip, ipopt| (nlip.alpha_y - ipopt.alpha_y).abs(),
                ),
            ]
            .join("; ")
        }

        fn accepted_direction_gap_ladder(
            accepted_nlip_solver_snapshots: &[optimization::InteriorPointIterationSnapshot],
            accepted_ipopt_solver_snapshots: &[optimization::IpoptIterationSnapshot],
            nlip_trace: &[TracePoint],
            ipopt_trace: &[TracePoint],
        ) -> String {
            fn metric_gap(
                metric: &str,
                thresholds: &[f64],
                compared: usize,
                gap: impl Fn(usize) -> f64,
                nlip_trace: &[TracePoint],
                ipopt_trace: &[TracePoint],
            ) -> String {
                let threshold_text = thresholds
                    .iter()
                    .map(|&threshold| {
                        let marker = (1..compared).find_map(|index| {
                            if is_restoration_bridge_trace_index(nlip_trace, ipopt_trace, index) {
                                return None;
                            }
                            let gap = gap(index);
                            (gap > threshold).then_some((index, gap))
                        });
                        match marker {
                            Some((index, gap)) => format!(
                                ">{threshold:.0e}:index={index},nlip_iter={},ipopt_iter={},gap={gap:.3e}",
                                nlip_trace[index].iteration, ipopt_trace[index].iteration
                            ),
                            None => format!(">{threshold:.0e}:none"),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{metric}[{threshold_text}]")
            }

            let compared = accepted_nlip_solver_snapshots
                .len()
                .min(accepted_ipopt_solver_snapshots.len())
                .min(nlip_trace.len())
                .min(ipopt_trace.len());
            let thresholds = [1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0e-1, 1.0, 1.0e1];

            [
                metric_gap(
                    "x",
                    &thresholds,
                    compared,
                    |index| {
                        max_vector_direction_diff(
                            &accepted_nlip_solver_snapshots[index - 1].x,
                            &accepted_nlip_solver_snapshots[index].x,
                            &accepted_ipopt_solver_snapshots[index - 1].x,
                            &accepted_ipopt_solver_snapshots[index].x,
                            nlip_trace[index].alpha_pr,
                            ipopt_trace[index].alpha_pr,
                        )
                    },
                    nlip_trace,
                    ipopt_trace,
                ),
                metric_gap(
                    "y_c",
                    &thresholds,
                    compared,
                    |index| {
                        max_vector_direction_diff(
                            accepted_nlip_solver_snapshots[index - 1]
                                .equality_multipliers
                                .as_deref()
                                .unwrap_or(&[]),
                            accepted_nlip_solver_snapshots[index]
                                .equality_multipliers
                                .as_deref()
                                .unwrap_or(&[]),
                            &accepted_ipopt_solver_snapshots[index - 1].equality_multipliers,
                            &accepted_ipopt_solver_snapshots[index].equality_multipliers,
                            nlip_trace[index].alpha_y,
                            ipopt_trace[index].alpha_y,
                        )
                    },
                    nlip_trace,
                    ipopt_trace,
                ),
                metric_gap(
                    "y_d",
                    &thresholds,
                    compared,
                    |index| {
                        max_vector_direction_diff(
                            accepted_nlip_solver_snapshots[index - 1]
                                .inequality_multipliers
                                .as_deref()
                                .unwrap_or(&[]),
                            accepted_nlip_solver_snapshots[index]
                                .inequality_multipliers
                                .as_deref()
                                .unwrap_or(&[]),
                            &accepted_ipopt_solver_snapshots[index - 1].inequality_multipliers,
                            &accepted_ipopt_solver_snapshots[index].inequality_multipliers,
                            nlip_trace[index].alpha_y,
                            ipopt_trace[index].alpha_y,
                        )
                    },
                    nlip_trace,
                    ipopt_trace,
                ),
                metric_gap(
                    "v_U",
                    &thresholds,
                    compared,
                    |index| {
                        max_vector_direction_diff(
                            accepted_nlip_solver_snapshots[index - 1]
                                .slack_multipliers
                                .as_deref()
                                .unwrap_or(&[]),
                            accepted_nlip_solver_snapshots[index]
                                .slack_multipliers
                                .as_deref()
                                .unwrap_or(&[]),
                            &accepted_ipopt_solver_snapshots[index - 1]
                                .slack_upper_bound_multipliers,
                            &accepted_ipopt_solver_snapshots[index].slack_upper_bound_multipliers,
                            nlip_trace[index].alpha_du,
                            ipopt_trace[index].alpha_du,
                        )
                    },
                    nlip_trace,
                    ipopt_trace,
                ),
            ]
            .join("; ")
        }

        fn expanded_compact_lower_bound_multipliers(
            compact: Option<&[f64]>,
            bounds: &VariableBoundView,
        ) -> Vec<f64> {
            let mut expanded = vec![0.0; bounds.lower.len()];
            let Some(compact) = compact else {
                return expanded;
            };
            let mut compact_index = 0;
            for (index, lower) in bounds.lower.iter().enumerate() {
                let upper = bounds.upper.get(index).copied().flatten();
                if lower.is_some() && *lower != upper {
                    if let Some(value) = compact.get(compact_index) {
                        expanded[index] = *value;
                    }
                    compact_index += 1;
                }
            }
            expanded
        }

        fn expanded_compact_upper_bound_multipliers(
            compact: Option<&[f64]>,
            bounds: &VariableBoundView,
        ) -> Vec<f64> {
            let mut expanded = vec![0.0; bounds.upper.len()];
            let Some(compact) = compact else {
                return expanded;
            };
            let mut compact_index = 0;
            for (index, upper) in bounds.upper.iter().enumerate() {
                let lower = bounds.lower.get(index).copied().flatten();
                if upper.is_some() && *upper != lower {
                    if let Some(value) = compact.get(compact_index) {
                        expanded[index] = *value;
                    }
                    compact_index += 1;
                }
            }
            expanded
        }

        fn ipopt_algorithmic_bound_multipliers(
            multipliers: &[f64],
            bounds: &VariableBoundView,
        ) -> Vec<f64> {
            let mut filtered = multipliers.to_vec();
            // Ipopt's TNLPAdapter default fixed_variable_treatment=make_parameter
            // removes fixed variables from the algorithmic NLP, then
            // ResortBoundMultipliers computes output-only fixed-variable z_L/z_U.
            for ((value, lower), upper) in filtered
                .iter_mut()
                .zip(bounds.lower.iter())
                .zip(bounds.upper.iter())
            {
                if lower.is_some() && lower == upper {
                    *value = 0.0;
                }
            }
            filtered
        }

        fn vector_inf_diff(lhs: &[f64], rhs: &[f64], rhs_sign: f64) -> f64 {
            lhs.iter().zip(rhs.iter()).fold(0.0_f64, |acc, (lhs, rhs)| {
                acc.max((lhs - rhs_sign * rhs).abs())
            })
        }

        fn vector_pair_difference(lhs: &[f64], rhs: &[f64]) -> Vec<f64> {
            lhs.iter()
                .zip(rhs.iter())
                .map(|(lhs, rhs)| lhs - rhs)
                .collect()
        }

        fn accepted_state_gap_ladder(
            accepted_nlip_solver_snapshots: &[optimization::InteriorPointIterationSnapshot],
            accepted_ipopt_solver_snapshots: &[optimization::IpoptIterationSnapshot],
        ) -> String {
            fn metric_gap(
                metric: &str,
                thresholds: &[f64],
                accepted_nlip_solver_snapshots: &[optimization::InteriorPointIterationSnapshot],
                accepted_ipopt_solver_snapshots: &[optimization::IpoptIterationSnapshot],
                gap: impl Fn(
                    &optimization::InteriorPointIterationSnapshot,
                    &optimization::IpoptIterationSnapshot,
                ) -> f64,
            ) -> String {
                let compared = accepted_nlip_solver_snapshots
                    .len()
                    .min(accepted_ipopt_solver_snapshots.len());
                let threshold_text = thresholds
                    .iter()
                    .map(|&threshold| {
                        let marker = (0..compared).find_map(|index| {
                            let nlip = &accepted_nlip_solver_snapshots[index];
                            let ipopt = &accepted_ipopt_solver_snapshots[index];
                            let gap = gap(nlip, ipopt);
                            (gap > threshold).then_some((index, gap, nlip.iteration, ipopt.iteration))
                        });
                        match marker {
                            Some((index, gap, nlip_iter, ipopt_iter)) => format!(
                                ">{threshold:.0e}:index={index},nlip_iter={nlip_iter},ipopt_iter={ipopt_iter},gap={gap:.3e}"
                            ),
                            None => format!(">{threshold:.0e}:none"),
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{metric}[{threshold_text}]")
            }

            let thresholds = [1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4];
            [
                metric_gap(
                    "x",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| vector_inf_diff(&nlip.x, &ipopt.x, 1.0),
                ),
                metric_gap(
                    "eq_y",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        vector_inf_diff(
                            nlip.equality_multipliers.as_deref().unwrap_or(&[]),
                            &ipopt.equality_multipliers,
                            1.0,
                        )
                    },
                ),
                metric_gap(
                    "ineq_y",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        vector_inf_diff(
                            nlip.inequality_multipliers.as_deref().unwrap_or(&[]),
                            &ipopt.inequality_multipliers,
                            1.0,
                        )
                    },
                ),
                metric_gap(
                    "slack_stat",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        vector_inf_diff(
                            nlip.kkt_slack_stationarity.as_deref().unwrap_or(&[]),
                            &ipopt.kkt_slack_stationarity,
                            1.0,
                        )
                    },
                ),
                metric_gap(
                    "x_stat",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        vector_inf_diff(
                            nlip.kkt_x_stationarity.as_deref().unwrap_or(&[]),
                            &ipopt.kkt_x_stationarity,
                            1.0,
                        )
                    },
                ),
                metric_gap(
                    "barrier_err",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        (nlip.barrier_subproblem_error.unwrap_or(nlip.overall_inf)
                            - ipopt.curr_barrier_error)
                            .abs()
                    },
                ),
                metric_gap(
                    "barrier_primal",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        let nlip_primal = nlip.barrier_primal_inf.unwrap_or_else(|| {
                            nlip.eq_inf.unwrap_or(0.0).max(nlip.ineq_inf.unwrap_or(0.0))
                        });
                        (nlip_primal - ipopt.curr_primal_infeasibility).abs()
                    },
                ),
                metric_gap(
                    "barrier_dual",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        (nlip.barrier_dual_inf.unwrap_or(nlip.dual_inf)
                            - ipopt.curr_dual_infeasibility)
                            .abs()
                    },
                ),
                metric_gap(
                    "barrier_comp",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        (nlip.barrier_complementarity_inf.unwrap_or(0.0)
                            - ipopt.curr_complementarity)
                            .abs()
                    },
                ),
                metric_gap(
                    "grad_f",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        vector_inf_diff(
                            nlip.curr_grad_f.as_deref().unwrap_or(&[]),
                            &ipopt.curr_grad_f,
                            1.0,
                        )
                    },
                ),
                metric_gap(
                    "jac_cT_y_c",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        vector_inf_diff(
                            nlip.curr_jac_c_t_y_c.as_deref().unwrap_or(&[]),
                            &ipopt.curr_jac_c_t_y_c,
                            1.0,
                        )
                    },
                ),
                metric_gap(
                    "jac_dT_y_d",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        vector_inf_diff(
                            nlip.curr_jac_d_t_y_d.as_deref().unwrap_or(&[]),
                            &ipopt.curr_jac_d_t_y_d,
                            1.0,
                        )
                    },
                ),
                metric_gap(
                    "grad_lag_x",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        vector_inf_diff(
                            nlip.curr_grad_lag_x.as_deref().unwrap_or(&[]),
                            &ipopt.curr_grad_lag_x,
                            1.0,
                        )
                    },
                ),
                metric_gap(
                    "grad_lag_s",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        vector_inf_diff(
                            nlip.curr_grad_lag_s.as_deref().unwrap_or(&[]),
                            &ipopt.curr_grad_lag_s,
                            1.0,
                        )
                    },
                ),
                metric_gap(
                    "x_damping",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        let nlip_damping = vector_pair_difference(
                            nlip.kkt_x_stationarity.as_deref().unwrap_or(&[]),
                            nlip.curr_grad_lag_x.as_deref().unwrap_or(&[]),
                        );
                        let ipopt_damping = vector_pair_difference(
                            &ipopt.kkt_x_stationarity,
                            &ipopt.curr_grad_lag_x,
                        );
                        vector_inf_diff(&nlip_damping, &ipopt_damping, 1.0)
                    },
                ),
                metric_gap(
                    "slack_damping",
                    &thresholds,
                    accepted_nlip_solver_snapshots,
                    accepted_ipopt_solver_snapshots,
                    |nlip, ipopt| {
                        let nlip_damping = vector_pair_difference(
                            nlip.kkt_slack_stationarity.as_deref().unwrap_or(&[]),
                            nlip.curr_grad_lag_s.as_deref().unwrap_or(&[]),
                        );
                        let ipopt_damping = vector_pair_difference(
                            &ipopt.kkt_slack_stationarity,
                            &ipopt.curr_grad_lag_s,
                        );
                        vector_inf_diff(&nlip_damping, &ipopt_damping, 1.0)
                    },
                ),
            ]
            .join("; ")
        }

        fn top_vector_diffs<F>(lhs: &[f64], rhs: &[f64], rhs_sign: f64, mut label: F) -> String
        where
            F: FnMut(usize) -> String,
        {
            let mut diffs = lhs
                .iter()
                .zip(rhs.iter())
                .enumerate()
                .map(|(index, (lhs, rhs))| {
                    let signed_rhs = rhs_sign * *rhs;
                    (index, *lhs, signed_rhs, (*lhs - signed_rhs).abs())
                })
                .collect::<Vec<_>>();
            diffs.sort_by(|lhs, rhs| rhs.3.total_cmp(&lhs.3));
            if diffs.is_empty() {
                return "--".to_string();
            }
            diffs
                .into_iter()
                .take(6)
                .map(|(index, lhs, rhs, diff)| {
                    format!(
                        "#{} {} n={lhs:.6e} i={rhs:.6e} d={diff:.3e}",
                        index,
                        label(index)
                    )
                })
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn top_equality_multiplier_output_scale_diffs(
            nlip_direction: &[f64],
            ipopt_direction: &[f64],
            params: &Params,
            intervals: usize,
            order: usize,
        ) -> String {
            let mut diffs = nlip_direction
                .iter()
                .zip(ipopt_direction.iter())
                .enumerate()
                .map(|(index, (nlip, ipopt))| {
                    let factor =
                        glider_equality_multiplier_output_factor(index, params, intervals, order)
                            .unwrap_or(1.0);
                    let external_gap = (*nlip - *ipopt).abs();
                    let internal_gap = if factor == 0.0 {
                        f64::NAN
                    } else {
                        external_gap / factor.abs()
                    };
                    (index, *nlip, *ipopt, factor, external_gap, internal_gap)
                })
                .collect::<Vec<_>>();
            diffs.sort_by(|lhs, rhs| rhs.4.total_cmp(&lhs.4));
            if diffs.is_empty() {
                return "--".to_string();
            }
            diffs
                .into_iter()
                .take(6)
                .map(
                    |(index, nlip, ipopt, factor, external_gap, internal_gap)| {
                        format!(
                            "#{} {} n={nlip:.6e} i={ipopt:.6e} external_d={external_gap:.3e} output_factor={factor:.3e} implied_internal_d={internal_gap:.3e}",
                            index,
                            glider_equality_label(index, intervals, order)
                        )
                    },
                )
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn ipopt_fraction_to_boundary_limiters(
            previous: &optimization::IpoptIterationSnapshot,
            current: &optimization::IpoptIterationSnapshot,
            bounds: &VariableBoundView,
            alpha_pr: f64,
            intervals: usize,
            order: usize,
        ) -> String {
            let alpha = alpha_pr.abs().max(1.0e-16);
            let tau = 0.99_f64.max(1.0 - current.barrier_parameter.max(0.0));
            let mut limiters = Vec::new();

            for (index, (&previous_value, &current_value)) in
                previous.x.iter().zip(current.x.iter()).enumerate()
            {
                let direction = (current_value - previous_value) / alpha;
                if direction < 0.0
                    && let Some(Some(lower)) = bounds.lower.get(index)
                {
                    let candidate = tau * (previous_value - lower) / -direction;
                    if candidate.is_finite() {
                        limiters.push((
                            candidate,
                            format!(
                                "x lower {} value={previous_value:.3e} dir={direction:.3e}",
                                glider_decision_label(index, intervals, order)
                            ),
                        ));
                    }
                }
                if direction > 0.0
                    && let Some(Some(upper)) = bounds.upper.get(index)
                {
                    let candidate = tau * (upper - previous_value) / direction;
                    if candidate.is_finite() {
                        limiters.push((
                            candidate,
                            format!(
                                "x upper {} value={previous_value:.3e} dir={direction:.3e}",
                                glider_decision_label(index, intervals, order)
                            ),
                        ));
                    }
                }
            }

            for (index, ((&previous_value, &current_value), &previous_distance)) in previous
                .internal_slack
                .iter()
                .zip(current.internal_slack.iter())
                .zip(previous.kkt_slack_distance.iter())
                .enumerate()
            {
                let direction = (current_value - previous_value) / alpha;
                // The OCP inequality adapter presents each normalized path row as upper-only.
                if direction > 0.0 {
                    let candidate = tau * previous_distance / direction;
                    if candidate.is_finite() {
                        limiters.push((
                            candidate,
                            format!(
                                "s upper {} value={previous_value:.3e} dir={direction:.3e}",
                                glider_inequality_label(index, intervals, order)
                            ),
                        ));
                    }
                }
            }

            limiters.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
            if limiters.is_empty() {
                return "--".to_string();
            }
            limiters
                .into_iter()
                .take(6)
                .map(|(alpha, label)| format!("a={alpha:.3e} {label}"))
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn direct_fraction_candidate(tau: f64, value: f64, direction: f64) -> f64 {
            // Mirrors IpIpoptCalculatedQuantities::CalcFracToBound through
            // DenseVector::FracToBoundImpl: candidates are evaluated as
            // `-tau / direction * value` for negative slack directions.
            -tau / direction * value
        }

        fn nlip_direct_primal_fraction_to_boundary_limiters(
            previous: &optimization::InteriorPointIterationSnapshot,
            current: &optimization::InteriorPointIterationSnapshot,
            previous_ipopt: &optimization::IpoptIterationSnapshot,
            bounds: &VariableBoundView,
            tau: f64,
            intervals: usize,
            order: usize,
        ) -> String {
            let Some(direction) = current.step_direction.as_ref() else {
                return "--".to_string();
            };
            let slack = previous.slack_primal.as_deref().unwrap_or(&[]);
            let mut limiters = Vec::new();

            for (index, maybe_lower) in bounds.lower.iter().enumerate() {
                let Some(lower) = *maybe_lower else {
                    continue;
                };
                let Some((&value, &delta)) = previous.x.get(index).zip(direction.x.get(index))
                else {
                    continue;
                };
                let slack_distance = value - lower;
                if delta < 0.0 && slack_distance.is_finite() {
                    let candidate = direct_fraction_candidate(tau, slack_distance, delta);
                    if candidate.is_finite() {
                        limiters.push((
                            candidate,
                            format!(
                                "x lower {} value={slack_distance:.17e} dir={delta:.17e}",
                                glider_decision_label(index, intervals, order)
                            ),
                        ));
                    }
                }
            }
            for (index, maybe_upper) in bounds.upper.iter().enumerate() {
                let Some(upper) = *maybe_upper else {
                    continue;
                };
                let Some((&value, &delta_x)) = previous.x.get(index).zip(direction.x.get(index))
                else {
                    continue;
                };
                let slack_distance = upper - value;
                let slack_direction = -delta_x;
                if slack_direction < 0.0 && slack_distance.is_finite() {
                    let candidate = direct_fraction_candidate(tau, slack_distance, slack_direction);
                    if candidate.is_finite() {
                        limiters.push((
                            candidate,
                            format!(
                                "x upper {} value={slack_distance:.17e} dir={slack_direction:.17e}",
                                glider_decision_label(index, intervals, order)
                            ),
                        ));
                    }
                }
            }

            for (index, ((&slack_value, &delta_s), (&ipopt_internal, &ipopt_distance))) in slack
                .iter()
                .zip(direction.slack.iter())
                .zip(
                    previous_ipopt
                        .internal_slack
                        .iter()
                        .zip(previous_ipopt.kkt_slack_distance.iter()),
                )
                .enumerate()
            {
                // NLIP snapshots do not currently expose the relaxed upper slack
                // bounds. Reconstruct the same comparison distance from IPOPT's
                // previous internal slack state so this probe isolates the
                // FracToBound arithmetic and direction values.
                let upper = ipopt_internal + ipopt_distance;
                let slack_distance = upper - slack_value;
                let slack_direction = -delta_s;
                if slack_direction < 0.0 && slack_distance.is_finite() {
                    let candidate = direct_fraction_candidate(tau, slack_distance, slack_direction);
                    if candidate.is_finite() {
                        limiters.push((
                            candidate,
                            format!(
                                "s upper {} value={slack_distance:.17e} dir={slack_direction:.17e}",
                                glider_inequality_label(index, intervals, order)
                            ),
                        ));
                    }
                }
            }

            limiters.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
            if limiters.is_empty() {
                return "--".to_string();
            }
            limiters
                .into_iter()
                .take(6)
                .map(|(alpha, label)| format!("a={alpha:.17e} {label}"))
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn ipopt_direct_primal_fraction_to_boundary_limiters(
            previous: &optimization::IpoptIterationSnapshot,
            current: &optimization::IpoptIterationSnapshot,
            bounds: &VariableBoundView,
            tau: f64,
            intervals: usize,
            order: usize,
        ) -> String {
            let mut limiters = Vec::new();

            for (index, maybe_lower) in bounds.lower.iter().enumerate() {
                let Some(lower) = *maybe_lower else {
                    continue;
                };
                let Some((&value, &delta)) =
                    previous.x.get(index).zip(current.direction_x.get(index))
                else {
                    continue;
                };
                let slack_distance = value - lower;
                if delta < 0.0 && slack_distance.is_finite() {
                    let candidate = direct_fraction_candidate(tau, slack_distance, delta);
                    if candidate.is_finite() {
                        limiters.push((
                            candidate,
                            format!(
                                "x lower {} value={slack_distance:.17e} dir={delta:.17e}",
                                glider_decision_label(index, intervals, order)
                            ),
                        ));
                    }
                }
            }
            for (index, maybe_upper) in bounds.upper.iter().enumerate() {
                let Some(upper) = *maybe_upper else {
                    continue;
                };
                let Some((&value, &delta_x)) =
                    previous.x.get(index).zip(current.direction_x.get(index))
                else {
                    continue;
                };
                let slack_distance = upper - value;
                let slack_direction = -delta_x;
                if slack_direction < 0.0 && slack_distance.is_finite() {
                    let candidate = direct_fraction_candidate(tau, slack_distance, slack_direction);
                    if candidate.is_finite() {
                        limiters.push((
                            candidate,
                            format!(
                                "x upper {} value={slack_distance:.17e} dir={slack_direction:.17e}",
                                glider_decision_label(index, intervals, order)
                            ),
                        ));
                    }
                }
            }

            for (index, (&distance, &delta_s)) in previous
                .kkt_slack_distance
                .iter()
                .zip(current.direction_slack.iter())
                .enumerate()
            {
                let slack_direction = -delta_s;
                if slack_direction < 0.0 && distance.is_finite() {
                    let candidate = direct_fraction_candidate(tau, distance, slack_direction);
                    if candidate.is_finite() {
                        limiters.push((
                            candidate,
                            format!(
                                "s upper {} value={distance:.17e} dir={slack_direction:.17e}",
                                glider_inequality_label(index, intervals, order)
                            ),
                        ));
                    }
                }
            }

            limiters.sort_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0));
            if limiters.is_empty() {
                return "--".to_string();
            }
            limiters
                .into_iter()
                .take(6)
                .map(|(alpha, label)| format!("a={alpha:.17e} {label}"))
                .collect::<Vec<_>>()
                .join("; ")
        }

        fn print_internal_ipopt_probe(
            index: usize,
            nlip_snapshot: &optimization::InteriorPointIterationSnapshot,
            ipopt_snapshot: &optimization::IpoptIterationSnapshot,
            bounds: &VariableBoundView,
            params: &Params,
            intervals: usize,
            order: usize,
        ) {
            let nlip_slack_primal = nlip_snapshot.slack_primal.as_deref().unwrap_or(&[]);
            let ipopt_slack_upper_bound = ipopt_snapshot
                .internal_slack
                .iter()
                .zip(ipopt_snapshot.kkt_slack_distance.iter())
                .map(|(&value, &distance)| value + distance)
                .collect::<Vec<_>>();
            let nlip_slack_distance = nlip_slack_primal
                .iter()
                .zip(ipopt_slack_upper_bound.iter())
                .map(|(&value, &upper)| upper - value)
                .collect::<Vec<_>>();
            let nlip_eq = nlip_snapshot.equality_multipliers.as_deref().unwrap_or(&[]);
            let nlip_ineq = nlip_snapshot
                .inequality_multipliers
                .as_deref()
                .unwrap_or(&[]);
            let nlip_slack_dual = nlip_snapshot.slack_multipliers.as_deref().unwrap_or(&[]);
            let nlip_kkt_ineq = nlip_snapshot
                .kkt_inequality_residual
                .as_deref()
                .unwrap_or(&[]);
            let nlip_kkt_x_stationarity =
                nlip_snapshot.kkt_x_stationarity.as_deref().unwrap_or(&[]);
            let nlip_kkt_slack_stationarity = nlip_snapshot
                .kkt_slack_stationarity
                .as_deref()
                .unwrap_or(&[]);
            let nlip_kkt_slack_complementarity = nlip_snapshot
                .kkt_slack_complementarity
                .as_deref()
                .unwrap_or(&[]);
            let nlip_kkt_slack_sigma = nlip_snapshot.kkt_slack_sigma.as_deref().unwrap_or(&[]);
            let nlip_curr_grad_f = nlip_snapshot.curr_grad_f.as_deref().unwrap_or(&[]);
            let nlip_curr_jac_c_t_y_c = nlip_snapshot.curr_jac_c_t_y_c.as_deref().unwrap_or(&[]);
            let nlip_curr_jac_d_t_y_d = nlip_snapshot.curr_jac_d_t_y_d.as_deref().unwrap_or(&[]);
            let nlip_curr_grad_lag_x = nlip_snapshot.curr_grad_lag_x.as_deref().unwrap_or(&[]);
            let nlip_curr_grad_lag_s = nlip_snapshot.curr_grad_lag_s.as_deref().unwrap_or(&[]);
            let nlip_x_damping =
                vector_pair_difference(nlip_kkt_x_stationarity, nlip_curr_grad_lag_x);
            let ipopt_x_damping = vector_pair_difference(
                &ipopt_snapshot.kkt_x_stationarity,
                &ipopt_snapshot.curr_grad_lag_x,
            );
            let nlip_slack_damping =
                vector_pair_difference(nlip_kkt_slack_stationarity, nlip_curr_grad_lag_s);
            let ipopt_slack_damping = vector_pair_difference(
                &ipopt_snapshot.kkt_slack_stationarity,
                &ipopt_snapshot.curr_grad_lag_s,
            );
            let nlip_lower = expanded_compact_lower_bound_multipliers(
                nlip_snapshot.lower_bound_multipliers.as_deref(),
                bounds,
            );
            let nlip_upper = expanded_compact_upper_bound_multipliers(
                nlip_snapshot.upper_bound_multipliers.as_deref(),
                bounds,
            );
            let ipopt_lower = ipopt_algorithmic_bound_multipliers(
                &ipopt_snapshot.lower_bound_multipliers,
                bounds,
            );
            let ipopt_upper = ipopt_algorithmic_bound_multipliers(
                &ipopt_snapshot.upper_bound_multipliers,
                bounds,
            );
            let nlip_primal_inf = nlip_snapshot.barrier_primal_inf.unwrap_or_else(|| {
                nlip_snapshot
                    .eq_inf
                    .unwrap_or(0.0)
                    .max(nlip_snapshot.ineq_inf.unwrap_or(0.0))
            });
            let nlip_barrier_error = nlip_snapshot
                .barrier_subproblem_error
                .unwrap_or(nlip_snapshot.overall_inf);
            let nlip_barrier_dual = nlip_snapshot
                .barrier_dual_inf
                .unwrap_or(nlip_snapshot.dual_inf);
            let nlip_complementarity = nlip_snapshot.barrier_complementarity_inf.unwrap_or(0.0);

            println!(
                "internal_probe[{index}] ipopt_iter={} x_diff={:.3e} slack_same={:.3e} slack_neg={:.3e} slack_dist={:.3e} eq_y_diff={:.3e} ineq_y_same={:.3e} ineq_y_neg={:.3e} lower_z_diff={:.3e} upper_z_diff={:.3e} slack_vl_same={:.3e} slack_vu_same={:.3e} slack_vu_neg={:.3e} kkt_ineq={:.3e} kkt_x_stat={:.3e} kkt_slack_stat={:.3e} kkt_slack_comp={:.3e} kkt_slack_sigma={:.3e} barrier_err={:.3e} barrier_primal={:.3e} barrier_dual={:.3e} barrier_comp={:.3e}",
                ipopt_snapshot.iteration,
                vector_inf_diff(&nlip_snapshot.x, &ipopt_snapshot.x, 1.0),
                vector_inf_diff(nlip_slack_primal, &ipopt_snapshot.internal_slack, 1.0),
                vector_inf_diff(nlip_slack_primal, &ipopt_snapshot.internal_slack, -1.0),
                vector_inf_diff(
                    &nlip_slack_distance,
                    &ipopt_snapshot.kkt_slack_distance,
                    1.0
                ),
                vector_inf_diff(nlip_eq, &ipopt_snapshot.equality_multipliers, 1.0),
                vector_inf_diff(nlip_ineq, &ipopt_snapshot.inequality_multipliers, 1.0),
                vector_inf_diff(nlip_ineq, &ipopt_snapshot.inequality_multipliers, -1.0),
                vector_inf_diff(&nlip_lower, &ipopt_lower, 1.0),
                vector_inf_diff(&nlip_upper, &ipopt_upper, 1.0),
                vector_inf_diff(
                    nlip_slack_dual,
                    &ipopt_snapshot.slack_lower_bound_multipliers,
                    1.0,
                ),
                vector_inf_diff(
                    nlip_slack_dual,
                    &ipopt_snapshot.slack_upper_bound_multipliers,
                    1.0,
                ),
                vector_inf_diff(
                    nlip_slack_dual,
                    &ipopt_snapshot.slack_upper_bound_multipliers,
                    -1.0,
                ),
                vector_inf_diff(nlip_kkt_ineq, &ipopt_snapshot.kkt_inequality_residual, 1.0),
                vector_inf_diff(
                    nlip_kkt_x_stationarity,
                    &ipopt_snapshot.kkt_x_stationarity,
                    1.0,
                ),
                vector_inf_diff(
                    nlip_kkt_slack_stationarity,
                    &ipopt_snapshot.kkt_slack_stationarity,
                    1.0,
                ),
                vector_inf_diff(
                    nlip_kkt_slack_complementarity,
                    &ipopt_snapshot.kkt_slack_complementarity,
                    1.0,
                ),
                vector_inf_diff(nlip_kkt_slack_sigma, &ipopt_snapshot.kkt_slack_sigma, 1.0),
                (nlip_barrier_error - ipopt_snapshot.curr_barrier_error).abs(),
                (nlip_primal_inf - ipopt_snapshot.curr_primal_infeasibility).abs(),
                (nlip_barrier_dual - ipopt_snapshot.curr_dual_infeasibility).abs(),
                (nlip_complementarity - ipopt_snapshot.curr_complementarity).abs(),
            );
            println!(
                "          component diffs grad_f={:.3e} jac_cT_y_c={:.3e} jac_dT_y_d={:.3e} grad_lag_x={:.3e} grad_lag_s={:.3e} x_damping={:.3e} slack_damping={:.3e}",
                vector_inf_diff(nlip_curr_grad_f, &ipopt_snapshot.curr_grad_f, 1.0),
                vector_inf_diff(nlip_curr_jac_c_t_y_c, &ipopt_snapshot.curr_jac_c_t_y_c, 1.0,),
                vector_inf_diff(nlip_curr_jac_d_t_y_d, &ipopt_snapshot.curr_jac_d_t_y_d, 1.0,),
                vector_inf_diff(nlip_curr_grad_lag_x, &ipopt_snapshot.curr_grad_lag_x, 1.0,),
                vector_inf_diff(nlip_curr_grad_lag_s, &ipopt_snapshot.curr_grad_lag_s, 1.0,),
                vector_inf_diff(&nlip_x_damping, &ipopt_x_damping, 1.0),
                vector_inf_diff(&nlip_slack_damping, &ipopt_slack_damping, 1.0),
            );
            println!(
                "          top internal slack diffs (same sign): {}",
                top_vector_diffs(
                    nlip_slack_primal,
                    &ipopt_snapshot.internal_slack,
                    1.0,
                    |idx| glider_inequality_label(idx, intervals, order),
                )
            );
            println!(
                "          top objective gradient diffs: {}",
                top_vector_diffs(nlip_curr_grad_f, &ipopt_snapshot.curr_grad_f, 1.0, |idx| {
                    glider_decision_label(idx, intervals, order)
                })
            );
            println!(
                "          top jac_cT*y_c diffs: {}",
                top_vector_diffs(
                    nlip_curr_jac_c_t_y_c,
                    &ipopt_snapshot.curr_jac_c_t_y_c,
                    1.0,
                    |idx| glider_decision_label(idx, intervals, order),
                )
            );
            println!(
                "          top jac_dT*y_d diffs: {}",
                top_vector_diffs(
                    nlip_curr_jac_d_t_y_d,
                    &ipopt_snapshot.curr_jac_d_t_y_d,
                    1.0,
                    |idx| glider_decision_label(idx, intervals, order),
                )
            );
            println!(
                "          top curr grad_lag_x diffs: {}",
                top_vector_diffs(
                    nlip_curr_grad_lag_x,
                    &ipopt_snapshot.curr_grad_lag_x,
                    1.0,
                    |idx| glider_decision_label(idx, intervals, order),
                )
            );
            println!(
                "          top curr grad_lag_s diffs: {}",
                top_vector_diffs(
                    nlip_curr_grad_lag_s,
                    &ipopt_snapshot.curr_grad_lag_s,
                    1.0,
                    |idx| glider_inequality_label(idx, intervals, order),
                )
            );
            println!(
                "          top x damping diffs: {}",
                top_vector_diffs(&nlip_x_damping, &ipopt_x_damping, 1.0, |idx| {
                    glider_decision_label(idx, intervals, order)
                })
            );
            println!(
                "          top slack damping diffs: {}",
                top_vector_diffs(&nlip_slack_damping, &ipopt_slack_damping, 1.0, |idx| {
                    glider_inequality_label(idx, intervals, order)
                },)
            );
            println!(
                "          top upper-slack distance diffs: {}",
                top_vector_diffs(
                    &nlip_slack_distance,
                    &ipopt_snapshot.kkt_slack_distance,
                    1.0,
                    |idx| glider_inequality_label(idx, intervals, order),
                )
            );
            println!(
                "          top y_d diffs: {}",
                top_vector_diffs(
                    nlip_ineq,
                    &ipopt_snapshot.inequality_multipliers,
                    1.0,
                    |idx| glider_inequality_label(idx, intervals, order),
                )
            );
            println!(
                "          top v_U diffs: {}",
                top_vector_diffs(
                    nlip_slack_dual,
                    &ipopt_snapshot.slack_upper_bound_multipliers,
                    1.0,
                    |idx| glider_inequality_label(idx, intervals, order),
                )
            );
            println!(
                "          top variable upper z diffs: {}",
                top_vector_diffs(&nlip_upper, &ipopt_upper, 1.0, |idx| glider_decision_label(
                    idx, intervals, order
                ),)
            );
            println!(
                "          top KKT inequality residual diffs: {}",
                top_vector_diffs(
                    nlip_kkt_ineq,
                    &ipopt_snapshot.kkt_inequality_residual,
                    1.0,
                    |idx| glider_inequality_label(idx, intervals, order),
                )
            );
            println!(
                "          top KKT x stationarity diffs: {}",
                top_vector_diffs(
                    nlip_kkt_x_stationarity,
                    &ipopt_snapshot.kkt_x_stationarity,
                    1.0,
                    |idx| glider_decision_label(idx, intervals, order),
                )
            );
            println!(
                "          top KKT slack stationarity diffs: {}",
                top_vector_diffs(
                    nlip_kkt_slack_stationarity,
                    &ipopt_snapshot.kkt_slack_stationarity,
                    1.0,
                    |idx| glider_inequality_label(idx, intervals, order),
                )
            );
            println!(
                "          top KKT slack sigma diffs: {}",
                top_vector_diffs(
                    nlip_kkt_slack_sigma,
                    &ipopt_snapshot.kkt_slack_sigma,
                    1.0,
                    |idx| glider_inequality_label(idx, intervals, order),
                )
            );
            println!(
                "          top upper-slack KKT RHS diffs: {}",
                top_upper_slack_kkt_rhs_diffs(
                    nlip_snapshot,
                    ipopt_snapshot,
                    params,
                    intervals,
                    order
                )
            );
        }

        fn print_internal_probe_window(
            label: &str,
            center_index: usize,
            accepted_nlip_solver_snapshots: &[optimization::InteriorPointIterationSnapshot],
            accepted_ipopt_solver_snapshots: &[optimization::IpoptIterationSnapshot],
            nlip_trace: &[TracePoint],
            ipopt_trace: &[TracePoint],
            params: &Params,
            kappa_d: f64,
            bounds: &VariableBoundView,
            intervals: usize,
            order: usize,
        ) {
            let probe_start = center_index.saturating_sub(4);
            let probe_end = (center_index + 3)
                .min(accepted_ipopt_solver_snapshots.len())
                .min(accepted_nlip_solver_snapshots.len());
            println!("{label} internal IPOPT/NLIP state probes range={probe_start}..{probe_end}");
            for probe_index in probe_start..probe_end {
                println!("          probe_index={probe_index}");
                if probe_index > 0 {
                    let tau = 0.99_f64.max(1.0 - ipopt_trace[probe_index].mu.max(0.0));
                    println!(
                        "          direct alpha_pr limiters: nlip={} || ipopt={}",
                        nlip_direct_primal_fraction_to_boundary_limiters(
                            &accepted_nlip_solver_snapshots[probe_index - 1],
                            &accepted_nlip_solver_snapshots[probe_index],
                            &accepted_ipopt_solver_snapshots[probe_index - 1],
                            bounds,
                            tau,
                            intervals,
                            order,
                        ),
                        ipopt_direct_primal_fraction_to_boundary_limiters(
                            &accepted_ipopt_solver_snapshots[probe_index - 1],
                            &accepted_ipopt_solver_snapshots[probe_index],
                            bounds,
                            tau,
                            intervals,
                            order,
                        )
                    );
                    println!(
                        "          ipopt alpha_pr limiters: {}",
                        ipopt_fraction_to_boundary_limiters(
                            &accepted_ipopt_solver_snapshots[probe_index - 1],
                            &accepted_ipopt_solver_snapshots[probe_index],
                            bounds,
                            ipopt_trace[probe_index].alpha_pr,
                            intervals,
                            order,
                        )
                    );
                }
                if probe_index > 0
                    && probe_index < nlip_trace.len()
                    && probe_index < ipopt_trace.len()
                {
                    // IPOPT BacktrackingLineSearch::PerformDualStep defaults
                    // alpha_for_y to alpha_primal. NLIP now records the
                    // computed alpha_y explicitly so non-default strategies do
                    // not get hidden behind alpha_primal in diagnostics.
                    let nlip_alpha_y = nlip_trace[probe_index].alpha_y;
                    let ipopt_alpha_y = ipopt_trace[probe_index].alpha_y;
                    let nlip_alpha_du = nlip_trace[probe_index].alpha_du;
                    let ipopt_alpha_du = ipopt_trace[probe_index].alpha_du;
                    let prev_nlip = &accepted_nlip_solver_snapshots[probe_index - 1];
                    let nlip = &accepted_nlip_solver_snapshots[probe_index];
                    let prev_ipopt = &accepted_ipopt_solver_snapshots[probe_index - 1];
                    let ipopt = &accepted_ipopt_solver_snapshots[probe_index];
                    println!(
                        "          alpha step sizes nlip(pr={:.17e},du={:.17e},y={:.17e}) ipopt(pr={:.17e},du={:.17e},y={:.17e})",
                        nlip_trace[probe_index].alpha_pr,
                        nlip_alpha_du,
                        nlip_alpha_y,
                        ipopt_trace[probe_index].alpha_pr,
                        ipopt_alpha_du,
                        ipopt_alpha_y,
                    );
                    println!(
                        "          top upper-slack next-solve RHS diffs: {}",
                        top_upper_slack_next_solve_rhs_diffs(
                            prev_nlip,
                            prev_ipopt,
                            nlip_trace[probe_index].mu,
                            ipopt_trace[probe_index].mu,
                            params,
                            kappa_d,
                            intervals,
                            order,
                        )
                    );
                    println!(
                        "          top y_c accepted-direction diffs: {}",
                        top_vector_direction_diffs(
                            prev_nlip.equality_multipliers.as_deref().unwrap_or(&[]),
                            nlip.equality_multipliers.as_deref().unwrap_or(&[]),
                            &prev_ipopt.equality_multipliers,
                            &ipopt.equality_multipliers,
                            nlip_alpha_y,
                            ipopt_alpha_y,
                            |idx| glider_equality_label(idx, intervals, order),
                        )
                    );
                    println!(
                        "          top y_d accepted-direction diffs: {}",
                        top_vector_direction_diffs(
                            prev_nlip.inequality_multipliers.as_deref().unwrap_or(&[]),
                            nlip.inequality_multipliers.as_deref().unwrap_or(&[]),
                            &prev_ipopt.inequality_multipliers,
                            &ipopt.inequality_multipliers,
                            nlip_alpha_y,
                            ipopt_alpha_y,
                            |idx| glider_inequality_label(idx, intervals, order),
                        )
                    );
                    println!(
                        "          top v_U accepted-direction diffs: {}",
                        top_vector_direction_diffs(
                            prev_nlip.slack_multipliers.as_deref().unwrap_or(&[]),
                            nlip.slack_multipliers.as_deref().unwrap_or(&[]),
                            &prev_ipopt.slack_upper_bound_multipliers,
                            &ipopt.slack_upper_bound_multipliers,
                            nlip_alpha_du,
                            ipopt_alpha_du,
                            |idx| glider_inequality_label(idx, intervals, order),
                        )
                    );
                    println!(
                        "          top upper-slack multiplier direction diffs: {}",
                        top_upper_slack_multiplier_direction_diffs(
                            prev_nlip,
                            nlip,
                            prev_ipopt,
                            ipopt,
                            nlip_alpha_y,
                            nlip_alpha_du,
                            nlip_trace[probe_index].mu,
                            ipopt_alpha_y,
                            ipopt_alpha_du,
                            ipopt_trace[probe_index].mu,
                            intervals,
                            order,
                        )
                    );
                    println!(
                        "          step application summary: {}",
                        step_application_summary(
                            prev_nlip,
                            nlip,
                            prev_ipopt,
                            ipopt,
                            nlip_alpha_y,
                            nlip_alpha_du,
                            ipopt_alpha_y,
                            ipopt_alpha_du,
                        )
                    );
                    if let Some(nlip_direction) = nlip.step_direction.as_ref() {
                        println!(
                            "          top x delta-snapshot diffs: {}",
                            top_vector_diffs(&nlip_direction.x, &ipopt.direction_x, 1.0, |idx| {
                                glider_decision_label(idx, intervals, order)
                            },)
                        );
                        println!(
                            "          top internal slack delta-snapshot diffs: {}",
                            top_vector_diffs(
                                &nlip_direction.slack,
                                &ipopt.direction_slack,
                                1.0,
                                |idx| glider_inequality_label(idx, intervals, order),
                            )
                        );
                        println!(
                            "          top y_c delta-snapshot diffs: {}",
                            top_vector_diffs(
                                &nlip_direction.equality_multipliers,
                                &ipopt.direction_equality_multipliers,
                                1.0,
                                |idx| glider_equality_label(idx, intervals, order),
                            )
                        );
                        println!(
                            "          top y_c output-scale delta-snapshot diffs: {}",
                            top_equality_multiplier_output_scale_diffs(
                                &nlip_direction.equality_multipliers,
                                &ipopt.direction_equality_multipliers,
                                params,
                                intervals,
                                order,
                            )
                        );
                        println!(
                            "          top y_d delta-snapshot diffs: {}",
                            top_vector_diffs(
                                &nlip_direction.inequality_multipliers,
                                &ipopt.direction_inequality_multipliers,
                                1.0,
                                |idx| glider_inequality_label(idx, intervals, order),
                            )
                        );
                        println!(
                            "          top v_U delta-snapshot diffs: {}",
                            top_vector_diffs(
                                &nlip_direction.slack_multipliers,
                                &ipopt.direction_slack_upper_bound_multipliers,
                                1.0,
                                |idx| glider_inequality_label(idx, intervals, order),
                            )
                        );
                    }
                }
                print_internal_ipopt_probe(
                    probe_index,
                    &accepted_nlip_solver_snapshots[probe_index],
                    &accepted_ipopt_solver_snapshots[probe_index],
                    bounds,
                    params,
                    intervals,
                    order,
                );
            }
        }

        fn print_trace_window(
            nlip_trace: &[TracePoint],
            ipopt_trace: &[TracePoint],
            center: usize,
            intervals: usize,
            order: usize,
        ) {
            let compared = nlip_trace.len().min(ipopt_trace.len());
            let start = if center <= 8 {
                0
            } else {
                center.saturating_sub(4)
            };
            let end = (center + 5).min(compared);
            println!(
                "trace_window index={} range={}..{} columns: idx | nlip(iter tag obj primal dual mu tf x_T reg step dx ds dz alpha_pr alpha_du ls inertia evt limiter_pr limiter_du) || ipopt(iter tag obj primal dual mu tf x_T reg step alpha_pr alpha_du ls)",
                center, start, end
            );
            for index in start..end {
                let nlip = &nlip_trace[index];
                let ipopt = &ipopt_trace[index];
                println!(
                    "trace[{index:02}] | nlip({:>3} {:>2} {:>12.5e} {:>10.3e} {:>10.3e} {:>9.2e} {:>9.2e} {:>12.5e} {:>8} {:>9.2e} {:>9.2e} {:>9.2e} {:>9.2e} {:>9.2e} {:>9.2e} {:>2} {:>13} {:<10} {:<34} {:<34}) || ipopt({:>3} {:>2} {:>12.5e} {:>10.3e} {:>10.3e} {:>9.2e} {:>9.2e} {:>12.5e} {:>8} {:>9.2e} {:>9.2e} {:>9.2e} {:>2})",
                    nlip.iteration,
                    nlip.step_tag,
                    nlip.objective,
                    nlip.primal_inf,
                    nlip.dual_inf,
                    nlip.mu,
                    nlip.tf,
                    nlip.terminal_x,
                    regularization_text(nlip.regularization),
                    nlip.step_inf,
                    nlip.dx_inf,
                    nlip.ds_inf,
                    nlip.dz_inf,
                    nlip.alpha_pr,
                    nlip.alpha_du,
                    nlip.trial_count,
                    nlip.inertia,
                    nlip.events,
                    nlip.alpha_pr_limiter,
                    nlip.alpha_du_limiter,
                    ipopt.iteration,
                    ipopt.step_tag,
                    ipopt.objective,
                    ipopt.primal_inf,
                    ipopt.dual_inf,
                    ipopt.mu,
                    ipopt.tf,
                    ipopt.terminal_x,
                    regularization_text(ipopt.regularization),
                    ipopt.step_inf,
                    ipopt.alpha_pr,
                    ipopt.alpha_du,
                    ipopt.trial_count,
                );
                if nlip.alpha_du_limiters != "--" {
                    println!("          nlip alpha_du top: {}", nlip.alpha_du_limiters);
                }
                if nlip.linear_detail != "--" {
                    println!("          nlip linear detail: {}", nlip.linear_detail);
                }
                if nlip.linear_stats != "--" {
                    println!("          nlip linear stats: {}", nlip.linear_stats);
                }
                println!(
                    "          nlip perturbations: primal_shift={:.6e} dual_regularization={:.6e}; ipopt regularization={}",
                    nlip.primal_shift,
                    nlip.dual_regularization,
                    regularization_text(ipopt.regularization),
                );
                println!(
                    "          alpha exact: nlip_pr={:.17e} ipopt_pr={:.17e} gap_pr={:.3e}; nlip_du={:.17e} ipopt_du={:.17e} gap_du={:.3e}",
                    nlip.alpha_pr,
                    ipopt.alpha_pr,
                    (nlip.alpha_pr - ipopt.alpha_pr).abs(),
                    nlip.alpha_du,
                    ipopt.alpha_du,
                    (nlip.alpha_du - ipopt.alpha_du).abs(),
                );
                println!(
                    "          top x diffs: {}",
                    top_x_diffs(nlip, ipopt, intervals, order)
                );
                if index > 0 {
                    println!(
                        "          top step delta diffs: {}",
                        top_step_delta_diffs(
                            &nlip_trace[index - 1],
                            nlip,
                            &ipopt_trace[index - 1],
                            ipopt,
                            intervals,
                            order
                        )
                    );
                    println!(
                        "          top accepted-direction diffs: {}",
                        top_direction_estimate_diffs(
                            &nlip_trace[index - 1],
                            nlip,
                            &ipopt_trace[index - 1],
                            ipopt,
                            intervals,
                            order
                        )
                    );
                }
            }
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
        let runtime = dc_runtime(&params);
        let variable_bounds = variable_bound_view(&params);
        let intervals = params.transcription.intervals;
        let order = params.transcription.collocation_degree;

        let mut nlip_options = crate::common::nlip_options(&params.solver);
        if std::env::var_os("GLIDER_PARITY_SOURCE_DEFAULTS").is_some() {
            optimization::apply_ipopt_source_exact_hessian_defaults_to_nlip_options(
                &mut nlip_options,
            );
        }
        optimization::apply_native_spral_parity_to_nlip_options(&mut nlip_options);
        if std::env::var_os("GLIDER_PARITY_SOURCE_DEFAULTS").is_none() {
            nlip_options.max_iters = 400;
            nlip_options.acceptable_iter = 0;
        }
        nlip_options.verbose = false;
        if let Some(max_iters) = std::env::var("GLIDER_PARITY_NLIP_MAX_ITERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
        {
            nlip_options.max_iters = max_iters;
        }
        let requested_nlip_augmented_dump_dir =
            std::env::var_os("GLIDER_PARITY_NLIP_AUGMENTED_DUMP_DIR").map(std::path::PathBuf::from);
        let nlip_augmented_temp_dir = if requested_nlip_augmented_dump_dir.is_none() {
            std::env::var_os("GLIDER_PARITY_PRINT_NLIP_AUGMENTED_FINGERPRINTS")
                .map(|_| TempDir::new().expect("NLIP augmented fingerprint dump dir"))
        } else {
            None
        };
        let nlip_augmented_dump_dir = requested_nlip_augmented_dump_dir.or_else(|| {
            nlip_augmented_temp_dir
                .as_ref()
                .map(|dir| dir.path().to_path_buf())
        });
        nlip_options.linear_debug = Some(optimization::InteriorPointLinearDebugOptions {
            compare_solvers: Vec::new(),
            schedule: optimization::InteriorPointLinearDebugSchedule::EveryIteration,
            dump_dir: nlip_augmented_dump_dir.clone(),
        });

        let mut ipopt_options = crate::common::ipopt_options(&params.solver);
        if std::env::var_os("GLIDER_PARITY_SOURCE_DEFAULTS").is_some() {
            ipopt_options = optimization::IpoptOptions::default();
        }
        optimization::apply_native_spral_parity_to_ipopt_options(&mut ipopt_options);
        if let Some(print_level) = std::env::var("GLIDER_PARITY_IPOPT_PRINT_LEVEL")
            .ok()
            .and_then(|value| value.parse::<i32>().ok())
        {
            ipopt_options.print_level = print_level;
        }
        if let Some(max_iters) = std::env::var("GLIDER_PARITY_IPOPT_MAX_ITERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
        {
            ipopt_options.max_iters = max_iters;
        }

        let mut nlip_initial_options = nlip_options.clone();
        nlip_initial_options.max_iters = 0;
        nlip_initial_options.linear_debug = None;
        let mut nlip_initial_ocp_snapshots = Vec::new();
        let _ = compiled.solve_interior_point_with_callback(
            &runtime,
            &nlip_initial_options,
            |snapshot| nlip_initial_ocp_snapshots.push(snapshot.clone()),
        );

        let mut nlip_ocp_snapshots = Vec::new();
        let nlip =
            compiled.solve_interior_point_with_callback(&runtime, &nlip_options, |snapshot| {
                nlip_ocp_snapshots.push(snapshot.clone());
            });
        if std::env::var_os("GLIDER_PARITY_PRINT_NLIP_RESTORATION_TRACE").is_some() {
            let start = std::env::var("GLIDER_PARITY_RESTORATION_TRACE_START")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            let end = std::env::var("GLIDER_PARITY_RESTORATION_TRACE_END")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(usize::MAX);
            for snapshot in nlip_ocp_snapshots.iter().filter(|snapshot| {
                snapshot.solver.phase == optimization::InteriorPointIterationPhase::Restoration
                    && (start..=end).contains(&snapshot.solver.iteration)
            }) {
                println!(
                    "nlip_resto iter={} obj={:.8e} primal={:.8e} dual={:.8e} comp={:.8e} overall={:.8e} mu={:.8e} tf={:.8e} xT={:.8e} alpha={:?} alpha_pr={:?} alpha_du={:?} alpha_y={:?} ls={} tag={:?} events={} barrier_obj={:?} theta={:?} step_inf={:?}",
                    snapshot.solver.iteration,
                    snapshot.solver.objective,
                    snapshot
                        .solver
                        .eq_inf
                        .unwrap_or(0.0)
                        .max(snapshot.solver.ineq_inf.unwrap_or(0.0)),
                    snapshot.solver.dual_inf,
                    snapshot.solver.comp_inf.unwrap_or(0.0),
                    snapshot.solver.overall_inf,
                    snapshot.solver.barrier_parameter.unwrap_or(0.0),
                    snapshot.trajectories.tf,
                    snapshot.trajectories.x.terminal.x,
                    snapshot.solver.alpha,
                    snapshot.solver.alpha_pr,
                    snapshot.solver.alpha_du,
                    snapshot.solver.alpha_y,
                    snapshot.solver.line_search_trials,
                    snapshot.solver.step_tag,
                    optimization::nlip_event_codes_for_events(&snapshot.solver.events),
                    snapshot.solver.barrier_objective,
                    snapshot.solver.filter_theta,
                    snapshot.solver.step_inf,
                );
            }
        }
        if std::env::var_os("GLIDER_PARITY_PRINT_NLIP_LINE_SEARCH_TRACE").is_some() {
            let start = std::env::var("GLIDER_PARITY_LINE_SEARCH_TRACE_START")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            let end = std::env::var("GLIDER_PARITY_LINE_SEARCH_TRACE_END")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(usize::MAX);
            for snapshot in nlip_ocp_snapshots.iter().filter(|snapshot| {
                matches!(
                    snapshot.solver.phase,
                    optimization::InteriorPointIterationPhase::AcceptedStep
                        | optimization::InteriorPointIterationPhase::Restoration
                ) && (start..=end).contains(&snapshot.solver.iteration)
            }) {
                let Some(line_search) = snapshot.solver.line_search.as_ref() else {
                    continue;
                };
                println!(
                    "nlip_ls iter={} phase={:?} tag={:?} obj={:.8e} primal={:.8e} dual={:.8e} mu={:.8e} tf={:.8e} xT={:.8e} alpha0={:.17e} alpha_du0={:?} alpha_y0={:?} accepted={:?} accepted_du={:?} accepted_y={:?} last={:.17e} last_du={:?} last_y={:?} backtracks={} alpha_min={:.17e} sigma={:.8e} current_barrier={:.8e} current_primal={:.8e} rejected={} soc_attempted={} soc_used={} watchdog_active={} watchdog_accepted={} tiny_step={} mode={:?}",
                    snapshot.solver.iteration,
                    snapshot.solver.phase,
                    snapshot.solver.step_tag,
                    snapshot.solver.objective,
                    snapshot
                        .solver
                        .eq_inf
                        .unwrap_or(0.0)
                        .max(snapshot.solver.ineq_inf.unwrap_or(0.0)),
                    snapshot.solver.dual_inf,
                    snapshot.solver.barrier_parameter.unwrap_or(0.0),
                    snapshot.trajectories.tf,
                    snapshot.trajectories.x.terminal.x,
                    line_search.initial_alpha_pr,
                    line_search.initial_alpha_du,
                    line_search.initial_alpha_y,
                    line_search.accepted_alpha,
                    line_search.accepted_alpha_du,
                    line_search.accepted_alpha_y,
                    line_search.last_tried_alpha,
                    line_search.last_tried_alpha_du,
                    line_search.last_tried_alpha_y,
                    line_search.backtrack_count,
                    line_search.alpha_min,
                    line_search.sigma,
                    line_search.current_barrier_objective,
                    line_search.current_primal_inf,
                    line_search.rejected_trials.len(),
                    line_search.second_order_correction_attempted,
                    line_search.second_order_correction_used,
                    line_search.watchdog_active,
                    line_search.watchdog_accepted,
                    line_search.tiny_step,
                    line_search.filter_acceptance_mode,
                );
                for (index, trial) in line_search.rejected_trials.iter().enumerate().take(8) {
                    println!(
                        "  nlip_ls_rejected[{index}] alpha={:.17e} alpha_du={:?} slack_positive={} multipliers_positive={} merit={:?} barrier={:?} primal={:?} dual={:?} comp={:?} local_filter={:?} filter={:?} dominated={:?} sufficient_phi={:?} sufficient_theta={:?} switching={:?}",
                        trial.alpha,
                        trial.alpha_du,
                        trial.slack_positive,
                        trial.multipliers_positive,
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
                if line_search.rejected_trials.len() > 8
                    && let Some(trial) = line_search.rejected_trials.last()
                {
                    println!(
                        "  nlip_ls_rejected[last] alpha={:.17e} alpha_du={:?} slack_positive={} multipliers_positive={} merit={:?} barrier={:?} primal={:?} dual={:?} comp={:?} local_filter={:?} filter={:?} dominated={:?} sufficient_phi={:?} sufficient_theta={:?} switching={:?}",
                        trial.alpha,
                        trial.alpha_du,
                        trial.slack_positive,
                        trial.multipliers_positive,
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
        if let Err(err) = &nlip {
            println!("nlip_err={err}");
            let context = match err {
                optimization::InteriorPointSolveError::LinearSolve { context, .. }
                | optimization::InteriorPointSolveError::LineSearchFailed { context, .. }
                | optimization::InteriorPointSolveError::RestorationFailed { context, .. }
                | optimization::InteriorPointSolveError::LocalInfeasibility { context }
                | optimization::InteriorPointSolveError::DivergingIterates { context, .. }
                | optimization::InteriorPointSolveError::CpuTimeExceeded { context, .. }
                | optimization::InteriorPointSolveError::WallTimeExceeded { context, .. }
                | optimization::InteriorPointSolveError::UserRequestedStop { context }
                | optimization::InteriorPointSolveError::SearchDirectionTooSmall { context }
                | optimization::InteriorPointSolveError::MaxIterations { context, .. } => {
                    Some(context.as_ref())
                }
                optimization::InteriorPointSolveError::InvalidInput(_) => None,
            };
            if let Some(context) = context {
                if let Some(last) = context.last_accepted_state.as_ref() {
                    println!(
                        "nlip_last_accepted iter={} phase={:?} obj={:.6e} primal={:.6e} dual={:.6e} comp={:.6e} mu={:.6e} alpha_pr={:?} alpha_du={:?} alpha_y={:?} tag={:?} events={:?}",
                        last.iteration,
                        last.phase,
                        last.objective,
                        last.eq_inf.unwrap_or(0.0).max(last.ineq_inf.unwrap_or(0.0)),
                        last.dual_inf,
                        last.comp_inf.unwrap_or(0.0),
                        last.barrier_parameter.unwrap_or(0.0),
                        last.alpha_pr,
                        last.alpha_du,
                        last.alpha_y,
                        last.step_tag,
                        last.events,
                    );
                }
                if let Some(line_search) = context.failed_line_search.as_ref() {
                    println!(
                        "nlip_line_search_failure alpha0={:.17e} alpha_du0={:?} alpha_y0={:?} last_alpha={:.17e} last_alpha_du={:?} last_alpha_y={:?} backtracks={} alpha_min={:.17e} rejected={} soc_attempted={} soc_used={} watchdog_active={} watchdog_accepted={} tiny_step={} current_merit={:.17e} barrier_obj={:.17e} primal={:.17e}",
                        line_search.initial_alpha_pr,
                        line_search.initial_alpha_du,
                        line_search.initial_alpha_y,
                        line_search.last_tried_alpha,
                        line_search.last_tried_alpha_du,
                        line_search.last_tried_alpha_y,
                        line_search.backtrack_count,
                        line_search.alpha_min,
                        line_search.rejected_trials.len(),
                        line_search.second_order_correction_attempted,
                        line_search.second_order_correction_used,
                        line_search.watchdog_active,
                        line_search.watchdog_accepted,
                        line_search.tiny_step,
                        line_search.current_merit,
                        line_search.current_barrier_objective,
                        line_search.current_primal_inf,
                    );
                    for (index, trial) in line_search.rejected_trials.iter().take(6).enumerate() {
                        println!(
                            "  rejected[{index}] alpha={:.17e} alpha_du={:?} slack_positive={} multipliers_positive={} merit={:?} primal={:?} dual={:?} comp={:?} local_filter={:?} filter={:?} dominated={:?} sufficient_phi={:?} sufficient_theta={:?} switching={:?}",
                            trial.alpha,
                            trial.alpha_du,
                            trial.slack_positive,
                            trial.multipliers_positive,
                            trial.merit,
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
                    if line_search.rejected_trials.len() > 6
                        && let Some(trial) = line_search.rejected_trials.last()
                    {
                        println!(
                            "  rejected[last] alpha={:.17e} alpha_du={:?} slack_positive={} multipliers_positive={} merit={:?} primal={:?} dual={:?} comp={:?} local_filter={:?} filter={:?} dominated={:?} sufficient_phi={:?} sufficient_theta={:?} switching={:?}",
                            trial.alpha,
                            trial.alpha_du,
                            trial.slack_positive,
                            trial.multipliers_positive,
                            trial.merit,
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
                if let Some(direction) = context.failed_direction_diagnostics.as_ref() {
                    println!(
                        "nlip_failed_direction dx_inf={:.17e} dlambda_inf={:.17e} ds_inf={:.17e} dz_inf={:.17e} alpha_pr_limiter={:?} alpha_du_limiter={:?}",
                        direction.dx_inf,
                        direction.d_lambda_inf,
                        direction.ds_inf,
                        direction.dz_inf,
                        direction.alpha_pr_limiter,
                        direction.alpha_du_limiter,
                    );
                }
                if let Some(diagnostics) = context.failed_linear_solve.as_ref() {
                    println!(
                        "nlip_linear_failure dim={} attempts={}",
                        diagnostics.matrix_dimension,
                        diagnostics.attempts.len()
                    );
                    for (index, attempt) in diagnostics.attempts.iter().enumerate().take(8) {
                        println!(
                            "  attempt[{index}] solver={} kind={} reg={:.3e} detail={}",
                            attempt.solver.label(),
                            attempt.failure_kind.label(),
                            attempt.regularization,
                            attempt.detail.as_deref().unwrap_or("--"),
                        );
                    }
                    if diagnostics.attempts.len() > 8
                        && let Some(attempt) = diagnostics.attempts.last()
                    {
                        println!(
                            "  attempt[last] solver={} kind={} reg={:.3e} detail={}",
                            attempt.solver.label(),
                            attempt.failure_kind.label(),
                            attempt.regularization,
                            attempt.detail.as_deref().unwrap_or("--"),
                        );
                    }
                }
            }
        }
        let mut ipopt_snapshots = Vec::new();
        let mut ipopt_ocp_snapshots = Vec::new();
        let ipopt = compiled.solve_ipopt_with_callback(&runtime, &ipopt_options, |snapshot| {
            ipopt_snapshots.push(snapshot.solver.clone());
            ipopt_ocp_snapshots.push(IpoptOcpTracePoint {
                solver: snapshot.solver.clone(),
                tf: snapshot.trajectories.tf,
                terminal_x: snapshot.trajectories.x.terminal.x,
            });
        });
        if let Err(err) = &ipopt {
            println!("ipopt_err={err}");
        }
        if ipopt_snapshots.is_empty() {
            match &ipopt {
                Ok(summary) => {
                    ipopt_snapshots = summary.solver.snapshots.clone();
                }
                Err(optimization::IpoptSolveError::Solve { snapshots, .. }) => {
                    ipopt_snapshots = snapshots.clone();
                }
                Err(_) => {}
            }
        }
        let ipopt_journal_output = match &ipopt {
            Ok(summary) => summary.solver.journal_output.as_deref(),
            Err(optimization::IpoptSolveError::Solve { journal_output, .. }) => {
                journal_output.as_deref()
            }
            Err(_) => None,
        };
        print_ipopt_linear_journal_excerpt(ipopt_journal_output);
        print_ipopt_augmented_journal_fingerprints(ipopt_journal_output);
        print_nlip_augmented_dump_fingerprints(
            nlip_augmented_dump_dir.as_deref(),
            ipopt_journal_output,
            intervals,
            order,
        );
        print_ipopt_spral_interface_dump_fingerprints(
            std::env::var_os("GLIDER_PARITY_IPOPT_SPRAL_DUMP_DIR")
                .as_deref()
                .map(Path::new),
            nlip_augmented_dump_dir.as_deref(),
            intervals,
            order,
        );
        print_ipopt_spral_restoration_dump_fingerprints(
            std::env::var_os("GLIDER_PARITY_IPOPT_SPRAL_DUMP_DIR")
                .as_deref()
                .map(Path::new),
            intervals,
            order,
        );
        print_ipopt_spral_solve_boundary_ladder(
            std::env::var_os("GLIDER_PARITY_IPOPT_SPRAL_DUMP_DIR")
                .as_deref()
                .map(Path::new),
            nlip_augmented_dump_dir.as_deref(),
        );
        print_ipopt_full_space_residual_dump_fingerprints(
            std::env::var_os("GLIDER_PARITY_IPOPT_RESIDUAL_DUMP_DIR")
                .as_deref()
                .map(Path::new),
            nlip_augmented_dump_dir.as_deref(),
            intervals,
            order,
        );
        let ipopt_step_tags = parse_ipopt_step_tags(ipopt_journal_output);

        if let (Some(nlip_initial), Some(ipopt_initial)) = (
            nlip_initial_ocp_snapshots.iter().find(|snapshot| {
                snapshot.solver.phase == optimization::InteriorPointIterationPhase::Initial
                    && snapshot.solver.iteration == 0
            }),
            ipopt_snapshots
                .iter()
                .find(|snapshot| snapshot.iteration == 0),
        ) {
            let nlip_primal = nlip_initial
                .solver
                .eq_inf
                .unwrap_or(0.0)
                .max(nlip_initial.solver.ineq_inf.unwrap_or(0.0));
            println!(
                "initial nlip obj={:.6e} primal={:.6e} dual={:.6e} comp={:.6e} mu={:.6e} tf={:.6e} x_T={:.6e} || ipopt obj={:.6e} primal={:.6e} dual={:.6e} mu={:.6e}",
                nlip_initial.solver.objective,
                nlip_primal,
                nlip_initial.solver.dual_inf,
                nlip_initial.solver.comp_inf.unwrap_or(0.0),
                nlip_initial.solver.barrier_parameter.unwrap_or(0.0),
                nlip_initial.trajectories.tf,
                nlip_initial.trajectories.x.terminal.x,
                ipopt_initial.objective,
                ipopt_initial.primal_inf,
                ipopt_initial.dual_inf,
                ipopt_initial.barrier_parameter,
            );
            println!(
                "initial top eq_y diffs: {}",
                top_vector_diffs(
                    nlip_initial
                        .solver
                        .equality_multipliers
                        .as_deref()
                        .unwrap_or(&[]),
                    &ipopt_initial.equality_multipliers,
                    1.0,
                    |index| glider_equality_label(index, intervals, order),
                )
            );
            println!(
                "initial top y_d diffs: {}",
                top_vector_diffs(
                    nlip_initial
                        .solver
                        .inequality_multipliers
                        .as_deref()
                        .unwrap_or(&[]),
                    &ipopt_initial.inequality_multipliers,
                    1.0,
                    |index| glider_inequality_label(index, intervals, order),
                )
            );
            let nlip_initial_slack = nlip_initial.solver.slack_primal.as_deref().unwrap_or(&[]);
            println!(
                "initial state diffs x={:.3e} slack={:.3e}",
                vector_inf_diff(&nlip_initial.solver.x, &ipopt_initial.x, 1.0),
                vector_inf_diff(nlip_initial_slack, &ipopt_initial.internal_slack, 1.0),
            );
            println!(
                "initial top x diffs: {}",
                top_vector_diffs(&nlip_initial.solver.x, &ipopt_initial.x, 1.0, |idx| {
                    glider_decision_label(idx, intervals, order)
                },)
            );
            println!(
                "initial top internal slack diffs: {}",
                top_vector_diffs(
                    nlip_initial_slack,
                    &ipopt_initial.internal_slack,
                    1.0,
                    |idx| glider_inequality_label(idx, intervals, order),
                )
            );
            let nlip_initial_lower = expanded_compact_lower_bound_multipliers(
                nlip_initial.solver.lower_bound_multipliers.as_deref(),
                &variable_bounds,
            );
            let nlip_initial_upper = expanded_compact_upper_bound_multipliers(
                nlip_initial.solver.upper_bound_multipliers.as_deref(),
                &variable_bounds,
            );
            let ipopt_initial_lower = ipopt_algorithmic_bound_multipliers(
                &ipopt_initial.lower_bound_multipliers,
                &variable_bounds,
            );
            let ipopt_initial_upper = ipopt_algorithmic_bound_multipliers(
                &ipopt_initial.upper_bound_multipliers,
                &variable_bounds,
            );
            let nlip_initial_grad_f = nlip_initial.solver.curr_grad_f.as_deref().unwrap_or(&[]);
            let nlip_initial_jac_c_t_y_c = nlip_initial
                .solver
                .curr_jac_c_t_y_c
                .as_deref()
                .unwrap_or(&[]);
            let nlip_initial_jac_d_t_y_d = nlip_initial
                .solver
                .curr_jac_d_t_y_d
                .as_deref()
                .unwrap_or(&[]);
            let nlip_initial_grad_lag_x = nlip_initial
                .solver
                .curr_grad_lag_x
                .as_deref()
                .unwrap_or(&[]);
            let nlip_initial_kkt_x = nlip_initial
                .solver
                .kkt_x_stationarity
                .as_deref()
                .unwrap_or(&[]);
            let nlip_initial_grad_lag_s = nlip_initial
                .solver
                .curr_grad_lag_s
                .as_deref()
                .unwrap_or(&[]);
            let nlip_initial_kkt_slack = nlip_initial
                .solver
                .kkt_slack_stationarity
                .as_deref()
                .unwrap_or(&[]);
            println!(
                "initial component diffs grad_f={:.3e} jac_cT_y_c={:.3e} jac_dT_y_d={:.3e} grad_lag_x={:.3e} kkt_x={:.3e} grad_lag_s={:.3e} kkt_slack={:.3e} lower_z={:.3e} upper_z={:.3e}",
                vector_inf_diff(nlip_initial_grad_f, &ipopt_initial.curr_grad_f, 1.0),
                vector_inf_diff(
                    nlip_initial_jac_c_t_y_c,
                    &ipopt_initial.curr_jac_c_t_y_c,
                    1.0,
                ),
                vector_inf_diff(
                    nlip_initial_jac_d_t_y_d,
                    &ipopt_initial.curr_jac_d_t_y_d,
                    1.0,
                ),
                vector_inf_diff(nlip_initial_grad_lag_x, &ipopt_initial.curr_grad_lag_x, 1.0),
                vector_inf_diff(nlip_initial_kkt_x, &ipopt_initial.kkt_x_stationarity, 1.0),
                vector_inf_diff(nlip_initial_grad_lag_s, &ipopt_initial.curr_grad_lag_s, 1.0),
                vector_inf_diff(
                    nlip_initial_kkt_slack,
                    &ipopt_initial.kkt_slack_stationarity,
                    1.0,
                ),
                vector_inf_diff(&nlip_initial_lower, &ipopt_initial_lower, 1.0),
                vector_inf_diff(&nlip_initial_upper, &ipopt_initial_upper, 1.0),
            );
            println!(
                "initial top objective gradient diffs: {}",
                top_vector_diffs(
                    nlip_initial_grad_f,
                    &ipopt_initial.curr_grad_f,
                    1.0,
                    |idx| { glider_decision_label(idx, intervals, order) }
                )
            );
            println!(
                "initial top jac_cT*y_c diffs: {}",
                top_vector_diffs(
                    nlip_initial_jac_c_t_y_c,
                    &ipopt_initial.curr_jac_c_t_y_c,
                    1.0,
                    |idx| glider_decision_label(idx, intervals, order),
                )
            );
            println!(
                "initial top jac_dT*y_d diffs: {}",
                top_vector_diffs(
                    nlip_initial_jac_d_t_y_d,
                    &ipopt_initial.curr_jac_d_t_y_d,
                    1.0,
                    |idx| glider_decision_label(idx, intervals, order),
                )
            );
            println!(
                "initial top curr grad_lag_x diffs: {}",
                top_vector_diffs(
                    nlip_initial_grad_lag_x,
                    &ipopt_initial.curr_grad_lag_x,
                    1.0,
                    |idx| glider_decision_label(idx, intervals, order),
                )
            );
            println!(
                "initial top KKT x stationarity diffs: {}",
                top_vector_diffs(
                    nlip_initial_kkt_x,
                    &ipopt_initial.kkt_x_stationarity,
                    1.0,
                    |idx| glider_decision_label(idx, intervals, order),
                )
            );
            println!(
                "initial top variable lower z diffs: {}",
                top_vector_diffs(&nlip_initial_lower, &ipopt_initial_lower, 1.0, |idx| {
                    glider_decision_label(idx, intervals, order)
                },)
            );
            println!(
                "initial top variable upper z diffs: {}",
                top_vector_diffs(&nlip_initial_upper, &ipopt_initial_upper, 1.0, |idx| {
                    glider_decision_label(idx, intervals, order)
                },)
            );
        }

        let nlip_trace = nlip_ocp_snapshots
            .iter()
            .filter(|snapshot| {
                snapshot.solver.phase == optimization::InteriorPointIterationPhase::AcceptedStep
                    && snapshot.solver.alpha.is_some()
            })
            .map(|snapshot| TracePoint {
                iteration: snapshot.solver.iteration,
                x: snapshot.solver.x.clone(),
                objective: snapshot.solver.objective,
                primal_inf: snapshot
                    .solver
                    .eq_inf
                    .unwrap_or(0.0)
                    .max(snapshot.solver.ineq_inf.unwrap_or(0.0)),
                dual_inf: snapshot.solver.dual_inf,
                mu: snapshot.solver.barrier_parameter.unwrap_or(0.0),
                tf: snapshot.trajectories.tf,
                terminal_x: snapshot.trajectories.x.terminal.x,
                regularization: snapshot
                    .solver
                    .regularization_size
                    .and_then(positive_regularization),
                primal_shift: snapshot
                    .solver
                    .direction_diagnostics
                    .as_ref()
                    .map_or(f64::NAN, |diagnostics| diagnostics.primal_diagonal_shift),
                dual_regularization: snapshot
                    .solver
                    .direction_diagnostics
                    .as_ref()
                    .map_or(f64::NAN, |diagnostics| diagnostics.dual_regularization),
                step_inf: snapshot.solver.step_inf.unwrap_or(f64::NAN),
                dx_inf: snapshot
                    .solver
                    .direction_diagnostics
                    .as_ref()
                    .map_or(f64::NAN, |diagnostics| diagnostics.dx_inf),
                ds_inf: snapshot
                    .solver
                    .direction_diagnostics
                    .as_ref()
                    .map_or(f64::NAN, |diagnostics| diagnostics.ds_inf),
                dz_inf: snapshot
                    .solver
                    .direction_diagnostics
                    .as_ref()
                    .map_or(f64::NAN, |diagnostics| diagnostics.dz_inf),
                alpha_pr: snapshot
                    .solver
                    .alpha_pr
                    .or(snapshot.solver.alpha)
                    .unwrap_or(f64::NAN),
                alpha_du: snapshot.solver.alpha_du.unwrap_or(f64::NAN),
                alpha_y: snapshot
                    .solver
                    .alpha_y
                    .or(snapshot.solver.alpha_pr)
                    .or(snapshot.solver.alpha)
                    .unwrap_or(f64::NAN),
                step_tag: snapshot
                    .solver
                    .step_tag
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                trial_count: snapshot.solver.line_search_trials,
                events: optimization::nlip_event_codes_for_events(&snapshot.solver.events),
                inertia: nlip_primary_inertia_text(&snapshot.solver),
                linear_stats: nlip_primary_linear_stats_text(&snapshot.solver),
                linear_detail: nlip_primary_detail_text(&snapshot.solver),
                alpha_pr_limiter: limiter_text(
                    snapshot
                        .solver
                        .direction_diagnostics
                        .as_ref()
                        .and_then(|diagnostics| diagnostics.alpha_pr_limiter.as_ref()),
                    &snapshot.solver.x,
                    snapshot.solver.slack_primal.as_deref(),
                    &variable_bounds,
                    params.transcription.intervals,
                    params.transcription.collocation_degree,
                ),
                alpha_du_limiter: limiter_text(
                    snapshot
                        .solver
                        .direction_diagnostics
                        .as_ref()
                        .and_then(|diagnostics| diagnostics.alpha_du_limiter.as_ref()),
                    &snapshot.solver.x,
                    None,
                    &variable_bounds,
                    params.transcription.intervals,
                    params.transcription.collocation_degree,
                ),
                alpha_du_limiters: snapshot.solver.direction_diagnostics.as_ref().map_or_else(
                    || "--".to_string(),
                    |diagnostics| limiters_text(&diagnostics.alpha_du_limiters),
                ),
            })
            .collect::<Vec<_>>();

        let ipopt_ocp_by_iteration = ipopt_ocp_snapshots
            .iter()
            .map(|snapshot| {
                (
                    snapshot.solver.iteration,
                    (snapshot.tf, snapshot.terminal_x),
                )
            })
            .collect::<BTreeMap<_, _>>();

        let ipopt_trace = ipopt_snapshots
            .iter()
            .filter(|snapshot| snapshot.iteration > 0)
            .map(|snapshot| {
                let (tf, terminal_x) = ipopt_ocp_by_iteration
                    .get(&snapshot.iteration)
                    .copied()
                    .unwrap_or((f64::NAN, f64::NAN));
                TracePoint {
                    iteration: snapshot.iteration,
                    x: snapshot.x.clone(),
                    objective: snapshot.objective,
                    primal_inf: snapshot.primal_inf,
                    dual_inf: snapshot.dual_inf,
                    mu: snapshot.barrier_parameter,
                    tf,
                    terminal_x,
                    regularization: positive_regularization(snapshot.regularization_size),
                    primal_shift: snapshot.regularization_size,
                    dual_regularization: f64::NAN,
                    step_inf: snapshot.step_inf,
                    dx_inf: f64::NAN,
                    ds_inf: f64::NAN,
                    dz_inf: f64::NAN,
                    alpha_pr: snapshot.alpha_pr,
                    alpha_du: snapshot.alpha_du,
                    alpha_y: snapshot.alpha_pr,
                    step_tag: ipopt_step_tags
                        .get(&snapshot.iteration)
                        .cloned()
                        .unwrap_or_else(|| "-".to_string()),
                    trial_count: snapshot.line_search_trials.saturating_sub(1),
                    events: String::new(),
                    inertia: String::new(),
                    linear_stats: String::new(),
                    linear_detail: String::new(),
                    alpha_pr_limiter: String::new(),
                    alpha_du_limiter: String::new(),
                    alpha_du_limiters: String::new(),
                }
            })
            .collect::<Vec<_>>();
        let accepted_nlip_solver_snapshots = nlip_ocp_snapshots
            .iter()
            .filter(|snapshot| {
                snapshot.solver.phase == optimization::InteriorPointIterationPhase::AcceptedStep
                    && snapshot.solver.alpha.is_some()
            })
            .map(|snapshot| snapshot.solver.clone())
            .collect::<Vec<_>>();
        let accepted_ipopt_solver_snapshots = ipopt_snapshots
            .iter()
            .filter(|snapshot| snapshot.iteration > 0)
            .cloned()
            .collect::<Vec<_>>();

        println!("\n=== glider native SPRAL NLIP/IPOPT first divergence ===");
        println!(
            "nlip_steps={} ipopt_steps={}",
            nlip_trace.len(),
            ipopt_trace.len()
        );
        let ipopt_snapshot_lengths = ipopt_snapshots
            .iter()
            .map(|snapshot| snapshot.x.len())
            .collect::<BTreeSet<_>>();
        println!(
            "ipopt_snapshot_count={} ipopt_ocp_snapshot_count={} ipopt_x_lengths={:?}",
            ipopt_snapshots.len(),
            ipopt_ocp_snapshots.len(),
            ipopt_snapshot_lengths
        );
        println!(
            "direction_gap_ladder {}",
            direction_gap_ladder(&nlip_trace, &ipopt_trace)
        );
        println!(
            "alpha_gap_ladder {}",
            alpha_gap_ladder(&nlip_trace, &ipopt_trace)
        );
        println!(
            "accepted_direction_gap_ladder {}",
            accepted_direction_gap_ladder(
                &accepted_nlip_solver_snapshots,
                &accepted_ipopt_solver_snapshots,
                &nlip_trace,
                &ipopt_trace,
            )
        );
        println!(
            "accepted_state_gap_ladder {}",
            accepted_state_gap_ladder(
                &accepted_nlip_solver_snapshots,
                &accepted_ipopt_solver_snapshots,
            )
        );
        let direction_window_threshold_env = std::env::var("GLIDER_PARITY_DIRECTION_THRESHOLD");
        let direction_window_threshold = direction_window_threshold_env
            .as_ref()
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(1.0e-1);
        if let Some((direction_index, direction_gap)) = (1..nlip_trace.len().min(ipopt_trace.len()))
            .find_map(|index| {
                if is_restoration_bridge_trace_index(&nlip_trace, &ipopt_trace, index) {
                    return None;
                }
                let gap = max_direction_estimate_diff(
                    &nlip_trace[index - 1],
                    &nlip_trace[index],
                    &ipopt_trace[index - 1],
                    &ipopt_trace[index],
                );
                (gap > direction_window_threshold).then_some((index, gap))
            })
        {
            println!(
                "first_direction_divergence threshold={:.3e} index={} nlip_iter={} ipopt_iter={} max_dir_gap={:.3e}",
                direction_window_threshold,
                direction_index,
                nlip_trace[direction_index].iteration,
                ipopt_trace[direction_index].iteration,
                direction_gap
            );
            println!(
                "first_direction_divergence top accepted-direction diffs: {}",
                top_direction_estimate_diffs(
                    &nlip_trace[direction_index - 1],
                    &nlip_trace[direction_index],
                    &ipopt_trace[direction_index - 1],
                    &ipopt_trace[direction_index],
                    params.transcription.intervals,
                    params.transcription.collocation_degree,
                )
            );
            print_trace_window(
                &nlip_trace,
                &ipopt_trace,
                direction_index,
                params.transcription.intervals,
                params.transcription.collocation_degree,
            );
            print_internal_probe_window(
                "first_direction_divergence",
                direction_index,
                &accepted_nlip_solver_snapshots,
                &accepted_ipopt_solver_snapshots,
                &nlip_trace,
                &ipopt_trace,
                &params,
                nlip_options.kappa_d,
                &variable_bounds,
                params.transcription.intervals,
                params.transcription.collocation_degree,
            );
            if direction_window_threshold_env.is_err() {
                panic!(
                    "glider native-SPRAL NLIP/IPOPT direction divergence exceeded default threshold {direction_window_threshold:.3e} at accepted index {direction_index}"
                );
            }
        }
        if let Some(probe_index) = std::env::var("GLIDER_PARITY_PROBE_INDEX")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
        {
            print_trace_window(
                &nlip_trace,
                &ipopt_trace,
                probe_index,
                params.transcription.intervals,
                params.transcription.collocation_degree,
            );
            print_internal_probe_window(
                "requested_probe",
                probe_index,
                &accepted_nlip_solver_snapshots,
                &accepted_ipopt_solver_snapshots,
                &nlip_trace,
                &ipopt_trace,
                &params,
                nlip_options.kappa_d,
                &variable_bounds,
                params.transcription.intervals,
                params.transcription.collocation_degree,
            );
        }
        for (index, (nlip_point, ipopt_point)) in
            nlip_trace.iter().zip(ipopt_trace.iter()).enumerate()
        {
            let primal_gap = log_gap(nlip_point.primal_inf, ipopt_point.primal_inf, 1.0e-12);
            let dual_gap = log_gap(nlip_point.dual_inf, ipopt_point.dual_inf, 1.0e-12);
            let mu_gap = log_gap(nlip_point.mu, ipopt_point.mu, 1.0e-16);
            let objective_gap = log_gap(nlip_point.objective, ipopt_point.objective, 1.0e-12);
            let regularization_gap =
                regularization_log_gap(nlip_point.regularization, ipopt_point.regularization);
            let step_gap = log_gap(nlip_point.step_inf, ipopt_point.step_inf, 1.0e-12);
            let alpha_pr_gap = (nlip_point.alpha_pr - ipopt_point.alpha_pr).abs();
            let alpha_du_gap = (nlip_point.alpha_du - ipopt_point.alpha_du).abs();
            let trial_gap = nlip_point.trial_count.abs_diff(ipopt_point.trial_count);
            let restoration_bridge = restoration_bridge_state_matches(nlip_point, ipopt_point);
            let step_tag_mismatch = nlip_point.step_tag != "-"
                && ipopt_point.step_tag != "-"
                && nlip_point.step_tag != ipopt_point.step_tag;
            let divergence_kind = if step_tag_mismatch {
                Some(DivergenceKind::StepTag)
            } else if !restoration_bridge && alpha_pr_gap > 1.0e-2 {
                Some(DivergenceKind::AlphaPr)
            } else if !restoration_bridge && alpha_du_gap > 1.0e-2 {
                Some(DivergenceKind::AlphaDu)
            } else if regularization_gap > 0.5 {
                Some(DivergenceKind::Regularization)
            } else if objective_gap > 0.3 {
                Some(DivergenceKind::Objective)
            } else if primal_gap > 2.0 {
                Some(DivergenceKind::Primal)
            } else if dual_gap > 2.5 {
                Some(DivergenceKind::Dual)
            } else if mu_gap > 2.5 {
                Some(DivergenceKind::Barrier)
            } else if trial_gap > 3 {
                Some(DivergenceKind::TrialCount)
            } else {
                None
            };
            if let Some(divergence_kind) = divergence_kind {
                println!(
                    "divergence_at_index={} kind={:?} nlip_iter={} ipopt_iter={} objective_gap={:.3e} primal_gap={:.3e} dual_gap={:.3e} mu_gap={:.3e} regularization_gap={:.3e} step_gap={:.3e} alpha_pr_gap={:.3e} alpha_du_gap={:.3e} trial_gap={} nlip_obj={:.6e} ipopt_obj={:.6e} nlip_primal={:.6e} ipopt_primal={:.6e} nlip_dual={:.6e} ipopt_dual={:.6e} nlip_mu={:.6e} ipopt_mu={:.6e} nlip_tf={:.6e} ipopt_tf={:.6e} nlip_x_T={:.6e} ipopt_x_T={:.6e} nlip_reg={} ipopt_reg={} nlip_primal_shift={:.6e} nlip_dual_reg={:.6e} nlip_step={:.6e} ipopt_step={:.6e} nlip_alpha_pr={:.6e} ipopt_alpha_pr={:.6e} nlip_alpha_du={:.6e} ipopt_alpha_du={:.6e} nlip_trials={} ipopt_trials={} nlip_tag={} ipopt_tag={}",
                    index,
                    divergence_kind,
                    nlip_point.iteration,
                    ipopt_point.iteration,
                    objective_gap,
                    primal_gap,
                    dual_gap,
                    mu_gap,
                    regularization_gap,
                    step_gap,
                    alpha_pr_gap,
                    alpha_du_gap,
                    trial_gap,
                    nlip_point.objective,
                    ipopt_point.objective,
                    nlip_point.primal_inf,
                    ipopt_point.primal_inf,
                    nlip_point.dual_inf,
                    ipopt_point.dual_inf,
                    nlip_point.mu,
                    ipopt_point.mu,
                    nlip_point.tf,
                    ipopt_point.tf,
                    nlip_point.terminal_x,
                    ipopt_point.terminal_x,
                    regularization_text(nlip_point.regularization),
                    regularization_text(ipopt_point.regularization),
                    nlip_point.primal_shift,
                    nlip_point.dual_regularization,
                    nlip_point.step_inf,
                    ipopt_point.step_inf,
                    nlip_point.alpha_pr,
                    ipopt_point.alpha_pr,
                    nlip_point.alpha_du,
                    ipopt_point.alpha_du,
                    nlip_point.trial_count,
                    ipopt_point.trial_count,
                    nlip_point.step_tag,
                    ipopt_point.step_tag,
                );
                print_trace_window(
                    &nlip_trace,
                    &ipopt_trace,
                    index,
                    params.transcription.intervals,
                    params.transcription.collocation_degree,
                );
                print_internal_probe_window(
                    "accepted-state divergence",
                    index,
                    &accepted_nlip_solver_snapshots,
                    &accepted_ipopt_solver_snapshots,
                    &nlip_trace,
                    &ipopt_trace,
                    &params,
                    nlip_options.kappa_d,
                    &variable_bounds,
                    params.transcription.intervals,
                    params.transcription.collocation_degree,
                );
                panic!(
                    "glider native-SPRAL NLIP/IPOPT accepted-trace divergence at index {index}: {divergence_kind:?}"
                );
            }
        }
        if nlip_trace.len() != ipopt_trace.len() {
            let compared = nlip_trace.len().min(ipopt_trace.len());
            println!(
                "trace_length_divergence compared={} nlip_steps={} ipopt_steps={}",
                compared,
                nlip_trace.len(),
                ipopt_trace.len()
            );
            if let Some(last) = nlip_trace.last() {
                println!(
                    "last_nlip iter={} obj={:.6e} primal={:.6e} dual={:.6e} mu={:.6e} trials={}",
                    last.iteration,
                    last.objective,
                    last.primal_inf,
                    last.dual_inf,
                    last.mu,
                    last.trial_count,
                );
            }
            if let Some(next) = ipopt_trace.get(compared) {
                println!(
                    "next_ipopt iter={} obj={:.6e} primal={:.6e} dual={:.6e} mu={:.6e} trials={}",
                    next.iteration,
                    next.objective,
                    next.primal_inf,
                    next.dual_inf,
                    next.mu,
                    next.trial_count,
                );
            }
            panic!(
                "glider native-SPRAL NLIP/IPOPT accepted-trace length mismatch: nlip_steps={} ipopt_steps={}",
                nlip_trace.len(),
                ipopt_trace.len()
            );
        }
        println!("no early divergence detected in compared accepted traces");
    }

    #[test]
    #[ignore = "manual profiling helper"]
    fn profile_reduced_direct_collocation_symbolic_setup() {
        let family = optimal_control::CollocationFamily::RadauIIA;
        let ocp = model(DirectCollocation {
            intervals: 6,
            order: 2,
            family,
            time_grid: Default::default(),
        });
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
        let ocp = model(DirectCollocation {
            intervals: DEFAULT_INTERVALS,
            order: DEFAULT_COLLOCATION_DEGREE,
            family,
            time_grid: Default::default(),
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
            !matches!(option_env!("OPTIVIBRE_OPT_LEVEL"), Some("0")),
            "manual reduced glider Jacobian policy checks must be run with an optimized binary; current opt-level=0\n\ntry:\n  cargo test -p optimal_control_problems --release reduced_direct_collocation_jacobian_policies_stay_clean -- --ignored"
        );
    }

    fn require_release_mode_for_manual_hessian_checks() {
        assert!(
            !matches!(option_env!("OPTIVIBRE_OPT_LEVEL"), Some("0")),
            "manual glider Hessian policy checks must be run with an optimized binary; current opt-level=0\n\ntry:\n  cargo test -p optimal_control_problems --release direct_collocation_hessian_policies_stay_clean -- --ignored --nocapture"
        );
    }

    fn dc_decision_layout_names() -> Vec<String> {
        let x_len = <State<SX> as optimization::Vectorize<SX>>::LEN;
        let u_len = <Control<SX> as optimization::Vectorize<SX>>::LEN;
        let decision_len = (DEFAULT_INTERVALS + 1) * (x_len + u_len)
            + DEFAULT_INTERVALS * DEFAULT_COLLOCATION_DEGREE * (x_len + 2 * u_len)
            + 1;
        (0..decision_len)
            .map(|index| format!("w[{index}]"))
            .collect()
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
        let runtime = ms_runtime(&params);

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
            let compiled = model(MultipleShooting {
                intervals: N,
                rk4_substeps: RK4_SUBSTEPS,
            })
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
        let runtime = dc_runtime(&params);

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
            let compiled = model(DirectCollocation {
                intervals: N,
                order: K,
                family: optimal_control::CollocationFamily::RadauIIA,
                time_grid: Default::default(),
            })
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

    #[test]
    #[ignore = "manual direct reproducer for native SPRAL webapp glider solve crash"]
    fn reproduce_native_spral_webapp_glider_solve() {
        if NativeSpral::load().is_err() {
            eprintln!(
                "skipping native SPRAL webapp-style glider solve reproducer: library unavailable"
            );
            return;
        }
        let mut values = BTreeMap::new();
        values.insert("solver_method".to_string(), 1.0);
        values.insert("solver_nlip_linear_solver".to_string(), 1.0);
        values.insert("solver_max_iters".to_string(), 200.0);
        let _ = crate::solve_problem(crate::ProblemId::OptimalDistanceGlider, &values);
    }
}
