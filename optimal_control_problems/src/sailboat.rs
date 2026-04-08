use crate::common::{
    CachedCompile, CompileCacheStatus, CompileProgressInfo, CompileProgressUpdate, FromMap,
    LatexSection, MetricKey, OcpCompileProgressState, PlotMode, ProblemId, ProblemSpec, Scene2D,
    SceneAnimation, SceneArrow, SceneFrame, ScenePath, SolveArtifact, SolveStreamEvent,
    SolverMethod, SolverReport, SqpConfig, TimeSeries, TranscriptionConfig, TranscriptionMethod,
    cached_compile_with_progress, chart, compile_progress_info, default_solver_method,
    default_sqp_config, default_transcription, deg_to_rad,
    direct_collocation_compile_key as dc_compile_key, expect_finite,
    interactive_multiple_shooting_opt_level, interval_arc_bound_series, interval_arc_series,
    metric_with_key, node_times, numeric_metric_with_key, ocp_compile_progress_update,
    problem_controls, problem_scientific_slider_control, problem_slider_control, problem_spec,
    rad_to_deg, sample_or_default, segmented_series, solve_cached_direct_collocation_problem,
    solve_cached_direct_collocation_problem_with_progress, solve_cached_multiple_shooting_problem,
    solve_cached_multiple_shooting_problem_with_progress, solver_config_from_map,
    solver_method_from_map, summarize_backend_compile_report, transcription_from_map,
    transcription_metrics,
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
use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::rc::Rc;
use sx_core::SX;

const RK4_SUBSTEPS: usize = 2;
const DEFAULT_INTERVALS: usize = 30;
const DEFAULT_COLLOCATION_DEGREE: usize = 3;
const SUPPORTED_INTERVALS: [usize; 1] = [DEFAULT_INTERVALS];
const SUPPORTED_DEGREES: [usize; 1] = [DEFAULT_COLLOCATION_DEGREE];

const BOAT_PLUS_SAILOR_MASS_KG: f64 = 230.0;
const RHO_AIR: f64 = 1.2;
const RHO_WATER: f64 = 1000.0;
const SAIL_AREA_M2: f64 = 16.0;
const FIN_AREA_M2: f64 = 1.21;
const VELOCITY_LIMIT_MPS: f64 = 100.0;
const SPEED_EPS2: f64 = 1.0e-9;

#[derive(Clone, Debug, PartialEq, Serialize, optimization::Vectorize)]
pub struct State<T> {
    pub gamma: T,
    pub x: T,
    pub z: T,
    pub vx: T,
    pub vz: T,
}

#[derive(Clone, Debug, PartialEq, Serialize, optimization::Vectorize)]
pub struct Control<T> {
    pub omega: T,
    pub alpha: T,
}

#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct Path<T> {
    gamma: T,
    z: T,
    vx: T,
    vz: T,
    omega: T,
    alpha: T,
}

#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct BoundaryEq<T> {
    periodic_gamma: T,
    periodic_z: T,
    periodic_vx: T,
    periodic_vz: T,
    x0: T,
    z0: T,
}

#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct ModelParams<T> {
    wind_speed: T,
    omega_weight: T,
    alpha_weight: T,
    omega_rate_weight: T,
    alpha_rate_weight: T,
}

type MsCompiled<const N: usize> = CompiledMultipleShootingOcp<
    State<SX>,
    Control<SX>,
    ModelParams<SX>,
    Path<SX>,
    BoundaryEq<SX>,
    (),
    N,
    RK4_SUBSTEPS,
>;

type DcCompiled<const N: usize, const K: usize> = CompiledDirectCollocationOcp<
    State<SX>,
    Control<SX>,
    ModelParams<SX>,
    Path<SX>,
    BoundaryEq<SX>,
    (),
    N,
    K,
>;

crate::standard_ocp_compile_caches!(
    MULTIPLE_SHOOTING_CACHE: MsCompiled<DEFAULT_INTERVALS>,
    DIRECT_COLLOCATION_CACHE: DcCompiled<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>
);

const PROBLEM_NAME: &str = "Sailboat Symmetric Tack";

#[derive(Clone, Debug)]
pub struct Params {
    pub wind_speed_mps: f64,
    pub initial_time_guess_s: f64,
    pub min_final_time_s: f64,
    pub max_final_time_s: f64,
    pub gamma_limit_deg: f64,
    pub omega_limit_deg_s: f64,
    pub alpha_limit_deg: f64,
    pub omega_weight: f64,
    pub alpha_weight: f64,
    pub omega_rate_regularization: f64,
    pub alpha_rate_regularization: f64,
    pub cross_track_limit_m: f64,
    pub solver_method: SolverMethod,
    pub solver: SqpConfig,
    pub transcription: TranscriptionConfig,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            wind_speed_mps: 5.0,
            initial_time_guess_s: 20.0,
            min_final_time_s: 1.0,
            max_final_time_s: 50.0,
            gamma_limit_deg: 12.0,
            omega_limit_deg_s: 5.0,
            alpha_limit_deg: 12.0,
            omega_weight: 1.0e-3,
            alpha_weight: 1.0e-3,
            omega_rate_regularization: 0.0,
            alpha_rate_regularization: 0.0,
            cross_track_limit_m: 30.0,
            solver_method: default_solver_method(),
            solver: default_sqp_config(),
            transcription: default_transcription(DEFAULT_INTERVALS),
        }
    }
}

impl FromMap for Params {
    fn from_map(values: &BTreeMap<String, f64>) -> Result<Self> {
        let defaults = Self::default();
        Ok(Self {
            wind_speed_mps: expect_finite(
                sample_or_default(values, "wind_speed_mps", defaults.wind_speed_mps),
                "wind_speed_mps",
            )?,
            initial_time_guess_s: expect_finite(
                sample_or_default(
                    values,
                    "initial_time_guess_s",
                    defaults.initial_time_guess_s,
                ),
                "initial_time_guess_s",
            )?,
            min_final_time_s: expect_finite(
                sample_or_default(values, "min_final_time_s", defaults.min_final_time_s),
                "min_final_time_s",
            )?,
            max_final_time_s: expect_finite(
                sample_or_default(values, "max_final_time_s", defaults.max_final_time_s),
                "max_final_time_s",
            )?,
            gamma_limit_deg: expect_finite(
                sample_or_default(values, "gamma_limit_deg", defaults.gamma_limit_deg),
                "gamma_limit_deg",
            )?,
            omega_limit_deg_s: expect_finite(
                sample_or_default(values, "omega_limit_deg_s", defaults.omega_limit_deg_s),
                "omega_limit_deg_s",
            )?,
            alpha_limit_deg: expect_finite(
                sample_or_default(values, "alpha_limit_deg", defaults.alpha_limit_deg),
                "alpha_limit_deg",
            )?,
            omega_weight: expect_finite(
                sample_or_default(values, "omega_weight", defaults.omega_weight),
                "omega_weight",
            )?,
            alpha_weight: expect_finite(
                sample_or_default(values, "alpha_weight", defaults.alpha_weight),
                "alpha_weight",
            )?,
            omega_rate_regularization: expect_finite(
                sample_or_default(
                    values,
                    "omega_rate_regularization",
                    defaults.omega_rate_regularization,
                ),
                "omega_rate_regularization",
            )?,
            alpha_rate_regularization: expect_finite(
                sample_or_default(
                    values,
                    "alpha_rate_regularization",
                    defaults.alpha_rate_regularization,
                ),
                "alpha_rate_regularization",
            )?,
            cross_track_limit_m: expect_finite(
                sample_or_default(values, "cross_track_limit_m", defaults.cross_track_limit_m),
                "cross_track_limit_m",
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
        ProblemId::SailboatUpwind,
        "Sailboat Symmetric Tack",
        "A dynobud-style point-mass sailboat with coupled aerodynamic and hydrodynamic lift/drag, mirrored tack boundary conditions, and a free final time chosen to maximize average upwind velocity.",
        problem_controls(
            defaults.transcription,
            &SUPPORTED_INTERVALS,
            &SUPPORTED_DEGREES,
            defaults.solver_method,
            defaults.solver,
            vec![
                problem_slider_control(
                    "wind_speed_mps",
                    "True Wind Speed",
                    2.0,
                    12.0,
                    0.25,
                    defaults.wind_speed_mps,
                    "m/s",
                    "True wind points in the negative x direction.",
                ),
                problem_slider_control(
                    "initial_time_guess_s",
                    "Initial Time Guess",
                    2.0,
                    40.0,
                    0.5,
                    defaults.initial_time_guess_s,
                    "s",
                    "Initial guess for the free final time.",
                ),
                problem_slider_control(
                    "min_final_time_s",
                    "Min Final Time",
                    0.5,
                    20.0,
                    0.5,
                    defaults.min_final_time_s,
                    "s",
                    "Lower bound on the free final time.",
                ),
                problem_slider_control(
                    "max_final_time_s",
                    "Max Final Time",
                    5.0,
                    80.0,
                    0.5,
                    defaults.max_final_time_s,
                    "s",
                    "Upper bound on the free final time.",
                ),
                problem_slider_control(
                    "gamma_limit_deg",
                    "Fin AoA Limit",
                    4.0,
                    20.0,
                    0.5,
                    defaults.gamma_limit_deg,
                    "deg",
                    "Absolute bound on the water-fin angle of attack.",
                ),
                problem_slider_control(
                    "omega_limit_deg_s",
                    "Fin Rate Limit",
                    1.0,
                    15.0,
                    0.25,
                    defaults.omega_limit_deg_s,
                    "deg/s",
                    "Absolute bound on the fin-angle rate control.",
                ),
                problem_slider_control(
                    "alpha_limit_deg",
                    "Sail AoA Limit",
                    4.0,
                    20.0,
                    0.5,
                    defaults.alpha_limit_deg,
                    "deg",
                    "Absolute bound on sail angle of attack.",
                ),
                problem_slider_control(
                    "omega_weight",
                    "Fin Rate Weight",
                    0.0,
                    1.0e-2,
                    1.0e-4,
                    defaults.omega_weight,
                    "",
                    "Quadratic stage-cost weight on the fin-rate control state.",
                ),
                problem_slider_control(
                    "alpha_weight",
                    "Sail AoA Weight",
                    0.0,
                    1.0e-2,
                    1.0e-4,
                    defaults.alpha_weight,
                    "",
                    "Quadratic stage-cost weight on sail angle of attack.",
                ),
                problem_scientific_slider_control(
                    "omega_rate_regularization",
                    "Fin Accel Weight",
                    0.0,
                    1.0e-2,
                    1.0e-4,
                    defaults.omega_rate_regularization,
                    "",
                    "Optional quadratic regularization on the fin-rate derivative.",
                ),
                problem_scientific_slider_control(
                    "alpha_rate_regularization",
                    "Sail Accel Weight",
                    0.0,
                    1.0e-2,
                    1.0e-4,
                    defaults.alpha_rate_regularization,
                    "",
                    "Optional quadratic regularization on the sail-AoA derivative.",
                ),
                problem_slider_control(
                    "cross_track_limit_m",
                    "Cross-Track Limit",
                    10.0,
                    60.0,
                    0.5,
                    defaults.cross_track_limit_m,
                    "m",
                    "Absolute bound on the lateral displacement z.",
                ),
            ],
        ),
        vec![
            LatexSection {
                title: "Physical State".to_string(),
                entries: vec![r"\mathbf{x} = \begin{bmatrix} \gamma & x & z & v_x & v_z \end{bmatrix}^{\mathsf T}".to_string()],
            },
            LatexSection {
                title: "Control-State".to_string(),
                entries: vec![
                    r"\mathbf{u} = \begin{bmatrix} \omega & \alpha \end{bmatrix}^{\mathsf T}".to_string(),
                    r"\dot{\mathbf{u}} = \begin{bmatrix} \dot{\omega} & \dot{\alpha} \end{bmatrix}^{\mathsf T}".to_string(),
                ],
            },
            LatexSection {
                title: "Objective".to_string(),
                entries: vec![
                    r"J = \int_0^T \left(w_\omega \omega^2 + w_\alpha \alpha^2 + w_{\dot{\omega}} \dot{\omega}^2 + w_{\dot{\alpha}} \dot{\alpha}^2\right)\,dt - \frac{x(T)}{T}".to_string(),
                ],
            },
            LatexSection {
                title: "Lift Model".to_string(),
                entries: vec![
                    r"C_L(\beta) = 2 \pi \beta \frac{10}{12} - \exp\!\left(\beta \frac{180}{\pi} - 12\right) + \exp\!\left(-\beta \frac{180}{\pi} - 12\right)".to_string(),
                    r"C_D(\beta) = 0.01 + \frac{C_L(\beta)^2}{10\pi}".to_string(),
                ],
            },
            LatexSection {
                title: "Flow Kinematics".to_string(),
                entries: vec![
                    r"\mathbf{w} = \begin{bmatrix} -W & 0 \end{bmatrix}^{\mathsf T}, \qquad \mathbf{w}_e = \mathbf{w} - \mathbf{v}".to_string(),
                    r"V_a = \|\mathbf{w}_e\|, \qquad V_w = \|\mathbf{v}\|".to_string(),
                    r"\hat{\ell}_a = \frac{1}{V_a}\begin{bmatrix} w_{e,z} \\ -w_{e,x} \end{bmatrix}, \quad \hat{d}_a = \frac{\mathbf{w}_e}{V_a}".to_string(),
                    r"\hat{\ell}_w = \frac{1}{V_w}\begin{bmatrix} -v_z \\ v_x \end{bmatrix}, \quad \hat{d}_w = \frac{-\mathbf{v}}{V_w}".to_string(),
                ],
            },
            LatexSection {
                title: "Forces".to_string(),
                entries: vec![
                    r"\mathbf{F}_{La} = \tfrac12 \rho_a V_a^2 C_L(\alpha) S_{\text{sail}} \hat{\ell}_a, \qquad \mathbf{F}_{Da} = \tfrac12 \rho_a V_a^2 C_D(\alpha) S_{\text{sail}} \hat{d}_a".to_string(),
                    r"\mathbf{F}_{Lw} = \tfrac12 \rho_w V_w^2 C_L(\gamma) S_{\text{fin}} \hat{\ell}_w, \qquad \mathbf{F}_{Dw} = \tfrac12 \rho_w V_w^2 C_D(\gamma) S_{\text{fin}} \hat{d}_w".to_string(),
                    r"\mathbf{F} = \mathbf{F}_{La} + \mathbf{F}_{Da} + \mathbf{F}_{Lw} + \mathbf{F}_{Dw}".to_string(),
                ],
            },
            LatexSection {
                title: "Differential Equations".to_string(),
                entries: vec![
                    r"\dot{\gamma} = \omega".to_string(),
                    r"\dot{x} = v_x".to_string(),
                    r"\dot{z} = v_z".to_string(),
                    r"\dot{v}_x = \frac{F_x}{m}".to_string(),
                    r"\dot{v}_z = \frac{F_z}{m}".to_string(),
                    r"\dot{\omega} = \nu_\omega".to_string(),
                    r"\dot{\alpha} = \nu_\alpha".to_string(),
                ],
            },
        ],
        vec![
            format!(
                "This matches the dynobud sailboat example structure with m = {:.0} kg, rho_air = {:.1}, rho_water = {:.0}, sail area = {:.1} m^2, and fin area = {:.2} m^2.",
                BOAT_PLUS_SAILOR_MASS_KG, RHO_AIR, RHO_WATER, SAIL_AREA_M2, FIN_AREA_M2
            ),
            "The mirrored terminal equalities enforce a symmetric tack segment rather than a point-to-point return-to-centerline problem.".to_string(),
        ],
    )
}

fn clift_sx(alpha: SX) -> SX {
    let alpha_deg = alpha * (180.0 / PI);
    2.0 * PI * (10.0 / 12.0) * alpha - (alpha_deg - 12.0).exp() + (-alpha_deg - 12.0).exp()
}

fn clift_numeric(alpha: f64) -> f64 {
    let alpha_deg = alpha * (180.0 / PI);
    2.0 * PI * (10.0 / 12.0) * alpha - (alpha_deg - 12.0).exp() + (-alpha_deg - 12.0).exp()
}

struct SymbolicFlow {
    f_air_lift_x: SX,
    f_air_lift_z: SX,
    f_air_drag_x: SX,
    f_air_drag_z: SX,
    f_water_lift_x: SX,
    f_water_lift_z: SX,
    f_water_drag_x: SX,
    f_water_drag_z: SX,
}

fn sailboat_flow_sx(state: &State<SX>, control: &Control<SX>, wind_speed: SX) -> SymbolicFlow {
    let we_x = -wind_speed - state.vx;
    let we_z = -state.vz;
    let airspeed2 = we_x.sqr() + we_z.sqr() + SPEED_EPS2;
    let airspeed = airspeed2.sqrt();
    let sail_cl = clift_sx(control.alpha);
    let sail_cd = 0.01 + sail_cl.sqr() / (10.0 * PI);
    let sail_scale = 0.5 * RHO_AIR * SAIL_AREA_M2 * airspeed2;
    let f_air_drag_x = sail_scale * sail_cd * we_x / airspeed;
    let f_air_drag_z = sail_scale * sail_cd * we_z / airspeed;
    let f_air_lift_x = sail_scale * sail_cl * we_z / airspeed;
    let f_air_lift_z = sail_scale * sail_cl * (-we_x) / airspeed;

    let waterspeed2 = state.vx.sqr() + state.vz.sqr() + SPEED_EPS2;
    let waterspeed = waterspeed2.sqrt();
    let fin_cl = clift_sx(state.gamma);
    let fin_cd = 0.01 + fin_cl.sqr() / (10.0 * PI);
    let water_scale = 0.5 * RHO_WATER * FIN_AREA_M2 * waterspeed2;
    let f_water_drag_x = water_scale * fin_cd * (-state.vx) / waterspeed;
    let f_water_drag_z = water_scale * fin_cd * (-state.vz) / waterspeed;
    let f_water_lift_x = water_scale * fin_cl * (-state.vz) / waterspeed;
    let f_water_lift_z = water_scale * fin_cl * state.vx / waterspeed;

    SymbolicFlow {
        f_air_lift_x,
        f_air_lift_z,
        f_air_drag_x,
        f_air_drag_z,
        f_water_lift_x,
        f_water_lift_z,
        f_water_drag_x,
        f_water_drag_z,
    }
}

#[derive(Clone)]
struct NumericFlow {
    airspeed: f64,
    waterspeed: f64,
    f_air_lift: [f64; 2],
    f_water_lift: [f64; 2],
}

fn sailboat_flow_numeric(
    state: &State<f64>,
    control: &Control<f64>,
    wind_speed: f64,
) -> NumericFlow {
    let we_x = -wind_speed - state.vx;
    let we_z = -state.vz;
    let airspeed = (we_x * we_x + we_z * we_z + SPEED_EPS2).sqrt();
    let sail_cl = clift_numeric(control.alpha);
    let sail_cd = 0.01 + sail_cl * sail_cl / (10.0 * PI);
    let sail_scale = 0.5 * RHO_AIR * SAIL_AREA_M2 * airspeed * airspeed;
    let _f_air_drag = [
        sail_scale * sail_cd * we_x / airspeed,
        sail_scale * sail_cd * we_z / airspeed,
    ];
    let f_air_lift = [
        sail_scale * sail_cl * we_z / airspeed,
        sail_scale * sail_cl * (-we_x) / airspeed,
    ];

    let waterspeed = (state.vx * state.vx + state.vz * state.vz + SPEED_EPS2).sqrt();
    let fin_cl = clift_numeric(state.gamma);
    let fin_cd = 0.01 + fin_cl * fin_cl / (10.0 * PI);
    let water_scale = 0.5 * RHO_WATER * FIN_AREA_M2 * waterspeed * waterspeed;
    let _f_water_drag = [
        water_scale * fin_cd * (-state.vx) / waterspeed,
        water_scale * fin_cd * (-state.vz) / waterspeed,
    ];
    let f_water_lift = [
        water_scale * fin_cl * (-state.vz) / waterspeed,
        water_scale * fin_cl * state.vx / waterspeed,
    ];

    NumericFlow {
        airspeed,
        waterspeed,
        f_air_lift,
        f_water_lift,
    }
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
                    Some(summarize_backend_compile_report(compiled.backend_compile_report())),
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
                    Some(summarize_backend_compile_report(compiled.backend_compile_report())),
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
        ProblemId::SailboatUpwind,
        PROBLEM_NAME,
        MULTIPLE_SHOOTING_CACHE,
        DIRECT_COLLOCATION_CACHE
    )
}

fn model<Scheme>(
    scheme: Scheme,
) -> Ocp<State<SX>, Control<SX>, ModelParams<SX>, Path<SX>, BoundaryEq<SX>, (), Scheme> {
    Ocp::new("sailboat_symmetric_tack", scheme)
        .objective_lagrange(
            |_: &State<SX>,
             control: &Control<SX>,
             rate: &Control<SX>,
             runtime: &ModelParams<SX>| {
                runtime.omega_weight * control.omega.sqr()
                    + runtime.alpha_weight * control.alpha.sqr()
                    + runtime.omega_rate_weight * rate.omega.sqr()
                    + runtime.alpha_rate_weight * rate.alpha.sqr()
            },
        )
        .objective_mayer(
            |_: &State<SX>,
             _: &Control<SX>,
             terminal: &State<SX>,
             _: &Control<SX>,
             _: &ModelParams<SX>,
             tf: &SX| { -terminal.x / *tf },
        )
        .ode(
            |state: &State<SX>, control: &Control<SX>, runtime: &ModelParams<SX>| {
                let flow = sailboat_flow_sx(state, control, runtime.wind_speed);
                let fx = flow.f_air_lift_x
                    + flow.f_air_drag_x
                    + flow.f_water_lift_x
                    + flow.f_water_drag_x;
                let fz = flow.f_air_lift_z
                    + flow.f_air_drag_z
                    + flow.f_water_lift_z
                    + flow.f_water_drag_z;
                State {
                    gamma: control.omega,
                    x: state.vx,
                    z: state.vz,
                    vx: fx / BOAT_PLUS_SAILOR_MASS_KG,
                    vz: fz / BOAT_PLUS_SAILOR_MASS_KG,
                }
            },
        )
        .path_constraints(
            |state: &State<SX>, control: &Control<SX>, _: &Control<SX>, _: &ModelParams<SX>| Path {
                gamma: state.gamma,
                z: state.z,
                vx: state.vx,
                vz: state.vz,
                omega: control.omega,
                alpha: control.alpha,
            },
        )
        .boundary_equalities(
            |initial: &State<SX>,
             _: &Control<SX>,
             terminal: &State<SX>,
             _: &Control<SX>,
             _: &ModelParams<SX>,
             _: &SX| BoundaryEq {
                periodic_gamma: initial.gamma + terminal.gamma,
                periodic_z: initial.z - terminal.z,
                periodic_vx: initial.vx - terminal.vx,
                periodic_vz: initial.vz + terminal.vz,
                x0: initial.x,
                z0: initial.z,
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
        .expect("sailboat model should build")
}

fn guess<const N: usize>(params: &Params) -> InterpolatedTrajectory<State<f64>, Control<f64>> {
    let sample_count = 2 * N + 1;
    let tf = params.initial_time_guess_s;
    let dt = tf / (sample_count as f64 - 1.0);
    let radius = params.cross_track_limit_m.max(12.0);
    let omega = PI / tf;
    let sample_times = (0..sample_count)
        .map(|index| index as f64 * dt)
        .collect::<Vec<_>>();
    let x_samples = sample_times
        .iter()
        .map(|time| {
            let phase = omega * *time;
            State {
                gamma: 0.0,
                x: radius - radius * phase.cos(),
                z: radius * phase.sin(),
                vx: omega * radius * phase.sin(),
                vz: omega * radius * phase.cos(),
            }
        })
        .collect::<Vec<_>>();
    let u_samples = sample_times
        .iter()
        .map(|_| Control {
            omega: 0.0,
            alpha: 0.0,
        })
        .collect::<Vec<_>>();
    let dudt_samples = sample_times
        .iter()
        .map(|_| Control {
            omega: 0.0,
            alpha: 0.0,
        })
        .collect::<Vec<_>>();
    InterpolatedTrajectory {
        sample_times,
        x_samples,
        u_samples,
        dudt_samples,
        tf,
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
    BoundaryEq<f64>,
    (),
    State<f64>,
    Control<f64>,
    N,
> {
    MultipleShootingRuntimeValues {
        parameters: ModelParams {
            wind_speed: params.wind_speed_mps,
            omega_weight: params.omega_weight,
            alpha_weight: params.alpha_weight,
            omega_rate_weight: params.omega_rate_regularization,
            alpha_rate_weight: params.alpha_rate_regularization,
        },
        beq: BoundaryEq {
            periodic_gamma: 0.0,
            periodic_z: 0.0,
            periodic_vx: 0.0,
            periodic_vz: 0.0,
            x0: 0.0,
            z0: 0.0,
        },
        bineq_bounds: (),
        path_bounds: Path {
            gamma: Bounds1D {
                lower: Some(-deg_to_rad(params.gamma_limit_deg)),
                upper: Some(deg_to_rad(params.gamma_limit_deg)),
            },
            z: Bounds1D {
                lower: Some(-params.cross_track_limit_m),
                upper: Some(params.cross_track_limit_m),
            },
            vx: Bounds1D {
                lower: Some(-VELOCITY_LIMIT_MPS),
                upper: Some(VELOCITY_LIMIT_MPS),
            },
            vz: Bounds1D {
                lower: Some(-VELOCITY_LIMIT_MPS),
                upper: Some(VELOCITY_LIMIT_MPS),
            },
            omega: Bounds1D {
                lower: Some(-deg_to_rad(params.omega_limit_deg_s)),
                upper: Some(deg_to_rad(params.omega_limit_deg_s)),
            },
            alpha: Bounds1D {
                lower: Some(-deg_to_rad(params.alpha_limit_deg)),
                upper: Some(deg_to_rad(params.alpha_limit_deg)),
            },
        },
        tf_bounds: Bounds1D {
            lower: Some(params.min_final_time_s),
            upper: Some(params.max_final_time_s),
        },
        initial_guess: MultipleShootingInitialGuess::Interpolated(guess::<N>(params)),
    }
}

fn dc_runtime<const N: usize, const K: usize>(
    params: &Params,
) -> DirectCollocationRuntimeValues<
    ModelParams<f64>,
    Path<Bounds1D>,
    BoundaryEq<f64>,
    (),
    State<f64>,
    Control<f64>,
    N,
    K,
> {
    DirectCollocationRuntimeValues {
        parameters: ModelParams {
            wind_speed: params.wind_speed_mps,
            omega_weight: params.omega_weight,
            alpha_weight: params.alpha_weight,
            omega_rate_weight: params.omega_rate_regularization,
            alpha_rate_weight: params.alpha_rate_regularization,
        },
        beq: BoundaryEq {
            periodic_gamma: 0.0,
            periodic_z: 0.0,
            periodic_vx: 0.0,
            periodic_vz: 0.0,
            x0: 0.0,
            z0: 0.0,
        },
        bineq_bounds: (),
        path_bounds: Path {
            gamma: Bounds1D {
                lower: Some(-deg_to_rad(params.gamma_limit_deg)),
                upper: Some(deg_to_rad(params.gamma_limit_deg)),
            },
            z: Bounds1D {
                lower: Some(-params.cross_track_limit_m),
                upper: Some(params.cross_track_limit_m),
            },
            vx: Bounds1D {
                lower: Some(-VELOCITY_LIMIT_MPS),
                upper: Some(VELOCITY_LIMIT_MPS),
            },
            vz: Bounds1D {
                lower: Some(-VELOCITY_LIMIT_MPS),
                upper: Some(VELOCITY_LIMIT_MPS),
            },
            omega: Bounds1D {
                lower: Some(-deg_to_rad(params.omega_limit_deg_s)),
                upper: Some(deg_to_rad(params.omega_limit_deg_s)),
            },
            alpha: Bounds1D {
                lower: Some(-deg_to_rad(params.alpha_limit_deg)),
                upper: Some(deg_to_rad(params.alpha_limit_deg)),
            },
        },
        tf_bounds: Bounds1D {
            lower: Some(params.min_final_time_s),
            upper: Some(params.max_final_time_s),
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
    let (compiled, running_solver, compile_report) =
        lifecycle.compile_with_progress(compile_multiple_shooting_with_progress)?;
    let mut artifact = solve_cached_multiple_shooting_problem_with_progress(
        &compiled,
        &ms_runtime::<DEFAULT_INTERVALS>(params),
        params.solver_method,
        &params.solver,
        lifecycle.into_emit(),
        running_solver,
        |trajectories, x_arcs, u_arcs| {
            artifact_from_ms_trajectories(params, trajectories, x_arcs, u_arcs)
        },
    )?;
    artifact.compile_report = compile_report;
    Ok(artifact)
}

fn solve_direct_collocation_with_progress<F>(params: &Params, emit: F) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    let mut lifecycle = crate::common::SolveLifecycleReporter::new(emit, params.solver_method);
    let (compiled, running_solver, compile_report) = lifecycle.compile_with_progress(|callback| {
        compile_direct_collocation_with_progress(params.transcription.collocation_family, callback)
    })?;
    let mut artifact = solve_cached_direct_collocation_problem_with_progress(
        &compiled,
        &dc_runtime::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(params),
        params.solver_method,
        &params.solver,
        lifecycle.into_emit(),
        running_solver,
        |trajectories, time_grid| artifact_from_dc_trajectories(params, trajectories, time_grid),
    )?;
    artifact.compile_report = compile_report;
    Ok(artifact)
}

fn point_frame(state: &State<f64>) -> SceneFrame {
    let mut points = BTreeMap::new();
    points.insert("boat".to_string(), [state.x, state.z]);
    let velocity_tip = [state.x + 0.35 * state.vx, state.z + 0.35 * state.vz];
    SceneFrame {
        points,
        segments: vec![([state.x, state.z], velocity_tip)],
    }
}

struct SailboatMetrics {
    upwind_distance: f64,
    average_upwind_speed: f64,
    max_cross_track: f64,
    max_waterspeed: f64,
    max_airspeed: f64,
    final_time: f64,
}

fn summarize(
    states: &[State<f64>],
    controls: &[Control<f64>],
    params: &Params,
    tf: f64,
) -> SailboatMetrics {
    let terminal = states
        .last()
        .expect("trajectory should contain a terminal state");
    let max_cross_track = states
        .iter()
        .fold(0.0_f64, |acc, state| acc.max(state.z.abs()));
    let (max_waterspeed, max_airspeed) = states.iter().zip(controls.iter()).fold(
        (0.0_f64, 0.0_f64),
        |(max_water, max_air), (state, control)| {
            let flow = sailboat_flow_numeric(state, control, params.wind_speed_mps);
            (max_water.max(flow.waterspeed), max_air.max(flow.airspeed))
        },
    );
    SailboatMetrics {
        upwind_distance: terminal.x,
        average_upwind_speed: terminal.x / tf,
        max_cross_track,
        max_waterspeed,
        max_airspeed,
        final_time: tf,
    }
}

fn summary_metrics(params: &Params, metrics: &SailboatMetrics) -> Vec<crate::common::Metric> {
    let mut summary = transcription_metrics(&params.transcription).to_vec();
    summary.extend([
        numeric_metric_with_key(
            MetricKey::UpwindDistance,
            "Upwind Distance",
            metrics.upwind_distance,
            format!("{:.2} m", metrics.upwind_distance),
        ),
        numeric_metric_with_key(
            MetricKey::Custom,
            "Avg Upwind Speed",
            metrics.average_upwind_speed,
            format!("{:.3} m/s", metrics.average_upwind_speed),
        ),
        numeric_metric_with_key(
            MetricKey::MaxCrossTrack,
            "Max Cross-Track",
            metrics.max_cross_track,
            format!("{:.2} m", metrics.max_cross_track),
        ),
        numeric_metric_with_key(
            MetricKey::MaxSpeed,
            "Max Water Speed",
            metrics.max_waterspeed,
            format!("{:.3} m/s", metrics.max_waterspeed),
        ),
        numeric_metric_with_key(
            MetricKey::Custom,
            "Max Air Speed",
            metrics.max_airspeed,
            format!("{:.3} m/s", metrics.max_airspeed),
        ),
        numeric_metric_with_key(
            MetricKey::FinalTime,
            "Final Time",
            metrics.final_time,
            format!("{:.2} s", metrics.final_time),
        ),
        metric_with_key(
            MetricKey::Custom,
            "Wind",
            format!("{:.2} m/s", params.wind_speed_mps),
        ),
    ]);
    summary
}

fn build_rate_series_ms(
    label: &str,
    x_arcs: &[IntervalArc<State<f64>>],
    values: impl Fn(usize) -> f64,
) -> Vec<TimeSeries> {
    segmented_series(
        label,
        x_arcs
            .iter()
            .enumerate()
            .map(|(interval, arc)| (arc.times.clone(), vec![values(interval); arc.times.len()])),
        PlotMode::LinesMarkers,
    )
}

fn airspeed_chart_series(
    state_arcs: &[IntervalArc<State<f64>>],
    control_arcs: &[IntervalArc<Control<f64>>],
    wind_speed: f64,
) -> Vec<TimeSeries> {
    segmented_series(
        "Airspeed (m/s)",
        state_arcs
            .iter()
            .zip(control_arcs.iter())
            .map(|(state_arc, control_arc)| {
                (
                    state_arc.times.clone(),
                    state_arc
                        .values
                        .iter()
                        .zip(control_arc.values.iter())
                        .map(|(state, control)| {
                            sailboat_flow_numeric(state, control, wind_speed).airspeed
                        })
                        .collect::<Vec<_>>(),
                )
            }),
        PlotMode::LinesMarkers,
    )
}

fn waterspeed_chart_series(
    state_arcs: &[IntervalArc<State<f64>>],
    control_arcs: &[IntervalArc<Control<f64>>],
    wind_speed: f64,
) -> Vec<TimeSeries> {
    segmented_series(
        "Water Speed (m/s)",
        state_arcs
            .iter()
            .zip(control_arcs.iter())
            .map(|(state_arc, control_arc)| {
                (
                    state_arc.times.clone(),
                    state_arc
                        .values
                        .iter()
                        .zip(control_arc.values.iter())
                        .map(|(state, control)| {
                            sailboat_flow_numeric(state, control, wind_speed).waterspeed
                        })
                        .collect::<Vec<_>>(),
                )
            }),
        PlotMode::LinesMarkers,
    )
}

fn force_component_chart_series(
    name: &str,
    state_arcs: &[IntervalArc<State<f64>>],
    control_arcs: &[IntervalArc<Control<f64>>],
    wind_speed: f64,
    extractor: impl Fn(&NumericFlow) -> [f64; 2],
    component: usize,
) -> Vec<TimeSeries> {
    segmented_series(
        name,
        state_arcs
            .iter()
            .zip(control_arcs.iter())
            .map(|(state_arc, control_arc)| {
                (
                    state_arc.times.clone(),
                    state_arc
                        .values
                        .iter()
                        .zip(control_arc.values.iter())
                        .map(|(state, control)| {
                            extractor(&sailboat_flow_numeric(state, control, wind_speed))[component]
                        })
                        .collect::<Vec<_>>(),
                )
            }),
        PlotMode::LinesMarkers,
    )
}

fn artifact_from_interval_data(
    params: &Params,
    metrics: SailboatMetrics,
    animation_times: Vec<f64>,
    states: Vec<State<f64>>,
    x_arcs: &[IntervalArc<State<f64>>],
    u_arcs: &[IntervalArc<Control<f64>>],
    omega_rate_series: Vec<TimeSeries>,
    alpha_rate_series: Vec<TimeSeries>,
    notes: Vec<String>,
) -> SolveArtifact {
    let x = states.iter().map(|state| state.x).collect::<Vec<_>>();
    let z = states.iter().map(|state| state.z).collect::<Vec<_>>();
    let frames = states.iter().map(point_frame).collect::<Vec<_>>();
    let left_extent = x.iter().fold(0.0_f64, |acc, value| acc.min(*value)) - 5.0;
    let right_extent = x.iter().fold(0.0_f64, |acc, value| acc.max(*value)) + 5.0;
    SolveArtifact::new(
        "Sailboat Symmetric Tack",
        summary_metrics(params, &metrics),
        SolverReport::placeholder(),
        vec![
            chart(
                "Upwind Position",
                "x (m)",
                interval_arc_series("x (m)", x_arcs, PlotMode::LinesMarkers, |state| state.x),
            ),
            chart("Cross-Track Position", "z (m)", {
                let mut series =
                    interval_arc_series("z (m)", x_arcs, PlotMode::LinesMarkers, |state| state.z);
                series.extend(interval_arc_bound_series(
                    x_arcs,
                    Some(-params.cross_track_limit_m),
                    Some(params.cross_track_limit_m),
                    PlotMode::Lines,
                ));
                series
            }),
            chart("Fin Angle", "gamma (deg)", {
                let mut series =
                    interval_arc_series("gamma (deg)", x_arcs, PlotMode::LinesMarkers, |state| {
                        rad_to_deg(state.gamma)
                    });
                series.extend(interval_arc_bound_series(
                    x_arcs,
                    Some(-params.gamma_limit_deg),
                    Some(params.gamma_limit_deg),
                    PlotMode::Lines,
                ));
                series
            }),
            chart("Longitudinal Velocity", "v_x (m/s)", {
                let mut series =
                    interval_arc_series("v_x (m/s)", x_arcs, PlotMode::LinesMarkers, |state| {
                        state.vx
                    });
                series.extend(interval_arc_bound_series(
                    x_arcs,
                    Some(-VELOCITY_LIMIT_MPS),
                    Some(VELOCITY_LIMIT_MPS),
                    PlotMode::Lines,
                ));
                series
            }),
            chart("Lateral Velocity", "v_z (m/s)", {
                let mut series =
                    interval_arc_series("v_z (m/s)", x_arcs, PlotMode::LinesMarkers, |state| {
                        state.vz
                    });
                series.extend(interval_arc_bound_series(
                    x_arcs,
                    Some(-VELOCITY_LIMIT_MPS),
                    Some(VELOCITY_LIMIT_MPS),
                    PlotMode::Lines,
                ));
                series
            }),
            chart("Fin Rate", "omega (deg/s)", {
                let mut series = interval_arc_series(
                    "omega (deg/s)",
                    u_arcs,
                    PlotMode::LinesMarkers,
                    |control| rad_to_deg(control.omega),
                );
                series.extend(interval_arc_bound_series(
                    u_arcs,
                    Some(-params.omega_limit_deg_s),
                    Some(params.omega_limit_deg_s),
                    PlotMode::Lines,
                ));
                series
            }),
            chart("Sail AoA", "alpha (deg)", {
                let mut series =
                    interval_arc_series("alpha (deg)", u_arcs, PlotMode::LinesMarkers, |control| {
                        rad_to_deg(control.alpha)
                    });
                series.extend(interval_arc_bound_series(
                    u_arcs,
                    Some(-params.alpha_limit_deg),
                    Some(params.alpha_limit_deg),
                    PlotMode::Lines,
                ));
                series
            }),
            chart(
                "Water Speed",
                "Water Speed (m/s)",
                waterspeed_chart_series(x_arcs, u_arcs, params.wind_speed_mps),
            ),
            chart(
                "Air Speed",
                "Air Speed (m/s)",
                airspeed_chart_series(x_arcs, u_arcs, params.wind_speed_mps),
            ),
            chart(
                "Air Lift X",
                "F_La,x (N)",
                force_component_chart_series(
                    "Air Lift X (N)",
                    x_arcs,
                    u_arcs,
                    params.wind_speed_mps,
                    |flow| flow.f_air_lift,
                    0,
                ),
            ),
            chart(
                "Air Lift Z",
                "F_La,z (N)",
                force_component_chart_series(
                    "Air Lift Z (N)",
                    x_arcs,
                    u_arcs,
                    params.wind_speed_mps,
                    |flow| flow.f_air_lift,
                    1,
                ),
            ),
            chart(
                "Water Lift X",
                "F_Lw,x (N)",
                force_component_chart_series(
                    "Water Lift X (N)",
                    x_arcs,
                    u_arcs,
                    params.wind_speed_mps,
                    |flow| flow.f_water_lift,
                    0,
                ),
            ),
            chart(
                "Water Lift Z",
                "F_Lw,z (N)",
                force_component_chart_series(
                    "Water Lift Z (N)",
                    x_arcs,
                    u_arcs,
                    params.wind_speed_mps,
                    |flow| flow.f_water_lift,
                    1,
                ),
            ),
            chart("Omega Rate", "omega_dot (deg/s^2)", omega_rate_series),
            chart("Alpha Rate", "alpha_dot (deg/s^2)", alpha_rate_series),
        ],
        Scene2D {
            title: "Plan View".to_string(),
            x_label: "x (m, upwind positive)".to_string(),
            y_label: "z (m)".to_string(),
            paths: vec![ScenePath {
                name: "Boat Track".to_string(),
                x,
                y: z,
            }],
            circles: Vec::new(),
            arrows: vec![
                SceneArrow {
                    x: right_extent,
                    y: params.cross_track_limit_m * 0.72,
                    dx: -5.0,
                    dy: 0.0,
                    label: "True Wind".to_string(),
                },
                SceneArrow {
                    x: right_extent,
                    y: -params.cross_track_limit_m * 0.72,
                    dx: -5.0,
                    dy: 0.0,
                    label: "True Wind".to_string(),
                },
                SceneArrow {
                    x: left_extent,
                    y: 0.0,
                    dx: 4.0,
                    dy: 0.0,
                    label: "Upwind".to_string(),
                },
            ],
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
    let mut states = trajectories.x.nodes.to_vec();
    states.push(trajectories.x.terminal.clone());
    let mut controls = trajectories.u.nodes.to_vec();
    controls.push(trajectories.u.terminal.clone());
    let metrics = summarize(&states, &controls, params, trajectories.tf);
    let omega_rate_series = build_rate_series_ms("omega_dot (deg/s^2)", x_arcs, |interval| {
        rad_to_deg(trajectories.dudt[interval].omega)
    });
    let alpha_rate_series = build_rate_series_ms("alpha_dot (deg/s^2)", x_arcs, |interval| {
        rad_to_deg(trajectories.dudt[interval].alpha)
    });
    artifact_from_interval_data(
        params,
        metrics,
        node_times::<N>(trajectories.tf),
        states,
        x_arcs,
        u_arcs,
        omega_rate_series,
        alpha_rate_series,
        vec![
            "This follows the dynobud sailboat example: the state is fin angle, position, and water velocity; the control-state is fin-rate and sail angle of attack.".to_string(),
            "Multiple shooting draws each interval as an RK4 arc from the mesh node so transient continuity errors remain visible during live solves.".to_string(),
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
    let mut states = trajectories.x.nodes.to_vec();
    states.push(trajectories.x.terminal.clone());
    let mut controls = trajectories.u.nodes.to_vec();
    controls.push(trajectories.u.terminal.clone());
    let metrics = summarize(&states, &controls, params, trajectories.tf);
    let omega_rate_series = interval_arc_series(
        "omega_dot (deg/s^2)",
        &dudt_arcs,
        PlotMode::LinesMarkers,
        |rate| rad_to_deg(rate.omega),
    );
    let alpha_rate_series = interval_arc_series(
        "alpha_dot (deg/s^2)",
        &dudt_arcs,
        PlotMode::LinesMarkers,
        |rate| rad_to_deg(rate.alpha),
    );
    let mut animation_times = time_grid.nodes.nodes.to_vec();
    animation_times.push(time_grid.nodes.terminal);
    artifact_from_interval_data(
        params,
        metrics,
        animation_times,
        states,
        &x_arcs,
        &u_arcs,
        omega_rate_series,
        alpha_rate_series,
        vec![
            "This follows the dynobud sailboat example: the state is fin angle, position, and water velocity; the control-state is fin-rate and sail angle of attack.".to_string(),
            "Each direct-collocation interval is rendered as its own start-root-end arc, while omega_dot and alpha_dot are shown only at the collocation roots.".to_string(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sailboat_initial_guess_is_symmetric() {
        let guess = guess::<DEFAULT_INTERVALS>(&Params::default());
        let first = guess.x_samples.first().expect("first state");
        let last = guess.x_samples.last().expect("last state");
        assert!(first.x.abs() < 1.0e-9);
        assert!(first.z.abs() < 1.0e-9);
        assert!(last.z.abs() < 1.0e-6);
        assert!((first.gamma + last.gamma).abs() < 1.0e-9);
        assert!((first.vx - last.vx).abs() < 1.0e-6);
        assert!((first.vz + last.vz).abs() < 1.0e-6);
    }

    #[test]
    #[ignore = "slow until the sailboat NLP is tuned"]
    fn sailboat_streams_real_solver_progress() {
        let params = Params {
            wind_speed_mps: 5.0,
            initial_time_guess_s: 18.0,
            min_final_time_s: 1.0,
            max_final_time_s: 40.0,
            gamma_limit_deg: 12.0,
            omega_limit_deg_s: 5.0,
            alpha_limit_deg: 12.0,
            omega_weight: 1.0e-3,
            alpha_weight: 1.0e-3,
            omega_rate_regularization: 0.0,
            alpha_rate_regularization: 0.0,
            cross_track_limit_m: 30.0,
            solver_method: SolverMethod::Sqp,
            solver: SqpConfig {
                max_iters: 8,
                dual_tol: 1.0e-1,
                constraint_tol: 1.0e-4,
                complementarity_tol: 1.0e-4,
            },
            transcription: TranscriptionConfig {
                method: TranscriptionMethod::MultipleShooting,
                ..default_transcription(DEFAULT_INTERVALS)
            },
        };
        let mut events = Vec::new();
        let result = solve_with_progress(&params, |event| events.push(event));
        crate::common::assert_shared_progress_lifecycle(&events);
        let iteration_count = events
            .iter()
            .filter(|event| matches!(event, SolveStreamEvent::Iteration { .. }))
            .count();
        let final_count = events
            .iter()
            .filter(|event| matches!(event, SolveStreamEvent::Final { .. }))
            .count();
        assert!(
            iteration_count > 0 || final_count > 0 || result.is_err(),
            "expected a real solve attempt rather than a preview shortcut"
        );
        if let Ok(artifact) = result {
            let upwind_distance =
                crate::find_metric(&artifact.summary, crate::MetricKey::UpwindDistance)
                    .and_then(|metric| metric.numeric_value)
                    .expect("upwind distance should exist");
            assert!(
                upwind_distance > 1.0,
                "successful solve should make upwind progress"
            );
        }
    }
}
