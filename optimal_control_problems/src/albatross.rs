use crate::common::{
    ArtifactVisualization, CompileCacheStatus, CompileProgressInfo, CompileProgressUpdate,
    CompiledDirectCollocationOcp, CompiledMultipleShootingOcp, DirectCollocationRuntimeValues,
    DirectCollocationTimeGrid, DirectCollocationTrajectories, FromMap, LatexSection, MetricKey,
    MultipleShootingRuntimeValues, MultipleShootingTrajectories, OcpCompileProgressState,
    OcpSxFunctionConfig, PlotMode, ProblemId, ProblemSpec, Scene2D, ScenePath, ScenePath3D,
    SolveArtifact, SolveStreamEvent, SolverConfig, SolverMethod, SolverReport, StandardOcpParams,
    TimeSeries, TranscriptionConfig, chart, compile_progress_info_from_compiled,
    default_solver_config, default_transcription, direct_collocation_compile_key_with_sx,
    direct_collocation_variant_with_sx, expect_finite, interval_arc_series, metric_with_key,
    multiple_shooting_compile_key, multiple_shooting_variant_with_sx, node_times,
    numeric_metric_with_key, ocp_compile_options, ocp_compile_progress_update,
    ocp_sx_function_config_from_map, problem_controls, problem_scientific_slider_control,
    problem_slider_control, problem_spec, rad_to_deg, sample_or_default, select_control,
    solver_config_from_map, solver_method_from_map, transcription_from_map, transcription_metrics,
};
use anyhow::{Result, anyhow};
use optimal_control::runtime::{
    DirectCollocation, MultipleShooting, direct_collocation_root_arcs,
    direct_collocation_state_like_arcs,
};
use optimal_control::{Bounds1D, InterpolatedTrajectory, IntervalArc, Ocp, OcpGlobalDesign};
use optimization::{InteriorPointLinearSolver, ScalarLeaf, Vectorize};
use serde::Serialize;
use std::collections::BTreeMap;
use sx_core::SX;

const PROBLEM_NAME: &str = "Albatross Dynamic Soaring";
const RK4_SUBSTEPS: usize = 2;
const DEFAULT_INTERVALS: usize = 28;
const DEFAULT_COLLOCATION_DEGREE: usize = 3;
const SUPPORTED_INTERVALS: [usize; 6] = [12, 18, DEFAULT_INTERVALS, 36, 48, 64];
const SUPPORTED_DEGREES: [usize; 4] = [2, DEFAULT_COLLOCATION_DEGREE, 4, 5];

const GRAVITY: f64 = 9.81;
const AIR_DENSITY: f64 = 1.225;
const MASS_KG: f64 = 8.5;
const REFERENCE_AREA_M2: f64 = 0.65;
const ASPECT_RATIO: f64 = 18.0;
const OSWALD: f64 = 0.85;
const CL_SLOPE: f64 = 5.7;
const CD0: f64 = 0.018;
const SPEED_EPS: f64 = 1.0e-3;
const FRAME_EPS: f64 = 1.0e-4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ObjectiveKind {
    AverageSpeed,
    WindWork,
    TerminalEnergy,
    ControlRegularization,
}

impl ObjectiveKind {
    fn from_value(value: f64) -> Result<Self> {
        match value.round() as i32 {
            0 => Ok(Self::AverageSpeed),
            1 => Ok(Self::WindWork),
            2 => Ok(Self::TerminalEnergy),
            3 => Ok(Self::ControlRegularization),
            _ => Err(anyhow!("objective must be one of 0, 1, 2, or 3")),
        }
    }

    const fn id(self) -> &'static str {
        match self {
            Self::AverageSpeed => "avg_speed",
            Self::WindWork => "wind_work",
            Self::TerminalEnergy => "terminal_energy",
            Self::ControlRegularization => "control_reg",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::AverageSpeed => "Average Speed",
            Self::WindWork => "Wind Work",
            Self::TerminalEnergy => "Terminal Energy",
            Self::ControlRegularization => "Control Regularization",
        }
    }

    const fn value(self) -> f64 {
        match self {
            Self::AverageSpeed => 0.0,
            Self::WindWork => 1.0,
            Self::TerminalEnergy => 2.0,
            Self::ControlRegularization => 3.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct AlbatrossMsKey {
    base: crate::common::MultipleShootingCompileKey,
    objective: ObjectiveKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct AlbatrossDcKey {
    base: crate::common::DirectCollocationCompileVariantKey,
    objective: ObjectiveKind,
}

#[derive(Clone, Debug, PartialEq, Serialize, Vectorize)]
pub struct State<T> {
    pub px: T,
    pub py: T,
    pub pz: T,
    pub vx: T,
    pub vy: T,
    pub vz: T,
}

#[derive(Clone, Debug, PartialEq, Serialize, Vectorize)]
pub struct Control<T> {
    pub alpha: T,
    pub roll: T,
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct Design<T> {
    delta_l: T,
    h0: T,
    vx0: T,
    tf: T,
}

impl<T> OcpGlobalDesign<T> for Design<T>
where
    T: Clone + ScalarLeaf,
{
    fn final_time(&self) -> T {
        self.tf.clone()
    }

    fn from_final_time(tf: T) -> Self {
        Self {
            delta_l: tf.clone(),
            h0: tf.clone(),
            vx0: tf.clone(),
            tf,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct ModelParams<T> {
    wind_azimuth_rad: T,
    wind_low_mps: T,
    wind_high_mps: T,
    wind_mid_altitude_m: T,
    wind_transition_height_m: T,
    rate_weight: T,
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct Path<T> {
    altitude: T,
    airspeed: T,
    cl: T,
    load_factor: T,
    frame_guard: T,
    alpha_rate: T,
    roll_rate: T,
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct Boundary<T> {
    px0: T,
    delta_l: T,
    py0: T,
    py_t: T,
    h0: T,
    pz_periodic: T,
    vx0: T,
    vx_periodic: T,
    vy0: T,
    vy_t: T,
    vz0: T,
    vz_t: T,
    alpha_periodic: T,
    roll_periodic: T,
}

type MsCompiled = CompiledMultipleShootingOcp<
    State<SX>,
    Control<SX>,
    ModelParams<SX>,
    Path<SX>,
    Boundary<SX>,
    (),
    Design<SX>,
>;

type DcCompiled = CompiledDirectCollocationOcp<
    State<SX>,
    Control<SX>,
    ModelParams<SX>,
    Path<SX>,
    Boundary<SX>,
    (),
    Design<SX>,
>;

thread_local! {
    static MULTIPLE_SHOOTING_CACHE: std::cell::RefCell<
        crate::common::SharedCompileCache<AlbatrossMsKey, MsCompiled>
    > = std::cell::RefCell::new(crate::common::SharedCompileCache::new());
    static DIRECT_COLLOCATION_CACHE: std::cell::RefCell<
        crate::common::SharedCompileCache<AlbatrossDcKey, DcCompiled>
    > = std::cell::RefCell::new(crate::common::SharedCompileCache::new());
}

#[derive(Clone, Debug)]
struct DesignControl {
    fixed: bool,
    value: f64,
    lower: f64,
    upper: f64,
}

#[derive(Clone, Debug)]
pub struct Params {
    objective: ObjectiveKind,
    delta_l: DesignControl,
    h0: DesignControl,
    vx0: DesignControl,
    tf: DesignControl,
    wind_azimuth_deg: f64,
    wind_low_mps: f64,
    wind_high_mps: f64,
    wind_mid_altitude_m: f64,
    wind_transition_height_m: f64,
    initial_wave_amplitude_m: f64,
    initial_wave_rotation_deg: f64,
    initial_alpha_deg: f64,
    initial_roll_amplitude_deg: f64,
    min_airspeed_mps: f64,
    max_airspeed_mps: f64,
    max_load_factor: f64,
    max_alpha_rate_deg_s: f64,
    max_roll_rate_deg_s: f64,
    rate_regularization: f64,
    scaling_enabled: bool,
    solver_method: SolverMethod,
    solver: SolverConfig,
    transcription: TranscriptionConfig,
    sx_functions: OcpSxFunctionConfig,
}

impl Default for Params {
    fn default() -> Self {
        let transcription = default_transcription(DEFAULT_INTERVALS);
        let mut solver = default_solver_config();
        solver.nlip.linear_solver = InteriorPointLinearSolver::SparseQdldl;
        Self {
            objective: ObjectiveKind::AverageSpeed,
            delta_l: DesignControl {
                fixed: true,
                value: 200.0,
                lower: 120.0,
                upper: 320.0,
            },
            h0: DesignControl {
                fixed: true,
                value: 25.0,
                lower: 10.0,
                upper: 60.0,
            },
            vx0: DesignControl {
                fixed: true,
                value: 20.0,
                lower: 12.0,
                upper: 35.0,
            },
            tf: DesignControl {
                fixed: true,
                value: 10.0,
                lower: 6.0,
                upper: 24.0,
            },
            wind_azimuth_deg: 0.0,
            wind_low_mps: 0.0,
            wind_high_mps: 12.0,
            wind_mid_altitude_m: 20.0,
            wind_transition_height_m: 8.0,
            initial_wave_amplitude_m: 14.0,
            initial_wave_rotation_deg: 45.0,
            initial_alpha_deg: 4.0,
            initial_roll_amplitude_deg: 35.0,
            min_airspeed_mps: 5.0,
            max_airspeed_mps: 70.0,
            max_load_factor: 8.0,
            max_alpha_rate_deg_s: 45.0,
            max_roll_rate_deg_s: 160.0,
            rate_regularization: 1.0e-3,
            scaling_enabled: true,
            solver_method: SolverMethod::Nlip,
            solver,
            transcription,
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

fn read_design(
    values: &BTreeMap<String, f64>,
    prefix: &str,
    defaults: &DesignControl,
) -> Result<DesignControl> {
    let fixed = sample_or_default(
        values,
        &format!("{prefix}_free"),
        if defaults.fixed { 0.0 } else { 1.0 },
    ) < 0.5;
    let value = expect_finite(
        sample_or_default(values, &format!("{prefix}_value"), defaults.value),
        &format!("{prefix}_value"),
    )?;
    let lower = expect_finite(
        sample_or_default(values, &format!("{prefix}_lower"), defaults.lower),
        &format!("{prefix}_lower"),
    )?;
    let upper = expect_finite(
        sample_or_default(values, &format!("{prefix}_upper"), defaults.upper),
        &format!("{prefix}_upper"),
    )?;
    if !fixed {
        if lower > upper {
            return Err(anyhow!(
                "{prefix}_lower must be less than or equal to {prefix}_upper"
            ));
        }
        if value < lower || value > upper {
            return Err(anyhow!(
                "{prefix}_value ({value}) must lie inside [{lower}, {upper}] when used as the initial free-design guess"
            ));
        }
    }
    Ok(DesignControl {
        fixed,
        value,
        lower,
        upper,
    })
}

impl FromMap for Params {
    fn from_map(values: &BTreeMap<String, f64>) -> Result<Self> {
        let defaults = Self::default();
        let params = Self {
            objective: ObjectiveKind::from_value(sample_or_default(
                values,
                "objective",
                defaults.objective.value(),
            ))?,
            delta_l: read_design(values, "delta_l", &defaults.delta_l)?,
            h0: read_design(values, "h0", &defaults.h0)?,
            vx0: read_design(values, "vx0", &defaults.vx0)?,
            tf: read_design(values, "tf", &defaults.tf)?,
            wind_azimuth_deg: expect_finite(
                sample_or_default(values, "wind_azimuth_deg", defaults.wind_azimuth_deg),
                "wind_azimuth_deg",
            )?,
            wind_low_mps: expect_finite(
                sample_or_default(values, "wind_low_mps", defaults.wind_low_mps),
                "wind_low_mps",
            )?,
            wind_high_mps: expect_finite(
                sample_or_default(values, "wind_high_mps", defaults.wind_high_mps),
                "wind_high_mps",
            )?,
            wind_mid_altitude_m: expect_finite(
                sample_or_default(values, "wind_mid_altitude_m", defaults.wind_mid_altitude_m),
                "wind_mid_altitude_m",
            )?,
            wind_transition_height_m: expect_finite(
                sample_or_default(
                    values,
                    "wind_transition_height_m",
                    defaults.wind_transition_height_m,
                ),
                "wind_transition_height_m",
            )?,
            initial_wave_amplitude_m: expect_finite(
                sample_or_default(
                    values,
                    "initial_wave_amplitude_m",
                    defaults.initial_wave_amplitude_m,
                ),
                "initial_wave_amplitude_m",
            )?,
            initial_wave_rotation_deg: expect_finite(
                sample_or_default(
                    values,
                    "initial_wave_rotation_deg",
                    defaults.initial_wave_rotation_deg,
                ),
                "initial_wave_rotation_deg",
            )?,
            initial_alpha_deg: expect_finite(
                sample_or_default(values, "initial_alpha_deg", defaults.initial_alpha_deg),
                "initial_alpha_deg",
            )?,
            initial_roll_amplitude_deg: expect_finite(
                sample_or_default(
                    values,
                    "initial_roll_amplitude_deg",
                    defaults.initial_roll_amplitude_deg,
                ),
                "initial_roll_amplitude_deg",
            )?,
            min_airspeed_mps: expect_finite(
                sample_or_default(values, "min_airspeed_mps", defaults.min_airspeed_mps),
                "min_airspeed_mps",
            )?,
            max_airspeed_mps: expect_finite(
                sample_or_default(values, "max_airspeed_mps", defaults.max_airspeed_mps),
                "max_airspeed_mps",
            )?,
            max_load_factor: expect_finite(
                sample_or_default(values, "max_load_factor", defaults.max_load_factor),
                "max_load_factor",
            )?,
            max_alpha_rate_deg_s: expect_finite(
                sample_or_default(
                    values,
                    "max_alpha_rate_deg_s",
                    defaults.max_alpha_rate_deg_s,
                ),
                "max_alpha_rate_deg_s",
            )?,
            max_roll_rate_deg_s: expect_finite(
                sample_or_default(values, "max_roll_rate_deg_s", defaults.max_roll_rate_deg_s),
                "max_roll_rate_deg_s",
            )?,
            rate_regularization: expect_finite(
                sample_or_default(values, "rate_regularization", defaults.rate_regularization),
                "rate_regularization",
            )?,
            scaling_enabled: sample_or_default(
                values,
                "scaling_enabled",
                if defaults.scaling_enabled { 1.0 } else { 0.0 },
            ) >= 0.5,
            solver_method: solver_method_from_map(values, defaults.solver_method)?,
            solver: solver_config_from_map(values, defaults.solver)?,
            transcription: transcription_from_map(
                values,
                defaults.transcription,
                &SUPPORTED_INTERVALS,
                &SUPPORTED_DEGREES,
            )?,
            sx_functions: ocp_sx_function_config_from_map(values, defaults.sx_functions)?,
        };
        if params.wind_transition_height_m <= 0.0 {
            return Err(anyhow!("wind_transition_height_m must be positive"));
        }
        if params.min_airspeed_mps > params.max_airspeed_mps {
            return Err(anyhow!(
                "min_airspeed_mps must be less than or equal to max_airspeed_mps"
            ));
        }
        validate_design_domain("delta_l", &params.delta_l, 0.0, false, "positive")?;
        validate_design_domain("h0", &params.h0, 0.0, true, "nonnegative")?;
        validate_design_domain("vx0", &params.vx0, 0.0, false, "positive")?;
        validate_design_domain("tf", &params.tf, 0.0, false, "positive")?;
        if params.initial_wave_amplitude_m < 0.0 {
            return Err(anyhow!("initial_wave_amplitude_m must be nonnegative"));
        }
        Ok(params)
    }
}

fn validate_design_domain(
    prefix: &str,
    control: &DesignControl,
    floor: f64,
    inclusive: bool,
    description: &str,
) -> Result<()> {
    let violates = |value: f64| {
        if inclusive {
            value < floor
        } else {
            value <= floor
        }
    };
    if control.fixed {
        if violates(control.value) {
            return Err(anyhow!("{prefix}_value must be {description}"));
        }
    } else if violates(control.lower) {
        return Err(anyhow!(
            "{prefix}_lower must be {description} when {prefix} is free"
        ));
    }
    Ok(())
}

fn checkbox_control(
    id: &str,
    label: &str,
    default_checked: bool,
    help: &str,
) -> crate::common::ControlSpec {
    crate::common::ControlSpec {
        id: id.to_string(),
        label: label.to_string(),
        min: 0.0,
        max: 1.0,
        step: 1.0,
        default: if default_checked { 1.0 } else { 0.0 },
        unit: String::new(),
        help: help.to_string(),
        section: crate::common::ControlSection::Problem,
        panel: None,
        editor: crate::common::ControlEditor::Checkbox,
        visibility: crate::common::ControlVisibility::Always,
        semantic: crate::common::ControlSemantic::ProblemParameter,
        value_display: crate::common::ControlValueDisplay::Scalar,
        choices: Vec::new(),
        profile_defaults: Vec::new(),
    }
}

fn design_controls(
    prefix: &str,
    label: &str,
    unit: &str,
    defaults: &DesignControl,
) -> Vec<crate::common::ControlSpec> {
    vec![
        checkbox_control(
            &format!("{prefix}_free"),
            &format!("Free {label}"),
            !defaults.fixed,
            "When checked, the lower and upper bound editors define this global design variable. When unchecked, the fixed value editor is used as an equality bound.",
        ),
        problem_slider_control(
            format!("{prefix}_value"),
            format!("{label} Fixed/Guess"),
            defaults.lower.min(defaults.value) * 0.5,
            defaults.upper.max(defaults.value) * 1.5,
            0.5,
            defaults.value,
            unit,
            "Fixed value when not free; initial guess when free. It must already satisfy active bounds.",
        ),
        problem_slider_control(
            format!("{prefix}_lower"),
            format!("{label} Lower"),
            defaults.lower * 0.5,
            defaults.upper * 1.5,
            0.5,
            defaults.lower,
            unit,
            "Lower bound used when this design variable is free.",
        ),
        problem_slider_control(
            format!("{prefix}_upper"),
            format!("{label} Upper"),
            defaults.lower * 0.5,
            defaults.upper * 1.5,
            0.5,
            defaults.upper,
            unit,
            "Upper bound used when this design variable is free.",
        ),
    ]
}

pub fn spec() -> ProblemSpec {
    let defaults = Params::default();
    let mut extra = vec![select_control(
        "objective",
        "Objective",
        defaults.objective.value(),
        "",
        "Symbolic objective variant. Changing this intentionally compiles a separate NLP.",
        &[
            (0.0, "Average speed"),
            (1.0, "Wind work"),
            (2.0, "Terminal energy"),
            (3.0, "Control regularization"),
        ],
        crate::common::ControlSection::Problem,
        crate::common::ControlVisibility::Always,
        crate::common::ControlSemantic::ProblemParameter,
    )];
    extra.extend(design_controls(
        "delta_l",
        "Delta L",
        "m",
        &defaults.delta_l,
    ));
    extra.extend(design_controls("h0", "h0", "m", &defaults.h0));
    extra.extend(design_controls("vx0", "vx0", "m/s", &defaults.vx0));
    extra.extend(design_controls("tf", "T", "s", &defaults.tf));
    extra.extend([
        problem_slider_control(
            "wind_azimuth_deg",
            "Wind Azimuth",
            -180.0,
            180.0,
            1.0,
            defaults.wind_azimuth_deg,
            "deg",
            "Horizontal wind direction. 0 deg is +X.",
        ),
        problem_slider_control(
            "wind_low_mps",
            "Low Wind",
            -25.0,
            25.0,
            0.5,
            defaults.wind_low_mps,
            "m/s",
            "Wind speed at low altitude along the configured azimuth.",
        ),
        problem_slider_control(
            "wind_high_mps",
            "High Wind",
            -25.0,
            35.0,
            0.5,
            defaults.wind_high_mps,
            "m/s",
            "Wind speed at high altitude along the configured azimuth.",
        ),
        problem_slider_control(
            "wind_mid_altitude_m",
            "Wind Mid Altitude",
            0.0,
            80.0,
            0.5,
            defaults.wind_mid_altitude_m,
            "m",
            "Altitude at the middle of the smooth tanh wind transition.",
        ),
        problem_slider_control(
            "wind_transition_height_m",
            "Wind Transition Height",
            1.0,
            40.0,
            0.5,
            defaults.wind_transition_height_m,
            "m",
            "Positive tanh scale height for the C-infinity wind shear.",
        ),
        problem_slider_control(
            "initial_wave_amplitude_m",
            "Guess Wave Amplitude",
            0.0,
            50.0,
            0.5,
            defaults.initial_wave_amplitude_m,
            "m",
            "Amplitude of the sin^2 initial trajectory wave.",
        ),
        problem_slider_control(
            "initial_wave_rotation_deg",
            "Guess Wave Rotation",
            -90.0,
            90.0,
            1.0,
            defaults.initial_wave_rotation_deg,
            "deg",
            "Rotation of the initial wave in the Y/Z plane about +X.",
        ),
        problem_slider_control(
            "initial_alpha_deg",
            "Initial AoA",
            -8.0,
            12.0,
            0.25,
            defaults.initial_alpha_deg,
            "deg",
            "Periodic constant angle-of-attack seed.",
        ),
        problem_slider_control(
            "initial_roll_amplitude_deg",
            "Initial Roll Amp",
            0.0,
            80.0,
            1.0,
            defaults.initial_roll_amplitude_deg,
            "deg",
            "Sinusoidal roll seed amplitude.",
        ),
        problem_slider_control(
            "min_airspeed_mps",
            "Min Airspeed",
            0.0,
            30.0,
            0.5,
            defaults.min_airspeed_mps,
            "m/s",
            "Lower path bound on air-relative speed.",
        ),
        problem_slider_control(
            "max_airspeed_mps",
            "Max Airspeed",
            20.0,
            120.0,
            1.0,
            defaults.max_airspeed_mps,
            "m/s",
            "Upper path bound on air-relative speed.",
        ),
        problem_slider_control(
            "max_load_factor",
            "Max Load",
            1.0,
            20.0,
            0.25,
            defaults.max_load_factor,
            "g",
            "Upper path bound on aerodynamic load factor.",
        ),
        problem_slider_control(
            "max_alpha_rate_deg_s",
            "AoA Rate Limit",
            1.0,
            180.0,
            1.0,
            defaults.max_alpha_rate_deg_s,
            "deg/s",
            "Path bound on alpha rate.",
        ),
        problem_slider_control(
            "max_roll_rate_deg_s",
            "Roll Rate Limit",
            5.0,
            360.0,
            5.0,
            defaults.max_roll_rate_deg_s,
            "deg/s",
            "Path bound on roll rate.",
        ),
        problem_scientific_slider_control(
            "rate_regularization",
            "Rate Weight",
            0.0,
            1.0,
            1.0e-4,
            defaults.rate_regularization,
            "",
            "Quadratic regularization on alpha and roll rates.",
        ),
        select_control(
            "scaling_enabled",
            "Scaling",
            1.0,
            "",
            "Enable OCP scaling.",
            &[(1.0, "On"), (0.0, "Off")],
            crate::common::ControlSection::Problem,
            crate::common::ControlVisibility::Always,
            crate::common::ControlSemantic::ProblemParameter,
        ),
    ]);
    problem_spec(
        ProblemId::AlbatrossDynamicSoaring,
        PROBLEM_NAME,
        "A 3D point-mass dynamic-soaring OCP with alpha and bank/roll controls, smooth tanh wind shear, and periodic translational/control boundary conditions.",
        problem_controls(
            defaults.transcription,
            &SUPPORTED_INTERVALS,
            &SUPPORTED_DEGREES,
            defaults.solver_method,
            defaults.solver,
            extra,
        ),
        vec![
            LatexSection {
                title: "States, Controls, and Globals".to_string(),
                entries: vec![
                    r"x = [p_x,p_y,p_z,v_x,v_y,v_z]^T,\quad u=[\alpha,\phi]^T".to_string(),
                    r"g=[\Delta L,h_0,v_{x0},T]^T".to_string(),
                ],
            },
            LatexSection {
                title: "Wind Shear".to_string(),
                entries: vec![r"W(p_z)=\hat w\left(W_\ell+\frac{1}{2}(W_h-W_\ell)(1+\tanh((p_z-z_m)/z_s))\right)".to_string()],
            },
            LatexSection {
                title: "Lift Frame".to_string(),
                entries: vec![r"\hat d=-a/V_a,\quad n_0=\mathrm{normalize}(\hat z-(\hat z\cdot\hat a)\hat a),\quad \hat l=\cos\phi\,n_0+\sin\phi\,(\hat a\times n_0)".to_string()],
            },
        ],
        vec![
            "The lift-frame reference uses projected inertial up, so no attitude state or quaternion is introduced.".to_string(),
            "The frame projection and airspeed use epsilon regularization; a separate frame-guard path constraint reports near-degenerate air-relative vertical flight.".to_string(),
        ],
    )
}

fn deg_to_rad(value: f64) -> f64 {
    value * std::f64::consts::PI / 180.0
}

fn cl_sx(alpha: SX) -> SX {
    CL_SLOPE * alpha
}

fn cd_sx(alpha: SX) -> SX {
    let cl = cl_sx(alpha);
    CD0 + cl.sqr() / (std::f64::consts::PI * ASPECT_RATIO * OSWALD)
}

fn cl_numeric(alpha: f64) -> f64 {
    CL_SLOPE * alpha
}

fn cd_numeric(alpha: f64) -> f64 {
    let cl = cl_numeric(alpha);
    CD0 + cl * cl / (std::f64::consts::PI * ASPECT_RATIO * OSWALD)
}

#[derive(Clone)]
struct AeroSx {
    ax: SX,
    ay: SX,
    az: SX,
    cl: SX,
    va: SX,
    load_factor: SX,
    frame_guard: SX,
    wind_work_rate: SX,
}

fn wind_sx(pz: SX, p: &ModelParams<SX>) -> (SX, SX, SX, SX) {
    let transition =
        ((pz - p.wind_mid_altitude_m.clone()) / p.wind_transition_height_m.clone()).tanh();
    let speed = p.wind_low_mps.clone()
        + 0.5 * (p.wind_high_mps.clone() - p.wind_low_mps.clone()) * (1.0 + transition);
    let wx = speed.clone() * p.wind_azimuth_rad.clone().cos();
    let wy = speed.clone() * p.wind_azimuth_rad.clone().sin();
    (wx, wy, SX::zero(), speed)
}

fn aero_sx(state: &State<SX>, control: &Control<SX>, p: &ModelParams<SX>) -> AeroSx {
    let (wx, wy, wz, _) = wind_sx(state.pz.clone(), p);
    let ax_rel = state.vx.clone() - wx.clone();
    let ay_rel = state.vy.clone() - wy.clone();
    let az_rel = state.vz.clone() - wz.clone();
    let va2 =
        ax_rel.clone().sqr() + ay_rel.clone().sqr() + az_rel.clone().sqr() + SPEED_EPS * SPEED_EPS;
    let va = va2.clone().sqrt();
    let inv_va = SX::one() / va.clone();
    let ahx = ax_rel * inv_va.clone();
    let ahy = ay_rel * inv_va.clone();
    let ahz = az_rel * inv_va.clone();
    let dhx = -ahx.clone();
    let dhy = -ahy.clone();
    let dhz = -ahz.clone();
    let dot_up = ahz.clone();
    let nrx = -dot_up.clone() * ahx.clone();
    let nry = -dot_up.clone() * ahy.clone();
    let nrz = SX::one() - dot_up * ahz.clone();
    let frame_guard = (nrx.clone().sqr() + nry.clone().sqr() + nrz.clone().sqr()).sqrt();
    let inv_n = SX::one() / (frame_guard.clone().sqr() + FRAME_EPS * FRAME_EPS).sqrt();
    let nx = nrx * inv_n.clone();
    let ny = nry * inv_n.clone();
    let nz = nrz * inv_n.clone();
    let sx_x = ahy.clone() * nz.clone() - ahz.clone() * ny.clone();
    let sx_y = ahz.clone() * nx.clone() - ahx.clone() * nz.clone();
    let sx_z = ahx * ny.clone() - ahy * nx.clone();
    let cr = control.roll.clone().cos();
    let sr = control.roll.clone().sin();
    let lx = cr.clone() * nx + sr.clone() * sx_x;
    let ly = cr.clone() * ny + sr.clone() * sx_y;
    let lz = cr * nz + sr * sx_z;
    let cl = cl_sx(control.alpha.clone());
    let cd = cd_sx(control.alpha.clone());
    let q_over_m = 0.5 * AIR_DENSITY * REFERENCE_AREA_M2 * va2 / MASS_KG;
    let aero_x = q_over_m.clone() * (cl.clone() * lx + cd.clone() * dhx);
    let aero_y = q_over_m.clone() * (cl.clone() * ly + cd.clone() * dhy);
    let aero_z = q_over_m * (cl.clone() * lz + cd.clone() * dhz);
    let load_factor =
        (aero_x.clone().sqr() + aero_y.clone().sqr() + aero_z.clone().sqr()).sqrt() / GRAVITY;
    let wind_work_rate =
        MASS_KG * (aero_x.clone() * wx + aero_y.clone() * wy + aero_z.clone() * wz);
    AeroSx {
        ax: aero_x,
        ay: aero_y,
        az: aero_z,
        cl,
        va,
        load_factor,
        frame_guard,
        wind_work_rate,
    }
}

fn model<Scheme>(
    scheme: Scheme,
    objective: ObjectiveKind,
) -> Ocp<State<SX>, Control<SX>, ModelParams<SX>, Path<SX>, Boundary<SX>, (), Scheme, Design<SX>> {
    Ocp::new("albatross_dynamic_soaring", scheme)
        .objective_lagrange(
            move |x: &State<SX>, u: &Control<SX>, dudt: &Control<SX>, p: &ModelParams<SX>| {
                let rate_cost =
                    p.rate_weight.clone() * (dudt.alpha.clone().sqr() + dudt.roll.clone().sqr());
                match objective {
                    ObjectiveKind::WindWork => rate_cost - aero_sx(x, u, p).wind_work_rate,
                    ObjectiveKind::ControlRegularization
                    | ObjectiveKind::AverageSpeed
                    | ObjectiveKind::TerminalEnergy => rate_cost,
                }
            },
        )
        .objective_mayer_global(
            move |_: &State<SX>,
                  _: &Control<SX>,
                  xf: &State<SX>,
                  _: &Control<SX>,
                  _: &ModelParams<SX>,
                  g: &Design<SX>| {
                match objective {
                    ObjectiveKind::AverageSpeed => -g.delta_l.clone() / g.tf.clone(),
                    ObjectiveKind::TerminalEnergy => {
                        -(0.5
                            * MASS_KG
                            * (xf.vx.clone().sqr() + xf.vy.clone().sqr() + xf.vz.clone().sqr())
                            + MASS_KG * GRAVITY * xf.pz.clone())
                    }
                    ObjectiveKind::WindWork | ObjectiveKind::ControlRegularization => SX::zero(),
                }
            },
        )
        .ode(|x, u, p| {
            let aero = aero_sx(x, u, p);
            State {
                px: x.vx.clone(),
                py: x.vy.clone(),
                pz: x.vz.clone(),
                vx: aero.ax,
                vy: aero.ay,
                vz: aero.az - GRAVITY,
            }
        })
        .path_constraints(|x, u, dudt, p| {
            let aero = aero_sx(x, u, p);
            Path {
                altitude: x.pz.clone(),
                airspeed: aero.va,
                cl: aero.cl,
                load_factor: aero.load_factor,
                frame_guard: aero.frame_guard,
                alpha_rate: dudt.alpha.clone(),
                roll_rate: dudt.roll.clone(),
            }
        })
        .boundary_equalities_global(|x0, u0, xt, ut, _p, g| Boundary {
            px0: x0.px.clone(),
            delta_l: xt.px.clone() - x0.px.clone() - g.delta_l.clone(),
            py0: x0.py.clone(),
            py_t: xt.py.clone(),
            h0: x0.pz.clone() - g.h0.clone(),
            pz_periodic: xt.pz.clone() - x0.pz.clone(),
            vx0: x0.vx.clone() - g.vx0.clone(),
            vx_periodic: xt.vx.clone() - x0.vx.clone(),
            vy0: x0.vy.clone(),
            vy_t: xt.vy.clone(),
            vz0: x0.vz.clone(),
            vz_t: xt.vz.clone(),
            alpha_periodic: ut.alpha.clone() - u0.alpha.clone(),
            roll_periodic: ut.roll.clone() - u0.roll.clone(),
        })
        .boundary_inequalities_global(|_, _, _, _, _, _| ())
        .build()
        .expect("albatross OCP should build")
}

fn active_design_value(name: &str, control: &DesignControl) -> Result<f64> {
    if control.fixed {
        Ok(control.value)
    } else if control.value < control.lower || control.value > control.upper {
        Err(anyhow!(
            "{name} initial guess {} is outside active free bounds [{}, {}]",
            control.value,
            control.lower,
            control.upper
        ))
    } else {
        Ok(control.value)
    }
}

fn active_design(params: &Params) -> Result<Design<f64>> {
    Ok(Design {
        delta_l: active_design_value("delta_l", &params.delta_l)?,
        h0: active_design_value("h0", &params.h0)?,
        vx0: active_design_value("vx0", &params.vx0)?,
        tf: active_design_value("tf", &params.tf)?,
    })
}

fn design_bounds(control: &DesignControl) -> Bounds1D {
    if control.fixed {
        Bounds1D {
            lower: Some(control.value),
            upper: Some(control.value),
        }
    } else {
        Bounds1D {
            lower: Some(control.lower),
            upper: Some(control.upper),
        }
    }
}

fn model_params(params: &Params) -> ModelParams<f64> {
    ModelParams {
        wind_azimuth_rad: deg_to_rad(params.wind_azimuth_deg),
        wind_low_mps: params.wind_low_mps,
        wind_high_mps: params.wind_high_mps,
        wind_mid_altitude_m: params.wind_mid_altitude_m,
        wind_transition_height_m: params.wind_transition_height_m,
        rate_weight: params.rate_regularization,
    }
}

fn wind_numeric(pz: f64, params: &Params) -> ([f64; 3], f64) {
    let az = deg_to_rad(params.wind_azimuth_deg);
    let speed = params.wind_low_mps
        + 0.5
            * (params.wind_high_mps - params.wind_low_mps)
            * (1.0 + ((pz - params.wind_mid_altitude_m) / params.wind_transition_height_m).tanh());
    ([speed * az.cos(), speed * az.sin(), 0.0], speed)
}

#[derive(Clone, Debug)]
struct AeroNum {
    aero_accel: [f64; 3],
    lift_accel: [f64; 3],
    drag_accel: [f64; 3],
    air_dir: [f64; 3],
    zero_bank_lift_dir: [f64; 3],
    side_dir: [f64; 3],
    va: f64,
    wind_speed: f64,
    wind: [f64; 3],
    cl: f64,
    cd: f64,
    load_factor: f64,
    frame_guard: f64,
    wind_work_rate: f64,
}

fn aero_numeric(state: &State<f64>, control: &Control<f64>, params: &Params) -> AeroNum {
    let (wind, wind_speed) = wind_numeric(state.pz, params);
    let rel = [state.vx - wind[0], state.vy - wind[1], state.vz - wind[2]];
    let va = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2] + SPEED_EPS * SPEED_EPS).sqrt();
    let ah = [rel[0] / va, rel[1] / va, rel[2] / va];
    let dh = [-ah[0], -ah[1], -ah[2]];
    let dot_up = ah[2];
    let nr = [-dot_up * ah[0], -dot_up * ah[1], 1.0 - dot_up * ah[2]];
    let frame_guard = (nr[0] * nr[0] + nr[1] * nr[1] + nr[2] * nr[2]).sqrt();
    let inv_n = 1.0 / (frame_guard * frame_guard + FRAME_EPS * FRAME_EPS).sqrt();
    let n = [nr[0] * inv_n, nr[1] * inv_n, nr[2] * inv_n];
    let side = [
        ah[1] * n[2] - ah[2] * n[1],
        ah[2] * n[0] - ah[0] * n[2],
        ah[0] * n[1] - ah[1] * n[0],
    ];
    let cr = control.roll.cos();
    let sr = control.roll.sin();
    let lift_dir = [
        cr * n[0] + sr * side[0],
        cr * n[1] + sr * side[1],
        cr * n[2] + sr * side[2],
    ];
    let cl = cl_numeric(control.alpha);
    let cd = cd_numeric(control.alpha);
    let q_over_m = 0.5 * AIR_DENSITY * REFERENCE_AREA_M2 * va * va / MASS_KG;
    let lift_accel = [
        q_over_m * cl * lift_dir[0],
        q_over_m * cl * lift_dir[1],
        q_over_m * cl * lift_dir[2],
    ];
    let drag_accel = [
        q_over_m * cd * dh[0],
        q_over_m * cd * dh[1],
        q_over_m * cd * dh[2],
    ];
    let aero_accel = [
        lift_accel[0] + drag_accel[0],
        lift_accel[1] + drag_accel[1],
        lift_accel[2] + drag_accel[2],
    ];
    let load_factor = (aero_accel[0] * aero_accel[0]
        + aero_accel[1] * aero_accel[1]
        + aero_accel[2] * aero_accel[2])
        .sqrt()
        / GRAVITY;
    let wind_work_rate =
        MASS_KG * (aero_accel[0] * wind[0] + aero_accel[1] * wind[1] + aero_accel[2] * wind[2]);
    AeroNum {
        aero_accel,
        lift_accel,
        drag_accel,
        air_dir: ah,
        zero_bank_lift_dir: n,
        side_dir: side,
        va,
        wind_speed,
        wind,
        cl,
        cd,
        load_factor,
        frame_guard,
        wind_work_rate,
    }
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
) -> Result<InterpolatedTrajectory<State<f64>, Control<f64>, Design<f64>>> {
    let g = active_design(params)?;
    let sample_count = 2 * params.transcription.intervals + 1;
    if sample_count < 3 {
        return Err(anyhow!("initial guess requires at least 2 intervals"));
    }
    let tf = g.tf;
    let dt = tf / (sample_count as f64 - 1.0);
    let theta = deg_to_rad(params.initial_wave_rotation_deg);
    let amp_y = params.initial_wave_amplitude_m * theta.cos();
    let amp_z = params.initial_wave_amplitude_m * theta.sin();
    let x_sine = (g.vx0 * tf - g.delta_l) / (2.0 * std::f64::consts::PI);
    let alpha0 = deg_to_rad(params.initial_alpha_deg);
    let roll_amp = deg_to_rad(params.initial_roll_amplitude_deg);
    let mut times = Vec::with_capacity(sample_count);
    let mut x_samples = Vec::with_capacity(sample_count);
    let mut u_samples = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let t = index as f64 * dt;
        let s = t / tf;
        let phase = 2.0 * std::f64::consts::PI * s;
        let bump = (std::f64::consts::PI * s).sin().powi(2);
        let bump_dot = std::f64::consts::PI * phase.sin() / tf;
        times.push(t);
        x_samples.push(State {
            px: g.delta_l * s + x_sine * phase.sin(),
            py: amp_y * bump,
            pz: g.h0 + amp_z * bump,
            vx: g.delta_l / tf + x_sine * (2.0 * std::f64::consts::PI / tf) * phase.cos(),
            vy: amp_y * bump_dot,
            vz: amp_z * bump_dot,
        });
        u_samples.push(Control {
            alpha: alpha0,
            roll: roll_amp * phase.sin(),
        });
    }
    let alpha_values = u_samples.iter().map(|u| u.alpha).collect::<Vec<_>>();
    let roll_values = u_samples.iter().map(|u| u.roll).collect::<Vec<_>>();
    let alpha_rate = finite_difference(&alpha_values, dt);
    let roll_rate = finite_difference(&roll_values, dt);
    let dudt_samples = alpha_rate
        .iter()
        .zip(roll_rate.iter())
        .map(|(alpha, roll)| Control {
            alpha: *alpha,
            roll: *roll,
        })
        .collect::<Vec<_>>();
    let trajectory = InterpolatedTrajectory {
        sample_times: times,
        x_samples,
        u_samples,
        dudt_samples,
        global: g.clone(),
        tf,
    };
    validate_initial_guess(params, &trajectory)?;
    Ok(trajectory)
}

fn validate_initial_guess(
    params: &Params,
    trajectory: &InterpolatedTrajectory<State<f64>, Control<f64>, Design<f64>>,
) -> Result<()> {
    let first = trajectory
        .x_samples
        .first()
        .ok_or_else(|| anyhow!("initial guess has no state samples"))?;
    let last = trajectory
        .x_samples
        .last()
        .ok_or_else(|| anyhow!("initial guess has no terminal sample"))?;
    let g = &trajectory.global;
    let tol = 1.0e-8;
    let checks = [
        ("px(0)", first.px),
        ("px(T)-px(0)-delta_l", last.px - first.px - g.delta_l),
        ("py(0)", first.py),
        ("py(T)", last.py),
        ("pz(0)-h0", first.pz - g.h0),
        ("pz(T)-pz(0)", last.pz - first.pz),
        ("vx(0)-vx0", first.vx - g.vx0),
        ("vx(T)-vx(0)", last.vx - first.vx),
        ("vy(0)", first.vy),
        ("vy(T)", last.vy),
        ("vz(0)", first.vz),
        ("vz(T)", last.vz),
    ];
    for (name, residual) in checks {
        if residual.abs() > tol {
            return Err(anyhow!(
                "initial guess violates {name}: residual {residual:.3e}"
            ));
        }
    }
    let path_bounds = path_bounds(params);
    for ((state, control), rate) in trajectory
        .x_samples
        .iter()
        .zip(trajectory.u_samples.iter())
        .zip(trajectory.dudt_samples.iter())
    {
        let aero = aero_numeric(state, control, params);
        let values = [
            ("altitude", state.pz, &path_bounds.altitude),
            ("airspeed", aero.va, &path_bounds.airspeed),
            ("CL", aero.cl, &path_bounds.cl),
            ("load_factor", aero.load_factor, &path_bounds.load_factor),
            ("frame_guard", aero.frame_guard, &path_bounds.frame_guard),
            ("alpha_rate", rate.alpha, &path_bounds.alpha_rate),
            ("roll_rate", rate.roll, &path_bounds.roll_rate),
        ];
        for (name, value, bounds) in values {
            if let Some(lower) = bounds.lower {
                if value < lower - 1.0e-8 {
                    return Err(anyhow!(
                        "initial guess violates lower {name} bound: {value:.6} < {lower:.6}"
                    ));
                }
            }
            if let Some(upper) = bounds.upper {
                if value > upper + 1.0e-8 {
                    return Err(anyhow!(
                        "initial guess violates upper {name} bound: {value:.6} > {upper:.6}"
                    ));
                }
            }
        }
    }
    Ok(())
}

fn path_bounds(params: &Params) -> Path<Bounds1D> {
    Path {
        altitude: Bounds1D {
            lower: Some(0.0),
            upper: None,
        },
        airspeed: Bounds1D {
            lower: Some(params.min_airspeed_mps),
            upper: Some(params.max_airspeed_mps),
        },
        cl: Bounds1D {
            lower: Some(-1.0),
            upper: Some(1.5),
        },
        load_factor: Bounds1D {
            lower: Some(0.0),
            upper: Some(params.max_load_factor),
        },
        frame_guard: Bounds1D {
            lower: Some(0.05),
            upper: None,
        },
        alpha_rate: Bounds1D {
            lower: Some(-deg_to_rad(params.max_alpha_rate_deg_s)),
            upper: Some(deg_to_rad(params.max_alpha_rate_deg_s)),
        },
        roll_rate: Bounds1D {
            lower: Some(-deg_to_rad(params.max_roll_rate_deg_s)),
            upper: Some(deg_to_rad(params.max_roll_rate_deg_s)),
        },
    }
}

fn global_bounds(params: &Params) -> Design<Bounds1D> {
    Design {
        delta_l: design_bounds(&params.delta_l),
        h0: design_bounds(&params.h0),
        vx0: design_bounds(&params.vx0),
        tf: design_bounds(&params.tf),
    }
}

fn ms_runtime(
    params: &Params,
) -> Result<
    MultipleShootingRuntimeValues<
        ModelParams<f64>,
        Path<Bounds1D>,
        Boundary<f64>,
        (),
        State<f64>,
        Control<f64>,
        Design<f64>,
        Design<Bounds1D>,
    >,
> {
    Ok(MultipleShootingRuntimeValues {
        parameters: model_params(params),
        beq: Boundary {
            px0: 0.0,
            delta_l: 0.0,
            py0: 0.0,
            py_t: 0.0,
            h0: 0.0,
            pz_periodic: 0.0,
            vx0: 0.0,
            vx_periodic: 0.0,
            vy0: 0.0,
            vy_t: 0.0,
            vz0: 0.0,
            vz_t: 0.0,
            alpha_periodic: 0.0,
            roll_periodic: 0.0,
        },
        bineq_bounds: (),
        path_bounds: path_bounds(params),
        global_bounds: global_bounds(params),
        initial_guess: optimal_control::runtime::MultipleShootingInitialGuess::Interpolated(
            continuous_guess(params)?,
        ),
        scaling: params.scaling_enabled.then(|| scaling(params)),
    })
}

fn dc_runtime(
    params: &Params,
) -> Result<
    DirectCollocationRuntimeValues<
        ModelParams<f64>,
        Path<Bounds1D>,
        Boundary<f64>,
        (),
        State<f64>,
        Control<f64>,
        Design<f64>,
        Design<Bounds1D>,
    >,
> {
    Ok(DirectCollocationRuntimeValues {
        parameters: model_params(params),
        beq: Boundary {
            px0: 0.0,
            delta_l: 0.0,
            py0: 0.0,
            py_t: 0.0,
            h0: 0.0,
            pz_periodic: 0.0,
            vx0: 0.0,
            vx_periodic: 0.0,
            vy0: 0.0,
            vy_t: 0.0,
            vz0: 0.0,
            vz_t: 0.0,
            alpha_periodic: 0.0,
            roll_periodic: 0.0,
        },
        bineq_bounds: (),
        path_bounds: path_bounds(params),
        global_bounds: global_bounds(params),
        initial_guess: optimal_control::runtime::DirectCollocationInitialGuess::Interpolated(
            continuous_guess(params)?,
        ),
        scaling: params.scaling_enabled.then(|| scaling(params)),
    })
}

fn scaling(
    params: &Params,
) -> optimal_control::OcpScaling<ModelParams<f64>, State<f64>, Control<f64>, Design<f64>> {
    optimal_control::OcpScaling {
        objective: 100.0,
        state: State {
            px: params.delta_l.value.abs().max(100.0),
            py: params.initial_wave_amplitude_m.max(10.0),
            pz: params.h0.value.abs().max(20.0),
            vx: params.vx0.value.abs().max(10.0),
            vy: params.vx0.value.abs().max(10.0),
            vz: params.vx0.value.abs().max(10.0),
        },
        control: Control {
            alpha: deg_to_rad(5.0),
            roll: deg_to_rad(45.0),
        },
        control_rate: Control {
            alpha: deg_to_rad(30.0),
            roll: deg_to_rad(120.0),
        },
        global: Design {
            delta_l: params.delta_l.value.abs().max(100.0),
            h0: params.h0.value.abs().max(20.0),
            vx0: params.vx0.value.abs().max(10.0),
            tf: params.tf.value.abs().max(5.0),
        },
        parameters: ModelParams {
            wind_azimuth_rad: 1.0,
            wind_low_mps: 10.0,
            wind_high_mps: 10.0,
            wind_mid_altitude_m: 20.0,
            wind_transition_height_m: 10.0,
            rate_weight: 1.0,
        },
        path: vec![
            20.0,
            20.0,
            1.0,
            5.0,
            1.0,
            deg_to_rad(30.0),
            deg_to_rad(120.0),
        ],
        boundary_equalities: vec![1.0; Boundary::<f64>::LEN],
        boundary_inequalities: Vec::new(),
    }
}

fn ms_key(params: &Params) -> AlbatrossMsKey {
    AlbatrossMsKey {
        base: multiple_shooting_compile_key(params.transcription.intervals, params.sx_functions),
        objective: params.objective,
    }
}

fn dc_key(params: &Params, family: optimal_control::CollocationFamily) -> AlbatrossDcKey {
    AlbatrossDcKey {
        base: direct_collocation_compile_key_with_sx(
            params.transcription.intervals,
            params.transcription.collocation_degree,
            family,
            params.transcription.time_grid,
            params.sx_functions,
        ),
        objective: params.objective,
    }
}

fn ms_variant(key: AlbatrossMsKey) -> (String, String) {
    let (id, label) = multiple_shooting_variant_with_sx(key.base);
    (
        format!("{id}__obj_{}", key.objective.id()),
        format!("{label} · {}", key.objective.label()),
    )
}

fn dc_variant(key: AlbatrossDcKey) -> (String, String) {
    let (id, label) = direct_collocation_variant_with_sx(key.base);
    (
        format!("{id}__obj_{}", key.objective.id()),
        format!("{label} · {}", key.objective.label()),
    )
}

pub(crate) fn compile_variant_for_values(
    values: &BTreeMap<String, f64>,
) -> Option<(String, String)> {
    let params = Params::from_map(values).ok()?;
    if params.transcription.method == crate::common::TranscriptionMethod::MultipleShooting {
        Some(ms_variant(ms_key(&params)))
    } else {
        Some(dc_variant(dc_key(
            &params,
            params.transcription.collocation_family,
        )))
    }
}

fn cached_multiple_shooting(params: &Params) -> Result<crate::common::CachedCompile<MsCompiled>> {
    MULTIPLE_SHOOTING_CACHE.with(|cache| {
        cache.borrow_mut().get_or_try_init(ms_key(params), || {
            model(
                MultipleShooting {
                    intervals: params.transcription.intervals,
                    rk4_substeps: RK4_SUBSTEPS,
                },
                params.objective,
            )
            .compile_jit_with_ocp_options(ocp_compile_options(
                crate::common::interactive_multiple_shooting_opt_level(),
                params.sx_functions,
            ))
            .map_err(Into::into)
        })
    })
}

fn cached_direct_collocation(
    params: &Params,
    family: optimal_control::CollocationFamily,
) -> Result<crate::common::CachedCompile<DcCompiled>> {
    DIRECT_COLLOCATION_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .get_or_try_init(dc_key(params, family), || {
                model(
                    DirectCollocation {
                        intervals: params.transcription.intervals,
                        order: params.transcription.collocation_degree,
                        family,
                        time_grid: params.transcription.time_grid,
                    },
                    params.objective,
                )
                .compile_jit_with_ocp_options(ocp_compile_options(
                    crate::common::interactive_direct_collocation_opt_level(),
                    params.sx_functions,
                ))
                .map_err(Into::into)
            })
    })
}

fn compile_multiple_shooting_with_progress(
    params: &Params,
    on_symbolic_ready: &mut dyn FnMut(CompileProgressUpdate),
) -> Result<(
    std::rc::Rc<std::cell::RefCell<MsCompiled>>,
    CompileProgressInfo,
)> {
    MULTIPLE_SHOOTING_CACHE.with(|cache| {
        crate::common::cached_compile_with_progress(
            &mut cache.borrow_mut(),
            ms_key(params),
            on_symbolic_ready,
            |on_progress| {
                let mut progress_state = OcpCompileProgressState::default();
                model(
                    MultipleShooting {
                        intervals: params.transcription.intervals,
                        rk4_substeps: RK4_SUBSTEPS,
                    },
                    params.objective,
                )
                .compile_jit_with_ocp_options_and_progress_callback(
                    ocp_compile_options(
                        crate::common::interactive_multiple_shooting_opt_level(),
                        params.sx_functions,
                    ),
                    &mut |progress| {
                        on_progress(ocp_compile_progress_update(progress, &mut progress_state))
                    },
                )
                .map_err(Into::into)
            },
            compile_progress_info_from_compiled,
        )
    })
}

fn compile_direct_collocation_with_progress(
    params: &Params,
    family: optimal_control::CollocationFamily,
    on_symbolic_ready: &mut dyn FnMut(CompileProgressUpdate),
) -> Result<(
    std::rc::Rc<std::cell::RefCell<DcCompiled>>,
    CompileProgressInfo,
)> {
    DIRECT_COLLOCATION_CACHE.with(|cache| {
        crate::common::cached_compile_with_progress(
            &mut cache.borrow_mut(),
            dc_key(params, family),
            on_symbolic_ready,
            |on_progress| {
                let mut progress_state = OcpCompileProgressState::default();
                model(
                    DirectCollocation {
                        intervals: params.transcription.intervals,
                        order: params.transcription.collocation_degree,
                        family,
                        time_grid: params.transcription.time_grid,
                    },
                    params.objective,
                )
                .compile_jit_with_ocp_options_and_progress_callback(
                    ocp_compile_options(
                        crate::common::interactive_direct_collocation_opt_level(),
                        params.sx_functions,
                    ),
                    &mut |progress| {
                        on_progress(ocp_compile_progress_update(progress, &mut progress_state))
                    },
                )
                .map_err(Into::into)
            },
            compile_progress_info_from_compiled,
        )
    })
}

pub fn compile_cache_statuses() -> Vec<CompileCacheStatus> {
    MULTIPLE_SHOOTING_CACHE.with(|cache| {
        DIRECT_COLLOCATION_CACHE.with(|dc_cache| {
            let mut out = crate::common::collect_compile_cache_statuses(
                ProblemId::AlbatrossDynamicSoaring,
                PROBLEM_NAME,
                &cache.borrow(),
                ms_variant,
            );
            out.extend(crate::common::collect_compile_cache_statuses(
                ProblemId::AlbatrossDynamicSoaring,
                PROBLEM_NAME,
                &dc_cache.borrow(),
                dc_variant,
            ));
            out
        })
    })
}

pub fn prewarm(params: &Params) -> Result<()> {
    match params.transcription.method {
        crate::common::TranscriptionMethod::MultipleShooting => {
            cached_multiple_shooting(params).map(|_| ())
        }
        crate::common::TranscriptionMethod::DirectCollocation => {
            cached_direct_collocation(params, params.transcription.collocation_family).map(|_| ())
        }
    }
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

fn runtime_ms_for_standard(
    params: &Params,
) -> MultipleShootingRuntimeValues<
    ModelParams<f64>,
    Path<Bounds1D>,
    Boundary<f64>,
    (),
    State<f64>,
    Control<f64>,
    Design<f64>,
    Design<Bounds1D>,
> {
    ms_runtime(params).expect("validated albatross multiple-shooting runtime")
}

fn runtime_dc_for_standard(
    params: &Params,
) -> DirectCollocationRuntimeValues<
    ModelParams<f64>,
    Path<Bounds1D>,
    Boundary<f64>,
    (),
    State<f64>,
    Control<f64>,
    Design<f64>,
    Design<Bounds1D>,
> {
    dc_runtime(params).expect("validated albatross direct-collocation runtime")
}

pub fn solve(params: &Params) -> Result<SolveArtifact> {
    active_design(params)?;
    match params.transcription.method {
        crate::common::TranscriptionMethod::MultipleShooting => {
            let runtime = ms_runtime(params)?;
            let compiled = cached_multiple_shooting(params)?;
            crate::common::solve_cached_multiple_shooting_problem(
                &compiled.compiled,
                &runtime,
                params.solver_method,
                &params.solver,
                |trajectories, x_arcs, u_arcs| {
                    artifact_from_ms(params, trajectories, x_arcs, u_arcs)
                },
            )
        }
        crate::common::TranscriptionMethod::DirectCollocation => {
            let runtime = dc_runtime(params)?;
            let compiled =
                cached_direct_collocation(params, params.transcription.collocation_family)?;
            crate::common::solve_cached_direct_collocation_problem(
                &compiled.compiled,
                &runtime,
                params.solver_method,
                &params.solver,
                |trajectories, time_grid| artifact_from_dc(params, trajectories, time_grid),
            )
        }
    }
}

pub fn solve_with_progress<F>(params: &Params, emit: F) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    active_design(params)?;
    crate::common::solve_standard_ocp_with_progress::<_, _, _, _, _, _, _, _, _, _>(
        params,
        params.transcription.method,
        params.transcription.collocation_family,
        params.solver_method,
        &params.solver,
        emit,
        compile_multiple_shooting_with_progress,
        compile_direct_collocation_with_progress,
        runtime_ms_for_standard,
        runtime_dc_for_standard,
        |trajectories, x_arcs, u_arcs| artifact_from_ms(params, trajectories, x_arcs, u_arcs),
        |trajectories, time_grid| artifact_from_dc(params, trajectories, time_grid),
    )
}

pub(crate) fn solve_from_map(values: &BTreeMap<String, f64>) -> Result<SolveArtifact> {
    crate::common::solve_from_value_map::<Params, _>(values, solve)
}

pub(crate) fn prewarm_from_map(values: &BTreeMap<String, f64>) -> Result<()> {
    crate::common::prewarm_from_value_map::<Params, _>(values, prewarm)
}

pub(crate) fn solve_with_progress_boxed(
    values: &BTreeMap<String, f64>,
    emit: Box<dyn FnMut(SolveStreamEvent) + Send>,
) -> Result<SolveArtifact> {
    crate::common::solve_with_progress_from_value_map::<Params, _>(values, emit, |params, emit| {
        solve_with_progress(params, emit)
    })
}

pub(crate) fn prewarm_with_progress_boxed(
    values: &BTreeMap<String, f64>,
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
        ProblemId::AlbatrossDynamicSoaring,
        PROBLEM_NAME,
        params,
        params.transcription.method,
        params.transcription.collocation_family,
        params.sx_functions,
        request,
        cached_multiple_shooting,
        cached_direct_collocation,
        runtime_ms_for_standard,
        runtime_dc_for_standard,
    )
}

pub(crate) fn validate_derivatives_from_request(
    request: &crate::common::DerivativeCheckRequest,
) -> Result<crate::common::ProblemDerivativeCheck> {
    let mut params = Params::from_map(&request.values)?;
    crate::common::apply_derivative_request_overrides(&mut params, request);
    validate_derivatives(&params, request)
}

pub(crate) fn benchmark_default_case_with_progress(
    transcription: crate::common::TranscriptionMethod,
    preset: crate::benchmark_report::OcpBenchmarkPreset,
    eval_options: optimization::NlpEvaluationBenchmarkOptions,
    on_progress: &mut dyn FnMut(crate::benchmark_report::BenchmarkCaseProgress),
) -> Result<crate::benchmark_report::OcpBenchmarkRecord> {
    crate::common::benchmark_standard_ocp_case_with_progress::<_, _, _, _, _, _, _, _>(
        ProblemId::AlbatrossDynamicSoaring,
        PROBLEM_NAME,
        transcription,
        preset,
        eval_options,
        on_progress,
        |options, on_progress| {
            model(
                MultipleShooting {
                    intervals: DEFAULT_INTERVALS,
                    rk4_substeps: RK4_SUBSTEPS,
                },
                ObjectiveKind::AverageSpeed,
            )
            .compile_jit_with_ocp_options_and_progress_callback(options, on_progress)
        },
        |family, options, on_progress| {
            model(
                DirectCollocation {
                    intervals: DEFAULT_INTERVALS,
                    order: DEFAULT_COLLOCATION_DEGREE,
                    family,
                    time_grid: Default::default(),
                },
                ObjectiveKind::AverageSpeed,
            )
            .compile_jit_with_ocp_options_and_progress_callback(options, on_progress)
        },
        runtime_ms_for_standard,
        runtime_dc_for_standard,
    )
}

fn series(name: &str, x: Vec<f64>, y: Vec<f64>) -> TimeSeries {
    TimeSeries {
        name: name.to_string(),
        x,
        y,
        mode: Some(PlotMode::LinesMarkers),
        legend_group: Some(name.to_string()),
        show_legend: true,
        role: crate::common::TimeSeriesRole::Data,
    }
}

fn constant_bound_series(
    name: &str,
    legend_group: &str,
    times: &[f64],
    value: f64,
    role: crate::common::TimeSeriesRole,
) -> TimeSeries {
    TimeSeries {
        name: name.to_string(),
        x: times.to_vec(),
        y: vec![value; times.len()],
        mode: Some(PlotMode::Lines),
        legend_group: Some(legend_group.to_string()),
        show_legend: true,
        role,
    }
}

fn diagnostics_for_nodes(
    params: &Params,
    states: &[State<f64>],
    controls: &[Control<f64>],
    rates: &[Control<f64>],
) -> Vec<(AeroNum, f64)> {
    states
        .iter()
        .zip(controls.iter())
        .zip(rates.iter())
        .map(|((state, control), _rate)| {
            let aero = aero_numeric(state, control, params);
            let ground_speed =
                (state.vx * state.vx + state.vy * state.vy + state.vz * state.vz).sqrt();
            (aero, ground_speed)
        })
        .collect()
}

fn artifact_common(
    params: &Params,
    states: Vec<State<f64>>,
    controls: Vec<Control<f64>>,
    rates: Vec<Control<f64>>,
    times: Vec<f64>,
    tf: f64,
    global: &Design<f64>,
    notes: Vec<String>,
) -> SolveArtifact {
    let diagnostics = diagnostics_for_nodes(params, &states, &controls, &rates);
    let px = states.iter().map(|s| s.px).collect::<Vec<_>>();
    let py = states.iter().map(|s| s.py).collect::<Vec<_>>();
    let pz = states.iter().map(|s| s.pz).collect::<Vec<_>>();
    let vx = states.iter().map(|s| s.vx).collect::<Vec<_>>();
    let vy = states.iter().map(|s| s.vy).collect::<Vec<_>>();
    let vz = states.iter().map(|s| s.vz).collect::<Vec<_>>();
    let airspeed = diagnostics.iter().map(|(a, _)| a.va).collect::<Vec<_>>();
    let ground_speed = diagnostics.iter().map(|(_, v)| *v).collect::<Vec<_>>();
    let wind_speed = diagnostics
        .iter()
        .map(|(a, _)| a.wind_speed)
        .collect::<Vec<_>>();
    let wind_x = diagnostics
        .iter()
        .map(|(a, _)| a.wind[0])
        .collect::<Vec<_>>();
    let wind_y = diagnostics
        .iter()
        .map(|(a, _)| a.wind[1])
        .collect::<Vec<_>>();
    let wind_z = diagnostics
        .iter()
        .map(|(a, _)| a.wind[2])
        .collect::<Vec<_>>();
    let cl = diagnostics.iter().map(|(a, _)| a.cl).collect::<Vec<_>>();
    let cd = diagnostics.iter().map(|(a, _)| a.cd).collect::<Vec<_>>();
    let ld = diagnostics
        .iter()
        .map(|(a, _)| a.cl / a.cd.max(1.0e-9))
        .collect::<Vec<_>>();
    let load = diagnostics
        .iter()
        .map(|(a, _)| a.load_factor)
        .collect::<Vec<_>>();
    let frame_guard = diagnostics
        .iter()
        .map(|(a, _)| a.frame_guard)
        .collect::<Vec<_>>();
    let wind_work = diagnostics
        .iter()
        .map(|(a, _)| a.wind_work_rate)
        .collect::<Vec<_>>();
    let q = airspeed
        .iter()
        .map(|v| 0.5 * AIR_DENSITY * v * v)
        .collect::<Vec<_>>();
    let regularization_density = rates
        .iter()
        .map(|u| params.rate_regularization * (u.alpha * u.alpha + u.roll * u.roll))
        .collect::<Vec<_>>();
    let objective_density = diagnostics
        .iter()
        .zip(rates.iter())
        .map(|((a, _), u)| {
            let regularization = params.rate_regularization * (u.alpha * u.alpha + u.roll * u.roll);
            match params.objective {
                ObjectiveKind::WindWork => regularization - a.wind_work_rate,
                ObjectiveKind::AverageSpeed
                | ObjectiveKind::TerminalEnergy
                | ObjectiveKind::ControlRegularization => regularization,
            }
        })
        .collect::<Vec<_>>();
    let kinetic = states
        .iter()
        .map(|s| 0.5 * MASS_KG * (s.vx * s.vx + s.vy * s.vy + s.vz * s.vz))
        .collect::<Vec<_>>();
    let potential = states
        .iter()
        .map(|s| MASS_KG * GRAVITY * s.pz)
        .collect::<Vec<_>>();
    let total_energy = kinetic
        .iter()
        .zip(potential.iter())
        .map(|(k, p)| k + p)
        .collect::<Vec<_>>();
    let path_bounds = path_bounds(params);
    let mut charts = vec![
        chart(
            "Position",
            "m",
            vec![
                series("px", times.clone(), px.clone()),
                series("py", times.clone(), py.clone()),
                series("pz", times.clone(), pz.clone()),
            ],
        ),
        chart(
            "Velocity",
            "m/s",
            vec![
                series("vx", times.clone(), vx),
                series("vy", times.clone(), vy),
                series("vz", times.clone(), vz),
            ],
        ),
        chart(
            "Controls",
            "deg",
            vec![
                series(
                    "alpha",
                    times.clone(),
                    controls.iter().map(|u| rad_to_deg(u.alpha)).collect(),
                ),
                series(
                    "roll",
                    times.clone(),
                    controls.iter().map(|u| rad_to_deg(u.roll)).collect(),
                ),
            ],
        ),
        chart(
            "Control Rates",
            "deg/s",
            vec![
                series(
                    "alpha_dot",
                    times.clone(),
                    rates.iter().map(|u| rad_to_deg(u.alpha)).collect(),
                ),
                series(
                    "roll_dot",
                    times.clone(),
                    rates.iter().map(|u| rad_to_deg(u.roll)).collect(),
                ),
            ],
        ),
        chart(
            "Airspeed and Wind",
            "m/s",
            vec![
                series("airspeed", times.clone(), airspeed.clone()),
                series("ground speed", times.clone(), ground_speed),
                series("wind speed", times.clone(), wind_speed),
            ],
        ),
        chart(
            "Wind Components",
            "m/s",
            vec![
                series("Wx", times.clone(), wind_x),
                series("Wy", times.clone(), wind_y),
                series("Wz", times.clone(), wind_z),
            ],
        ),
        chart(
            "Aero Coefficients",
            "-",
            vec![
                series("CL", times.clone(), cl.clone()),
                series("CD", times.clone(), cd),
                series("L/D", times.clone(), ld),
            ],
        ),
        chart(
            "Path Constraints",
            "-",
            vec![
                series("load factor", times.clone(), load),
                series("frame guard", times.clone(), frame_guard),
            ],
        ),
        chart(
            "Energy",
            "J",
            vec![
                series("kinetic", times.clone(), kinetic),
                series("potential", times.clone(), potential),
                series("total", times.clone(), total_energy),
            ],
        ),
        chart(
            "Wind Work Rate",
            "W",
            vec![series("F_aero · W", times.clone(), wind_work)],
        ),
        chart(
            "Objective Contribution",
            "-",
            vec![
                series("objective density", times.clone(), objective_density),
                series("rate regularization", times.clone(), regularization_density),
            ],
        ),
        chart(
            "Dynamic Pressure",
            "Pa",
            vec![series("q", times.clone(), q)],
        ),
    ];
    if let Some(lower) = path_bounds.altitude.lower {
        charts[0].series.push(constant_bound_series(
            "pz lower",
            "pz",
            &times,
            lower,
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.altitude.upper {
        charts[0].series.push(constant_bound_series(
            "pz upper",
            "pz",
            &times,
            upper,
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    if let Some(lower) = path_bounds.airspeed.lower {
        charts[4].series.push(constant_bound_series(
            "airspeed lower",
            "airspeed",
            &times,
            lower,
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.airspeed.upper {
        charts[4].series.push(constant_bound_series(
            "airspeed upper",
            "airspeed",
            &times,
            upper,
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    if let Some(lower) = path_bounds.cl.lower {
        charts[2].series.push(constant_bound_series(
            "alpha lower from CL",
            "alpha",
            &times,
            rad_to_deg(lower / CL_SLOPE),
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.cl.upper {
        charts[2].series.push(constant_bound_series(
            "alpha upper from CL",
            "alpha",
            &times,
            rad_to_deg(upper / CL_SLOPE),
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    if let Some(lower) = path_bounds.alpha_rate.lower {
        charts[3].series.push(constant_bound_series(
            "alpha_dot lower",
            "alpha_dot",
            &times,
            rad_to_deg(lower),
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.alpha_rate.upper {
        charts[3].series.push(constant_bound_series(
            "alpha_dot upper",
            "alpha_dot",
            &times,
            rad_to_deg(upper),
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    if let Some(lower) = path_bounds.roll_rate.lower {
        charts[3].series.push(constant_bound_series(
            "roll_dot lower",
            "roll_dot",
            &times,
            rad_to_deg(lower),
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.roll_rate.upper {
        charts[3].series.push(constant_bound_series(
            "roll_dot upper",
            "roll_dot",
            &times,
            rad_to_deg(upper),
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    if let Some(lower) = path_bounds.cl.lower {
        charts[6].series.push(constant_bound_series(
            "CL lower",
            "CL",
            &times,
            lower,
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.cl.upper {
        charts[6].series.push(constant_bound_series(
            "CL upper",
            "CL",
            &times,
            upper,
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    if let Some(lower) = path_bounds.load_factor.lower {
        charts[7].series.push(constant_bound_series(
            "load lower",
            "load factor",
            &times,
            lower,
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.load_factor.upper {
        charts[7].series.push(constant_bound_series(
            "load upper",
            "load factor",
            &times,
            upper,
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    if let Some(lower) = path_bounds.frame_guard.lower {
        charts[7].series.push(constant_bound_series(
            "frame guard lower",
            "frame guard",
            &times,
            lower,
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.frame_guard.upper {
        charts[7].series.push(constant_bound_series(
            "frame guard upper",
            "frame guard",
            &times,
            upper,
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    let mut paths = vec![ScenePath3D {
        name: "trajectory".to_string(),
        x: px.clone(),
        y: py.clone(),
        z: pz.clone(),
    }];
    let stride = (states.len() / 12).max(1);
    for (idx, (state, (aero, _))) in states
        .iter()
        .zip(diagnostics.iter())
        .enumerate()
        .step_by(stride)
    {
        let scale = 2.0;
        let frame_scale = 6.0;
        paths.push(ScenePath3D {
            name: format!("lift {idx}"),
            x: vec![state.px, state.px + scale * aero.lift_accel[0]],
            y: vec![state.py, state.py + scale * aero.lift_accel[1]],
            z: vec![state.pz, state.pz + scale * aero.lift_accel[2]],
        });
        paths.push(ScenePath3D {
            name: format!("drag {idx}"),
            x: vec![state.px, state.px + scale * aero.drag_accel[0]],
            y: vec![state.py, state.py + scale * aero.drag_accel[1]],
            z: vec![state.pz, state.pz + scale * aero.drag_accel[2]],
        });
        paths.push(ScenePath3D {
            name: format!("aero accel {idx}"),
            x: vec![state.px, state.px + scale * aero.aero_accel[0]],
            y: vec![state.py, state.py + scale * aero.aero_accel[1]],
            z: vec![state.pz, state.pz + scale * aero.aero_accel[2]],
        });
        paths.push(ScenePath3D {
            name: format!("wind {idx}"),
            x: vec![state.px, state.px + aero.wind[0]],
            y: vec![state.py, state.py + aero.wind[1]],
            z: vec![state.pz, state.pz + aero.wind[2]],
        });
        paths.push(ScenePath3D {
            name: format!("air axis {idx}"),
            x: vec![state.px, state.px + frame_scale * aero.air_dir[0]],
            y: vec![state.py, state.py + frame_scale * aero.air_dir[1]],
            z: vec![state.pz, state.pz + frame_scale * aero.air_dir[2]],
        });
        paths.push(ScenePath3D {
            name: format!("zero-bank frame {idx}"),
            x: vec![
                state.px,
                state.px + frame_scale * aero.zero_bank_lift_dir[0],
            ],
            y: vec![
                state.py,
                state.py + frame_scale * aero.zero_bank_lift_dir[1],
            ],
            z: vec![
                state.pz,
                state.pz + frame_scale * aero.zero_bank_lift_dir[2],
            ],
        });
        paths.push(ScenePath3D {
            name: format!("side frame {idx}"),
            x: vec![state.px, state.px + frame_scale * aero.side_dir[0]],
            y: vec![state.py, state.py + frame_scale * aero.side_dir[1]],
            z: vec![state.pz, state.pz + frame_scale * aero.side_dir[2]],
        });
    }
    for y_offset in [-30.0, 0.0, 30.0] {
        let z_values = (0..=24).map(|i| i as f64 * 3.0).collect::<Vec<_>>();
        let wind_vectors = z_values
            .iter()
            .map(|z| wind_numeric(*z, params).0)
            .collect::<Vec<_>>();
        let x_values = wind_vectors
            .iter()
            .map(|wind| wind[0] * 3.0)
            .collect::<Vec<_>>();
        let y_values = wind_vectors
            .iter()
            .map(|wind| y_offset + wind[1] * 3.0)
            .collect::<Vec<_>>();
        paths.push(ScenePath3D {
            name: format!("wind shear y={y_offset:.0}"),
            x: x_values,
            y: y_values,
            z: z_values,
        });
    }
    let average_speed = global.delta_l / tf;
    let mut summary = transcription_metrics(&params.transcription).to_vec();
    summary.extend([
        numeric_metric_with_key(
            MetricKey::Distance,
            "Delta L",
            global.delta_l,
            format!("{:.1} m", global.delta_l),
        ),
        numeric_metric_with_key(MetricKey::FinalTime, "T", tf, format!("{tf:.2} s")),
        numeric_metric_with_key(
            MetricKey::MaxSpeed,
            "Average Speed",
            average_speed,
            format!("{average_speed:.2} m/s"),
        ),
        metric_with_key(MetricKey::Custom, "Objective", params.objective.label()),
    ]);
    let mut artifact = SolveArtifact::new(
        PROBLEM_NAME,
        summary,
        SolverReport::placeholder(),
        charts,
        Scene2D {
            title: "Top View".to_string(),
            x_label: "px (m)".to_string(),
            y_label: "py (m)".to_string(),
            paths: vec![ScenePath {
                name: "trajectory".to_string(),
                x: px,
                y: py,
            }],
            circles: Vec::new(),
            arrows: Vec::new(),
            animation: None,
        },
        notes,
    );
    artifact
        .visualizations
        .push(ArtifactVisualization::Paths3D {
            title: "3D Dynamic Soaring Trajectory".to_string(),
            x_label: "px (m)".to_string(),
            y_label: "py (m)".to_string(),
            z_label: "pz (m)".to_string(),
            paths,
        });
    artifact
}

fn artifact_from_ms(
    params: &Params,
    trajectories: &MultipleShootingTrajectories<State<f64>, Control<f64>, Design<f64>>,
    x_arcs: &[IntervalArc<State<f64>>],
    _u_arcs: &[IntervalArc<Control<f64>>],
) -> SolveArtifact {
    let intervals = trajectories.interval_count();
    let times = node_times(trajectories.tf, intervals);
    let mut states = trajectories.x.nodes.clone();
    states.push(trajectories.x.terminal.clone());
    let mut controls = trajectories.u.nodes.clone();
    controls.push(trajectories.u.terminal.clone());
    let mut rates = trajectories.dudt.clone();
    rates.push(rates.last().cloned().unwrap_or(Control {
        alpha: 0.0,
        roll: 0.0,
    }));
    let mut artifact = artifact_common(
        params,
        states,
        controls,
        rates,
        times,
        trajectories.tf,
        &trajectories.global,
        vec!["Multiple shooting renders mesh-node diagnostics; RK4 interval arcs are still used for state continuity internally.".to_string()],
    );
    artifact.charts.push(chart(
        "px Arc Reconstruction",
        "px (m)",
        interval_arc_series("px", x_arcs, PlotMode::LinesMarkers, |x| x.px),
    ));
    artifact
}

fn artifact_from_dc(
    params: &Params,
    trajectories: &DirectCollocationTrajectories<State<f64>, Control<f64>, Design<f64>>,
    time_grid: &DirectCollocationTimeGrid,
) -> SolveArtifact {
    let x_arcs =
        direct_collocation_state_like_arcs(&trajectories.x, &trajectories.root_x, time_grid)
            .expect("collocation state arcs should match");
    let u_arcs =
        direct_collocation_state_like_arcs(&trajectories.u, &trajectories.root_u, time_grid)
            .expect("collocation control arcs should match");
    let dudt_arcs = direct_collocation_root_arcs(&trajectories.root_dudt, time_grid);
    let times = x_arcs
        .iter()
        .flat_map(|arc| arc.times.iter().copied())
        .collect::<Vec<_>>();
    let states = x_arcs
        .iter()
        .flat_map(|arc| arc.values.iter().cloned())
        .collect::<Vec<_>>();
    let controls = u_arcs
        .iter()
        .flat_map(|arc| arc.values.iter().cloned())
        .collect::<Vec<_>>();
    let mut rates = dudt_arcs
        .iter()
        .flat_map(|arc| arc.values.iter().cloned())
        .collect::<Vec<_>>();
    while rates.len() < states.len() {
        rates.push(rates.last().cloned().unwrap_or(Control {
            alpha: 0.0,
            roll: 0.0,
        }));
    }
    artifact_common(
        params,
        states,
        controls,
        rates,
        times,
        trajectories.tf,
        &trajectories.global,
        vec!["Direct collocation charts include interval-local start/root/end state and control samples; rate diagnostics use root controls padded to state-like samples for plotting.".to_string()],
    )
}

pub(crate) fn problem_entry() -> crate::ProblemEntry {
    crate::ProblemEntry {
        id: ProblemId::AlbatrossDynamicSoaring,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_transcription_is_direct_collocation() {
        assert_eq!(
            Params::default().transcription.method,
            crate::common::TranscriptionMethod::DirectCollocation,
        );
        let spec = spec();
        let transcription = spec
            .controls
            .iter()
            .find(|control| control.id == "transcription_method")
            .expect("transcription method control should be present");
        assert_eq!(transcription.default, 1.0);
    }

    #[test]
    fn parses_wind_controls_and_design_bounds() {
        let mut values = BTreeMap::new();
        values.insert("wind_azimuth_deg".to_string(), 45.0);
        values.insert("wind_high_mps".to_string(), 18.0);
        values.insert("delta_l_free".to_string(), 1.0);
        values.insert("delta_l_value".to_string(), 180.0);
        values.insert("delta_l_lower".to_string(), 100.0);
        values.insert("delta_l_upper".to_string(), 250.0);
        let params = Params::from_map(&values).expect("params should parse");
        assert_eq!(params.wind_azimuth_deg, 45.0);
        assert!(!params.delta_l.fixed);
        assert_eq!(
            active_design(&params)
                .expect("design should be active")
                .delta_l,
            180.0
        );
    }

    #[test]
    fn invalid_design_bounds_fail() {
        let mut values = BTreeMap::new();
        values.insert("h0_free".to_string(), 1.0);
        values.insert("h0_lower".to_string(), 30.0);
        values.insert("h0_upper".to_string(), 20.0);
        let err = Params::from_map(&values).expect_err("invalid h0 bounds should fail");
        assert!(err.to_string().contains("h0_lower"));
    }

    #[test]
    fn fixed_design_ignores_stale_free_bound_editors() {
        let mut values = BTreeMap::new();
        values.insert("delta_l_free".to_string(), 0.0);
        values.insert("delta_l_value".to_string(), 180.0);
        values.insert("delta_l_lower".to_string(), 250.0);
        values.insert("delta_l_upper".to_string(), 100.0);
        let params = Params::from_map(&values).expect("fixed design should ignore free bounds");
        let bounds = global_bounds(&params);
        assert_eq!(bounds.delta_l.lower, Some(180.0));
        assert_eq!(bounds.delta_l.upper, Some(180.0));
    }

    #[test]
    fn free_final_time_must_have_positive_lower_bound() {
        let mut values = BTreeMap::new();
        values.insert("tf_free".to_string(), 1.0);
        values.insert("tf_value".to_string(), 10.0);
        values.insert("tf_lower".to_string(), 0.0);
        values.insert("tf_upper".to_string(), 20.0);
        let err =
            Params::from_map(&values).expect_err("nonpositive free T lower bound should fail");
        assert!(err.to_string().contains("tf_lower"));
    }

    #[test]
    fn initial_guess_satisfies_boundary_and_active_bounds() {
        let params = Params::default();
        let guess = continuous_guess(&params).expect("default initial guess should be feasible");
        validate_initial_guess(&params, &guess).expect("initial guess should validate");
    }

    #[test]
    fn objective_changes_compile_key() {
        let mut values = BTreeMap::new();
        values.insert("objective".to_string(), 0.0);
        let avg = compile_variant_for_values(&values).expect("avg variant");
        values.insert("objective".to_string(), 1.0);
        let wind = compile_variant_for_values(&values).expect("wind variant");
        assert_ne!(avg.0, wind.0);
    }

    #[test]
    fn artifact_visualization_includes_force_wind_and_frame_paths() {
        let params = Params::default();
        let guess = continuous_guess(&params).expect("default guess should build");
        let artifact = artifact_common(
            &params,
            guess.x_samples.clone(),
            guess.u_samples.clone(),
            guess.dudt_samples.clone(),
            guess.sample_times.clone(),
            guess.tf,
            &guess.global,
            Vec::new(),
        );
        let paths = artifact
            .visualizations
            .iter()
            .find_map(|visualization| match visualization {
                ArtifactVisualization::Paths3D { paths, .. } => Some(paths),
                _ => None,
            })
            .expect("albatross artifact should include a 3D path visualization");
        assert!(paths.iter().any(|path| path.name == "trajectory"));
        assert!(paths.iter().any(|path| path.name.starts_with("lift ")));
        assert!(paths.iter().any(|path| path.name.starts_with("drag ")));
        assert!(
            paths
                .iter()
                .any(|path| path.name.starts_with("aero accel "))
        );
        assert!(paths.iter().any(|path| path.name.starts_with("wind ")));
        assert!(
            paths
                .iter()
                .any(|path| path.name.starts_with("wind shear "))
        );
        assert!(paths.iter().any(|path| path.name.starts_with("air axis ")));
        assert!(
            paths
                .iter()
                .any(|path| path.name.starts_with("zero-bank frame "))
        );
        assert!(
            paths
                .iter()
                .any(|path| path.name.starts_with("side frame "))
        );

        let position = artifact
            .charts
            .iter()
            .find(|chart| chart.title == "Position")
            .expect("position chart should exist");
        let altitude_lower = position
            .series
            .iter()
            .find(|series| series.name == "pz lower")
            .expect("position chart should label the altitude lower bound");
        assert_eq!(altitude_lower.legend_group.as_deref(), Some("pz"));

        let aero_coefficients = artifact
            .charts
            .iter()
            .find(|chart| chart.title == "Aero Coefficients")
            .expect("aero coefficient chart should exist");
        let cl_upper = aero_coefficients
            .series
            .iter()
            .find(|series| series.name == "CL upper")
            .expect("aero coefficient chart should label the CL upper bound");
        assert_eq!(cl_upper.legend_group.as_deref(), Some("CL"));
    }
}
