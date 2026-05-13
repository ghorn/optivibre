use crate::common::{
    ArtifactVisualization, Chart, CompileCacheStatus, CompileProgressInfo, CompileProgressUpdate,
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
const DEFAULT_INTERVALS: usize = 50;
const DEFAULT_COLLOCATION_DEGREE: usize = 3;
const SUPPORTED_INTERVALS: [usize; 7] = [12, 18, 28, 36, 48, DEFAULT_INTERVALS, 64];
const SUPPORTED_DEGREES: [usize; 4] = [2, DEFAULT_COLLOCATION_DEGREE, 4, 5];

const DEFAULT_GRAVITY_MPS2: f64 = 9.81;
const DEFAULT_AIR_DENSITY_KG_M3: f64 = 1.2;
const DEFAULT_MASS_KG: f64 = 8.5;
const DEFAULT_REFERENCE_AREA_M2: f64 = 0.65;
const DEFAULT_CL_SLOPE_PER_RAD: f64 = 5.7;
const DEFAULT_CD0: f64 = 0.0125;
const DEFAULT_ASPECT_RATIO: f64 = 10.0;
const DEFAULT_OSWALD_EFFICIENCY: f64 = 0.65;
const DEFAULT_SPEED_EPS_MPS: f64 = 1.0e-3;
const DEFAULT_FRAME_EPS: f64 = 1.0e-4;
const WIND_SHEAR_PROFILE_SAMPLES: usize = 73;
const WIND_SHEAR_PROFILE_LANES: usize = 5;
const DRAG_GLYPH_EXAGGERATION: f64 = 5.0;

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
            Self::WindWork => "Average Wind Power",
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
    vy0: T,
    vz0: T,
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
            vy0: tf.clone(),
            vz0: tf.clone(),
            tf,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct ModelParams<T> {
    gravity_mps2: T,
    air_density_kg_m3: T,
    mass_kg: T,
    reference_area_m2: T,
    cl_slope_per_rad: T,
    cd0: T,
    aspect_ratio: T,
    oswald_efficiency: T,
    speed_eps_mps: T,
    frame_eps: T,
    wind_dir_x: T,
    wind_dir_y: T,
    wind_low_mps: T,
    wind_high_mps: T,
    wind_mid_altitude_m: T,
    wind_transition_height_m: T,
    alpha_rate_weight: T,
    roll_rate_weight: T,
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
    vy_periodic: T,
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
    constrain_vy0_zero: bool,
    constrain_vz0_zero: bool,
    gravity_mps2: f64,
    air_density_kg_m3: f64,
    mass_kg: f64,
    reference_area_m2: f64,
    cl_slope_per_rad: f64,
    cd0: f64,
    aspect_ratio: f64,
    oswald_efficiency: f64,
    speed_eps_mps: f64,
    frame_eps: f64,
    wind_azimuth_deg: f64,
    wind_low_mps: f64,
    wind_high_mps: f64,
    wind_mid_altitude_m: f64,
    wind_transition_height_m: f64,
    initial_wave_amplitude_m: f64,
    initial_wave_rotation_deg: f64,
    initial_alpha_deg: f64,
    initial_roll_amplitude_deg: f64,
    min_altitude_m: f64,
    min_airspeed_mps: f64,
    max_airspeed_mps: f64,
    max_load_factor: f64,
    max_alpha_rate_deg_s: f64,
    max_roll_rate_deg_s: f64,
    alpha_rate_regularization: f64,
    roll_rate_regularization: f64,
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
        solver.nlip.linear_solver = InteriorPointLinearSolver::SsidsRs;
        Self {
            objective: ObjectiveKind::WindWork,
            delta_l: DesignControl {
                fixed: true,
                value: 90.0,
                lower: 50.0,
                upper: 220.0,
            },
            h0: DesignControl {
                fixed: true,
                value: 1.5,
                lower: 0.5,
                upper: 5.0,
            },
            vx0: DesignControl {
                fixed: true,
                value: 15.0,
                lower: 10.0,
                upper: 22.0,
            },
            tf: DesignControl {
                fixed: true,
                value: 6.0,
                lower: 3.0,
                upper: 12.0,
            },
            constrain_vy0_zero: true,
            constrain_vz0_zero: false,
            gravity_mps2: DEFAULT_GRAVITY_MPS2,
            air_density_kg_m3: DEFAULT_AIR_DENSITY_KG_M3,
            mass_kg: DEFAULT_MASS_KG,
            reference_area_m2: DEFAULT_REFERENCE_AREA_M2,
            cl_slope_per_rad: DEFAULT_CL_SLOPE_PER_RAD,
            cd0: DEFAULT_CD0,
            aspect_ratio: DEFAULT_ASPECT_RATIO,
            oswald_efficiency: DEFAULT_OSWALD_EFFICIENCY,
            speed_eps_mps: DEFAULT_SPEED_EPS_MPS,
            frame_eps: DEFAULT_FRAME_EPS,
            wind_azimuth_deg: -90.0,
            wind_low_mps: 0.0,
            wind_high_mps: 6.0,
            wind_mid_altitude_m: 3.0,
            wind_transition_height_m: 1.5,
            initial_wave_amplitude_m: 10.0,
            initial_wave_rotation_deg: 45.0,
            initial_alpha_deg: 5.0,
            initial_roll_amplitude_deg: 35.0,
            min_altitude_m: 0.5,
            min_airspeed_mps: 5.0,
            max_airspeed_mps: 70.0,
            max_load_factor: 8.0,
            max_alpha_rate_deg_s: 45.0,
            max_roll_rate_deg_s: 160.0,
            alpha_rate_regularization: 13.0,
            roll_rate_regularization: 23.0,
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

fn sample_regularization_weight(
    values: &BTreeMap<String, f64>,
    id: &str,
    legacy_id: &str,
    default: f64,
) -> Result<f64> {
    let legacy_or_default = sample_or_default(values, legacy_id, default);
    expect_finite(sample_or_default(values, id, legacy_or_default), id)
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
            constrain_vy0_zero: sample_or_default(
                values,
                "constrain_vy0_zero",
                if defaults.constrain_vy0_zero {
                    1.0
                } else {
                    0.0
                },
            ) >= 0.5,
            constrain_vz0_zero: sample_or_default(
                values,
                "constrain_vz0_zero",
                if defaults.constrain_vz0_zero {
                    1.0
                } else {
                    0.0
                },
            ) >= 0.5,
            gravity_mps2: expect_finite(
                sample_or_default(values, "gravity_mps2", defaults.gravity_mps2),
                "gravity_mps2",
            )?,
            air_density_kg_m3: expect_finite(
                sample_or_default(values, "air_density_kg_m3", defaults.air_density_kg_m3),
                "air_density_kg_m3",
            )?,
            mass_kg: expect_finite(
                sample_or_default(values, "mass_kg", defaults.mass_kg),
                "mass_kg",
            )?,
            reference_area_m2: expect_finite(
                sample_or_default(values, "reference_area_m2", defaults.reference_area_m2),
                "reference_area_m2",
            )?,
            cl_slope_per_rad: expect_finite(
                sample_or_default(values, "cl_slope_per_rad", defaults.cl_slope_per_rad),
                "cl_slope_per_rad",
            )?,
            cd0: expect_finite(sample_or_default(values, "cd0", defaults.cd0), "cd0")?,
            aspect_ratio: expect_finite(
                sample_or_default(values, "aspect_ratio", defaults.aspect_ratio),
                "aspect_ratio",
            )?,
            oswald_efficiency: expect_finite(
                sample_or_default(values, "oswald_efficiency", defaults.oswald_efficiency),
                "oswald_efficiency",
            )?,
            speed_eps_mps: expect_finite(
                sample_or_default(values, "speed_eps_mps", defaults.speed_eps_mps),
                "speed_eps_mps",
            )?,
            frame_eps: expect_finite(
                sample_or_default(values, "frame_eps", defaults.frame_eps),
                "frame_eps",
            )?,
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
            min_altitude_m: expect_finite(
                sample_or_default(values, "min_altitude_m", defaults.min_altitude_m),
                "min_altitude_m",
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
            alpha_rate_regularization: sample_regularization_weight(
                values,
                "alpha_rate_regularization",
                "rate_regularization",
                defaults.alpha_rate_regularization,
            )?,
            roll_rate_regularization: sample_regularization_weight(
                values,
                "roll_rate_regularization",
                "rate_regularization",
                defaults.roll_rate_regularization,
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
        if params.gravity_mps2 <= 0.0 {
            return Err(anyhow!("gravity_mps2 must be positive"));
        }
        if params.air_density_kg_m3 <= 0.0 {
            return Err(anyhow!("air_density_kg_m3 must be positive"));
        }
        if params.mass_kg <= 0.0 {
            return Err(anyhow!("mass_kg must be positive"));
        }
        if params.reference_area_m2 <= 0.0 {
            return Err(anyhow!("reference_area_m2 must be positive"));
        }
        if params.cl_slope_per_rad <= 0.0 {
            return Err(anyhow!("cl_slope_per_rad must be positive"));
        }
        if params.cd0 < 0.0 {
            return Err(anyhow!("cd0 must be nonnegative"));
        }
        if params.aspect_ratio <= 0.0 {
            return Err(anyhow!("aspect_ratio must be positive"));
        }
        if params.oswald_efficiency <= 0.0 {
            return Err(anyhow!("oswald_efficiency must be positive"));
        }
        if params.oswald_efficiency > 1.0 {
            return Err(anyhow!("oswald_efficiency must be less than or equal to 1"));
        }
        if params.speed_eps_mps <= 0.0 {
            return Err(anyhow!("speed_eps_mps must be positive"));
        }
        if params.frame_eps <= 0.0 {
            return Err(anyhow!("frame_eps must be positive"));
        }
        if params.min_altitude_m < 0.0 {
            return Err(anyhow!("min_altitude_m must be nonnegative"));
        }
        if params.min_airspeed_mps > params.max_airspeed_mps {
            return Err(anyhow!(
                "min_airspeed_mps must be less than or equal to max_airspeed_mps"
            ));
        }
        if params.alpha_rate_regularization < 0.0 {
            return Err(anyhow!("alpha_rate_regularization must be nonnegative"));
        }
        if params.roll_rate_regularization < 0.0 {
            return Err(anyhow!("roll_rate_regularization must be nonnegative"));
        }
        validate_design_domain("delta_l", &params.delta_l, 0.0, false, "positive")?;
        validate_design_domain("h0", &params.h0, 0.0, true, "nonnegative")?;
        validate_design_domain("vx0", &params.vx0, 0.0, false, "positive")?;
        validate_design_domain("tf", &params.tf, 0.0, false, "positive")?;
        if params.h0.fixed && params.h0.value < params.min_altitude_m {
            return Err(anyhow!(
                "h0_value must be greater than or equal to min_altitude_m"
            ));
        }
        if !params.h0.fixed && params.h0.upper < params.min_altitude_m {
            return Err(anyhow!(
                "h0_upper must be greater than or equal to min_altitude_m when h0 is free"
            ));
        }
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
            (1.0, "Average wind power"),
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
        checkbox_control(
            "constrain_vy0_zero",
            "vy(0) Anchor",
            defaults.constrain_vy0_zero,
            "Choose whether vy(0) is constrained to zero or left as a periodic free global value.",
        ),
        checkbox_control(
            "constrain_vz0_zero",
            "vz(0) Anchor",
            defaults.constrain_vz0_zero,
            "Choose whether vz(0) is constrained to zero or left as a periodic free global value.",
        ),
        problem_slider_control(
            "gravity_mps2",
            "Gravity",
            1.0,
            20.0,
            0.01,
            defaults.gravity_mps2,
            "m/s^2",
            "Runtime gravity parameter used in dynamics, load factor, and mechanical energy.",
        ),
        problem_slider_control(
            "air_density_kg_m3",
            "Air Density",
            0.2,
            2.0,
            0.01,
            defaults.air_density_kg_m3,
            "kg/m^3",
            "Runtime air-density parameter in dynamic pressure.",
        ),
        problem_slider_control(
            "mass_kg",
            "Mass",
            0.5,
            40.0,
            0.1,
            defaults.mass_kg,
            "kg",
            "Runtime aircraft mass parameter.",
        ),
        problem_slider_control(
            "reference_area_m2",
            "Reference Area",
            0.05,
            3.0,
            0.01,
            defaults.reference_area_m2,
            "m^2",
            "Runtime aerodynamic reference area parameter.",
        ),
        problem_slider_control(
            "cl_slope_per_rad",
            "CL Slope",
            0.5,
            10.0,
            0.05,
            defaults.cl_slope_per_rad,
            "1/rad",
            "Runtime lift-curve slope used by CL = CL_slope * alpha.",
        ),
        problem_scientific_slider_control(
            "cd0",
            "CD0",
            0.0,
            0.1,
            0.0005,
            defaults.cd0,
            "",
            "Runtime zero-lift drag coefficient.",
        ),
        problem_slider_control(
            "aspect_ratio",
            "Aspect Ratio",
            2.0,
            30.0,
            0.1,
            defaults.aspect_ratio,
            "",
            "Runtime wing aspect ratio used in CD = CD0 + CL^2/(pi AR e).",
        ),
        problem_slider_control(
            "oswald_efficiency",
            "Oswald Efficiency",
            0.1,
            1.0,
            0.01,
            defaults.oswald_efficiency,
            "",
            "Runtime Oswald span efficiency used in CD = CD0 + CL^2/(pi AR e).",
        ),
        problem_scientific_slider_control(
            "speed_eps_mps",
            "Airspeed Eps",
            1.0e-6,
            1.0e-1,
            1.0e-4,
            defaults.speed_eps_mps,
            "m/s",
            "Runtime epsilon inside the smooth airspeed norm.",
        ),
        problem_scientific_slider_control(
            "frame_eps",
            "Frame Eps",
            1.0e-8,
            1.0e-2,
            1.0e-5,
            defaults.frame_eps,
            "",
            "Runtime epsilon for the projected-up lift-frame normalization.",
        ),
        problem_slider_control(
            "wind_azimuth_deg",
            "Wind Azimuth",
            -180.0,
            180.0,
            1.0,
            defaults.wind_azimuth_deg,
            "deg",
            "Horizontal wind direction. 0 deg is downwind with +X travel; -90 deg is crosswind along -Y and matches the default 45 deg guess climbing upwind. Converted to runtime wind-direction parameters, so changing it reuses the compiled NLP.",
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
            "Signed rotation away from vertical for the initial climbing wave. 0 deg is pure altitude; positive and negative values choose opposite Y sides.",
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
            "min_altitude_m",
            "Min Altitude",
            0.0,
            10.0,
            0.1,
            defaults.min_altitude_m,
            "m",
            "Lower path bound on center-of-mass altitude above the water.",
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
            "alpha_rate_regularization",
            "Alpha Rate Weight",
            1.0e-6,
            1.0e6,
            1.0e-2,
            defaults.alpha_rate_regularization,
            "",
            "Time-averaged quadratic regularization weight on alpha_dot^2.",
        ),
        problem_scientific_slider_control(
            "roll_rate_regularization",
            "Roll Rate Weight",
            1.0e-6,
            1.0e6,
            1.0e-2,
            defaults.roll_rate_regularization,
            "",
            "Time-averaged quadratic regularization weight on roll_dot^2.",
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
                    r"g=[\Delta L,h_0,v_{x0},v_{y0},T]^T".to_string(),
                ],
            },
            LatexSection {
                title: "Wind Shear".to_string(),
                entries: vec![
                    r"W(p_z)=\hat w\left(W_\ell+\frac{1}{2}(W_h-W_\ell)(1+\tanh((p_z-z_m)/z_s))\right)".to_string(),
                    r"a=v-W(p_z),\quad V_a=\sqrt{a^Ta+\epsilon_V^2},\quad \hat a=a/V_a".to_string(),
                ],
            },
            LatexSection {
                title: "Aero Polar".to_string(),
                entries: vec![r"C_L=C_{L_\alpha}\alpha,\quad C_D=C_{D0}+\frac{C_L^2}{\pi ARe}".to_string()],
            },
            LatexSection {
                title: "Lift and Drag Axes".to_string(),
                entries: vec![
                    r"\hat d=-\hat a\quad\text{(drag force axis, opposite air-relative velocity)}".to_string(),
                    r"n_0=\mathrm{normalize}\left(\hat z-(\hat z\cdot\hat a)\hat a\right)\quad\text{(zero-bank lift axis)}".to_string(),
                    r"\hat l=\cos\phi\,n_0+\sin\phi\,(\hat a\times n_0),\quad \hat l^T\hat a=0".to_string(),
                    r"\dot v=\frac{qS}{m}\left(C_L\hat l+C_D\hat d\right)-g\hat z".to_string(),
                ],
            },
            LatexSection {
                title: "Boundary Conditions".to_string(),
                entries: vec![
                    r"p_x(0)=0,\quad p_x(T)-p_x(0)=\Delta L".to_string(),
                    r"p_y(0)=0,\quad p_y(T)=0".to_string(),
                    r"p_z(0)=h_0,\quad p_z(T)-p_z(0)=0".to_string(),
                    r"v_x(0)=v_{x0},\quad v_x(T)-v_x(0)=0".to_string(),
                    r"v_y(0)=v_{y0},\quad v_y(T)-v_y(0)=0,\quad v_{y0}=0\ \text{when the }v_y\text{ anchor is enabled}".to_string(),
                    r"v_z(0)=v_{z0},\quad v_z(T)=0,\quad v_{z0}=0\ \text{when the }v_z\text{ anchor is enabled}".to_string(),
                    r"\alpha(T)-\alpha(0)=0,\quad \phi(T)-\phi(0)=0".to_string(),
                ],
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

fn cl_sx(alpha: SX, p: &ModelParams<SX>) -> SX {
    p.cl_slope_per_rad.clone() * alpha
}

fn cd_sx(alpha: SX, p: &ModelParams<SX>) -> SX {
    let cl = cl_sx(alpha, p);
    p.cd0.clone() + induced_drag_factor_sx(p) * cl.sqr()
}

fn cl_numeric(alpha: f64, params: &Params) -> f64 {
    params.cl_slope_per_rad * alpha
}

fn cd_numeric(alpha: f64, params: &Params) -> f64 {
    let cl = cl_numeric(alpha, params);
    params.cd0 + induced_drag_factor_numeric(params) * cl * cl
}

fn induced_drag_factor_sx(p: &ModelParams<SX>) -> SX {
    1.0 / (std::f64::consts::PI * p.aspect_ratio.clone() * p.oswald_efficiency.clone())
}

fn induced_drag_factor_numeric(params: &Params) -> f64 {
    1.0 / (std::f64::consts::PI * params.aspect_ratio * params.oswald_efficiency)
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
    let wx = speed.clone() * p.wind_dir_x.clone();
    let wy = speed.clone() * p.wind_dir_y.clone();
    (wx, wy, SX::zero(), speed)
}

fn aero_sx(state: &State<SX>, control: &Control<SX>, p: &ModelParams<SX>) -> AeroSx {
    let (wx, wy, wz, _) = wind_sx(state.pz.clone(), p);
    let ax_rel = state.vx.clone() - wx.clone();
    let ay_rel = state.vy.clone() - wy.clone();
    let az_rel = state.vz.clone() - wz.clone();
    let va2 = ax_rel.clone().sqr()
        + ay_rel.clone().sqr()
        + az_rel.clone().sqr()
        + p.speed_eps_mps.clone().sqr();
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
    let inv_n = SX::one() / (frame_guard.clone().sqr() + p.frame_eps.clone().sqr()).sqrt();
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
    let cl = cl_sx(control.alpha.clone(), p);
    let cd = cd_sx(control.alpha.clone(), p);
    let q_over_m =
        0.5 * p.air_density_kg_m3.clone() * p.reference_area_m2.clone() * va2 / p.mass_kg.clone();
    let aero_x = q_over_m.clone() * (cl.clone() * lx + cd.clone() * dhx);
    let aero_y = q_over_m.clone() * (cl.clone() * ly + cd.clone() * dhy);
    let aero_z = q_over_m * (cl.clone() * lz + cd.clone() * dhz);
    let load_factor = (aero_x.clone().sqr() + aero_y.clone().sqr() + aero_z.clone().sqr()).sqrt()
        / p.gravity_mps2.clone();
    let wind_work_rate =
        p.mass_kg.clone() * (aero_x.clone() * wx + aero_y.clone() * wy + aero_z.clone() * wz);
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
        .objective_lagrange_global(
            move |x: &State<SX>,
                  u: &Control<SX>,
                  dudt: &Control<SX>,
                  p: &ModelParams<SX>,
                  g: &Design<SX>| {
                let rate_cost = (p.alpha_rate_weight.clone() * dudt.alpha.clone().sqr()
                    + p.roll_rate_weight.clone() * dudt.roll.clone().sqr())
                    / g.tf.clone();
                match objective {
                    ObjectiveKind::WindWork => {
                        rate_cost - aero_sx(x, u, p).wind_work_rate / g.tf.clone()
                    }
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
                  p: &ModelParams<SX>,
                  g: &Design<SX>| {
                match objective {
                    ObjectiveKind::AverageSpeed => -g.delta_l.clone() / g.tf.clone(),
                    ObjectiveKind::TerminalEnergy => {
                        -(0.5
                            * p.mass_kg.clone()
                            * (xf.vx.clone().sqr() + xf.vy.clone().sqr() + xf.vz.clone().sqr())
                            + p.mass_kg.clone() * p.gravity_mps2.clone() * xf.pz.clone())
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
                vz: aero.az - p.gravity_mps2.clone(),
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
            vy0: x0.vy.clone() - g.vy0.clone(),
            vy_periodic: xt.vy.clone() - x0.vy.clone(),
            vz0: x0.vz.clone() - g.vz0.clone(),
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
        vy0: 0.0,
        vz0: 0.0,
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

fn finite_min_max(values: impl IntoIterator<Item = f64>) -> Option<(f64, f64)> {
    let mut iter = values.into_iter().filter(|value| value.is_finite());
    let first = iter.next()?;
    let mut min_value = first;
    let mut max_value = first;
    for value in iter {
        min_value = min_value.min(value);
        max_value = max_value.max(value);
    }
    Some((min_value, max_value))
}

fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    match count {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (end - start) / (count - 1) as f64;
            (0..count).map(|idx| start + idx as f64 * step).collect()
        }
    }
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn finite_span(values: &[f64]) -> f64 {
    finite_min_max(values.iter().copied())
        .map(|(lower, upper)| upper - lower)
        .unwrap_or(0.0)
}

fn scene_glyph_target_length(params: &Params, px: &[f64], py: &[f64], pz: &[f64]) -> f64 {
    let span = finite_span(px)
        .max(finite_span(py))
        .max(finite_span(pz))
        .max(params.delta_l.value.abs())
        .max(params.h0.value.abs())
        .max(40.0);
    (0.07 * span).clamp(3.0, 9.0)
}

fn acceleration_glyph_scale(diagnostics: &[(AeroNum, f64)], target_length: f64) -> f64 {
    let max_accel = diagnostics
        .iter()
        .flat_map(|(aero, _)| [aero.lift_accel, aero.drag_accel, aero.aero_accel])
        .map(norm3)
        .filter(|value| value.is_finite())
        .fold(0.0, f64::max);
    if max_accel > 1.0e-12 {
        target_length / max_accel
    } else {
        0.0
    }
}

fn wind_glyph_scale(diagnostics: &[(AeroNum, f64)], target_length: f64) -> f64 {
    let max_wind = diagnostics
        .iter()
        .map(|(aero, _)| norm3(aero.wind))
        .filter(|value| value.is_finite())
        .fold(0.0, f64::max);
    if max_wind > target_length && max_wind > 1.0e-12 {
        target_length / max_wind
    } else {
        1.0
    }
}

fn model_params(params: &Params) -> ModelParams<f64> {
    let az = deg_to_rad(params.wind_azimuth_deg);
    ModelParams {
        gravity_mps2: params.gravity_mps2,
        air_density_kg_m3: params.air_density_kg_m3,
        mass_kg: params.mass_kg,
        reference_area_m2: params.reference_area_m2,
        cl_slope_per_rad: params.cl_slope_per_rad,
        cd0: params.cd0,
        aspect_ratio: params.aspect_ratio,
        oswald_efficiency: params.oswald_efficiency,
        speed_eps_mps: params.speed_eps_mps,
        frame_eps: params.frame_eps,
        wind_dir_x: az.cos(),
        wind_dir_y: az.sin(),
        wind_low_mps: params.wind_low_mps,
        wind_high_mps: params.wind_high_mps,
        wind_mid_altitude_m: params.wind_mid_altitude_m,
        wind_transition_height_m: params.wind_transition_height_m,
        alpha_rate_weight: params.alpha_rate_regularization,
        roll_rate_weight: params.roll_rate_regularization,
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

fn wind_shear_z_upper(params: &Params, pz: &[f64]) -> f64 {
    let max_state_altitude = finite_min_max(pz.iter().copied())
        .map(|(_, upper)| upper)
        .unwrap_or(params.h0.value);
    let design_altitude = if params.h0.fixed {
        params.h0.value
    } else {
        params.h0.upper.max(params.h0.value)
    };
    max_state_altitude
        .max(design_altitude + params.initial_wave_amplitude_m.abs())
        .max(params.wind_mid_altitude_m + 5.0 * params.wind_transition_height_m)
        .max(params.min_altitude_m)
        .max(1.0)
}

fn xy_from_wind_basis(
    along: f64,
    cross: f64,
    along_dir: [f64; 2],
    cross_dir: [f64; 2],
) -> (f64, f64) {
    (
        along * along_dir[0] + cross * cross_dir[0],
        along * along_dir[1] + cross * cross_dir[1],
    )
}

fn wind_shear_scene_paths(params: &Params, px: &[f64], py: &[f64], pz: &[f64]) -> Vec<ScenePath3D> {
    let az = deg_to_rad(params.wind_azimuth_deg);
    let along_dir = [az.cos(), az.sin()];
    let cross_dir = [-along_dir[1], along_dir[0]];
    let along_projection = px
        .iter()
        .zip(py.iter())
        .map(|(x, y)| x * along_dir[0] + y * along_dir[1]);
    let cross_projection = px
        .iter()
        .zip(py.iter())
        .map(|(x, y)| x * cross_dir[0] + y * cross_dir[1]);
    let (along_min, along_max) = finite_min_max(along_projection).unwrap_or((0.0, 1.0));
    let (cross_min, cross_max) = finite_min_max(cross_projection).unwrap_or((-15.0, 15.0));
    let along_span = (along_max - along_min)
        .abs()
        .max(params.delta_l.value.abs())
        .max(40.0);
    let cross_center = 0.5 * (cross_min + cross_max);
    let cross_span = (cross_max - cross_min).abs().max(40.0);
    let panel_along = along_max + 0.10 * along_span;
    let cross_left = cross_center - 0.55 * cross_span;
    let cross_right = cross_center + 0.55 * cross_span;
    let profile_cross_offsets = linspace(cross_left, cross_right, WIND_SHEAR_PROFILE_LANES);
    let z_upper = wind_shear_z_upper(params, pz);
    let profile_z = linspace(0.0, z_upper, WIND_SHEAR_PROFILE_SAMPLES);
    let profile_scale = 2.2;
    let mut paths = Vec::new();
    let (left_x, left_y) = xy_from_wind_basis(panel_along, cross_left, along_dir, cross_dir);
    let (right_x, right_y) = xy_from_wind_basis(panel_along, cross_right, along_dir, cross_dir);
    paths.push(ScenePath3D {
        name: "wind shear frame".to_string(),
        x: vec![left_x, right_x, right_x, left_x, left_x],
        y: vec![left_y, right_y, right_y, left_y, left_y],
        z: vec![0.0, 0.0, z_upper, z_upper, 0.0],
    });
    for (lane, cross) in profile_cross_offsets.iter().enumerate() {
        let (base_x, base_y) = xy_from_wind_basis(panel_along, *cross, along_dir, cross_dir);
        let wind_vectors = profile_z
            .iter()
            .map(|z| wind_numeric(*z, params).0)
            .collect::<Vec<_>>();
        paths.push(ScenePath3D {
            name: format!("wind shear profile lane={lane}"),
            x: wind_vectors
                .iter()
                .map(|wind| base_x + wind[0] * profile_scale)
                .collect(),
            y: wind_vectors
                .iter()
                .map(|wind| base_y + wind[1] * profile_scale)
                .collect(),
            z: profile_z.clone(),
        });
    }
    paths
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
    let va = (rel[0] * rel[0]
        + rel[1] * rel[1]
        + rel[2] * rel[2]
        + params.speed_eps_mps * params.speed_eps_mps)
        .sqrt();
    let ah = [rel[0] / va, rel[1] / va, rel[2] / va];
    let dh = [-ah[0], -ah[1], -ah[2]];
    let dot_up = ah[2];
    let nr = [-dot_up * ah[0], -dot_up * ah[1], 1.0 - dot_up * ah[2]];
    let frame_guard = (nr[0] * nr[0] + nr[1] * nr[1] + nr[2] * nr[2]).sqrt();
    let inv_n = 1.0 / (frame_guard * frame_guard + params.frame_eps * params.frame_eps).sqrt();
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
    let cl = cl_numeric(control.alpha, params);
    let cd = cd_numeric(control.alpha, params);
    let q_over_m =
        0.5 * params.air_density_kg_m3 * params.reference_area_m2 * va * va / params.mass_kg;
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
        / params.gravity_mps2;
    let wind_work_rate = params.mass_kg
        * (aero_accel[0] * wind[0] + aero_accel[1] * wind[1] + aero_accel[2] * wind[2]);
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
    let amp_y = params.initial_wave_amplitude_m * theta.sin();
    let amp_z = params.initial_wave_amplitude_m * theta.cos().abs();
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
        ("vy(0)-vy0", first.vy - g.vy0),
        ("vy(T)-vy(0)", last.vy - first.vy),
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
            lower: Some(params.min_altitude_m),
            upper: None,
        },
        airspeed: Bounds1D {
            lower: Some(params.min_airspeed_mps),
            upper: Some(params.max_airspeed_mps),
        },
        cl: Bounds1D {
            lower: Some(0.0),
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
    let initial_velocity_bound = params
        .max_airspeed_mps
        .abs()
        .max(params.min_airspeed_mps.abs())
        .max(1.0);
    Design {
        delta_l: design_bounds(&params.delta_l),
        h0: design_bounds(&params.h0),
        vx0: design_bounds(&params.vx0),
        vy0: if params.constrain_vy0_zero {
            Bounds1D {
                lower: Some(0.0),
                upper: Some(0.0),
            }
        } else {
            Bounds1D {
                lower: Some(-initial_velocity_bound),
                upper: Some(initial_velocity_bound),
            }
        },
        vz0: if params.constrain_vz0_zero {
            Bounds1D {
                lower: Some(0.0),
                upper: Some(0.0),
            }
        } else {
            Bounds1D {
                lower: Some(-initial_velocity_bound),
                upper: Some(initial_velocity_bound),
            }
        },
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
        Path<f64>,
        (),
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
            vy_periodic: 0.0,
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
        Path<f64>,
        (),
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
            vy_periodic: 0.0,
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
) -> optimal_control::OcpScaling<
    ModelParams<f64>,
    State<f64>,
    Control<f64>,
    Path<f64>,
    Boundary<f64>,
    (),
    Design<f64>,
> {
    optimal_control::OcpScaling {
        objective: 100.0,
        state: State {
            px: params.delta_l.value.abs().max(100.0),
            py: params.initial_wave_amplitude_m.max(25.0),
            pz: params
                .h0
                .value
                .abs()
                .max(params.initial_wave_amplitude_m)
                .max(20.0),
            vx: params.vx0.value.abs().max(30.0),
            vy: params.vx0.value.abs().max(25.0),
            vz: params.vx0.value.abs().max(15.0),
        },
        state_derivative: State {
            px: 30.0,
            py: 25.0,
            pz: 20.0,
            vx: 30.0,
            vy: 40.0,
            vz: 30.0,
        },
        control: Control {
            alpha: deg_to_rad(15.0),
            roll: deg_to_rad(90.0),
        },
        control_rate: Control {
            alpha: deg_to_rad(30.0),
            roll: deg_to_rad(160.0),
        },
        global: Design {
            delta_l: params.delta_l.value.abs().max(100.0),
            h0: params.h0.value.abs().max(5.0),
            vx0: params.vx0.value.abs().max(25.0),
            vy0: 15.0,
            vz0: 15.0,
            tf: params.tf.value.abs().max(5.0),
        },
        parameters: ModelParams {
            gravity_mps2: params.gravity_mps2.abs().max(10.0),
            air_density_kg_m3: params.air_density_kg_m3.abs().max(1.0),
            mass_kg: params.mass_kg.abs().max(10.0),
            reference_area_m2: params.reference_area_m2.abs().max(1.0),
            cl_slope_per_rad: params.cl_slope_per_rad.abs().max(1.0),
            cd0: params.cd0.abs().max(0.01),
            aspect_ratio: params.aspect_ratio.abs().max(10.0),
            oswald_efficiency: params.oswald_efficiency.abs().max(0.5),
            speed_eps_mps: params.speed_eps_mps.abs().max(1.0e-3),
            frame_eps: params.frame_eps.abs().max(1.0e-4),
            wind_dir_x: 1.0,
            wind_dir_y: 0.0,
            wind_low_mps: 5.0,
            wind_high_mps: 5.0,
            wind_mid_altitude_m: 5.0,
            wind_transition_height_m: 2.0,
            alpha_rate_weight: 1.0,
            roll_rate_weight: 1.0,
        },
        path: Path {
            altitude: 20.0,
            airspeed: params.vx0.value.abs().max(30.0),
            cl: 1.5,
            load_factor: 5.0,
            frame_guard: 1.0,
            alpha_rate: deg_to_rad(30.0),
            roll_rate: deg_to_rad(160.0),
        },
        boundary_equalities: Boundary {
            px0: params.delta_l.value.abs().max(100.0),
            delta_l: params.delta_l.value.abs().max(100.0),
            py0: params.initial_wave_amplitude_m.max(25.0),
            py_t: params.initial_wave_amplitude_m.max(25.0),
            h0: params.h0.value.abs().max(5.0),
            pz_periodic: params
                .h0
                .value
                .abs()
                .max(params.initial_wave_amplitude_m)
                .max(20.0),
            vx0: params.vx0.value.abs().max(25.0),
            vx_periodic: params.vx0.value.abs().max(30.0),
            vy0: 15.0,
            vy_periodic: params.vx0.value.abs().max(25.0),
            vz0: params.vx0.value.abs().max(15.0),
            vz_t: params.vx0.value.abs().max(15.0),
            alpha_periodic: deg_to_rad(15.0),
            roll_periodic: deg_to_rad(90.0),
        },
        boundary_inequalities: (),
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
    Path<f64>,
    (),
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
    Path<f64>,
    (),
> {
    dc_runtime(params).expect("validated albatross direct-collocation runtime")
}

fn validate_runtime(params: &Params) -> Result<()> {
    match params.transcription.method {
        crate::common::TranscriptionMethod::MultipleShooting => ms_runtime(params).map(|_| ()),
        crate::common::TranscriptionMethod::DirectCollocation => dc_runtime(params).map(|_| ()),
    }
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
    validate_runtime(params)?;
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
    validate_runtime(params)?;
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

fn full_segment(len: usize) -> Vec<std::ops::Range<usize>> {
    std::iter::once(0..len).collect()
}

fn push_segmented_trajectory_paths(
    paths: &mut Vec<ScenePath3D>,
    px: &[f64],
    py: &[f64],
    pz: &[f64],
    segments: &[std::ops::Range<usize>],
) {
    for (index, segment) in segments.iter().enumerate() {
        let start = segment.start.min(px.len()).min(py.len()).min(pz.len());
        let end = segment.end.min(px.len()).min(py.len()).min(pz.len());
        if start >= end {
            continue;
        }
        paths.push(ScenePath3D {
            name: if segments.len() == 1 {
                "trajectory".to_string()
            } else {
                format!("trajectory arc {index}")
            },
            x: px[start..end].to_vec(),
            y: py[start..end].to_vec(),
            z: pz[start..end].to_vec(),
        });
    }
}

fn segmented_series_from_values(
    name: &str,
    legend_group: &str,
    times: &[f64],
    y: &[f64],
    segments: &[std::ops::Range<usize>],
    mode: PlotMode,
    role: crate::common::TimeSeriesRole,
) -> Vec<TimeSeries> {
    segments
        .iter()
        .enumerate()
        .filter_map(|(index, segment)| {
            let start = segment.start.min(times.len()).min(y.len());
            let end = segment.end.min(times.len()).min(y.len());
            (start < end).then(|| TimeSeries {
                name: name.to_string(),
                x: times[start..end].to_vec(),
                y: y[start..end].to_vec(),
                mode: Some(mode),
                legend_group: Some(legend_group.to_string()),
                show_legend: index == 0,
                role,
            })
        })
        .collect()
}

fn constant_bound_series(
    name: &str,
    legend_group: &str,
    times: &[f64],
    segments: &[std::ops::Range<usize>],
    value: f64,
    role: crate::common::TimeSeriesRole,
) -> Vec<TimeSeries> {
    let y = vec![value; times.len()];
    segmented_series_from_values(
        name,
        legend_group,
        times,
        &y,
        segments,
        PlotMode::Lines,
        role,
    )
}

fn scalar_chart(
    title: &str,
    y_label: &str,
    name: &str,
    times: &[f64],
    y: Vec<f64>,
    segments: &[std::ops::Range<usize>],
) -> Chart {
    chart(
        title,
        y_label,
        segmented_series_from_values(
            name,
            name,
            times,
            &y,
            segments,
            PlotMode::LinesMarkers,
            crate::common::TimeSeriesRole::Data,
        ),
    )
}

fn diagnostics_for_nodes(
    params: &Params,
    states: &[State<f64>],
    controls: &[Control<f64>],
) -> Vec<(AeroNum, f64)> {
    states
        .iter()
        .zip(controls.iter())
        .map(|(state, control)| {
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
    rate_times: Vec<f64>,
    sample_segments: Vec<std::ops::Range<usize>>,
    rate_segments: Vec<std::ops::Range<usize>>,
    tf: f64,
    global: &Design<f64>,
    notes: Vec<String>,
) -> SolveArtifact {
    let diagnostics = diagnostics_for_nodes(params, &states, &controls);
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
        .map(|v| 0.5 * params.air_density_kg_m3 * v * v)
        .collect::<Vec<_>>();
    let inv_tf = 1.0 / tf;
    let alpha_rate_regularization_density = rates
        .iter()
        .map(|u| params.alpha_rate_regularization * u.alpha * u.alpha * inv_tf)
        .collect::<Vec<_>>();
    let roll_rate_regularization_density = rates
        .iter()
        .map(|u| params.roll_rate_regularization * u.roll * u.roll * inv_tf)
        .collect::<Vec<_>>();
    let wind_power_objective_density = wind_work
        .iter()
        .map(|power| -power * inv_tf)
        .collect::<Vec<_>>();
    let kinetic = states
        .iter()
        .map(|s| 0.5 * params.mass_kg * (s.vx * s.vx + s.vy * s.vy + s.vz * s.vz))
        .collect::<Vec<_>>();
    let potential = states
        .iter()
        .map(|s| params.mass_kg * params.gravity_mps2 * s.pz)
        .collect::<Vec<_>>();
    let total_energy = kinetic
        .iter()
        .zip(potential.iter())
        .map(|(k, p)| k + p)
        .collect::<Vec<_>>();
    let path_bounds = path_bounds(params);
    let px_chart = scalar_chart(
        "Position px",
        "m",
        "px",
        &times,
        px.clone(),
        &sample_segments,
    );
    let py_chart = scalar_chart(
        "Position py",
        "m",
        "py",
        &times,
        py.clone(),
        &sample_segments,
    );
    let mut pz_chart = scalar_chart(
        "Altitude pz",
        "m",
        "pz",
        &times,
        pz.clone(),
        &sample_segments,
    );
    let vx_chart = scalar_chart("Velocity vx", "m/s", "vx", &times, vx, &sample_segments);
    let vy_chart = scalar_chart("Velocity vy", "m/s", "vy", &times, vy, &sample_segments);
    let vz_chart = scalar_chart("Velocity vz", "m/s", "vz", &times, vz, &sample_segments);
    let mut alpha_chart = scalar_chart(
        "Alpha",
        "deg",
        "alpha",
        &times,
        controls.iter().map(|u| rad_to_deg(u.alpha)).collect(),
        &sample_segments,
    );
    let roll_chart = scalar_chart(
        "Roll",
        "deg",
        "roll",
        &times,
        controls.iter().map(|u| rad_to_deg(u.roll)).collect(),
        &sample_segments,
    );
    let mut alpha_rate_chart = scalar_chart(
        "Alpha Rate",
        "deg/s",
        "alpha_dot",
        &rate_times,
        rates.iter().map(|u| rad_to_deg(u.alpha)).collect(),
        &rate_segments,
    );
    let mut roll_rate_chart = scalar_chart(
        "Roll Rate",
        "deg/s",
        "roll_dot",
        &rate_times,
        rates.iter().map(|u| rad_to_deg(u.roll)).collect(),
        &rate_segments,
    );
    let mut airspeed_chart = scalar_chart(
        "Airspeed",
        "m/s",
        "airspeed",
        &times,
        airspeed.clone(),
        &sample_segments,
    );
    let ground_speed_chart = scalar_chart(
        "Ground Speed",
        "m/s",
        "ground speed",
        &times,
        ground_speed,
        &sample_segments,
    );
    let wind_speed_chart = scalar_chart(
        "Wind Speed",
        "m/s",
        "wind speed",
        &times,
        wind_speed,
        &sample_segments,
    );
    let wind_x_chart = scalar_chart("Wind Wx", "m/s", "Wx", &times, wind_x, &sample_segments);
    let wind_y_chart = scalar_chart("Wind Wy", "m/s", "Wy", &times, wind_y, &sample_segments);
    let wind_z_chart = scalar_chart("Wind Wz", "m/s", "Wz", &times, wind_z, &sample_segments);
    let mut cl_chart = scalar_chart("CL", "-", "CL", &times, cl.clone(), &sample_segments);
    let cd_chart = scalar_chart("CD", "-", "CD", &times, cd, &sample_segments);
    let ld_chart = scalar_chart("L/D", "-", "L/D", &times, ld, &sample_segments);
    let mut load_chart = scalar_chart(
        "Load Factor",
        "-",
        "load factor",
        &times,
        load,
        &sample_segments,
    );
    let mut frame_guard_chart = scalar_chart(
        "Frame Guard",
        "-",
        "frame guard",
        &times,
        frame_guard,
        &sample_segments,
    );
    let kinetic_chart = scalar_chart(
        "Kinetic Energy",
        "J",
        "kinetic",
        &times,
        kinetic,
        &sample_segments,
    );
    let potential_chart = scalar_chart(
        "Potential Energy",
        "J",
        "potential",
        &times,
        potential,
        &sample_segments,
    );
    let total_energy_chart = scalar_chart(
        "Total Energy",
        "J",
        "total",
        &times,
        total_energy,
        &sample_segments,
    );
    let wind_work_chart = scalar_chart(
        "Wind Work Rate",
        "W",
        "F_aero · W",
        &times,
        wind_work,
        &sample_segments,
    );
    let mut objective_contribution_series = Vec::new();
    if params.objective == ObjectiveKind::WindWork {
        objective_contribution_series.extend(segmented_series_from_values(
            "-wind power/T",
            "-wind power/T",
            &times,
            &wind_power_objective_density,
            &sample_segments,
            PlotMode::LinesMarkers,
            crate::common::TimeSeriesRole::Data,
        ));
    }
    objective_contribution_series.extend(segmented_series_from_values(
        "alpha rate^2/T",
        "alpha rate^2/T",
        &rate_times,
        &alpha_rate_regularization_density,
        &rate_segments,
        PlotMode::LinesMarkers,
        crate::common::TimeSeriesRole::Data,
    ));
    objective_contribution_series.extend(segmented_series_from_values(
        "roll rate^2/T",
        "roll rate^2/T",
        &rate_times,
        &roll_rate_regularization_density,
        &rate_segments,
        PlotMode::LinesMarkers,
        crate::common::TimeSeriesRole::Data,
    ));
    let objective_contributions_chart = chart(
        "Objective Contributions",
        "-",
        objective_contribution_series,
    );
    let alpha_rate_regularization_chart = scalar_chart(
        "Alpha Rate Regularization",
        "-",
        "alpha rate^2/T",
        &rate_times,
        alpha_rate_regularization_density,
        &rate_segments,
    );
    let roll_rate_regularization_chart = scalar_chart(
        "Roll Rate Regularization",
        "-",
        "roll rate^2/T",
        &rate_times,
        roll_rate_regularization_density,
        &rate_segments,
    );
    let q_chart = scalar_chart("Dynamic Pressure", "Pa", "q", &times, q, &sample_segments);
    if let Some(lower) = path_bounds.altitude.lower {
        pz_chart.series.extend(constant_bound_series(
            "pz lower",
            "pz",
            &times,
            &sample_segments,
            lower,
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.altitude.upper {
        pz_chart.series.extend(constant_bound_series(
            "pz upper",
            "pz",
            &times,
            &sample_segments,
            upper,
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    if let Some(lower) = path_bounds.airspeed.lower {
        airspeed_chart.series.extend(constant_bound_series(
            "airspeed lower",
            "airspeed",
            &times,
            &sample_segments,
            lower,
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.airspeed.upper {
        airspeed_chart.series.extend(constant_bound_series(
            "airspeed upper",
            "airspeed",
            &times,
            &sample_segments,
            upper,
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    if let Some(lower) = path_bounds.cl.lower {
        alpha_chart.series.extend(constant_bound_series(
            "alpha lower from CL",
            "alpha",
            &times,
            &sample_segments,
            rad_to_deg(lower / params.cl_slope_per_rad),
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.cl.upper {
        alpha_chart.series.extend(constant_bound_series(
            "alpha upper from CL",
            "alpha",
            &times,
            &sample_segments,
            rad_to_deg(upper / params.cl_slope_per_rad),
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    if let Some(lower) = path_bounds.alpha_rate.lower {
        alpha_rate_chart.series.extend(constant_bound_series(
            "alpha_dot lower",
            "alpha_dot",
            &rate_times,
            &rate_segments,
            rad_to_deg(lower),
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.alpha_rate.upper {
        alpha_rate_chart.series.extend(constant_bound_series(
            "alpha_dot upper",
            "alpha_dot",
            &rate_times,
            &rate_segments,
            rad_to_deg(upper),
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    if let Some(lower) = path_bounds.roll_rate.lower {
        roll_rate_chart.series.extend(constant_bound_series(
            "roll_dot lower",
            "roll_dot",
            &rate_times,
            &rate_segments,
            rad_to_deg(lower),
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.roll_rate.upper {
        roll_rate_chart.series.extend(constant_bound_series(
            "roll_dot upper",
            "roll_dot",
            &rate_times,
            &rate_segments,
            rad_to_deg(upper),
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    if let Some(lower) = path_bounds.cl.lower {
        cl_chart.series.extend(constant_bound_series(
            "CL lower",
            "CL",
            &times,
            &sample_segments,
            lower,
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.cl.upper {
        cl_chart.series.extend(constant_bound_series(
            "CL upper",
            "CL",
            &times,
            &sample_segments,
            upper,
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    if let Some(lower) = path_bounds.load_factor.lower {
        load_chart.series.extend(constant_bound_series(
            "load lower",
            "load factor",
            &times,
            &sample_segments,
            lower,
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.load_factor.upper {
        load_chart.series.extend(constant_bound_series(
            "load upper",
            "load factor",
            &times,
            &sample_segments,
            upper,
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    if let Some(lower) = path_bounds.frame_guard.lower {
        frame_guard_chart.series.extend(constant_bound_series(
            "frame guard lower",
            "frame guard",
            &times,
            &sample_segments,
            lower,
            crate::common::TimeSeriesRole::LowerBound,
        ));
    }
    if let Some(upper) = path_bounds.frame_guard.upper {
        frame_guard_chart.series.extend(constant_bound_series(
            "frame guard upper",
            "frame guard",
            &times,
            &sample_segments,
            upper,
            crate::common::TimeSeriesRole::UpperBound,
        ));
    }
    let charts = vec![
        px_chart,
        py_chart,
        pz_chart,
        vx_chart,
        vy_chart,
        vz_chart,
        alpha_chart,
        roll_chart,
        alpha_rate_chart,
        roll_rate_chart,
        airspeed_chart,
        ground_speed_chart,
        wind_speed_chart,
        wind_x_chart,
        wind_y_chart,
        wind_z_chart,
        cl_chart,
        cd_chart,
        ld_chart,
        load_chart,
        frame_guard_chart,
        kinetic_chart,
        potential_chart,
        total_energy_chart,
        wind_work_chart,
        objective_contributions_chart,
        alpha_rate_regularization_chart,
        roll_rate_regularization_chart,
        q_chart,
    ];
    let mut paths = Vec::new();
    push_segmented_trajectory_paths(&mut paths, &px, &py, &pz, &sample_segments);
    let glyph_target = scene_glyph_target_length(params, &px, &py, &pz);
    let accel_scale = acceleration_glyph_scale(&diagnostics, glyph_target);
    let drag_scale = DRAG_GLYPH_EXAGGERATION * accel_scale;
    let wind_scale = wind_glyph_scale(&diagnostics, 0.85 * glyph_target);
    let frame_scale = (0.70 * glyph_target).clamp(3.0, 5.5);
    let stride = (states.len() / 24).max(1);
    for (idx, (state, (aero, _))) in states
        .iter()
        .zip(diagnostics.iter())
        .enumerate()
        .step_by(stride)
    {
        paths.push(ScenePath3D {
            name: format!("lift {idx}"),
            x: vec![state.px, state.px + accel_scale * aero.lift_accel[0]],
            y: vec![state.py, state.py + accel_scale * aero.lift_accel[1]],
            z: vec![state.pz, state.pz + accel_scale * aero.lift_accel[2]],
        });
        paths.push(ScenePath3D {
            name: format!("drag {idx}"),
            x: vec![state.px, state.px + drag_scale * aero.drag_accel[0]],
            y: vec![state.py, state.py + drag_scale * aero.drag_accel[1]],
            z: vec![state.pz, state.pz + drag_scale * aero.drag_accel[2]],
        });
        paths.push(ScenePath3D {
            name: format!("aero accel {idx}"),
            x: vec![state.px, state.px + accel_scale * aero.aero_accel[0]],
            y: vec![state.py, state.py + accel_scale * aero.aero_accel[1]],
            z: vec![state.pz, state.pz + accel_scale * aero.aero_accel[2]],
        });
        paths.push(ScenePath3D {
            name: format!("wind {idx}"),
            x: vec![state.px, state.px + wind_scale * aero.wind[0]],
            y: vec![state.py, state.py + wind_scale * aero.wind[1]],
            z: vec![state.pz, state.pz + wind_scale * aero.wind[2]],
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
    paths.extend(wind_shear_scene_paths(params, &px, &py, &pz));
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
    let sample_segments = full_segment(times.len());
    let rate_segments = sample_segments.clone();
    let mut artifact = artifact_common(
        params,
        states,
        controls,
        rates,
        times.clone(),
        times,
        sample_segments,
        rate_segments,
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
    let (times, states, sample_segments) = flatten_interval_arcs(&x_arcs);
    let (_, controls, _) = flatten_interval_arcs(&u_arcs);
    let (rate_times, rates, rate_segments) = flatten_interval_arcs(&dudt_arcs);
    artifact_common(
        params,
        states,
        controls,
        rates,
        times,
        rate_times,
        sample_segments,
        rate_segments,
        trajectories.tf,
        &trajectories.global,
        vec!["Direct collocation charts render each interval as a separate arc; control-rate diagnostics are plotted only at collocation roots.".to_string()],
    )
}

fn flatten_interval_arcs<T: Clone>(
    arcs: &[IntervalArc<T>],
) -> (Vec<f64>, Vec<T>, Vec<std::ops::Range<usize>>) {
    let total_len = arcs.iter().map(|arc| arc.values.len()).sum();
    let mut times = Vec::with_capacity(total_len);
    let mut values = Vec::with_capacity(total_len);
    let mut segments = Vec::with_capacity(arcs.len());
    for arc in arcs {
        let start = values.len();
        times.extend(arc.times.iter().copied());
        values.extend(arc.values.iter().cloned());
        let end = values.len();
        if start < end {
            segments.push(start..end);
        }
    }
    (times, values, segments)
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
    fn model_description_documents_force_axes_and_boundaries() {
        let spec = spec();
        let force_axes = spec
            .math_sections
            .iter()
            .find(|section| section.title == "Lift and Drag Axes")
            .expect("force-axis section should be present");
        assert!(
            force_axes
                .entries
                .iter()
                .any(|entry| entry.contains(r"\hat d=-\hat a"))
        );
        assert!(
            force_axes
                .entries
                .iter()
                .any(|entry| entry.contains(r"\hat l^T\hat a=0"))
        );

        let boundaries = spec
            .math_sections
            .iter()
            .find(|section| section.title == "Boundary Conditions")
            .expect("boundary section should be present");
        assert!(
            boundaries
                .entries
                .iter()
                .any(|entry| entry.contains(r"p_x(T)-p_x(0)=\Delta L"))
        );
        assert!(
            boundaries
                .entries
                .iter()
                .any(|entry| entry.contains(r"v_y(T)-v_y(0)=0"))
        );
        assert!(
            boundaries
                .entries
                .iter()
                .any(|entry| entry.contains(r"\alpha(T)-\alpha(0)=0"))
        );
    }

    #[test]
    fn default_albatross_parameters_match_crosswind_reference_seed() {
        let params = Params::default();
        assert_eq!(params.objective, ObjectiveKind::WindWork);
        assert_eq!(params.delta_l.value, 90.0);
        assert_eq!(params.delta_l.lower, 50.0);
        assert_eq!(params.delta_l.upper, 220.0);
        assert_eq!(params.h0.value, 1.5);
        assert_eq!(params.h0.lower, 0.5);
        assert_eq!(params.h0.upper, 5.0);
        assert_eq!(params.vx0.value, 15.0);
        assert_eq!(params.vx0.lower, 10.0);
        assert_eq!(params.vx0.upper, 22.0);
        assert_eq!(params.tf.value, 6.0);
        assert_eq!(params.tf.lower, 3.0);
        assert_eq!(params.tf.upper, 12.0);
        assert!(params.constrain_vy0_zero);
        assert!(!params.constrain_vz0_zero);
        assert_eq!(params.gravity_mps2, DEFAULT_GRAVITY_MPS2);
        assert_eq!(params.air_density_kg_m3, DEFAULT_AIR_DENSITY_KG_M3);
        assert_eq!(params.mass_kg, DEFAULT_MASS_KG);
        assert_eq!(params.reference_area_m2, DEFAULT_REFERENCE_AREA_M2);
        assert_eq!(params.cl_slope_per_rad, DEFAULT_CL_SLOPE_PER_RAD);
        assert_eq!(params.cd0, DEFAULT_CD0);
        assert_eq!(params.aspect_ratio, DEFAULT_ASPECT_RATIO);
        assert_eq!(params.oswald_efficiency, DEFAULT_OSWALD_EFFICIENCY);
        assert_eq!(params.speed_eps_mps, DEFAULT_SPEED_EPS_MPS);
        assert_eq!(params.frame_eps, DEFAULT_FRAME_EPS);
        assert_eq!(params.wind_azimuth_deg, -90.0);
        assert_eq!(params.wind_high_mps, 6.0);
        assert_eq!(params.wind_mid_altitude_m, 3.0);
        assert_eq!(params.wind_transition_height_m, 1.5);
        assert_eq!(params.initial_wave_amplitude_m, 10.0);
        assert_eq!(params.initial_alpha_deg, 5.0);
        assert_eq!(params.min_altitude_m, 0.5);
        assert_eq!(params.alpha_rate_regularization, 13.0);
        assert_eq!(params.roll_rate_regularization, 23.0);

        let alpha_best_glide = 0.5 / params.cl_slope_per_rad;
        let best_glide = 0.5 / cd_numeric(alpha_best_glide, &params);
        let expected_drag_factor =
            1.0 / (std::f64::consts::PI * DEFAULT_ASPECT_RATIO * DEFAULT_OSWALD_EFFICIENCY);
        let expected_best_glide = 0.5 / (DEFAULT_CD0 + expected_drag_factor * 0.25);
        assert!((induced_drag_factor_numeric(&params) - expected_drag_factor).abs() < 1.0e-12);
        assert!((best_glide - expected_best_glide).abs() < 1.0e-12);
    }

    #[test]
    fn default_albatross_scaling_matches_physical_nominals() {
        let params = Params::default();
        let scaling = scaling(&params);

        assert_eq!(scaling.state.px, 100.0);
        assert_eq!(scaling.state.py, 25.0);
        assert_eq!(scaling.state.pz, 20.0);
        assert_eq!(scaling.state.vx, 30.0);
        assert_eq!(scaling.state.vy, 25.0);
        assert_eq!(scaling.state.vz, 15.0);

        assert_eq!(scaling.state_derivative.px, 30.0);
        assert_eq!(scaling.state_derivative.py, 25.0);
        assert_eq!(scaling.state_derivative.pz, 20.0);
        assert_eq!(scaling.state_derivative.vx, 30.0);
        assert_eq!(scaling.state_derivative.vy, 40.0);
        assert_eq!(scaling.state_derivative.vz, 30.0);

        assert_eq!(scaling.control.alpha, deg_to_rad(15.0));
        assert_eq!(scaling.control.roll, deg_to_rad(90.0));
        assert_eq!(scaling.control_rate.alpha, deg_to_rad(30.0));
        assert_eq!(scaling.control_rate.roll, deg_to_rad(160.0));

        assert_eq!(scaling.global.delta_l, 100.0);
        assert_eq!(scaling.global.h0, 5.0);
        assert_eq!(scaling.global.vx0, 25.0);
        assert_eq!(scaling.global.vy0, 15.0);
        assert_eq!(scaling.global.tf, 6.0);

        assert_eq!(
            scaling.path,
            Path {
                altitude: 20.0,
                airspeed: 30.0,
                cl: 1.5,
                load_factor: 5.0,
                frame_guard: 1.0,
                alpha_rate: deg_to_rad(30.0),
                roll_rate: deg_to_rad(160.0),
            }
        );
        assert_eq!(
            scaling.boundary_equalities,
            Boundary {
                px0: 100.0,
                delta_l: 100.0,
                py0: 25.0,
                py_t: 25.0,
                h0: 5.0,
                pz_periodic: 20.0,
                vx0: 25.0,
                vx_periodic: 30.0,
                vy0: 15.0,
                vy_periodic: 25.0,
                vz0: 15.0,
                vz_t: 15.0,
                alpha_periodic: deg_to_rad(15.0),
                roll_periodic: deg_to_rad(90.0),
            }
        );
        assert_eq!(scaling.boundary_inequalities, ());
    }

    #[test]
    fn parses_wind_controls_and_design_bounds() {
        let mut values = BTreeMap::new();
        values.insert("wind_azimuth_deg".to_string(), 45.0);
        values.insert("wind_high_mps".to_string(), 18.0);
        values.insert("min_altitude_m".to_string(), 0.75);
        values.insert("delta_l_free".to_string(), 1.0);
        values.insert("delta_l_value".to_string(), 180.0);
        values.insert("delta_l_lower".to_string(), 100.0);
        values.insert("delta_l_upper".to_string(), 250.0);
        let params = Params::from_map(&values).expect("params should parse");
        assert_eq!(params.wind_azimuth_deg, 45.0);
        assert_eq!(params.min_altitude_m, 0.75);
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
    fn invalid_aero_parameters_fail() {
        let mut values = BTreeMap::new();
        values.insert("aspect_ratio".to_string(), 0.0);
        let err = Params::from_map(&values).expect_err("nonpositive aspect ratio should fail");
        assert!(err.to_string().contains("aspect_ratio"));

        values.clear();
        values.insert("oswald_efficiency".to_string(), 1.1);
        let err = Params::from_map(&values).expect_err("Oswald efficiency above one should fail");
        assert!(err.to_string().contains("oswald_efficiency"));
    }

    #[test]
    fn fixed_h0_below_min_altitude_fails() {
        let mut values = BTreeMap::new();
        values.insert("h0_value".to_string(), 0.25);
        values.insert("min_altitude_m".to_string(), 0.5);
        let err = Params::from_map(&values).expect_err("h0 below minimum altitude should fail");
        assert!(err.to_string().contains("min_altitude_m"));
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
    fn initial_velocity_anchor_controls_switch_global_bounds_only() {
        let mut values = BTreeMap::new();
        let default_params = Params::from_map(&values).expect("default params should parse");
        let default_bounds = global_bounds(&default_params);
        let default_variant = compile_variant_for_values(&values).expect("default variant");

        values.insert("constrain_vy0_zero".to_string(), 0.0);
        values.insert("constrain_vz0_zero".to_string(), 1.0);
        let switched = Params::from_map(&values).expect("switched anchor params should parse");
        let switched_bounds = global_bounds(&switched);
        let switched_variant = compile_variant_for_values(&values).expect("switched variant");

        assert!(default_params.constrain_vy0_zero);
        assert!(!default_params.constrain_vz0_zero);
        assert!(!switched.constrain_vy0_zero);
        assert!(switched.constrain_vz0_zero);
        assert_eq!(switched_variant, default_variant);
        assert_eq!(
            active_design(&default_params).expect("active design").vy0,
            0.0
        );
        assert_eq!(
            active_design(&default_params).expect("active design").vz0,
            0.0
        );
        assert_eq!(default_bounds.vy0.lower, Some(0.0));
        assert_eq!(default_bounds.vy0.upper, Some(0.0));
        assert_eq!(
            default_bounds.vz0.lower,
            Some(-default_params.max_airspeed_mps)
        );
        assert_eq!(
            default_bounds.vz0.upper,
            Some(default_params.max_airspeed_mps)
        );
        assert_eq!(switched_bounds.vy0.lower, Some(-switched.max_airspeed_mps));
        assert_eq!(switched_bounds.vy0.upper, Some(switched.max_airspeed_mps));
        assert_eq!(switched_bounds.vz0.lower, Some(0.0));
        assert_eq!(switched_bounds.vz0.upper, Some(0.0));
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
    fn negative_guess_rotation_stays_above_altitude_floor() {
        let values = BTreeMap::from([("initial_wave_rotation_deg".to_string(), -45.0)]);
        let params = Params::from_map(&values).expect("params should parse");
        let guess = continuous_guess(&params).expect("negative rotation guess should be feasible");
        validate_initial_guess(&params, &guess).expect("negative rotation guess should validate");

        let min_altitude = guess
            .x_samples
            .iter()
            .map(|state| state.pz)
            .fold(f64::INFINITY, f64::min);
        assert!(min_altitude >= params.min_altitude_m);
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
    fn wind_azimuth_changes_runtime_parameters_not_compile_key() {
        let mut values = BTreeMap::new();
        values.insert("wind_azimuth_deg".to_string(), 0.0);
        let x_wind = compile_variant_for_values(&values).expect("x-wind variant");
        let x_params = Params::from_map(&values).expect("x-wind params");
        let x_model = model_params(&x_params);

        values.insert("wind_azimuth_deg".to_string(), 90.0);
        let y_wind = compile_variant_for_values(&values).expect("y-wind variant");
        let y_params = Params::from_map(&values).expect("y-wind params");
        let y_model = model_params(&y_params);

        assert_eq!(x_wind, y_wind);
        assert!((x_model.wind_dir_x - 1.0).abs() < 1.0e-12);
        assert!(x_model.wind_dir_y.abs() < 1.0e-12);
        assert!(y_model.wind_dir_x.abs() < 1.0e-12);
        assert!((y_model.wind_dir_y - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn aircraft_aero_controls_change_runtime_parameters_not_compile_key() {
        let mut values = BTreeMap::new();
        let default_variant = compile_variant_for_values(&values).expect("default variant");

        values.insert("gravity_mps2".to_string(), 9.7);
        values.insert("air_density_kg_m3".to_string(), 1.0);
        values.insert("mass_kg".to_string(), 10.0);
        values.insert("reference_area_m2".to_string(), 0.8);
        values.insert("cl_slope_per_rad".to_string(), 6.0);
        values.insert("cd0".to_string(), 0.02);
        values.insert("aspect_ratio".to_string(), 12.0);
        values.insert("oswald_efficiency".to_string(), 0.8);
        values.insert("speed_eps_mps".to_string(), 0.002);
        values.insert("frame_eps".to_string(), 0.0002);
        values.insert("alpha_rate_regularization".to_string(), 0.002);
        values.insert("roll_rate_regularization".to_string(), 0.004);

        let changed_variant = compile_variant_for_values(&values).expect("changed variant");
        let params = Params::from_map(&values).expect("changed params");
        let model = model_params(&params);

        assert_eq!(default_variant, changed_variant);
        assert_eq!(model.gravity_mps2, 9.7);
        assert_eq!(model.air_density_kg_m3, 1.0);
        assert_eq!(model.mass_kg, 10.0);
        assert_eq!(model.reference_area_m2, 0.8);
        assert_eq!(model.cl_slope_per_rad, 6.0);
        assert_eq!(model.cd0, 0.02);
        assert_eq!(model.aspect_ratio, 12.0);
        assert_eq!(model.oswald_efficiency, 0.8);
        assert_eq!(model.speed_eps_mps, 0.002);
        assert_eq!(model.frame_eps, 0.0002);
        assert_eq!(model.alpha_rate_weight, 0.002);
        assert_eq!(model.roll_rate_weight, 0.004);
        assert!(
            (induced_drag_factor_numeric(&params) - 1.0 / (std::f64::consts::PI * 12.0 * 0.8))
                .abs()
                < 1.0e-12
        );
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
            guess.sample_times.clone(),
            full_segment(guess.sample_times.len()),
            full_segment(guess.sample_times.len()),
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
        let shear_profiles = paths
            .iter()
            .filter(|path| path.name.starts_with("wind shear profile"))
            .collect::<Vec<_>>();
        assert_eq!(shear_profiles.len(), WIND_SHEAR_PROFILE_LANES);
        assert!(paths.iter().any(|path| path.name == "wind shear frame"));
        assert!(
            shear_profiles
                .iter()
                .all(|path| path.z.len() == WIND_SHEAR_PROFILE_SAMPLES)
        );
        assert!(
            paths
                .iter()
                .filter(|path| path.name.starts_with("wind shear vector"))
                .count()
                == 0
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
        let segment_length = |path: &ScenePath3D| {
            let dx = path.x[1] - path.x[0];
            let dy = path.y[1] - path.y[0];
            let dz = path.z[1] - path.z[0];
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        let max_accel_glyph = paths
            .iter()
            .filter(|path| path.name.starts_with("lift ") || path.name.starts_with("aero accel "))
            .map(segment_length)
            .fold(0.0, f64::max);
        assert!(
            max_accel_glyph <= 9.0 + 1.0e-9,
            "acceleration glyphs should be scene-scaled, got {max_accel_glyph}"
        );
        let max_drag_glyph = paths
            .iter()
            .filter(|path| path.name.starts_with("drag "))
            .map(segment_length)
            .fold(0.0, f64::max);
        assert!(
            max_drag_glyph <= DRAG_GLYPH_EXAGGERATION * 9.0 + 1.0e-9,
            "drag glyphs should only exceed normal scale by the configured exaggeration, got {max_drag_glyph}"
        );
        let max_wind_glyph = paths
            .iter()
            .filter(|path| path.name.starts_with("wind ") && !path.name.starts_with("wind shear"))
            .map(segment_length)
            .fold(0.0, f64::max);
        assert!(
            max_wind_glyph <= 7.65 + 1.0e-9,
            "wind glyphs should be scene-scaled, got {max_wind_glyph}"
        );

        let position = artifact
            .charts
            .iter()
            .find(|chart| chart.title == "Altitude pz")
            .expect("altitude chart should exist");
        let altitude_lower = position
            .series
            .iter()
            .find(|series| series.name == "pz lower")
            .expect("altitude chart should label the altitude lower bound");
        assert_eq!(altitude_lower.legend_group.as_deref(), Some("pz"));

        let aero_coefficients = artifact
            .charts
            .iter()
            .find(|chart| chart.title == "CL")
            .expect("CL chart should exist");
        let cl_upper = aero_coefficients
            .series
            .iter()
            .find(|series| series.name == "CL upper")
            .expect("aero coefficient chart should label the CL upper bound");
        assert_eq!(cl_upper.legend_group.as_deref(), Some("CL"));

        let objective_contributions = artifact
            .charts
            .iter()
            .find(|chart| chart.title == "Objective Contributions")
            .expect("combined objective contribution chart should exist");
        assert!(
            !objective_contributions
                .series
                .iter()
                .any(|series| series.name == "objective density")
        );
        assert!(
            objective_contributions
                .series
                .iter()
                .any(|series| series.name == "alpha rate^2/T")
        );
        assert!(
            objective_contributions
                .series
                .iter()
                .any(|series| series.name == "roll rate^2/T")
        );
        assert!(
            objective_contributions
                .series
                .iter()
                .any(|series| series.name == "-wind power/T")
        );
        assert_eq!(objective_contributions.series.len(), 3);
    }

    #[test]
    fn direct_collocation_rate_plot_samples_stay_interval_local() {
        let root_rates = vec![
            IntervalArc {
                times: vec![0.2, 0.5],
                values: vec![
                    Control {
                        alpha: 1.0,
                        roll: 10.0,
                    },
                    Control {
                        alpha: 2.0,
                        roll: 20.0,
                    },
                ],
            },
            IntervalArc {
                times: vec![1.2, 1.5],
                values: vec![
                    Control {
                        alpha: 3.0,
                        roll: 30.0,
                    },
                    Control {
                        alpha: 4.0,
                        roll: 40.0,
                    },
                ],
            },
        ];

        let (times, rates, segments) = flatten_interval_arcs(&root_rates);
        let alpha = rates.iter().map(|rate| rate.alpha).collect::<Vec<_>>();
        let roll = rates.iter().map(|rate| rate.roll).collect::<Vec<_>>();

        assert_eq!(times, vec![0.2, 0.5, 1.2, 1.5]);
        assert_eq!(alpha, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(roll, vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(segments, vec![0..2, 2..4]);
    }

    #[test]
    fn segmented_trajectory_paths_do_not_bridge_intervals() {
        let mut paths = Vec::new();
        let px = vec![0.0, 1.0, 10.0, 11.0];
        let py = vec![0.0, 0.0, 0.0, 0.0];
        let pz = vec![1.0, 2.0, 3.0, 4.0];
        push_segmented_trajectory_paths(&mut paths, &px, &py, &pz, &[0..2, 2..4]);

        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].name, "trajectory arc 0");
        assert_eq!(paths[1].name, "trajectory arc 1");
        assert_eq!(paths[0].x, vec![0.0, 1.0]);
        assert_eq!(paths[1].x, vec![10.0, 11.0]);
    }

    #[test]
    fn wind_work_objective_contributions_label_running_term() {
        let mut params = Params::default();
        params.objective = ObjectiveKind::WindWork;
        let guess = continuous_guess(&params).expect("default guess should build");
        let artifact = artifact_common(
            &params,
            guess.x_samples.clone(),
            guess.u_samples.clone(),
            guess.dudt_samples.clone(),
            guess.sample_times.clone(),
            guess.sample_times.clone(),
            full_segment(guess.sample_times.len()),
            full_segment(guess.sample_times.len()),
            guess.tf,
            &guess.global,
            Vec::new(),
        );
        let objective_contributions = artifact
            .charts
            .iter()
            .find(|chart| chart.title == "Objective Contributions")
            .expect("combined objective contribution chart should exist");
        assert!(
            objective_contributions
                .series
                .iter()
                .any(|series| series.name == "-wind power/T")
        );
        assert_eq!(objective_contributions.series.len(), 3);
    }

    #[test]
    fn wind_shear_profile_follows_azimuth_direction() {
        let px = vec![0.0, 90.0];
        let py = vec![0.0, 0.0];
        let pz = vec![1.5, 1.5];

        let mut values = BTreeMap::new();
        values.insert("wind_azimuth_deg".to_string(), 0.0);
        let x_params = Params::from_map(&values).expect("x wind params");
        let x_paths = wind_shear_scene_paths(&x_params, &px, &py, &pz);
        let x_profile = x_paths
            .iter()
            .find(|path| path.name.starts_with("wind shear profile"))
            .expect("x wind profile should be present");
        assert!(x_profile.x.last().unwrap() > x_profile.x.first().unwrap());
        assert!((x_profile.y.last().unwrap() - x_profile.y.first().unwrap()).abs() < 1.0e-10);

        values.insert("wind_azimuth_deg".to_string(), 90.0);
        let y_params = Params::from_map(&values).expect("y wind params");
        let y_paths = wind_shear_scene_paths(&y_params, &px, &py, &pz);
        let y_profile = y_paths
            .iter()
            .find(|path| path.name.starts_with("wind shear profile"))
            .expect("y wind profile should be present");
        assert!((y_profile.x.last().unwrap() - y_profile.x.first().unwrap()).abs() < 1.0e-10);
        assert!(y_profile.y.last().unwrap() > y_profile.y.first().unwrap());
    }
}
