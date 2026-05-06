use nalgebra::{Quaternion, Vector3};
use optimization::Vectorize;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseMode {
    Adaptive,
    OpenLoop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LongitudinalMode {
    TotalEnergy,
    MaxThrottleAltitudePitch,
}

impl Default for LongitudinalMode {
    fn default() -> Self {
        Self::TotalEnergy
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LateralOuterMode {
    Orbit,
    ForwardFormation,
    TimedTransition,
}

impl Default for LateralOuterMode {
    fn default() -> Self {
        Self::Orbit
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForwardFrameMode {
    WorldFixed,
    MeanVelocity,
}

impl Default for ForwardFrameMode {
    fn default() -> Self {
        Self::WorldFixed
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrydenConfig {
    pub seed: u64,
    pub intensity_scale: f64,
    pub length_scale: f64,
    pub altitude_intensity_enabled: bool,
    pub altitude_length_scale_enabled: bool,
}

impl Default for DrydenConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            intensity_scale: 1.0,
            length_scale: 1.0,
            altitude_intensity_enabled: true,
            altitude_length_scale_enabled: true,
        }
    }
}

impl DrydenConfig {
    pub fn finite_or_default(self, default: &Self) -> Self {
        Self {
            seed: self.seed,
            intensity_scale: if self.intensity_scale.is_finite() && self.intensity_scale >= 0.0 {
                self.intensity_scale
            } else {
                default.intensity_scale
            },
            length_scale: if self.length_scale.is_finite() && self.length_scale > 0.0 {
                self.length_scale
            } else {
                default.length_scale
            },
            altitude_intensity_enabled: self.altitude_intensity_enabled,
            altitude_length_scale_enabled: self.altitude_length_scale_enabled,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Preset {
    Swarm,
    FreeFlight1,
    SimpleTether,
}

pub const DEFAULT_SWARM_KITES: usize = 3;
pub const MIN_SWARM_KITES: usize = 1;
pub const MAX_SWARM_KITES: usize = 12;
pub const DEFAULT_SWARM_DISK_ALTITUDE_M: f64 = 350.0;
pub const DEFAULT_SWARM_DISK_RADIUS_M: f64 = 70.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub duration: f64,
    pub dt_control: f64,
    pub rk_abs_tol: f64,
    pub rk_rel_tol: f64,
    pub max_substeps: usize,
    pub phase_mode: PhaseMode,
    #[serde(default)]
    pub lateral_outer_mode: LateralOuterMode,
    #[serde(default)]
    pub forward_frame_mode: ForwardFrameMode,
    #[serde(default = "default_transition_to_forward_s")]
    pub transition_to_forward_s: f64,
    #[serde(default)]
    pub transition_to_orbit_s: Option<f64>,
    #[serde(default = "default_timed_transition_recenter_lead_radii")]
    pub timed_transition_recenter_lead_radii: f64,
    pub sample_stride: usize,
    pub sim_noise_enabled: bool,
    #[serde(default)]
    pub dryden: DrydenConfig,
    pub bridle_enabled: bool,
    pub longitudinal_mode: LongitudinalMode,
    pub controller_tuning: ControllerTuning<f64>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            duration: 30.0,
            dt_control: 0.01,
            rk_abs_tol: 1.0e-6,
            rk_rel_tol: 1.0e-6,
            max_substeps: 1000,
            phase_mode: PhaseMode::Adaptive,
            lateral_outer_mode: LateralOuterMode::Orbit,
            forward_frame_mode: ForwardFrameMode::WorldFixed,
            transition_to_forward_s: default_transition_to_forward_s(),
            transition_to_orbit_s: Some(65.0),
            timed_transition_recenter_lead_radii: default_timed_transition_recenter_lead_radii(),
            sample_stride: 1,
            sim_noise_enabled: true,
            dryden: DrydenConfig::default(),
            bridle_enabled: true,
            longitudinal_mode: LongitudinalMode::TotalEnergy,
            controller_tuning: ControllerTuning::default(),
        }
    }
}

fn default_transition_to_forward_s() -> f64 {
    5.0
}

fn default_timed_transition_recenter_lead_radii() -> f64 {
    1.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InitRequest {
    pub preset: Preset,
    pub payload_mass_kg: Option<f64>,
    pub performance_scale_percent: Option<f64>,
    pub wind_speed_mps: Option<f64>,
    pub swarm_kites: usize,
    #[serde(default)]
    pub swarm_forward_flight_init: bool,
    pub swarm_disk_altitude_m: Option<f64>,
    pub swarm_disk_radius_m: Option<f64>,
    pub swarm_aircraft_altitude_m: Option<f64>,
    pub swarm_upper_tether_length_m: Option<f64>,
    pub swarm_common_tether_length_m: Option<f64>,
}

impl Default for InitRequest {
    fn default() -> Self {
        Self {
            preset: Preset::Swarm,
            payload_mass_kg: None,
            performance_scale_percent: None,
            wind_speed_mps: None,
            swarm_kites: DEFAULT_SWARM_KITES,
            swarm_forward_flight_init: false,
            swarm_disk_altitude_m: None,
            swarm_disk_radius_m: None,
            swarm_aircraft_altitude_m: None,
            swarm_upper_tether_length_m: None,
            swarm_common_tether_length_m: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VehiclePerformanceSnapshot {
    pub scale_percent: f64,
    pub mu: f64,
    pub mass_kg: f64,
    pub s_ref_m2: f64,
    pub b_ref_m: f64,
    pub c_ref_m: f64,
    pub thrust_capacity_scale: f64,
    pub power_capacity_scale: f64,
    pub inertia_kg_m2: [f64; 3],
    pub trim_motor_torque_nm: f64,
    pub motor_torque_max_nm: f64,
    pub tecs_thrust_integrator_limit_nm: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VehiclePerformanceScalingPreview {
    pub baseline: VehiclePerformanceSnapshot,
    pub scaled: VehiclePerformanceSnapshot,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PresetInfo {
    pub preset: Preset,
    pub name: &'static str,
    pub description: &'static str,
    pub kites: usize,
    pub common_nodes: usize,
    pub upper_nodes: usize,
}

#[derive(Clone, Debug, Vectorize)]
pub struct ControlSurfaces<T> {
    pub aileron: T,
    pub flap: T,
    pub winglet: T,
    pub elevator: T,
    pub rudder: T,
}

impl ControlSurfaces<f64> {
    pub fn zero() -> Self {
        Self {
            aileron: 0.0,
            flap: 0.0,
            winglet: 0.0,
            elevator: 0.0,
            rudder: 0.0,
        }
    }
}

#[derive(Clone, Debug, Vectorize)]
pub struct KiteControls<T> {
    pub surfaces: ControlSurfaces<T>,
    pub motor_torque: T,
}

impl KiteControls<f64> {
    pub fn zero() -> Self {
        Self {
            surfaces: ControlSurfaces::zero(),
            motor_torque: 0.0,
        }
    }
}

#[derive(Clone, Debug, Vectorize)]
pub struct Controls<T, const NK: usize> {
    pub kites: [KiteControls<T>; NK],
}

#[derive(Clone, Debug, Vectorize)]
pub struct TetherNode<T> {
    pub pos_n: Vector3<T>,
    pub vel_n: Vector3<T>,
}

#[derive(Clone, Debug, Vectorize)]
pub struct BodyState<T> {
    pub pos_n: Vector3<T>,
    pub vel_b: Vector3<T>,
    pub quat_n2b: Quaternion<T>,
    pub omega_b: Vector3<T>,
}

#[derive(Clone, Debug, Vectorize)]
pub struct KiteState<T, const N_UPPER: usize> {
    pub body: BodyState<T>,
    pub rotor_speed: T,
    pub actuators: KiteControls<T>,
    pub tether: [TetherNode<T>; N_UPPER],
}

#[derive(Clone, Debug, Vectorize)]
pub struct State<T, const NK: usize, const N_COMMON: usize, const N_UPPER: usize> {
    pub kites: [KiteState<T, N_UPPER>; NK],
    pub splitter: TetherNode<T>,
    pub common_tether: [TetherNode<T>; N_COMMON],
    pub payload: TetherNode<T>,
    pub total_work: T,
    pub total_dissipated_work: T,
    pub mechanical_energy_reference: T,
}

impl<const NK: usize, const N_COMMON: usize, const N_UPPER: usize>
    State<f64, NK, N_COMMON, N_UPPER>
{
    pub fn renormalize_attitudes(&mut self) {
        for kite in &mut self.kites {
            let quat = kite.body.quat_n2b;
            let norm = (quat.coords[3] * quat.coords[3]
                + quat.coords[0] * quat.coords[0]
                + quat.coords[1] * quat.coords[1]
                + quat.coords[2] * quat.coords[2])
                .sqrt()
                .max(1.0e-9);
            kite.body.quat_n2b = Quaternion::new(
                quat.coords[3] / norm,
                quat.coords[0] / norm,
                quat.coords[1] / norm,
                quat.coords[2] / norm,
            );
        }
    }
}

#[derive(Clone, Debug, Vectorize)]
pub struct MassContactParams<T> {
    pub zeta: T,
    pub enable_length: T,
    pub ground_altitude: T,
}

#[derive(Clone, Debug, Vectorize)]
pub struct TetherParams<T> {
    pub natural_length: T,
    pub total_mass: T,
    pub ea: T,
    pub viscous_damping_coeff: T,
    pub cd_phi: T,
    pub diameter: T,
    pub contact: MassContactParams<T>,
}

#[derive(Clone, Debug, Vectorize)]
pub struct RigidBodyParams<T> {
    pub mass: T,
    pub inertia_diagonal: Vector3<T>,
    pub cad_offset_b: Vector3<T>,
}

#[derive(Clone, Debug, Vectorize)]
pub struct BridleParams<T> {
    pub pivot_b: Vector3<T>,
    pub radius: T,
}

#[derive(Clone, Debug, Vectorize)]
pub struct AeroParams<T> {
    pub ref_area: T,
    pub ref_span: T,
    pub ref_chord: T,
    pub cl0: T,
    pub cl_alpha: T,
    pub cl_elevator: T,
    pub cl_flap: T,
    pub cd0: T,
    pub cd_induced: T,
    pub cd_surface_abs: T,
    pub cy_beta: T,
    pub cy_rudder: T,
    pub roll_beta: T,
    pub roll_p: T,
    pub roll_r: T,
    pub roll_aileron: T,
    pub pitch0: T,
    pub pitch_alpha: T,
    pub pitch_q: T,
    pub pitch_elevator: T,
    pub pitch_flap: T,
    pub yaw_beta: T,
    pub yaw_p: T,
    pub yaw_r: T,
    pub yaw_rudder: T,
}

#[derive(Clone, Debug, Vectorize)]
pub struct RotorParams<T> {
    pub axis_b: Vector3<T>,
    pub position_b: Vector3<T>,
    pub radius: T,
    pub thrust_scale: T,
    pub torque_scale: T,
    pub inertia: T,
    pub sign: T,
    pub initial_speed: T,
}

#[derive(Clone, Debug, Vectorize)]
pub struct KiteParams<T> {
    pub rigid_body: RigidBodyParams<T>,
    pub aero: AeroParams<T>,
    pub bridle: BridleParams<T>,
    pub tether: TetherParams<T>,
    pub rotor: RotorParams<T>,
}

#[derive(Clone, Debug, Vectorize)]
pub struct Environment<T> {
    pub rho: T,
    pub g: T,
    pub wind_n: Vector3<T>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Vectorize)]
pub struct ControllerTuning<T> {
    pub speed_phase_gain: T,
    pub speed_min_mps: T,
    pub speed_max_mps: T,
    pub rabbit_speed_to_distance_s: T,
    pub rabbit_min_distance_m: T,
    pub rabbit_max_distance_m: T,
    pub roll_feedforward_gain: T,
    pub rabbit_bearing_roll_p: T,
    pub rabbit_bearing_roll_i: T,
    pub roll_curvature_p: T,
    pub roll_curvature_i: T,
    pub roll_curvature_integrator_limit: T,
    pub roll_ref_limit_deg: T,
    #[serde(default)]
    pub guidance_mode: T,
    #[serde(default)]
    pub forward_heading_deg: T,
    #[serde(default)]
    pub forward_lookahead_scale: T,
    #[serde(default)]
    pub formation_spacing_m: T,
    #[serde(default)]
    pub formation_lateral_offset_i_per_s: T,
    #[serde(default)]
    pub formation_lateral_offset_limit_m: T,
    #[serde(default)]
    pub formation_lateral_error_limit_m: T,
    pub tethered_roll_ref_rate_limit_degps: T,
    pub free_aileron_roll_p: T,
    pub free_aileron_roll_d: T,
    pub tethered_aileron_roll_p: T,
    pub tethered_aileron_roll_d: T,
    pub free_rudder_beta_p: T,
    pub free_rudder_world_z_p: T,
    pub tethered_rudder_beta_p: T,
    pub tethered_rudder_rate_d: T,
    pub tethered_rudder_world_z_p: T,
    pub tethered_rudder_trim_offset_deg: T,
    pub guidance_min_lookahead_fraction: T,
    pub guidance_lateral_lookahead_ratio_limit: T,
    pub guidance_curvature_limit: T,
    pub free_pitch_ref_limit_deg: T,
    pub tethered_pitch_ref_limit_deg: T,
    pub elevator_pitch_p: T,
    pub elevator_pitch_d: T,
    pub altitude_pitch_p: T,
    pub altitude_pitch_i: T,
    pub tecs_altitude_error_limit_m: T,
    pub tecs_thrust_kinetic_p: T,
    pub tecs_thrust_kinetic_i: T,
    pub tecs_thrust_integrator_limit_nm: T,
    pub tethered_thrust_positive_potential_blend: T,
    pub tethered_tecs_potential_error_limit: T,
    pub tethered_tecs_potential_balance_weight: T,
    pub tethered_tecs_kinetic_deficit_balance_weight: T,
    pub tethered_tecs_kinetic_surplus_balance_weight: T,
    pub tecs_pitch_balance_p: T,
    pub tecs_pitch_balance_i: T,
    pub tecs_pitch_integrator_limit_deg: T,
    pub alpha_protection_min_deg: T,
    pub alpha_protection_max_deg: T,
    pub alpha_to_elevator: T,
    pub surface_limit_aileron_deg: T,
    pub surface_limit_rudder_deg: T,
    pub surface_limit_elevator_deg: T,
    pub motor_torque_max_nm: T,
    pub actuator_surface_tau_s: T,
    pub actuator_motor_tau_s: T,
    pub rotor_speed_soft_limit_radps: T,
    pub rotor_speed_hard_limit_radps: T,
}

impl Default for ControllerTuning<f64> {
    fn default() -> Self {
        Self {
            speed_phase_gain: 10.0,
            speed_min_mps: 30.0,
            speed_max_mps: 35.0,
            rabbit_speed_to_distance_s: 3.6,
            rabbit_min_distance_m: 20.0,
            rabbit_max_distance_m: 25.0,
            roll_feedforward_gain: 1.0,
            rabbit_bearing_roll_p: 1.0,
            rabbit_bearing_roll_i: 0.2,
            roll_curvature_p: 18.0,
            roll_curvature_i: 2.0,
            roll_curvature_integrator_limit: 0.04,
            roll_ref_limit_deg: 35.0,
            guidance_mode: 0.0,
            forward_heading_deg: 0.0,
            forward_lookahead_scale: 1.0,
            formation_spacing_m: 0.0,
            formation_lateral_offset_i_per_s: 0.08,
            formation_lateral_offset_limit_m: 20.0,
            formation_lateral_error_limit_m: 40.0,
            tethered_roll_ref_rate_limit_degps: 90.0,
            free_aileron_roll_p: 2.0,
            free_aileron_roll_d: 0.35,
            tethered_aileron_roll_p: 2.0,
            tethered_aileron_roll_d: 0.35,
            free_rudder_beta_p: 1.8,
            free_rudder_world_z_p: 0.0,
            tethered_rudder_beta_p: 1.8,
            tethered_rudder_rate_d: 0.35,
            tethered_rudder_world_z_p: 0.0,
            tethered_rudder_trim_offset_deg: 0.0,
            guidance_min_lookahead_fraction: 0.25,
            guidance_lateral_lookahead_ratio_limit: 0.75,
            guidance_curvature_limit: 0.04,
            free_pitch_ref_limit_deg: 22.0,
            tethered_pitch_ref_limit_deg: 22.0,
            elevator_pitch_p: 0.85,
            elevator_pitch_d: 0.25,
            altitude_pitch_p: 0.01,
            altitude_pitch_i: 0.001,
            tecs_altitude_error_limit_m: 25.0,
            tecs_thrust_kinetic_p: 0.3,
            tecs_thrust_kinetic_i: 0.06,
            tecs_thrust_integrator_limit_nm: 8.0,
            tethered_thrust_positive_potential_blend: 0.02,
            tethered_tecs_potential_error_limit: 245.0,
            tethered_tecs_potential_balance_weight: 0.9,
            tethered_tecs_kinetic_deficit_balance_weight: 1.1,
            tethered_tecs_kinetic_surplus_balance_weight: 1.0,
            tecs_pitch_balance_p: 0.0012,
            tecs_pitch_balance_i: 0.00035,
            tecs_pitch_integrator_limit_deg: 22.0,
            alpha_protection_min_deg: -8.0,
            alpha_protection_max_deg: 10.0,
            alpha_to_elevator: 20.0,
            surface_limit_aileron_deg: 28.64788975654116,
            surface_limit_rudder_deg: 25.0,
            surface_limit_elevator_deg: 17.188733853924695,
            motor_torque_max_nm: 45.6,
            actuator_surface_tau_s: 0.04,
            actuator_motor_tau_s: 0.08,
            rotor_speed_soft_limit_radps: 800.0,
            rotor_speed_hard_limit_radps: 900.0,
        }
    }
}

impl ControllerTuning<f64> {
    pub fn finite_or_default(self, default: &Self) -> Self {
        fn finite(value: f64, default: f64) -> f64 {
            if value.is_finite() { value } else { default }
        }

        Self {
            speed_phase_gain: finite(self.speed_phase_gain, default.speed_phase_gain),
            speed_min_mps: finite(self.speed_min_mps, default.speed_min_mps),
            speed_max_mps: finite(self.speed_max_mps, default.speed_max_mps),
            rabbit_speed_to_distance_s: finite(
                self.rabbit_speed_to_distance_s,
                default.rabbit_speed_to_distance_s,
            )
            .max(0.0),
            rabbit_min_distance_m: finite(
                self.rabbit_min_distance_m,
                default.rabbit_min_distance_m,
            )
            .max(0.0),
            rabbit_max_distance_m: finite(
                self.rabbit_max_distance_m,
                default.rabbit_max_distance_m,
            )
            .max(0.0),
            roll_feedforward_gain: finite(
                self.roll_feedforward_gain,
                default.roll_feedforward_gain,
            ),
            rabbit_bearing_roll_p: finite(
                self.rabbit_bearing_roll_p,
                default.rabbit_bearing_roll_p,
            ),
            rabbit_bearing_roll_i: finite(
                self.rabbit_bearing_roll_i,
                default.rabbit_bearing_roll_i,
            ),
            roll_curvature_p: finite(self.roll_curvature_p, default.roll_curvature_p),
            roll_curvature_i: finite(self.roll_curvature_i, default.roll_curvature_i),
            roll_curvature_integrator_limit: finite(
                self.roll_curvature_integrator_limit,
                default.roll_curvature_integrator_limit,
            )
            .abs(),
            roll_ref_limit_deg: finite(self.roll_ref_limit_deg, default.roll_ref_limit_deg).abs(),
            guidance_mode: finite(self.guidance_mode, default.guidance_mode).clamp(0.0, 2.0),
            forward_heading_deg: finite(self.forward_heading_deg, default.forward_heading_deg),
            forward_lookahead_scale: {
                let scale = finite(
                    self.forward_lookahead_scale,
                    default.forward_lookahead_scale,
                );
                if scale > 0.0 {
                    scale
                } else {
                    default.forward_lookahead_scale
                }
            },
            formation_spacing_m: finite(self.formation_spacing_m, default.formation_spacing_m)
                .max(0.0),
            formation_lateral_offset_i_per_s: finite(
                self.formation_lateral_offset_i_per_s,
                default.formation_lateral_offset_i_per_s,
            ),
            formation_lateral_offset_limit_m: finite(
                self.formation_lateral_offset_limit_m,
                default.formation_lateral_offset_limit_m,
            )
            .abs(),
            formation_lateral_error_limit_m: finite(
                self.formation_lateral_error_limit_m,
                default.formation_lateral_error_limit_m,
            )
            .abs(),
            tethered_roll_ref_rate_limit_degps: finite(
                self.tethered_roll_ref_rate_limit_degps,
                default.tethered_roll_ref_rate_limit_degps,
            )
            .abs(),
            free_aileron_roll_p: finite(self.free_aileron_roll_p, default.free_aileron_roll_p),
            free_aileron_roll_d: finite(self.free_aileron_roll_d, default.free_aileron_roll_d),
            tethered_aileron_roll_p: finite(
                self.tethered_aileron_roll_p,
                default.tethered_aileron_roll_p,
            ),
            tethered_aileron_roll_d: finite(
                self.tethered_aileron_roll_d,
                default.tethered_aileron_roll_d,
            ),
            free_rudder_beta_p: finite(self.free_rudder_beta_p, default.free_rudder_beta_p),
            free_rudder_world_z_p: finite(
                self.free_rudder_world_z_p,
                default.free_rudder_world_z_p,
            ),
            tethered_rudder_beta_p: finite(
                self.tethered_rudder_beta_p,
                default.tethered_rudder_beta_p,
            ),
            tethered_rudder_rate_d: finite(
                self.tethered_rudder_rate_d,
                default.tethered_rudder_rate_d,
            ),
            tethered_rudder_world_z_p: finite(
                self.tethered_rudder_world_z_p,
                default.tethered_rudder_world_z_p,
            ),
            tethered_rudder_trim_offset_deg: finite(
                self.tethered_rudder_trim_offset_deg,
                default.tethered_rudder_trim_offset_deg,
            ),
            guidance_min_lookahead_fraction: finite(
                self.guidance_min_lookahead_fraction,
                default.guidance_min_lookahead_fraction,
            )
            .max(0.0),
            guidance_lateral_lookahead_ratio_limit: finite(
                self.guidance_lateral_lookahead_ratio_limit,
                default.guidance_lateral_lookahead_ratio_limit,
            )
            .abs(),
            guidance_curvature_limit: finite(
                self.guidance_curvature_limit,
                default.guidance_curvature_limit,
            )
            .abs(),
            free_pitch_ref_limit_deg: finite(
                self.free_pitch_ref_limit_deg,
                default.free_pitch_ref_limit_deg,
            )
            .abs(),
            tethered_pitch_ref_limit_deg: finite(
                self.tethered_pitch_ref_limit_deg,
                default.tethered_pitch_ref_limit_deg,
            )
            .abs(),
            elevator_pitch_p: finite(self.elevator_pitch_p, default.elevator_pitch_p),
            elevator_pitch_d: finite(self.elevator_pitch_d, default.elevator_pitch_d),
            altitude_pitch_p: finite(self.altitude_pitch_p, default.altitude_pitch_p),
            altitude_pitch_i: finite(self.altitude_pitch_i, default.altitude_pitch_i),
            tecs_altitude_error_limit_m: finite(
                self.tecs_altitude_error_limit_m,
                default.tecs_altitude_error_limit_m,
            )
            .abs(),
            tecs_thrust_kinetic_p: finite(
                self.tecs_thrust_kinetic_p,
                default.tecs_thrust_kinetic_p,
            ),
            tecs_thrust_kinetic_i: finite(
                self.tecs_thrust_kinetic_i,
                default.tecs_thrust_kinetic_i,
            ),
            tecs_thrust_integrator_limit_nm: finite(
                self.tecs_thrust_integrator_limit_nm,
                default.tecs_thrust_integrator_limit_nm,
            )
            .abs(),
            tethered_thrust_positive_potential_blend: finite(
                self.tethered_thrust_positive_potential_blend,
                default.tethered_thrust_positive_potential_blend,
            ),
            tethered_tecs_potential_error_limit: finite(
                self.tethered_tecs_potential_error_limit,
                default.tethered_tecs_potential_error_limit,
            )
            .abs(),
            tethered_tecs_potential_balance_weight: finite(
                self.tethered_tecs_potential_balance_weight,
                default.tethered_tecs_potential_balance_weight,
            ),
            tethered_tecs_kinetic_deficit_balance_weight: finite(
                self.tethered_tecs_kinetic_deficit_balance_weight,
                default.tethered_tecs_kinetic_deficit_balance_weight,
            ),
            tethered_tecs_kinetic_surplus_balance_weight: finite(
                self.tethered_tecs_kinetic_surplus_balance_weight,
                default.tethered_tecs_kinetic_surplus_balance_weight,
            ),
            tecs_pitch_balance_p: finite(self.tecs_pitch_balance_p, default.tecs_pitch_balance_p),
            tecs_pitch_balance_i: finite(self.tecs_pitch_balance_i, default.tecs_pitch_balance_i),
            tecs_pitch_integrator_limit_deg: finite(
                self.tecs_pitch_integrator_limit_deg,
                default.tecs_pitch_integrator_limit_deg,
            )
            .abs(),
            alpha_protection_min_deg: finite(
                self.alpha_protection_min_deg,
                default.alpha_protection_min_deg,
            ),
            alpha_protection_max_deg: finite(
                self.alpha_protection_max_deg,
                default.alpha_protection_max_deg,
            ),
            alpha_to_elevator: finite(self.alpha_to_elevator, default.alpha_to_elevator),
            surface_limit_aileron_deg: finite(
                self.surface_limit_aileron_deg,
                default.surface_limit_aileron_deg,
            )
            .abs(),
            surface_limit_rudder_deg: finite(
                self.surface_limit_rudder_deg,
                default.surface_limit_rudder_deg,
            )
            .abs(),
            surface_limit_elevator_deg: finite(
                self.surface_limit_elevator_deg,
                default.surface_limit_elevator_deg,
            )
            .abs(),
            motor_torque_max_nm: finite(self.motor_torque_max_nm, default.motor_torque_max_nm)
                .max(0.0),
            actuator_surface_tau_s: finite(
                self.actuator_surface_tau_s,
                default.actuator_surface_tau_s,
            )
            .max(0.0),
            actuator_motor_tau_s: finite(self.actuator_motor_tau_s, default.actuator_motor_tau_s)
                .max(0.0),
            rotor_speed_soft_limit_radps: finite(
                self.rotor_speed_soft_limit_radps,
                default.rotor_speed_soft_limit_radps,
            )
            .max(0.0),
            rotor_speed_hard_limit_radps: finite(
                self.rotor_speed_hard_limit_radps,
                default.rotor_speed_hard_limit_radps,
            )
            .max(0.0),
        }
    }
}

#[derive(Clone, Debug, Vectorize)]
pub struct ControllerGains<T> {
    pub trim: KiteControls<T>,
    pub wx_to_ail: T,
    pub wy_to_elev: T,
    pub wz_to_rudder: T,
    pub speed_to_torque_p: T,
    pub speed_to_torque_i: T,
    pub rabbit_distance: T,
    pub phase_lag_to_radius: T,
    pub vert_vel_to_rabbit_height: T,
    pub gain_int_y: T,
    pub gain_int_z: T,
    pub speed_ref: T,
    pub disk_center_n: Vector3<T>,
    pub disk_radius: T,
    pub tuning: ControllerTuning<T>,
}

#[derive(Clone, Debug, Vectorize)]
pub struct Params<T, const NK: usize> {
    pub kites: [KiteParams<T>; NK],
    pub common_tether: TetherParams<T>,
    pub splitter_mass: T,
    pub payload_mass: T,
    pub environment: Environment<T>,
    pub kite_gusts_n: [Vector3<T>; NK],
    pub controller: ControllerGains<T>,
}

#[derive(Clone, Debug, Vectorize)]
pub struct KiteDiagnostics<T> {
    pub cad_position_n: Vector3<T>,
    pub cad_velocity_n: Vector3<T>,
    pub body_accel_b: Vector3<T>,
    pub body_accel_n: Vector3<T>,
    pub felt_accel_g_b: Vector3<T>,
    pub tether_load_g_b: Vector3<T>,
    pub aero_load_g_b: Vector3<T>,
    pub omega_b: Vector3<T>,
    pub airspeed: T,
    pub alpha: T,
    pub beta: T,
    pub top_tension: T,
    pub phase_angle: T,
    pub phase_error: T,
    pub speed_target: T,
    pub altitude: T,
    pub altitude_ref: T,
    pub kinetic_energy_specific: T,
    pub kinetic_energy_ref_specific: T,
    pub kinetic_energy_error_specific: T,
    pub potential_energy_specific: T,
    pub potential_energy_ref_specific: T,
    pub potential_energy_error_specific: T,
    pub total_energy_error_specific: T,
    pub energy_balance_error_specific: T,
    pub thrust_energy_integrator: T,
    pub pitch_energy_integrator: T,
    pub rabbit_phase: T,
    pub rabbit_radius: T,
    pub rabbit_distance: T,
    pub rabbit_target_distance: T,
    pub rabbit_bearing_y: T,
    pub rabbit_vector_b: Vector3<T>,
    pub rabbit_target_n: Vector3<T>,
    pub lateral_outer_mode: T,
    pub forward_frame_heading: T,
    pub forward_lane_y: T,
    pub forward_cross_track_error: T,
    pub forward_neighbor_prev_y_f: T,
    pub forward_neighbor_next_y_f: T,
    pub forward_formation_error: T,
    pub forward_formation_spacing: T,
    pub forward_lateral_offset: T,
    pub forward_lane_point_n: Vector3<T>,
    pub forward_formation_error_tip_n: Vector3<T>,
    pub orbit_radius: T,
    pub curvature_y_b: T,
    pub curvature_y_ref: T,
    pub curvature_y_est: T,
    pub omega_world_z_ref: T,
    pub omega_world_z: T,
    pub beta_ref: T,
    pub roll_ref: T,
    pub roll_ff: T,
    pub roll_p: T,
    pub roll_i: T,
    pub pitch_ref: T,
    pub pitch_ref_p: T,
    pub pitch_ref_i: T,
    pub curvature_z_b: T,
    pub curvature_z_ref: T,
    pub aileron_trim: T,
    pub aileron_roll_p: T,
    pub aileron_roll_d: T,
    pub rudder_trim: T,
    pub rudder_beta_p: T,
    pub rudder_rate_d: T,
    pub rudder_world_z_p: T,
    pub elevator_trim: T,
    pub elevator_pitch_p: T,
    pub elevator_pitch_d: T,
    pub elevator_alpha_protection: T,
    pub motor_torque_trim: T,
    pub motor_torque_p: T,
    pub motor_torque_i: T,
    pub motor_force: T,
    pub motor_power: T,
    pub aero_dissipated_power: T,
    pub tether_dissipated_power: T,
    pub total_force_b: Vector3<T>,
    pub aero_force_b: Vector3<T>,
    pub aero_force_drag_b: Vector3<T>,
    pub aero_force_side_b: Vector3<T>,
    pub aero_force_lift_b: Vector3<T>,
    pub tether_force_b: Vector3<T>,
    pub gravity_force_b: Vector3<T>,
    pub motor_force_b: Vector3<T>,
    pub total_moment_b: Vector3<T>,
    pub aero_moment_b: Vector3<T>,
    pub rudder_force_b: Vector3<T>,
    pub rudder_moment_b: Vector3<T>,
    pub tether_moment_b: Vector3<T>,
    pub motor_moment_b: Vector3<T>,
    pub cl_total: T,
    pub cl_0_term: T,
    pub cl_alpha_term: T,
    pub cl_elevator_term: T,
    pub cl_flap_term: T,
    pub cd_total: T,
    pub cd_0_term: T,
    pub cd_induced_term: T,
    pub cd_surface_term: T,
    pub cy_total: T,
    pub cy_beta_term: T,
    pub cy_rudder_term: T,
    pub roll_coeff_total: T,
    pub roll_beta_term: T,
    pub roll_p_term: T,
    pub roll_r_term: T,
    pub roll_aileron_term: T,
    pub pitch_coeff_total: T,
    pub pitch_0_term: T,
    pub pitch_alpha_term: T,
    pub pitch_q_term: T,
    pub pitch_elevator_term: T,
    pub pitch_flap_term: T,
    pub yaw_coeff_total: T,
    pub yaw_beta_term: T,
    pub yaw_p_term: T,
    pub yaw_r_term: T,
    pub yaw_rudder_term: T,
    pub kinetic_energy: T,
    pub potential_energy: T,
    pub tether_strain_energy: T,
}

#[derive(Clone, Debug, Vectorize)]
pub struct Diagnostics<T, const NK: usize> {
    pub kites: [KiteDiagnostics<T>; NK],
    pub payload_position_n: Vector3<T>,
    pub splitter_position_n: Vector3<T>,
    pub total_kinetic_energy: T,
    pub total_potential_energy: T,
    pub total_tether_strain_energy: T,
    pub total_motor_power: T,
    pub total_dissipated_power: T,
    pub total_mechanical_energy: T,
    pub energy_conservation_residual: T,
    pub work_minus_potential: T,
}

#[derive(Clone, Debug)]
pub struct SimulationFrame<T, const NK: usize, const N_COMMON: usize, const N_UPPER: usize> {
    pub time: T,
    pub state: State<T, NK, N_COMMON, N_UPPER>,
    pub controls: Controls<T, NK>,
    pub diagnostics: Diagnostics<T, NK>,
    pub clean_wind_n: Vector3<T>,
    pub kite_gusts_n: [Vector3<T>; NK],
    pub kite_ref_spans: [T; NK],
    pub kite_ref_chords: [T; NK],
    pub kite_ref_areas: [T; NK],
    pub kite_cad_offsets_b: [Vector3<T>; NK],
    pub kite_bridle_pivots_b: [Vector3<T>; NK],
    pub kite_bridle_radii: [T; NK],
    pub kite_bridle_positions_n: [Vector3<T>; NK],
    pub control_ring_center_n: Vector3<T>,
    pub control_ring_radius: T,
    pub kite_masses_kg: [T; NK],
    pub sum_kite_masses_kg: T,
    pub tether_mass_kg: T,
    pub payload_mass_kg: T,
    pub payload_to_lifter_mass_ratio: T,
    pub common_tether_tensions: Vec<T>,
    pub upper_tether_tensions: Vec<Vec<T>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    pub duration: f64,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub max_phase_error: f64,
    pub final_total_work: f64,
    pub final_total_dissipated_work: f64,
    pub final_total_kinetic_energy: f64,
    pub final_total_potential_energy: f64,
    pub final_total_tether_strain_energy: f64,
    pub final_total_mechanical_energy: f64,
    pub final_energy_conservation_residual: f64,
    pub failure: Option<SimulationFailure>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimulationProgress {
    pub iteration: usize,
    pub time: f64,
    pub duration: f64,
    pub interval_dt: f64,
    pub sample_count: usize,
    pub accepted_steps_total: usize,
    pub rejected_steps_total: usize,
    pub accepted_steps_interval: usize,
    pub rejected_steps_interval: usize,
    pub substeps_interval: usize,
    pub substep_budget: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimulationFailure {
    pub time: f64,
    pub kite_index: usize,
    pub quantity: String,
    pub value_deg: f64,
    pub lower_limit_deg: f64,
    pub upper_limit_deg: f64,
    pub alpha_deg: f64,
    pub beta_deg: f64,
    pub message: String,
}

#[derive(Clone, Debug)]
pub struct RunResult<const NK: usize, const N_COMMON: usize, const N_UPPER: usize> {
    pub frames: Vec<SimulationFrame<f64, NK, N_COMMON, N_UPPER>>,
    pub summary: RunSummary,
}
