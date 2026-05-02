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
pub enum Preset {
    FreeFlight1,
    Star1,
    Y2Low,
    Y2,
    Y2High,
    Star3,
    Star4,
    SimpleTether,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub duration: f64,
    pub dt_control: f64,
    pub rk_abs_tol: f64,
    pub rk_rel_tol: f64,
    pub max_substeps: usize,
    pub phase_mode: PhaseMode,
    pub sample_stride: usize,
    pub sim_noise_enabled: bool,
    pub bridle_enabled: bool,
    pub longitudinal_mode: LongitudinalMode,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            duration: 10.0,
            dt_control: 0.02,
            rk_abs_tol: 1.0e-4,
            rk_rel_tol: 1.0e-4,
            max_substeps: 4096,
            phase_mode: PhaseMode::Adaptive,
            sample_stride: 1,
            sim_noise_enabled: false,
            bridle_enabled: true,
            longitudinal_mode: LongitudinalMode::TotalEnergy,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InitRequest {
    pub preset: Preset,
    pub payload_mass_kg: Option<f64>,
    pub wind_speed_mps: Option<f64>,
}

impl Default for InitRequest {
    fn default() -> Self {
        Self {
            preset: Preset::Y2,
            payload_mass_kg: None,
            wind_speed_mps: None,
        }
    }
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
    pub rabbit_target_n: Vector3<T>,
    pub orbit_radius: T,
    pub curvature_y_b: T,
    pub curvature_y_ref: T,
    pub curvature_y_est: T,
    pub omega_world_z_ref: T,
    pub omega_world_z: T,
    pub beta_ref: T,
    pub roll_ref: T,
    pub roll_ff: T,
    pub pitch_ref: T,
    pub curvature_z_b: T,
    pub curvature_z_ref: T,
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
