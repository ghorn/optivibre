use crate::types::Diagnostics;
use nalgebra::Vector3;

#[derive(Clone, Debug)]
pub(super) struct KiteControllerState {
    pub(super) thrust_energy_integrator: f64,
    pub(super) pitch_energy_integrator: f64,
    pub(super) curvature_to_roll_integrator: f64,
    pub(super) rabbit_bearing_to_roll_integrator: f64,
    pub(super) forward_lateral_offset_m: f64,
    pub(super) roll_ref_command: f64,
    pub(super) roll_ref_initialized: bool,
    pub(super) curvature_y_integrator: f64,
    pub(super) curvature_z_integrator: f64,
}

/// Persistent controller memory for the swarm scheduler plus each single-kite controller.
#[derive(Clone, Debug)]
pub struct ControllerState<const NK: usize> {
    pub(super) kites: [KiteControllerState; NK],
    pub(super) initial_phase: [f64; NK],
    pub(super) initial_altitude: [f64; NK],
}

/// Telemetry for one kite controller invocation. Keeping this per-kite matches the
/// ownership boundary used by `compute_kite_control`.
#[derive(Clone, Debug)]
pub struct KiteControllerTrace {
    pub phase_error: f64,
    pub speed_target: f64,
    pub altitude: f64,
    pub altitude_ref: f64,
    pub kinetic_energy_specific: f64,
    pub kinetic_energy_ref_specific: f64,
    pub kinetic_energy_error_specific: f64,
    pub potential_energy_specific: f64,
    pub potential_energy_ref_specific: f64,
    pub potential_energy_error_specific: f64,
    pub total_energy_error_specific: f64,
    pub energy_balance_error_specific: f64,
    pub thrust_energy_integrator: f64,
    pub pitch_energy_integrator: f64,
    pub rabbit_phase: f64,
    pub rabbit_radius: f64,
    pub rabbit_distance: f64,
    pub rabbit_target_distance: f64,
    pub rabbit_bearing_y: f64,
    pub rabbit_vector_b: Vector3<f64>,
    pub rabbit_target_n: Vector3<f64>,
    pub lateral_outer_mode: f64,
    pub forward_frame_heading: f64,
    pub forward_lane_y: f64,
    pub forward_cross_track_error: f64,
    pub forward_neighbor_prev_y_f: f64,
    pub forward_neighbor_next_y_f: f64,
    pub forward_formation_error: f64,
    pub forward_formation_spacing: f64,
    pub forward_lateral_offset: f64,
    pub forward_lane_point_n: Vector3<f64>,
    pub forward_formation_error_tip_n: Vector3<f64>,
    pub curvature_y_ref: f64,
    pub curvature_y_estimate: f64,
    pub omega_world_z_ref: f64,
    pub omega_world_z: f64,
    pub beta_ref: f64,
    pub roll_ref: f64,
    pub roll_feedforward: f64,
    pub roll_proportional: f64,
    pub roll_integrator: f64,
    pub pitch_ref: f64,
    pub pitch_ref_proportional: f64,
    pub pitch_ref_integrator: f64,
    pub curvature_z_ref: f64,
    pub aileron_trim: f64,
    pub aileron_roll_proportional: f64,
    pub aileron_roll_derivative: f64,
    pub rudder_trim: f64,
    pub rudder_beta_proportional: f64,
    pub rudder_rate_derivative: f64,
    pub rudder_world_z_proportional: f64,
    pub elevator_trim: f64,
    pub elevator_pitch_proportional: f64,
    pub elevator_pitch_derivative: f64,
    pub elevator_alpha_protection: f64,
    pub motor_torque_trim: f64,
    pub motor_torque_proportional: f64,
    pub motor_torque_integrator: f64,
}

/// Swarm-level trace wrapper; fields live on the per-kite trace, not as parallel arrays.
#[derive(Clone, Debug)]
pub struct ControllerTrace<const NK: usize> {
    pub kites: [KiteControllerTrace; NK],
}

impl<const NK: usize> ControllerState<NK> {
    pub fn new(initial_diag: &Diagnostics<f64, NK>) -> Self {
        Self {
            kites: std::array::from_fn(|_| KiteControllerState {
                thrust_energy_integrator: 0.0,
                pitch_energy_integrator: 0.0,
                curvature_to_roll_integrator: 0.0,
                rabbit_bearing_to_roll_integrator: 0.0,
                forward_lateral_offset_m: 0.0,
                roll_ref_command: 0.0,
                roll_ref_initialized: false,
                curvature_y_integrator: 0.0,
                curvature_z_integrator: 0.0,
            }),
            initial_phase: std::array::from_fn(|index| initial_diag.kites[index].phase_angle),
            initial_altitude: std::array::from_fn(|index| {
                (-initial_diag.kites[index].cad_position_n[2]).max(0.0)
            }),
        }
    }
}
