use crate::math::{clamp, wrap_angle};
use crate::types::ControllerTuning;

pub(super) fn tethered_rudder_command(
    trim: f64,
    beta: f64,
    omega_z: f64,
    omega_world_z: f64,
    omega_world_z_ref: f64,
    tuning: &ControllerTuning<f64>,
) -> f64 {
    let limit = tuning.surface_limit_rudder_deg.to_radians().abs();
    clamp(
        trim - tuning.tethered_rudder_beta_p * beta
            + tuning.tethered_rudder_trim_offset_deg.to_radians()
            + tuning.tethered_rudder_rate_d * omega_z
            + tuning.tethered_rudder_world_z_p * (omega_world_z - omega_world_z_ref),
        -limit,
        limit,
    )
}

pub(super) fn tethered_aileron_command(
    trim: f64,
    roll_ref: f64,
    roll_angle: f64,
    omega_x: f64,
    tuning: &ControllerTuning<f64>,
) -> f64 {
    let surface_limit = tuning.surface_limit_aileron_deg.to_radians().abs();
    clamp(
        trim - tuning.tethered_aileron_roll_p * wrap_angle(roll_ref - roll_angle)
            + tuning.tethered_aileron_roll_d * omega_x,
        -surface_limit,
        surface_limit,
    )
}

pub(super) fn elevator_command(
    trim: f64,
    pitch_ref: f64,
    pitch_angle: f64,
    omega_y: f64,
    alpha_protection: f64,
    tuning: &ControllerTuning<f64>,
) -> f64 {
    let elevator_limit = tuning.surface_limit_elevator_deg.to_radians().abs();
    clamp(
        trim - tuning.elevator_pitch_p * (pitch_ref - pitch_angle)
            + tuning.elevator_pitch_d * omega_y
            + tuning.alpha_to_elevator * alpha_protection,
        -elevator_limit,
        elevator_limit,
    )
}
