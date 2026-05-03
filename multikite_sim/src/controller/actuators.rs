use crate::math::{clamp, wrap_angle};
use crate::types::ControllerTuning;

#[derive(Clone, Copy, Debug)]
pub(super) struct AileronCommandBreakdown {
    pub trim: f64,
    pub roll_p: f64,
    pub roll_d: f64,
    pub total: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RudderCommandBreakdown {
    pub trim: f64,
    pub beta_p: f64,
    pub rate_d: f64,
    pub world_z_p: f64,
    pub total: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ElevatorCommandBreakdown {
    pub trim: f64,
    pub pitch_p: f64,
    pub pitch_d: f64,
    pub alpha_protection: f64,
    pub total: f64,
}

pub(super) fn tethered_rudder_breakdown(
    trim: f64,
    beta: f64,
    omega_z: f64,
    omega_world_z: f64,
    omega_world_z_ref: f64,
    tuning: &ControllerTuning<f64>,
) -> RudderCommandBreakdown {
    let limit = tuning.surface_limit_rudder_deg.to_radians().abs();
    let trim = trim + tuning.tethered_rudder_trim_offset_deg.to_radians();
    let beta_p = -tuning.tethered_rudder_beta_p * beta;
    let rate_d = tuning.tethered_rudder_rate_d * omega_z;
    let world_z_p = tuning.tethered_rudder_world_z_p * (omega_world_z - omega_world_z_ref);
    RudderCommandBreakdown {
        trim,
        beta_p,
        rate_d,
        world_z_p,
        total: clamp(trim + beta_p + rate_d + world_z_p, -limit, limit),
    }
}

pub(super) fn tethered_aileron_breakdown(
    trim: f64,
    roll_ref: f64,
    roll_angle: f64,
    omega_x: f64,
    tuning: &ControllerTuning<f64>,
) -> AileronCommandBreakdown {
    let surface_limit = tuning.surface_limit_aileron_deg.to_radians().abs();
    let roll_p = -tuning.tethered_aileron_roll_p * wrap_angle(roll_ref - roll_angle);
    let roll_d = tuning.tethered_aileron_roll_d * omega_x;
    AileronCommandBreakdown {
        trim,
        roll_p,
        roll_d,
        total: clamp(trim + roll_p + roll_d, -surface_limit, surface_limit),
    }
}

pub(super) fn elevator_breakdown(
    trim: f64,
    pitch_ref: f64,
    pitch_angle: f64,
    omega_y: f64,
    alpha_protection: f64,
    tuning: &ControllerTuning<f64>,
) -> ElevatorCommandBreakdown {
    let elevator_limit = tuning.surface_limit_elevator_deg.to_radians().abs();
    let pitch_p = -tuning.elevator_pitch_p * (pitch_ref - pitch_angle);
    let pitch_d = tuning.elevator_pitch_d * omega_y;
    let alpha_protection = tuning.alpha_to_elevator * alpha_protection;
    ElevatorCommandBreakdown {
        trim,
        pitch_p,
        pitch_d,
        alpha_protection,
        total: clamp(
            trim + pitch_p + pitch_d + alpha_protection,
            -elevator_limit,
            elevator_limit,
        ),
    }
}
