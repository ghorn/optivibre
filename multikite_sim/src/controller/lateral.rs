use super::state::KiteControllerState;
use crate::math::clamp;
use crate::types::ControllerTuning;
use nalgebra::Vector3;

pub(super) const TETHERED_CURVATURE_Y_INTEGRATOR_OUTPUT_LIMIT: f64 = 0.1;
pub(super) const TETHERED_CURVATURE_Z_INTEGRATOR_OUTPUT_LIMIT: f64 = 0.05;
pub(super) const GUIDANCE_MODE_RABBIT: u8 = 0;
pub(super) const GUIDANCE_MODE_CURVATURE: u8 = 1;
pub(super) const GUIDANCE_MODE_SWITCH: u8 = 2;

#[derive(Clone, Copy, Debug)]
pub(super) struct RollReferenceBreakdown {
    pub total: f64,
    pub feedforward: f64,
    pub proportional: f64,
    pub integrator: f64,
}

pub(super) fn speed_integrator_target(
    phase_error: f64,
    speed_ref: f64,
    tuning: &ControllerTuning<f64>,
) -> f64 {
    let min_speed = tuning.speed_min_mps;
    let max_speed = tuning.speed_max_mps;
    clamp(
        speed_ref + tuning.speed_phase_gain * phase_error,
        min_speed.min(max_speed),
        max_speed.max(min_speed),
    )
}

pub(super) fn orbit_roll_feedforward(
    inertial_speed: f64,
    orbit_radius: f64,
    g: f64,
    tuning: &ControllerTuning<f64>,
) -> f64 {
    tuning.roll_feedforward_gain
        * (inertial_speed * inertial_speed * orbit_curvature_y_reference(orbit_radius) / g).atan()
}

pub(super) fn scheduled_rabbit_distance(
    speed_target: f64,
    fallback_distance: f64,
    tuning: &ControllerTuning<f64>,
) -> f64 {
    let scheduled = speed_target.max(0.0) * tuning.rabbit_speed_to_distance_s;
    let default_distance = fallback_distance.max(1.0);
    let min_distance = if tuning.rabbit_min_distance_m > 0.0 {
        tuning.rabbit_min_distance_m
    } else {
        default_distance
    };
    let max_distance = if tuning.rabbit_max_distance_m > 0.0 {
        tuning.rabbit_max_distance_m
    } else {
        default_distance
    };
    clamp(
        scheduled,
        min_distance.min(max_distance),
        min_distance.max(max_distance),
    )
}

pub(super) fn rate_limit(current: f64, target: f64, max_rate: f64, dt: f64) -> f64 {
    current + clamp(target - current, -max_rate * dt, max_rate * dt)
}

pub(super) fn direct_rabbit_bearing_y(rabbit_vector_b: &Vector3<f64>) -> f64 {
    if rabbit_vector_b[0].hypot(rabbit_vector_b[1]) <= 1.0e-9 {
        0.0
    } else {
        rabbit_vector_b[1].atan2(rabbit_vector_b[0])
    }
}

pub(super) fn direct_rabbit_roll_reference_breakdown(
    rabbit_vector_b: &Vector3<f64>,
    control_state: &mut KiteControllerState,
    dt_control: f64,
    tuning: &ControllerTuning<f64>,
) -> RollReferenceBreakdown {
    let bearing_y = direct_rabbit_bearing_y(rabbit_vector_b);
    let limit = tuning.roll_ref_limit_deg.to_radians().abs();
    if tuning.rabbit_bearing_roll_i.abs() > 1.0e-9 {
        control_state.rabbit_bearing_to_roll_integrator += bearing_y * dt_control;
        let integrator_limit = limit / tuning.rabbit_bearing_roll_i.abs();
        control_state.rabbit_bearing_to_roll_integrator = clamp(
            control_state.rabbit_bearing_to_roll_integrator,
            -integrator_limit,
            integrator_limit,
        );
    } else {
        control_state.rabbit_bearing_to_roll_integrator = 0.0;
    }
    let proportional = tuning.rabbit_bearing_roll_p * bearing_y;
    let integrator = tuning.rabbit_bearing_roll_i * control_state.rabbit_bearing_to_roll_integrator;
    RollReferenceBreakdown {
        total: clamp(proportional + integrator, -limit, limit),
        feedforward: 0.0,
        proportional,
        integrator,
    }
}

pub(super) fn lateral_guidance_curvatures(
    rabbit_vector_b: &Vector3<f64>,
    _orbit_radius: f64,
    rabbit_distance: f64,
    tuning: &ControllerTuning<f64>,
) -> (f64, f64) {
    // Rabbit mode bypasses this curvature conversion; switch mode uses it only after the target falls behind.
    let minimum_lookahead = guidance_minimum_lookahead(rabbit_distance, tuning);
    let rabbit_curvature =
        pursuit_curvatures_from_body_vector(rabbit_vector_b, minimum_lookahead, tuning);
    match guidance_mode(tuning) {
        GUIDANCE_MODE_CURVATURE => rabbit_curvature,
        GUIDANCE_MODE_SWITCH => {
            if rabbit_vector_b[0] >= minimum_lookahead.max(1.0) {
                (0.0, 0.0)
            } else {
                rabbit_curvature
            }
        }
        _ => (0.0, 0.0),
    }
}

pub(super) fn guidance_uses_direct_rabbit(
    rabbit_vector_b: &Vector3<f64>,
    rabbit_distance: f64,
    tuning: &ControllerTuning<f64>,
) -> bool {
    match guidance_mode(tuning) {
        GUIDANCE_MODE_CURVATURE => false,
        GUIDANCE_MODE_SWITCH => {
            rabbit_vector_b[0] >= guidance_minimum_lookahead(rabbit_distance, tuning).max(1.0)
        }
        _ => true,
    }
}

fn orbit_curvature_y_reference(radius: f64) -> f64 {
    1.0 / radius.max(1.0)
}

fn guidance_mode(tuning: &ControllerTuning<f64>) -> u8 {
    if !tuning.guidance_mode.is_finite() {
        return GUIDANCE_MODE_RABBIT;
    }
    match tuning.guidance_mode.round() as i32 {
        1 => GUIDANCE_MODE_CURVATURE,
        2 => GUIDANCE_MODE_SWITCH,
        _ => GUIDANCE_MODE_RABBIT,
    }
}

fn guidance_minimum_lookahead(rabbit_distance: f64, tuning: &ControllerTuning<f64>) -> f64 {
    tuning.guidance_min_lookahead_fraction * rabbit_distance.max(1.0)
}

fn pursuit_curvatures_from_body_vector(
    rabbit_vector_b: &Vector3<f64>,
    minimum_lookahead: f64,
    tuning: &ControllerTuning<f64>,
) -> (f64, f64) {
    let lookahead = rabbit_vector_b[0].max(minimum_lookahead.max(1.0));
    let lateral_limit = tuning.guidance_lateral_lookahead_ratio_limit * lookahead;
    let lateral_y = clamp(rabbit_vector_b[1], -lateral_limit, lateral_limit);
    let lateral_z = clamp(rabbit_vector_b[2], -lateral_limit, lateral_limit);
    let denominator = lookahead * lookahead;
    (
        clamp(
            2.0 * lateral_y / denominator,
            -tuning.guidance_curvature_limit,
            tuning.guidance_curvature_limit,
        ),
        clamp(
            2.0 * lateral_z / denominator,
            -tuning.guidance_curvature_limit,
            tuning.guidance_curvature_limit,
        ),
    )
}
