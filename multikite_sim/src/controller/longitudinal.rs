use crate::math::clamp;
use crate::types::ControllerTuning;

const FREE_FLIGHT_DEMO_STAGE_S: f64 = 10.0;
const FREE_FLIGHT_DEMO_SPEED_STEP_MPS: f64 = 3.0;
const FREE_FLIGHT_DEMO_ALTITUDE_STEP_UP_M: f64 = 12.0;
const FREE_FLIGHT_DEMO_ALTITUDE_STEP_DOWN_M: f64 = -8.0;

#[derive(Clone, Copy, Debug)]
pub(crate) struct FreeFlightReference {
    pub speed_target: f64,
    pub altitude_ref_raw: f64,
    pub roll_ref: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct TecsTerms {
    pub altitude_ref: f64,
    pub kinetic_energy: f64,
    pub kinetic_energy_ref: f64,
    pub kinetic_energy_error: f64,
    pub potential_energy: f64,
    pub potential_energy_ref: f64,
    pub potential_energy_error: f64,
    pub total_energy_error: f64,
    pub energy_balance_error: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SaturatedPiConfig {
    pub bias: f64,
    pub kp: f64,
    pub ki: f64,
    pub output_min: f64,
    pub output_max: f64,
    pub integrator_min: f64,
    pub integrator_max: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct SaturatedPiBreakdown {
    pub bias: f64,
    pub proportional: f64,
    pub integrator: f64,
    pub total: f64,
}

pub(super) fn default_free_flight_reference(
    _index: usize,
    time: f64,
    initial_altitude: f64,
    speed_ref: f64,
) -> FreeFlightReference {
    FreeFlightReference {
        speed_target: free_flight_speed_reference(time, speed_ref),
        altitude_ref_raw: free_flight_altitude_reference(time, initial_altitude),
        roll_ref: free_flight_roll_reference(time),
    }
}

pub(super) fn tecs_terms(
    altitude: f64,
    altitude_ref_raw: f64,
    airspeed: f64,
    speed_ref: f64,
    gravity: f64,
    tethered: bool,
    tuning: &ControllerTuning<f64>,
) -> TecsTerms {
    let altitude_error = clamp(
        altitude_ref_raw - altitude,
        -tuning.tecs_altitude_error_limit_m,
        tuning.tecs_altitude_error_limit_m,
    );
    let kinetic_energy = 0.5 * airspeed * airspeed;
    let kinetic_energy_ref = 0.5 * speed_ref * speed_ref;
    let kinetic_energy_error = kinetic_energy_ref - kinetic_energy;
    let potential_energy = gravity * altitude;
    let potential_energy_error_raw = gravity * altitude_error;
    let potential_energy_error = if tethered {
        clamp(
            potential_energy_error_raw,
            -tuning.tethered_tecs_potential_error_limit,
            tuning.tethered_tecs_potential_error_limit,
        )
    } else {
        potential_energy_error_raw
    };
    let altitude_ref = altitude + potential_energy_error / gravity.max(1.0e-6);
    let potential_energy_ref = potential_energy + potential_energy_error;
    let kinetic_balance_weight = if tethered && kinetic_energy_error > 0.0 {
        tuning.tethered_tecs_kinetic_deficit_balance_weight
    } else if tethered {
        tuning.tethered_tecs_kinetic_surplus_balance_weight
    } else {
        1.0
    };
    let potential_balance_weight = if tethered {
        tuning.tethered_tecs_potential_balance_weight
    } else {
        1.0
    };
    TecsTerms {
        altitude_ref,
        kinetic_energy,
        kinetic_energy_ref,
        kinetic_energy_error,
        potential_energy,
        potential_energy_ref,
        potential_energy_error,
        total_energy_error: kinetic_energy_error + potential_energy_error,
        energy_balance_error: potential_balance_weight * potential_energy_error
            - kinetic_balance_weight * kinetic_energy_error,
    }
}

pub(super) fn saturated_pi_breakdown(
    integrator: &mut f64,
    error: f64,
    dt: f64,
    config: SaturatedPiConfig,
) -> SaturatedPiBreakdown {
    // Hold the integrator when it would drive an already-saturated output farther into saturation.
    let candidate_integrator = clamp(
        *integrator + config.ki * error * dt,
        config.integrator_min,
        config.integrator_max,
    );
    let candidate_unsat = config.bias + config.kp * error + candidate_integrator;
    let blocked_high = candidate_unsat > config.output_max && error > 0.0;
    let blocked_low = candidate_unsat < config.output_min && error < 0.0;
    if !(blocked_high || blocked_low) {
        *integrator = candidate_integrator;
    }
    let proportional = config.kp * error;
    SaturatedPiBreakdown {
        bias: config.bias,
        proportional,
        integrator: *integrator,
        total: clamp(
            config.bias + proportional + *integrator,
            config.output_min,
            config.output_max,
        ),
    }
}

pub(super) fn limit_motor_torque_for_rotor_speed(
    commanded_torque: f64,
    rotor_speed: f64,
    tuning: &ControllerTuning<f64>,
) -> f64 {
    if rotor_speed <= tuning.rotor_speed_soft_limit_radps {
        return commanded_torque;
    }

    let denominator = (tuning.rotor_speed_hard_limit_radps - tuning.rotor_speed_soft_limit_radps)
        .abs()
        .max(1.0e-9);
    let fade = ((tuning.rotor_speed_hard_limit_radps - rotor_speed) / denominator).clamp(0.0, 1.0);
    commanded_torque
        .min(tuning.motor_torque_max_nm * fade)
        .max(0.0)
}

fn free_flight_roll_reference(time: f64) -> f64 {
    if time < 60.0 {
        0.0
    } else if time < 70.0 {
        20.0_f64.to_radians()
    } else if time < 80.0 {
        -20.0_f64.to_radians()
    } else {
        0.0
    }
}

fn free_flight_speed_reference(time: f64, speed_ref: f64) -> f64 {
    match free_flight_demo_stage(time) {
        1 | 2 | 4 => speed_ref + FREE_FLIGHT_DEMO_SPEED_STEP_MPS,
        3 => speed_ref - FREE_FLIGHT_DEMO_SPEED_STEP_MPS,
        _ => speed_ref,
    }
}

fn free_flight_altitude_reference(time: f64, initial_altitude: f64) -> f64 {
    match free_flight_demo_stage(time) {
        2 | 4 => initial_altitude + FREE_FLIGHT_DEMO_ALTITUDE_STEP_UP_M,
        3 => initial_altitude + FREE_FLIGHT_DEMO_ALTITUDE_STEP_DOWN_M,
        _ => initial_altitude,
    }
}

fn free_flight_demo_stage(time: f64) -> usize {
    (time.max(0.0) / FREE_FLIGHT_DEMO_STAGE_S).floor() as usize
}
