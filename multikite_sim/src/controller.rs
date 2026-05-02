use crate::math::{circular_mean, clamp, rotate_body_to_nav, rotate_nav_to_body, sub, wrap_angle};
use crate::types::{
    ControlSurfaces, Controls, Diagnostics, KiteControls, LongitudinalMode, Params, PhaseMode,
    State,
};
use nalgebra::Vector3;

const SPEED_INTEGRATOR_PHASE_GAIN: f64 = 10.0;
const SPEED_INTEGRATOR_MIN_MPS: f64 = 18.0;
const SPEED_INTEGRATOR_MAX_MPS: f64 = 35.0;
const ROLL_FEEDFORWARD_GAIN: f64 = 1.0;
const ROLL_FROM_CURVATURE_P: f64 = 18.0;
const ROLL_FROM_CURVATURE_I: f64 = 2.0;
const ROLL_CURVATURE_INTEGRATOR_LIMIT: f64 = 0.04;
const ROLL_REF_LIMIT_RAD: f64 = 35.0_f64.to_radians();
const TETHERED_ROLL_REF_RATE_LIMIT_RADPS: f64 = 90.0_f64.to_radians();
const AILERON_ROLL_P: f64 = 1.8;
const AILERON_ROLL_D: f64 = 0.25;
const RABBIT_CURVATURE_LAG_S: f64 = 0.1;
const TETHERED_AILERON_ROLL_P: f64 = 3.5;
const TETHERED_AILERON_ROLL_D: f64 = 0.35;
const TETHERED_RUDDER_BETA_P: f64 = 1.8;
const TETHERED_RUDDER_RATE_D: f64 = 0.35;
const TETHERED_RUDDER_WORLD_Z_P: f64 = 0.0;
const TETHERED_RUDDER_TRIM_OFFSET_RAD: f64 = 0.0;
const TETHERED_CURVATURE_Y_INTEGRATOR_OUTPUT_LIMIT: f64 = 0.1;
const TETHERED_CURVATURE_Z_INTEGRATOR_OUTPUT_LIMIT: f64 = 0.05;
const GUIDANCE_MIN_LOOKAHEAD_FRACTION: f64 = 0.25;
const GUIDANCE_LATERAL_LOOKAHEAD_RATIO_LIMIT: f64 = 0.75;
const GUIDANCE_CURVATURE_LIMIT: f64 = 0.04;
const FREE_FLIGHT_PITCH_REF_LIMIT_RAD: f64 = 14.0_f64.to_radians();
const TETHERED_PITCH_REF_LIMIT_RAD: f64 = 3.8_f64.to_radians();
const ELEVATOR_PITCH_P: f64 = 0.85;
const ELEVATOR_PITCH_D: f64 = 0.25;
const ALTITUDE_PITCH_P: f64 = 0.01;
const ALTITUDE_PITCH_I: f64 = 0.001;
const TECS_ALTITUDE_ERROR_LIMIT_M: f64 = 25.0;
const TECS_THRUST_KINETIC_P: f64 = 0.3;
const TECS_THRUST_KINETIC_I: f64 = 0.06;
const TECS_THRUST_INTEGRATOR_LIMIT_NM: f64 = 8.0;
const TETHERED_THRUST_POSITIVE_POTENTIAL_BLEND: f64 = 0.02;
const TETHERED_TECS_POTENTIAL_ERROR_LIMIT: f64 = 245.0;
const TETHERED_TECS_POTENTIAL_BALANCE_WEIGHT: f64 = 0.9;
const TETHERED_TECS_KINETIC_DEFICIT_BALANCE_WEIGHT: f64 = 1.1;
const TETHERED_TECS_KINETIC_SURPLUS_BALANCE_WEIGHT: f64 = 1.0;
const TECS_PITCH_BALANCE_P: f64 = 0.0012;
const TECS_PITCH_BALANCE_I: f64 = 0.00035;
const TECS_PITCH_INTEGRATOR_LIMIT_RAD: f64 = 7.0_f64.to_radians();
const FREE_FLIGHT_DEMO_STAGE_S: f64 = 10.0;
const FREE_FLIGHT_DEMO_SPEED_STEP_MPS: f64 = 3.0;
const FREE_FLIGHT_DEMO_ALTITUDE_STEP_UP_M: f64 = 12.0;
const FREE_FLIGHT_DEMO_ALTITUDE_STEP_DOWN_M: f64 = -8.0;
const ALPHA_PROTECTION_MIN_RAD: f64 = -8.0_f64.to_radians();
const ALPHA_PROTECTION_MAX_RAD: f64 = 10.0_f64.to_radians();
const ALPHA_TO_ELEVATOR: f64 = 20.0;
const RUDDER_BETA_P: f64 = -0.3;
const RUDDER_OMEGA_WORLD_Z_P: f64 = 0.0;
const SURFACE_LIMIT_AILERON_RAD: f64 = 0.5;
const SURFACE_LIMIT_RUDDER_RAD: f64 = 25.0_f64.to_radians();
const SURFACE_LIMIT_ELEVATOR_RAD: f64 = 0.3;
const MOTOR_TORQUE_MAX_NM: f64 = 45.6;
const ROTOR_SPEED_SOFT_LIMIT_RADPS: f64 = 800.0;
const ROTOR_SPEED_HARD_LIMIT_RADPS: f64 = 900.0;

fn tuning_value(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

#[derive(Clone, Debug)]
struct KiteControllerState {
    thrust_energy_integrator: f64,
    pitch_energy_integrator: f64,
    curvature_to_roll_integrator: f64,
    roll_ref_command: f64,
    rabbit_lag_n: Vector3<f64>,
    rabbit_lag2_n: Vector3<f64>,
    rabbit_lag_initialized: bool,
    curvature_y_integrator: f64,
    curvature_z_integrator: f64,
}

#[derive(Clone, Debug)]
pub struct ControllerState<const NK: usize> {
    kites: [KiteControllerState; NK],
    initial_phase: [f64; NK],
    initial_altitude: [f64; NK],
}

#[derive(Clone, Debug)]
pub struct ControllerTrace<const NK: usize> {
    pub phase_errors: [f64; NK],
    pub speed_targets: [f64; NK],
    pub altitudes: [f64; NK],
    pub altitude_refs: [f64; NK],
    pub kinetic_energy_specific: [f64; NK],
    pub kinetic_energy_ref_specific: [f64; NK],
    pub kinetic_energy_error_specific: [f64; NK],
    pub potential_energy_specific: [f64; NK],
    pub potential_energy_ref_specific: [f64; NK],
    pub potential_energy_error_specific: [f64; NK],
    pub total_energy_error_specific: [f64; NK],
    pub energy_balance_error_specific: [f64; NK],
    pub thrust_energy_integrators: [f64; NK],
    pub pitch_energy_integrators: [f64; NK],
    pub rabbit_phases: [f64; NK],
    pub rabbit_radii: [f64; NK],
    pub rabbit_targets_n: [Vector3<f64>; NK],
    pub curvature_y_refs: [f64; NK],
    pub curvature_y_estimates: [f64; NK],
    pub omega_world_z_refs: [f64; NK],
    pub omega_world_z: [f64; NK],
    pub beta_refs: [f64; NK],
    pub roll_refs: [f64; NK],
    pub roll_feedforwards: [f64; NK],
    pub pitch_refs: [f64; NK],
    pub curvature_z_refs: [f64; NK],
}

impl<const NK: usize> ControllerState<NK> {
    pub fn new(initial_diag: &Diagnostics<f64, NK>) -> Self {
        Self {
            kites: std::array::from_fn(|_| KiteControllerState {
                thrust_energy_integrator: 0.0,
                pitch_energy_integrator: 0.0,
                curvature_to_roll_integrator: 0.0,
                roll_ref_command: 0.0,
                rabbit_lag_n: Vector3::zeros(),
                rabbit_lag2_n: Vector3::zeros(),
                rabbit_lag_initialized: false,
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

fn roll_angle_from_quat_n2b(quat_n2b: &nalgebra::Quaternion<f64>) -> f64 {
    let down_b = rotate_nav_to_body(quat_n2b, &Vector3::new(0.0, 0.0, 1.0));
    down_b[1].atan2(down_b[2])
}

fn pitch_angle_from_quat_n2b(quat_n2b: &nalgebra::Quaternion<f64>) -> f64 {
    let down_b = rotate_nav_to_body(quat_n2b, &Vector3::new(0.0, 0.0, 1.0));
    (-down_b[0]).atan2((down_b[1] * down_b[1] + down_b[2] * down_b[2]).sqrt())
}

fn pairwise_phase_errors<const NK: usize>(diag: &Diagnostics<f64, NK>) -> [f64; NK] {
    let slot_errors: [f64; NK] = std::array::from_fn(|index| {
        let desired_slot = 2.0 * std::f64::consts::PI * index as f64 / NK as f64;
        wrap_angle(diag.kites[index].phase_angle - desired_slot)
    });
    let mean_error = circular_mean(&slot_errors);
    std::array::from_fn(|index| wrap_angle(mean_error - slot_errors[index]))
}

fn open_loop_phase_errors<const NK: usize>(
    state: &ControllerState<NK>,
    diag: &Diagnostics<f64, NK>,
    params: &Params<f64, NK>,
    time: f64,
) -> [f64; NK] {
    let omega_ref = params.controller.speed_ref / params.controller.disk_radius.max(1.0e-6);
    std::array::from_fn(|index| {
        let desired_phase = state.initial_phase[index] + omega_ref * time;
        wrap_angle(desired_phase - diag.kites[index].phase_angle)
    })
}

fn speed_integrator_target(phase_error: f64, speed_ref: f64) -> f64 {
    let phase_gain = tuning_value(
        "MULTIKITE_SPEED_INTEGRATOR_PHASE_GAIN",
        SPEED_INTEGRATOR_PHASE_GAIN,
    );
    let min_speed = tuning_value(
        "MULTIKITE_SPEED_INTEGRATOR_MIN_MPS",
        SPEED_INTEGRATOR_MIN_MPS,
    );
    let max_speed = tuning_value(
        "MULTIKITE_SPEED_INTEGRATOR_MAX_MPS",
        SPEED_INTEGRATOR_MAX_MPS,
    );
    clamp(
        speed_ref + phase_gain * phase_error,
        min_speed.min(max_speed),
        max_speed.max(min_speed),
    )
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

#[derive(Clone, Copy, Debug)]
pub(crate) struct FreeFlightReference {
    pub speed_target: f64,
    pub altitude_ref_raw: f64,
    pub roll_ref: f64,
}

fn default_free_flight_reference(
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

#[derive(Clone, Copy, Debug)]
struct TecsTerms {
    altitude_ref: f64,
    kinetic_energy: f64,
    kinetic_energy_error: f64,
    potential_energy_ref: f64,
    potential_energy_error: f64,
    total_energy_error: f64,
    energy_balance_error: f64,
}

fn tecs_terms(
    altitude: f64,
    altitude_ref_raw: f64,
    airspeed: f64,
    speed_ref: f64,
    gravity: f64,
    tethered: bool,
) -> TecsTerms {
    let altitude_error = clamp(
        altitude_ref_raw - altitude,
        -TECS_ALTITUDE_ERROR_LIMIT_M,
        TECS_ALTITUDE_ERROR_LIMIT_M,
    );
    let kinetic_energy = 0.5 * airspeed * airspeed;
    let kinetic_energy_ref = 0.5 * speed_ref * speed_ref;
    let kinetic_energy_error = kinetic_energy_ref - kinetic_energy;
    let potential_energy = gravity * altitude;
    let potential_energy_error_raw = gravity * altitude_error;
    let potential_energy_error = if tethered {
        clamp(
            potential_energy_error_raw,
            -TETHERED_TECS_POTENTIAL_ERROR_LIMIT,
            TETHERED_TECS_POTENTIAL_ERROR_LIMIT,
        )
    } else {
        potential_energy_error_raw
    };
    let altitude_ref = altitude + potential_energy_error / gravity.max(1.0e-6);
    let potential_energy_ref = potential_energy + potential_energy_error;
    let kinetic_balance_weight = if tethered && kinetic_energy_error > 0.0 {
        TETHERED_TECS_KINETIC_DEFICIT_BALANCE_WEIGHT
    } else if tethered {
        TETHERED_TECS_KINETIC_SURPLUS_BALANCE_WEIGHT
    } else {
        1.0
    };
    let potential_balance_weight = if tethered {
        TETHERED_TECS_POTENTIAL_BALANCE_WEIGHT
    } else {
        1.0
    };
    TecsTerms {
        altitude_ref,
        kinetic_energy,
        kinetic_energy_error,
        potential_energy_ref,
        potential_energy_error,
        total_energy_error: kinetic_energy_error + potential_energy_error,
        energy_balance_error: potential_balance_weight * potential_energy_error
            - kinetic_balance_weight * kinetic_energy_error,
    }
}

#[derive(Clone, Copy, Debug)]
struct SaturatedPiConfig {
    bias: f64,
    kp: f64,
    ki: f64,
    output_min: f64,
    output_max: f64,
    integrator_min: f64,
    integrator_max: f64,
}

fn saturated_pi(integrator: &mut f64, error: f64, dt: f64, config: SaturatedPiConfig) -> f64 {
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
    clamp(
        config.bias + config.kp * error + *integrator,
        config.output_min,
        config.output_max,
    )
}

fn altitude_from_position_n(pos_n: &Vector3<f64>, ground_altitude: f64) -> f64 {
    (-pos_n[2] - ground_altitude).max(0.0)
}

fn tethered_rudder_command(
    trim: f64,
    beta: f64,
    omega_z: f64,
    omega_world_z: f64,
    omega_world_z_ref: f64,
) -> f64 {
    let beta_p = tuning_value("MULTIKITE_TETHERED_RUDDER_BETA_P", TETHERED_RUDDER_BETA_P);
    let rate_d = tuning_value("MULTIKITE_TETHERED_RUDDER_RATE_D", TETHERED_RUDDER_RATE_D);
    let world_z_p = tuning_value(
        "MULTIKITE_TETHERED_RUDDER_WORLD_Z_P",
        TETHERED_RUDDER_WORLD_Z_P,
    );
    let limit = tethered_rudder_limit_rad();
    clamp(
        trim - beta_p * beta
            + TETHERED_RUDDER_TRIM_OFFSET_RAD
            + rate_d * omega_z
            + world_z_p * (omega_world_z - omega_world_z_ref),
        -limit,
        limit,
    )
}

fn tethered_aileron_command(trim: f64, roll_ref: f64, roll_angle: f64, omega_x: f64) -> f64 {
    let roll_p = tuning_value("MULTIKITE_TETHERED_AILERON_ROLL_P", TETHERED_AILERON_ROLL_P);
    let roll_d = tuning_value("MULTIKITE_TETHERED_AILERON_ROLL_D", TETHERED_AILERON_ROLL_D);
    clamp(
        trim - roll_p * wrap_angle(roll_ref - roll_angle) + roll_d * omega_x,
        -SURFACE_LIMIT_AILERON_RAD,
        SURFACE_LIMIT_AILERON_RAD,
    )
}

fn orbit_curvature_y_reference(radius: f64) -> f64 {
    1.0 / radius.max(1.0)
}

fn orbit_roll_feedforward(inertial_speed: f64, orbit_radius: f64, g: f64) -> f64 {
    tuning_value("MULTIKITE_ROLL_FEEDFORWARD_GAIN", ROLL_FEEDFORWARD_GAIN)
        * (inertial_speed * inertial_speed * orbit_curvature_y_reference(orbit_radius) / g).atan()
}

fn roll_curvature_gain_p() -> f64 {
    tuning_value("MULTIKITE_ROLL_FROM_CURVATURE_P", ROLL_FROM_CURVATURE_P)
}

fn roll_curvature_gain_i() -> f64 {
    tuning_value("MULTIKITE_ROLL_FROM_CURVATURE_I", ROLL_FROM_CURVATURE_I)
}

fn tethered_rudder_limit_rad() -> f64 {
    tuning_value(
        "MULTIKITE_TETHERED_RUDDER_LIMIT_DEG",
        SURFACE_LIMIT_RUDDER_RAD.to_degrees(),
    )
    .to_radians()
    .abs()
}

fn rate_limit(current: f64, target: f64, max_rate: f64, dt: f64) -> f64 {
    current + clamp(target - current, -max_rate * dt, max_rate * dt)
}

fn guidance_minimum_lookahead(rabbit_distance: f64) -> f64 {
    GUIDANCE_MIN_LOOKAHEAD_FRACTION * rabbit_distance.max(1.0)
}

fn pursuit_curvatures_from_body_vector(
    rabbit_vector_b: &Vector3<f64>,
    minimum_lookahead: f64,
) -> (f64, f64) {
    let lookahead = rabbit_vector_b[0].max(minimum_lookahead.max(1.0));
    let lateral_limit = GUIDANCE_LATERAL_LOOKAHEAD_RATIO_LIMIT * lookahead;
    let lateral_y = clamp(rabbit_vector_b[1], -lateral_limit, lateral_limit);
    let lateral_z = clamp(rabbit_vector_b[2], -lateral_limit, lateral_limit);
    let denominator = lookahead * lookahead;
    (
        clamp(
            2.0 * lateral_y / denominator,
            -GUIDANCE_CURVATURE_LIMIT,
            GUIDANCE_CURVATURE_LIMIT,
        ),
        clamp(
            2.0 * lateral_z / denominator,
            -GUIDANCE_CURVATURE_LIMIT,
            GUIDANCE_CURVATURE_LIMIT,
        ),
    )
}

fn limit_motor_torque_for_rotor_speed(commanded_torque: f64, rotor_speed: f64) -> f64 {
    if rotor_speed <= ROTOR_SPEED_SOFT_LIMIT_RADPS {
        return commanded_torque;
    }

    let fade = ((ROTOR_SPEED_HARD_LIMIT_RADPS - rotor_speed)
        / (ROTOR_SPEED_HARD_LIMIT_RADPS - ROTOR_SPEED_SOFT_LIMIT_RADPS))
        .clamp(0.0, 1.0);
    commanded_torque.min(MOTOR_TORQUE_MAX_NM * fade).max(0.0)
}

pub fn controller_step<const NK: usize, const N_COMMON: usize, const N_UPPER: usize>(
    state: &mut ControllerState<NK>,
    plant_state: &State<f64, NK, N_COMMON, N_UPPER>,
    diag: &Diagnostics<f64, NK>,
    params: &Params<f64, NK>,
    dt_control: f64,
    phase_mode: PhaseMode,
    longitudinal_mode: LongitudinalMode,
    time: f64,
) -> (Controls<f64, NK>, ControllerTrace<NK>) {
    controller_step_impl(
        state,
        plant_state,
        diag,
        params,
        dt_control,
        phase_mode,
        longitudinal_mode,
        time,
        default_free_flight_reference,
    )
}

#[cfg(test)]
pub(crate) fn controller_step_with_free_flight_reference<
    const NK: usize,
    const N_COMMON: usize,
    const N_UPPER: usize,
    F,
>(
    state: &mut ControllerState<NK>,
    plant_state: &State<f64, NK, N_COMMON, N_UPPER>,
    diag: &Diagnostics<f64, NK>,
    params: &Params<f64, NK>,
    dt_control: f64,
    phase_mode: PhaseMode,
    longitudinal_mode: LongitudinalMode,
    time: f64,
    free_flight_reference: F,
) -> (Controls<f64, NK>, ControllerTrace<NK>)
where
    F: Fn(usize, f64, f64, f64) -> FreeFlightReference + Copy,
{
    controller_step_impl(
        state,
        plant_state,
        diag,
        params,
        dt_control,
        phase_mode,
        longitudinal_mode,
        time,
        free_flight_reference,
    )
}

fn controller_step_impl<const NK: usize, const N_COMMON: usize, const N_UPPER: usize, F>(
    state: &mut ControllerState<NK>,
    plant_state: &State<f64, NK, N_COMMON, N_UPPER>,
    diag: &Diagnostics<f64, NK>,
    params: &Params<f64, NK>,
    dt_control: f64,
    phase_mode: PhaseMode,
    longitudinal_mode: LongitudinalMode,
    time: f64,
    free_flight_reference: F,
) -> (Controls<f64, NK>, ControllerTrace<NK>)
where
    F: Fn(usize, f64, f64, f64) -> FreeFlightReference + Copy,
{
    let adaptive_phase_errors = pairwise_phase_errors(diag);
    let open_loop_phase_errors = open_loop_phase_errors(state, diag, params, time);
    let phase_errors = match phase_mode {
        PhaseMode::Adaptive => adaptive_phase_errors,
        PhaseMode::OpenLoop => open_loop_phase_errors,
    };
    let max_throttle_altitude_pitch =
        longitudinal_mode == LongitudinalMode::MaxThrottleAltitudePitch;

    let rabbit_phases = std::array::from_fn(|index| {
        diag.kites[index].phase_angle
            + params.controller.rabbit_distance / params.controller.disk_radius.max(1.0e-6)
    });
    let rabbit_radii = std::array::from_fn(|index| {
        params.controller.disk_radius
            * (1.0
                + phase_errors[index] / std::f64::consts::PI
                    * params.controller.phase_lag_to_radius)
    });
    let rabbit_targets_n = std::array::from_fn(|index| {
        let kite_diag = &diag.kites[index];
        Vector3::new(
            params.controller.disk_center_n[0] + rabbit_radii[index] * rabbit_phases[index].cos(),
            params.controller.disk_center_n[1] + rabbit_radii[index] * rabbit_phases[index].sin(),
            params.controller.disk_center_n[2]
                - kite_diag.cad_velocity_n[2] * params.controller.vert_vel_to_rabbit_height,
        )
    });

    let controls = Controls {
        kites: std::array::from_fn(|index| {
            let kite_diag = &diag.kites[index];
            let free_reference = if N_COMMON == 0 && N_UPPER == 0 {
                Some(free_flight_reference(
                    index,
                    time,
                    state.initial_altitude[index],
                    params.controller.speed_ref,
                ))
            } else {
                None
            };
            let control_state = &mut state.kites[index];
            let inertial_speed = kite_diag.cad_velocity_n.norm();
            let airspeed = kite_diag.airspeed;
            let roll_angle = roll_angle_from_quat_n2b(&plant_state.kites[index].body.quat_n2b);
            let pitch_angle = pitch_angle_from_quat_n2b(&plant_state.kites[index].body.quat_n2b);
            let omega_n =
                rotate_body_to_nav(&plant_state.kites[index].body.quat_n2b, &kite_diag.omega_b);
            let tethered_lateral = N_COMMON != 0 || N_UPPER != 0;
            let rudder_beta_p = if N_COMMON == 0 && N_UPPER == 0 {
                0.0
            } else {
                RUDDER_BETA_P
            };
            let alpha_protection = if kite_diag.alpha > ALPHA_PROTECTION_MAX_RAD {
                kite_diag.alpha - ALPHA_PROTECTION_MAX_RAD
            } else if kite_diag.alpha < ALPHA_PROTECTION_MIN_RAD {
                kite_diag.alpha - ALPHA_PROTECTION_MIN_RAD
            } else {
                0.0
            };
            let alpha_exceeded = clamp(kite_diag.alpha, 0.0, 0.15);
            let (
                speed_target,
                altitude_ref_raw,
                raw_roll_ref,
                _roll_feedforward,
                omega_world_z_ref,
                k_tg_y,
                k_tg_z,
                _rabbit_k_b_y,
            ) = if N_COMMON == 0 && N_UPPER == 0 {
                let reference = free_reference.expect("free-flight reference");
                (
                    reference.speed_target,
                    reference.altitude_ref_raw,
                    reference.roll_ref,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            } else {
                let speed_target =
                    speed_integrator_target(phase_errors[index], params.controller.speed_ref);
                let altitude_ref_raw = -rabbit_targets_n[index][2]
                    - params.kites[index].tether.contact.ground_altitude;
                let rabbit_from_center_n =
                    sub(&rabbit_targets_n[index], &params.controller.disk_center_n);
                if !control_state.rabbit_lag_initialized {
                    control_state.rabbit_lag_n = rabbit_from_center_n;
                    control_state.rabbit_lag2_n = rabbit_from_center_n;
                    control_state.roll_ref_command = roll_angle;
                    control_state.rabbit_lag_initialized = true;
                }
                let rabbit_lag = clamp(dt_control / RABBIT_CURVATURE_LAG_S, 0.0, 1.0);
                control_state.rabbit_lag_n = rabbit_from_center_n * rabbit_lag
                    + control_state.rabbit_lag_n * (1.0 - rabbit_lag);
                control_state.rabbit_lag2_n = rabbit_from_center_n * (0.5 * rabbit_lag)
                    + control_state.rabbit_lag2_n * (1.0 - 0.5 * rabbit_lag);
                let rc_y = (rabbit_from_center_n - control_state.rabbit_lag2_n)
                    .norm()
                    .max(1.0e-9);
                let rc_vect = (rabbit_from_center_n + control_state.rabbit_lag2_n) * 0.5
                    - control_state.rabbit_lag_n;
                let rabbit_k_n = rc_vect * (rc_vect.norm() / rc_y);
                let rabbit_k_b =
                    rotate_nav_to_body(&plant_state.kites[index].body.quat_n2b, &rabbit_k_n);
                let rabbit_vector_n = sub(&rabbit_targets_n[index], &kite_diag.cad_position_n);
                let rabbit_vector_b =
                    rotate_nav_to_body(&plant_state.kites[index].body.quat_n2b, &rabbit_vector_n);
                let (_pursuit_k_tg_y, k_tg_z) = pursuit_curvatures_from_body_vector(
                    &rabbit_vector_b,
                    guidance_minimum_lookahead(params.controller.rabbit_distance),
                );
                let k_tg_y = orbit_curvature_y_reference(rabbit_radii[index]);
                let curvature_y_est = omega_n[2] / inertial_speed.max(1.0);
                let gain_int_y = params.controller.gain_int_y.abs().max(1.0e-9);
                let gain_int_z = params.controller.gain_int_z.abs().max(1.0e-9);
                control_state.curvature_y_integrator = clamp(
                    control_state.curvature_y_integrator
                        + (kite_diag.curvature_y_b - k_tg_y) * dt_control
                        - alpha_exceeded * 0.5,
                    -TETHERED_CURVATURE_Y_INTEGRATOR_OUTPUT_LIMIT / gain_int_y,
                    TETHERED_CURVATURE_Y_INTEGRATOR_OUTPUT_LIMIT / gain_int_y,
                );
                control_state.curvature_z_integrator = clamp(
                    control_state.curvature_z_integrator
                        + (kite_diag.curvature_z_b - k_tg_z) * dt_control
                        - alpha_exceeded * 0.5,
                    -TETHERED_CURVATURE_Z_INTEGRATOR_OUTPUT_LIMIT / gain_int_z,
                    TETHERED_CURVATURE_Z_INTEGRATOR_OUTPUT_LIMIT / gain_int_z,
                );
                let roll_feedforward = orbit_roll_feedforward(
                    inertial_speed,
                    rabbit_radii[index],
                    params.environment.g,
                );
                control_state.curvature_to_roll_integrator = clamp(
                    control_state.curvature_to_roll_integrator
                        + (k_tg_y - curvature_y_est) * dt_control,
                    -ROLL_CURVATURE_INTEGRATOR_LIMIT,
                    ROLL_CURVATURE_INTEGRATOR_LIMIT,
                );
                let roll_ref = clamp(
                    roll_feedforward
                        + roll_curvature_gain_p() * (k_tg_y - curvature_y_est)
                        + roll_curvature_gain_i() * control_state.curvature_to_roll_integrator,
                    -ROLL_REF_LIMIT_RAD,
                    ROLL_REF_LIMIT_RAD,
                );
                (
                    speed_target,
                    altitude_ref_raw,
                    roll_ref,
                    roll_feedforward,
                    inertial_speed * k_tg_y,
                    k_tg_y,
                    k_tg_z,
                    rabbit_k_b[1],
                )
            };
            let roll_ref = if tethered_lateral {
                control_state.roll_ref_command = rate_limit(
                    control_state.roll_ref_command,
                    raw_roll_ref,
                    tuning_value(
                        "MULTIKITE_TETHERED_ROLL_REF_RATE_LIMIT_DEGPS",
                        TETHERED_ROLL_REF_RATE_LIMIT_RADPS.to_degrees(),
                    )
                    .to_radians(),
                    dt_control,
                );
                control_state.roll_ref_command
            } else {
                control_state.roll_ref_command = raw_roll_ref;
                raw_roll_ref
            };

            let altitude = altitude_from_position_n(
                &kite_diag.cad_position_n,
                params.kites[index].tether.contact.ground_altitude,
            );
            let pitch_ref_limit = if tethered_lateral {
                TETHERED_PITCH_REF_LIMIT_RAD
            } else {
                FREE_FLIGHT_PITCH_REF_LIMIT_RAD
            };
            let motor_torque_max = MOTOR_TORQUE_MAX_NM;
            let tecs = tecs_terms(
                altitude,
                altitude_ref_raw,
                airspeed,
                speed_target,
                params.environment.g,
                tethered_lateral,
            );
            let pitch_ref = if max_throttle_altitude_pitch {
                let altitude_error = clamp(
                    altitude_ref_raw - altitude,
                    -TECS_ALTITUDE_ERROR_LIMIT_M,
                    TECS_ALTITUDE_ERROR_LIMIT_M,
                );
                saturated_pi(
                    &mut control_state.pitch_energy_integrator,
                    altitude_error,
                    dt_control,
                    SaturatedPiConfig {
                        bias: 0.0,
                        kp: ALTITUDE_PITCH_P,
                        ki: ALTITUDE_PITCH_I,
                        output_min: -pitch_ref_limit,
                        output_max: pitch_ref_limit,
                        integrator_min: -TECS_PITCH_INTEGRATOR_LIMIT_RAD,
                        integrator_max: TECS_PITCH_INTEGRATOR_LIMIT_RAD,
                    },
                )
            } else {
                saturated_pi(
                    &mut control_state.pitch_energy_integrator,
                    tecs.energy_balance_error,
                    dt_control,
                    SaturatedPiConfig {
                        bias: 0.0,
                        kp: TECS_PITCH_BALANCE_P,
                        ki: TECS_PITCH_BALANCE_I,
                        output_min: -pitch_ref_limit,
                        output_max: pitch_ref_limit,
                        integrator_min: -TECS_PITCH_INTEGRATOR_LIMIT_RAD,
                        integrator_max: TECS_PITCH_INTEGRATOR_LIMIT_RAD,
                    },
                )
            };
            let thrust_energy_error = if tethered_lateral {
                tecs.kinetic_energy_error
                    + TETHERED_THRUST_POSITIVE_POTENTIAL_BLEND
                        * tecs.potential_energy_error.max(0.0)
            } else {
                tecs.kinetic_energy_error
            };
            let commanded_motor_torque = if max_throttle_altitude_pitch {
                control_state.thrust_energy_integrator = motor_torque_max;
                motor_torque_max
            } else {
                saturated_pi(
                    &mut control_state.thrust_energy_integrator,
                    thrust_energy_error,
                    dt_control,
                    SaturatedPiConfig {
                        bias: params.controller.trim.motor_torque,
                        kp: TECS_THRUST_KINETIC_P,
                        ki: TECS_THRUST_KINETIC_I,
                        output_min: 0.0,
                        output_max: motor_torque_max,
                        integrator_min: -TECS_THRUST_INTEGRATOR_LIMIT_NM,
                        integrator_max: TECS_THRUST_INTEGRATOR_LIMIT_NM,
                    },
                )
            };
            let motor_torque = limit_motor_torque_for_rotor_speed(
                commanded_motor_torque,
                plant_state.kites[index].rotor_speed,
            );
            if max_throttle_altitude_pitch {
                control_state.thrust_energy_integrator = motor_torque;
            }
            let (aileron, rudder) = if tethered_lateral {
                (
                    tethered_aileron_command(
                        params.controller.trim.surfaces.aileron,
                        roll_ref,
                        roll_angle,
                        kite_diag.omega_b[0],
                    ),
                    tethered_rudder_command(
                        params.controller.trim.surfaces.rudder,
                        kite_diag.beta,
                        kite_diag.omega_b[2],
                        omega_n[2],
                        omega_world_z_ref,
                    ),
                )
            } else {
                (
                    clamp(
                        params.controller.trim.surfaces.aileron
                            - AILERON_ROLL_P * wrap_angle(roll_ref - roll_angle)
                            + AILERON_ROLL_D * kite_diag.omega_b[0],
                        -SURFACE_LIMIT_AILERON_RAD,
                        SURFACE_LIMIT_AILERON_RAD,
                    ),
                    clamp(
                        params.controller.trim.surfaces.rudder - rudder_beta_p * kite_diag.beta
                            + RUDDER_OMEGA_WORLD_Z_P * (omega_n[2] - omega_world_z_ref),
                        -SURFACE_LIMIT_RUDDER_RAD,
                        SURFACE_LIMIT_RUDDER_RAD,
                    ),
                )
            };

            let surfaces = ControlSurfaces {
                aileron,
                flap: params.controller.trim.surfaces.flap,
                winglet: params.controller.trim.surfaces.winglet,
                elevator: clamp(
                    params.controller.trim.surfaces.elevator
                        - ELEVATOR_PITCH_P * (pitch_ref - pitch_angle)
                        + ELEVATOR_PITCH_D * kite_diag.omega_b[1]
                        + ALPHA_TO_ELEVATOR * alpha_protection,
                    -SURFACE_LIMIT_ELEVATOR_RAD,
                    SURFACE_LIMIT_ELEVATOR_RAD,
                ),
                rudder,
            };

            let _ = (k_tg_y, k_tg_z);

            KiteControls {
                surfaces,
                motor_torque,
            }
        }),
    };

    let is_free_flight = N_COMMON == 0 && N_UPPER == 0;
    let free_reference_for = |index: usize| {
        free_flight_reference(
            index,
            time,
            state.initial_altitude[index],
            params.controller.speed_ref,
        )
    };
    let speed_target_for = |index: usize| {
        if is_free_flight {
            free_reference_for(index).speed_target
        } else {
            speed_integrator_target(phase_errors[index], params.controller.speed_ref)
        }
    };
    let altitude_ref_raw_for = |index: usize| {
        if is_free_flight {
            free_reference_for(index).altitude_ref_raw
        } else {
            -rabbit_targets_n[index][2] - params.kites[index].tether.contact.ground_altitude
        }
    };

    (
        controls,
        ControllerTrace {
            phase_errors,
            speed_targets: std::array::from_fn(speed_target_for),
            altitudes: std::array::from_fn(|index| {
                altitude_from_position_n(
                    &diag.kites[index].cad_position_n,
                    params.kites[index].tether.contact.ground_altitude,
                )
            }),
            altitude_refs: std::array::from_fn(|index| {
                let speed_target = speed_target_for(index);
                let altitude = altitude_from_position_n(
                    &diag.kites[index].cad_position_n,
                    params.kites[index].tether.contact.ground_altitude,
                );
                let altitude_ref_raw = altitude_ref_raw_for(index);
                tecs_terms(
                    altitude,
                    altitude_ref_raw,
                    diag.kites[index].airspeed,
                    speed_target,
                    params.environment.g,
                    !is_free_flight,
                )
                .altitude_ref
            }),
            kinetic_energy_specific: std::array::from_fn(|index| {
                let speed_target = speed_target_for(index);
                let altitude = altitude_from_position_n(
                    &diag.kites[index].cad_position_n,
                    params.kites[index].tether.contact.ground_altitude,
                );
                let altitude_ref_raw = altitude_ref_raw_for(index);
                tecs_terms(
                    altitude,
                    altitude_ref_raw,
                    diag.kites[index].airspeed,
                    speed_target,
                    params.environment.g,
                    !is_free_flight,
                )
                .kinetic_energy
            }),
            kinetic_energy_ref_specific: std::array::from_fn(|index| {
                let speed_target = speed_target_for(index);
                0.5 * speed_target * speed_target
            }),
            kinetic_energy_error_specific: std::array::from_fn(|index| {
                let speed_target = speed_target_for(index);
                let altitude = altitude_from_position_n(
                    &diag.kites[index].cad_position_n,
                    params.kites[index].tether.contact.ground_altitude,
                );
                let altitude_ref_raw = altitude_ref_raw_for(index);
                tecs_terms(
                    altitude,
                    altitude_ref_raw,
                    diag.kites[index].airspeed,
                    speed_target,
                    params.environment.g,
                    !is_free_flight,
                )
                .kinetic_energy_error
            }),
            potential_energy_specific: std::array::from_fn(|index| {
                let altitude = altitude_from_position_n(
                    &diag.kites[index].cad_position_n,
                    params.kites[index].tether.contact.ground_altitude,
                );
                params.environment.g * altitude
            }),
            potential_energy_ref_specific: std::array::from_fn(|index| {
                let speed_target = speed_target_for(index);
                let altitude = altitude_from_position_n(
                    &diag.kites[index].cad_position_n,
                    params.kites[index].tether.contact.ground_altitude,
                );
                let altitude_ref_raw = altitude_ref_raw_for(index);
                tecs_terms(
                    altitude,
                    altitude_ref_raw,
                    diag.kites[index].airspeed,
                    speed_target,
                    params.environment.g,
                    !is_free_flight,
                )
                .potential_energy_ref
            }),
            potential_energy_error_specific: std::array::from_fn(|index| {
                let speed_target = speed_target_for(index);
                let altitude = altitude_from_position_n(
                    &diag.kites[index].cad_position_n,
                    params.kites[index].tether.contact.ground_altitude,
                );
                let altitude_ref_raw = altitude_ref_raw_for(index);
                tecs_terms(
                    altitude,
                    altitude_ref_raw,
                    diag.kites[index].airspeed,
                    speed_target,
                    params.environment.g,
                    !is_free_flight,
                )
                .potential_energy_error
            }),
            total_energy_error_specific: std::array::from_fn(|index| {
                let speed_target = speed_target_for(index);
                let altitude = altitude_from_position_n(
                    &diag.kites[index].cad_position_n,
                    params.kites[index].tether.contact.ground_altitude,
                );
                let altitude_ref_raw = altitude_ref_raw_for(index);
                tecs_terms(
                    altitude,
                    altitude_ref_raw,
                    diag.kites[index].airspeed,
                    speed_target,
                    params.environment.g,
                    !is_free_flight,
                )
                .total_energy_error
            }),
            energy_balance_error_specific: std::array::from_fn(|index| {
                let speed_target = speed_target_for(index);
                let altitude = altitude_from_position_n(
                    &diag.kites[index].cad_position_n,
                    params.kites[index].tether.contact.ground_altitude,
                );
                let altitude_ref_raw = altitude_ref_raw_for(index);
                tecs_terms(
                    altitude,
                    altitude_ref_raw,
                    diag.kites[index].airspeed,
                    speed_target,
                    params.environment.g,
                    !is_free_flight,
                )
                .energy_balance_error
            }),
            thrust_energy_integrators: std::array::from_fn(|index| {
                state.kites[index].thrust_energy_integrator
            }),
            pitch_energy_integrators: std::array::from_fn(|index| {
                state.kites[index].pitch_energy_integrator
            }),
            rabbit_phases,
            rabbit_radii,
            rabbit_targets_n,
            curvature_y_refs: std::array::from_fn(|index| {
                if N_COMMON == 0 && N_UPPER == 0 {
                    0.0
                } else {
                    orbit_curvature_y_reference(rabbit_radii[index])
                }
            }),
            curvature_y_estimates: std::array::from_fn(|index| {
                let kite_diag = &diag.kites[index];
                let omega_n =
                    rotate_body_to_nav(&plant_state.kites[index].body.quat_n2b, &kite_diag.omega_b);
                omega_n[2] / kite_diag.cad_velocity_n.norm().max(1.0)
            }),
            omega_world_z_refs: std::array::from_fn(|index| {
                if N_COMMON == 0 && N_UPPER == 0 {
                    0.0
                } else {
                    let kite_diag = &diag.kites[index];
                    kite_diag.cad_velocity_n.norm()
                        * orbit_curvature_y_reference(rabbit_radii[index])
                }
            }),
            omega_world_z: std::array::from_fn(|index| {
                let kite_diag = &diag.kites[index];
                let omega_n =
                    rotate_body_to_nav(&plant_state.kites[index].body.quat_n2b, &kite_diag.omega_b);
                omega_n[2]
            }),
            beta_refs: std::array::from_fn(|_| 0.0),
            roll_refs: std::array::from_fn(|index| {
                if is_free_flight {
                    free_reference_for(index).roll_ref
                } else {
                    state.kites[index].roll_ref_command
                }
            }),
            roll_feedforwards: std::array::from_fn(|index| {
                if N_COMMON == 0 && N_UPPER == 0 {
                    0.0
                } else {
                    let kite_diag = &diag.kites[index];
                    let inertial_speed = kite_diag.cad_velocity_n.norm().max(1.0);
                    orbit_roll_feedforward(
                        inertial_speed,
                        rabbit_radii[index],
                        params.environment.g,
                    )
                }
            }),
            pitch_refs: std::array::from_fn(|index| {
                let speed_target = speed_target_for(index);
                let altitude = altitude_from_position_n(
                    &diag.kites[index].cad_position_n,
                    params.kites[index].tether.contact.ground_altitude,
                );
                let altitude_ref_raw = altitude_ref_raw_for(index);
                let tecs = tecs_terms(
                    altitude,
                    altitude_ref_raw,
                    diag.kites[index].airspeed,
                    speed_target,
                    params.environment.g,
                    !is_free_flight,
                );
                let pitch_ref_limit = if is_free_flight {
                    FREE_FLIGHT_PITCH_REF_LIMIT_RAD
                } else {
                    TETHERED_PITCH_REF_LIMIT_RAD
                };
                if max_throttle_altitude_pitch {
                    let altitude_error = clamp(
                        altitude_ref_raw - altitude,
                        -TECS_ALTITUDE_ERROR_LIMIT_M,
                        TECS_ALTITUDE_ERROR_LIMIT_M,
                    );
                    clamp(
                        ALTITUDE_PITCH_P * altitude_error
                            + state.kites[index].pitch_energy_integrator,
                        -pitch_ref_limit,
                        pitch_ref_limit,
                    )
                } else {
                    clamp(
                        TECS_PITCH_BALANCE_P * tecs.energy_balance_error
                            + state.kites[index].pitch_energy_integrator,
                        -pitch_ref_limit,
                        pitch_ref_limit,
                    )
                }
            }),
            curvature_z_refs: std::array::from_fn(|index| {
                if N_COMMON == 0 && N_UPPER == 0 {
                    0.0
                } else {
                    let kite_diag = &diag.kites[index];
                    let rabbit_vector_n = sub(&rabbit_targets_n[index], &kite_diag.cad_position_n);
                    let rabbit_vector_b = rotate_nav_to_body(
                        &plant_state.kites[index].body.quat_n2b,
                        &rabbit_vector_n,
                    );
                    pursuit_curvatures_from_body_vector(
                        &rabbit_vector_b,
                        guidance_minimum_lookahead(params.controller.rabbit_distance),
                    )
                    .1
                }
            }),
        },
    )
}

pub fn apply_trace<const NK: usize>(
    diagnostics: &mut Diagnostics<f64, NK>,
    trace: &ControllerTrace<NK>,
) {
    for index in 0..NK {
        diagnostics.kites[index].phase_error = trace.phase_errors[index];
        diagnostics.kites[index].speed_target = trace.speed_targets[index];
        diagnostics.kites[index].altitude = trace.altitudes[index];
        diagnostics.kites[index].altitude_ref = trace.altitude_refs[index];
        diagnostics.kites[index].kinetic_energy_specific = trace.kinetic_energy_specific[index];
        diagnostics.kites[index].kinetic_energy_ref_specific =
            trace.kinetic_energy_ref_specific[index];
        diagnostics.kites[index].kinetic_energy_error_specific =
            trace.kinetic_energy_error_specific[index];
        diagnostics.kites[index].potential_energy_specific = trace.potential_energy_specific[index];
        diagnostics.kites[index].potential_energy_ref_specific =
            trace.potential_energy_ref_specific[index];
        diagnostics.kites[index].potential_energy_error_specific =
            trace.potential_energy_error_specific[index];
        diagnostics.kites[index].total_energy_error_specific =
            trace.total_energy_error_specific[index];
        diagnostics.kites[index].energy_balance_error_specific =
            trace.energy_balance_error_specific[index];
        diagnostics.kites[index].thrust_energy_integrator = trace.thrust_energy_integrators[index];
        diagnostics.kites[index].pitch_energy_integrator = trace.pitch_energy_integrators[index];
        diagnostics.kites[index].rabbit_phase = trace.rabbit_phases[index];
        diagnostics.kites[index].rabbit_radius = trace.rabbit_radii[index];
        diagnostics.kites[index].rabbit_target_n = trace.rabbit_targets_n[index];
        diagnostics.kites[index].curvature_y_ref = trace.curvature_y_refs[index];
        diagnostics.kites[index].curvature_y_est = trace.curvature_y_estimates[index];
        diagnostics.kites[index].omega_world_z_ref = trace.omega_world_z_refs[index];
        diagnostics.kites[index].omega_world_z = trace.omega_world_z[index];
        diagnostics.kites[index].beta_ref = trace.beta_refs[index];
        diagnostics.kites[index].roll_ref = trace.roll_refs[index];
        diagnostics.kites[index].roll_ff = trace.roll_feedforwards[index];
        diagnostics.kites[index].pitch_ref = trace.pitch_refs[index];
        diagnostics.kites[index].curvature_z_ref = trace.curvature_z_refs[index];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tethered_rudder_feedback_damps_beta_and_body_z_rate() {
        let beta_error = 5.0_f64.to_radians();
        assert!(tethered_rudder_command(0.0, beta_error, 0.0, 0.0, 0.0) < 0.0);
        assert!(tethered_rudder_command(0.0, -beta_error, 0.0, 0.0, 0.0) > 0.0);
        assert!(tethered_rudder_command(0.0, 0.0, 0.5, 0.0, 0.0) > 0.0);
        assert!(tethered_rudder_command(0.0, 0.0, -0.5, 0.0, 0.0) < 0.0);
    }

    #[test]
    fn tethered_aileron_feedback_closes_roll_error_and_damps_roll_rate() {
        let roll_error = 5.0_f64.to_radians();
        assert!(tethered_aileron_command(0.0, roll_error, 0.0, 0.0) < 0.0);
        assert!(tethered_aileron_command(0.0, -roll_error, 0.0, 0.0) > 0.0);
        assert!(tethered_aileron_command(0.0, 0.0, 0.0, 0.5) > 0.0);
        assert!(tethered_aileron_command(0.0, 0.0, 0.0, -0.5) < 0.0);
    }

    #[test]
    fn positive_phase_error_schedules_higher_speed() {
        let speed_ref = 25.0;
        let phase_error = 0.1;
        assert!(speed_integrator_target(phase_error, speed_ref) > speed_ref);
        assert!(speed_integrator_target(-phase_error, speed_ref) < speed_ref);
    }

    #[test]
    fn rotor_speed_limiter_fades_torque_above_fit_range() {
        let commanded = MOTOR_TORQUE_MAX_NM;
        assert_eq!(
            limit_motor_torque_for_rotor_speed(commanded, ROTOR_SPEED_SOFT_LIMIT_RADPS - 1.0),
            commanded
        );
        assert!(
            limit_motor_torque_for_rotor_speed(commanded, 850.0) < commanded,
            "torque should fade inside the overspeed band"
        );
        assert_eq!(
            limit_motor_torque_for_rotor_speed(commanded, ROTOR_SPEED_HARD_LIMIT_RADPS + 1.0),
            0.0
        );
    }

    #[test]
    fn tethered_tecs_clamps_potential_and_weights_kinetic_deficit() {
        let tethered = tecs_terms(100.0, 200.0, 10.0, 20.0, 9.81, true);
        assert!(
            (tethered.potential_energy_error - TETHERED_TECS_POTENTIAL_ERROR_LIMIT).abs() < 1.0e-9
        );

        let free = tecs_terms(100.0, 200.0, 10.0, 20.0, 9.81, false);
        assert!(free.potential_energy_error > tethered.potential_energy_error);
        assert!(tethered.energy_balance_error < free.energy_balance_error);
    }
}
