mod actuators;
mod lateral;
mod longitudinal;
mod state;

pub use state::{ControllerState, ControllerTrace, KiteControllerTrace};

use actuators::{elevator_breakdown, tethered_aileron_breakdown, tethered_rudder_breakdown};
use lateral::{
    RollReferenceBreakdown, TETHERED_CURVATURE_Y_INTEGRATOR_OUTPUT_LIMIT,
    TETHERED_CURVATURE_Z_INTEGRATOR_OUTPUT_LIMIT, direct_rabbit_roll_reference_breakdown,
    guidance_uses_direct_rabbit, lateral_guidance_curvatures, orbit_roll_feedforward, rate_limit,
    scheduled_rabbit_distance, speed_integrator_target,
};
pub(crate) use longitudinal::FreeFlightReference;
use longitudinal::{
    SaturatedPiBreakdown, SaturatedPiConfig, default_free_flight_reference,
    limit_motor_torque_for_rotor_speed, saturated_pi_breakdown, tecs_terms,
};
use state::KiteControllerState;

use crate::math::{circular_mean, clamp, rotate_body_to_nav, rotate_nav_to_body, sub, wrap_angle};
use crate::types::{
    ControlSurfaces, Controls, Diagnostics, KiteControls, LongitudinalMode, Params, PhaseMode,
    State,
};
use nalgebra::Vector3;

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

fn altitude_from_position_n(pos_n: &Vector3<f64>, ground_altitude: f64) -> f64 {
    (-pos_n[2] - ground_altitude).max(0.0)
}

#[derive(Clone, Debug)]
struct KiteSchedule {
    phase_error: f64,
    speed_target: f64,
    rabbit_distance: f64,
    rabbit_phase: f64,
    rabbit_radius: f64,
    rabbit_target_n: Vector3<f64>,
}

struct KiteControlRequest<'a, const NK: usize, const N_COMMON: usize, const N_UPPER: usize> {
    index: usize,
    schedule: KiteSchedule,
    control_state: &'a mut KiteControllerState,
    plant_state: &'a State<f64, NK, N_COMMON, N_UPPER>,
    diag: &'a Diagnostics<f64, NK>,
    params: &'a Params<f64, NK>,
    initial_altitude: f64,
    dt_control: f64,
    longitudinal_mode: LongitudinalMode,
    time: f64,
}

#[derive(Clone, Debug)]
struct KiteControlOutput {
    controls: KiteControls<f64>,
    trace: KiteControllerTrace,
}

fn compute_kite_control<const NK: usize, const N_COMMON: usize, const N_UPPER: usize, F>(
    request: KiteControlRequest<'_, NK, N_COMMON, N_UPPER>,
    free_flight_reference: F,
) -> KiteControlOutput
where
    F: Fn(usize, f64, f64, f64) -> FreeFlightReference + Copy,
{
    let index = request.index;
    let kite_diag = &request.diag.kites[index];
    let tuning = &request.params.controller.tuning;
    let control_state = request.control_state;
    let is_free_flight = N_COMMON == 0 && N_UPPER == 0;
    let tethered_lateral = !is_free_flight;
    let max_throttle_altitude_pitch =
        request.longitudinal_mode == LongitudinalMode::MaxThrottleAltitudePitch;

    let inertial_speed = kite_diag.cad_velocity_n.norm();
    let airspeed = kite_diag.airspeed;
    let roll_angle = roll_angle_from_quat_n2b(&request.plant_state.kites[index].body.quat_n2b);
    let pitch_angle = pitch_angle_from_quat_n2b(&request.plant_state.kites[index].body.quat_n2b);
    let omega_n = rotate_body_to_nav(
        &request.plant_state.kites[index].body.quat_n2b,
        &kite_diag.omega_b,
    );

    let alpha_min = tuning.alpha_protection_min_deg.to_radians();
    let alpha_max = tuning.alpha_protection_max_deg.to_radians();
    let alpha_protection = if kite_diag.alpha > alpha_max {
        kite_diag.alpha - alpha_max
    } else if kite_diag.alpha < alpha_min {
        kite_diag.alpha - alpha_min
    } else {
        0.0
    };
    let alpha_exceeded = clamp(kite_diag.alpha, 0.0, 0.15);

    let (altitude_ref_raw, raw_roll_breakdown, omega_world_z_ref, k_tg_y, k_tg_z) =
        if is_free_flight {
            let reference = free_flight_reference(
                index,
                request.time,
                request.initial_altitude,
                request.params.controller.speed_ref,
            );
            (
                reference.altitude_ref_raw,
                RollReferenceBreakdown {
                    total: reference.roll_ref,
                    feedforward: 0.0,
                    proportional: 0.0,
                    integrator: 0.0,
                },
                0.0,
                0.0,
                0.0,
            )
        } else {
            let altitude_ref_raw = -request.params.controller.disk_center_n[2]
                + kite_diag.cad_velocity_n[2] * request.params.controller.vert_vel_to_rabbit_height
                - request.params.kites[index].tether.contact.ground_altitude;

            if !control_state.roll_ref_initialized {
                control_state.roll_ref_command = roll_angle;
                control_state.roll_ref_initialized = true;
            }

            // The swarm scheduler has already chosen this kite's rabbit point.
            // Below this boundary the controller only sees one aircraft.
            let rabbit_vector_n = sub(&request.schedule.rabbit_target_n, &kite_diag.cad_position_n);
            let rabbit_vector_b = rotate_nav_to_body(
                &request.plant_state.kites[index].body.quat_n2b,
                &rabbit_vector_n,
            );
            let uses_direct_rabbit = guidance_uses_direct_rabbit(
                &rabbit_vector_b,
                request.schedule.rabbit_distance,
                tuning,
            );
            let (roll_breakdown, omega_world_z_ref, k_tg_y, k_tg_z) = if uses_direct_rabbit {
                (
                    direct_rabbit_roll_reference_breakdown(
                        &rabbit_vector_b,
                        control_state,
                        request.dt_control,
                        tuning,
                    ),
                    0.0,
                    0.0,
                    0.0,
                )
            } else {
                let (k_tg_y, k_tg_z) = lateral_guidance_curvatures(
                    &rabbit_vector_b,
                    request.schedule.rabbit_radius,
                    request.schedule.rabbit_distance,
                    tuning,
                );
                let curvature_y_est = omega_n[2] / inertial_speed.max(1.0);
                let gain_int_y = request.params.controller.gain_int_y.abs().max(1.0e-9);
                let gain_int_z = request.params.controller.gain_int_z.abs().max(1.0e-9);
                control_state.curvature_y_integrator = clamp(
                    control_state.curvature_y_integrator
                        + (kite_diag.curvature_y_b - k_tg_y) * request.dt_control
                        - alpha_exceeded * 0.5,
                    -TETHERED_CURVATURE_Y_INTEGRATOR_OUTPUT_LIMIT / gain_int_y,
                    TETHERED_CURVATURE_Y_INTEGRATOR_OUTPUT_LIMIT / gain_int_y,
                );
                control_state.curvature_z_integrator = clamp(
                    control_state.curvature_z_integrator
                        + (kite_diag.curvature_z_b - k_tg_z) * request.dt_control
                        - alpha_exceeded * 0.5,
                    -TETHERED_CURVATURE_Z_INTEGRATOR_OUTPUT_LIMIT / gain_int_z,
                    TETHERED_CURVATURE_Z_INTEGRATOR_OUTPUT_LIMIT / gain_int_z,
                );
                let roll_feedforward = orbit_roll_feedforward(
                    inertial_speed,
                    request.schedule.rabbit_radius,
                    request.params.environment.g,
                    tuning,
                );
                control_state.curvature_to_roll_integrator = clamp(
                    control_state.curvature_to_roll_integrator
                        + (k_tg_y - curvature_y_est) * request.dt_control,
                    -tuning.roll_curvature_integrator_limit,
                    tuning.roll_curvature_integrator_limit,
                );
                let roll_proportional = tuning.roll_curvature_p * (k_tg_y - curvature_y_est);
                let roll_integrator =
                    tuning.roll_curvature_i * control_state.curvature_to_roll_integrator;
                let roll_ref = clamp(
                    roll_feedforward + roll_proportional + roll_integrator,
                    -tuning.roll_ref_limit_deg.to_radians(),
                    tuning.roll_ref_limit_deg.to_radians(),
                );
                (
                    RollReferenceBreakdown {
                        total: roll_ref,
                        feedforward: roll_feedforward,
                        proportional: roll_proportional,
                        integrator: roll_integrator,
                    },
                    inertial_speed * k_tg_y,
                    k_tg_y,
                    k_tg_z,
                )
            };
            (
                altitude_ref_raw,
                roll_breakdown,
                omega_world_z_ref,
                k_tg_y,
                k_tg_z,
            )
        };

    let roll_ref = if tethered_lateral {
        control_state.roll_ref_command = rate_limit(
            control_state.roll_ref_command,
            raw_roll_breakdown.total,
            tuning.tethered_roll_ref_rate_limit_degps.to_radians(),
            request.dt_control,
        );
        control_state.roll_ref_command
    } else {
        control_state.roll_ref_command = raw_roll_breakdown.total;
        raw_roll_breakdown.total
    };

    let altitude = altitude_from_position_n(
        &kite_diag.cad_position_n,
        request.params.kites[index].tether.contact.ground_altitude,
    );
    let pitch_ref_limit = if tethered_lateral {
        tuning.tethered_pitch_ref_limit_deg.to_radians()
    } else {
        tuning.free_pitch_ref_limit_deg.to_radians()
    };
    let pitch_integrator_limit = tuning.tecs_pitch_integrator_limit_deg.to_radians();
    let motor_torque_max = tuning.motor_torque_max_nm;
    let tecs = tecs_terms(
        altitude,
        altitude_ref_raw,
        airspeed,
        request.schedule.speed_target,
        request.params.environment.g,
        tethered_lateral,
        tuning,
    );
    let pitch_ref_breakdown = if max_throttle_altitude_pitch {
        let altitude_error = clamp(
            altitude_ref_raw - altitude,
            -tuning.tecs_altitude_error_limit_m,
            tuning.tecs_altitude_error_limit_m,
        );
        saturated_pi_breakdown(
            &mut control_state.pitch_energy_integrator,
            altitude_error,
            request.dt_control,
            SaturatedPiConfig {
                bias: 0.0,
                kp: tuning.altitude_pitch_p,
                ki: tuning.altitude_pitch_i,
                output_min: -pitch_ref_limit,
                output_max: pitch_ref_limit,
                integrator_min: -pitch_integrator_limit,
                integrator_max: pitch_integrator_limit,
            },
        )
    } else {
        saturated_pi_breakdown(
            &mut control_state.pitch_energy_integrator,
            tecs.energy_balance_error,
            request.dt_control,
            SaturatedPiConfig {
                bias: 0.0,
                kp: tuning.tecs_pitch_balance_p,
                ki: tuning.tecs_pitch_balance_i,
                output_min: -pitch_ref_limit,
                output_max: pitch_ref_limit,
                integrator_min: -pitch_integrator_limit,
                integrator_max: pitch_integrator_limit,
            },
        )
    };
    let pitch_ref = pitch_ref_breakdown.total;

    let thrust_energy_error = if tethered_lateral {
        tecs.kinetic_energy_error
            + tuning.tethered_thrust_positive_potential_blend * tecs.potential_energy_error.max(0.0)
    } else {
        tecs.kinetic_energy_error
    };
    let motor_torque_breakdown = if max_throttle_altitude_pitch {
        control_state.thrust_energy_integrator = motor_torque_max;
        SaturatedPiBreakdown {
            bias: 0.0,
            proportional: 0.0,
            integrator: 0.0,
            total: motor_torque_max,
        }
    } else {
        saturated_pi_breakdown(
            &mut control_state.thrust_energy_integrator,
            thrust_energy_error,
            request.dt_control,
            SaturatedPiConfig {
                bias: request.params.controller.trim.motor_torque,
                kp: tuning.tecs_thrust_kinetic_p,
                ki: tuning.tecs_thrust_kinetic_i,
                output_min: 0.0,
                output_max: motor_torque_max,
                integrator_min: -tuning.tecs_thrust_integrator_limit_nm,
                integrator_max: tuning.tecs_thrust_integrator_limit_nm,
            },
        )
    };
    let commanded_motor_torque = motor_torque_breakdown.total;
    let motor_torque = limit_motor_torque_for_rotor_speed(
        commanded_motor_torque,
        request.plant_state.kites[index].rotor_speed,
        tuning,
    );
    if max_throttle_altitude_pitch {
        control_state.thrust_energy_integrator = motor_torque;
    }

    let aileron_breakdown = tethered_aileron_breakdown(
        request.params.controller.trim.surfaces.aileron,
        roll_ref,
        roll_angle,
        kite_diag.omega_b[0],
        tuning,
    );
    let rudder_breakdown = tethered_rudder_breakdown(
        request.params.controller.trim.surfaces.rudder,
        kite_diag.beta,
        kite_diag.omega_b[2],
        omega_n[2],
        omega_world_z_ref,
        tuning,
    );
    let elevator_cmd_breakdown = elevator_breakdown(
        request.params.controller.trim.surfaces.elevator,
        pitch_ref,
        pitch_angle,
        kite_diag.omega_b[1],
        alpha_protection,
        tuning,
    );
    let surfaces = ControlSurfaces {
        aileron: aileron_breakdown.total,
        flap: request.params.controller.trim.surfaces.flap,
        winglet: request.params.controller.trim.surfaces.winglet,
        elevator: elevator_cmd_breakdown.total,
        rudder: rudder_breakdown.total,
    };

    KiteControlOutput {
        controls: KiteControls {
            surfaces,
            motor_torque,
        },
        trace: KiteControllerTrace {
            phase_error: request.schedule.phase_error,
            speed_target: request.schedule.speed_target,
            altitude,
            altitude_ref: tecs.altitude_ref,
            kinetic_energy_specific: tecs.kinetic_energy,
            kinetic_energy_ref_specific: tecs.kinetic_energy_ref,
            kinetic_energy_error_specific: tecs.kinetic_energy_error,
            potential_energy_specific: tecs.potential_energy,
            potential_energy_ref_specific: tecs.potential_energy_ref,
            potential_energy_error_specific: tecs.potential_energy_error,
            total_energy_error_specific: tecs.total_energy_error,
            energy_balance_error_specific: tecs.energy_balance_error,
            thrust_energy_integrator: control_state.thrust_energy_integrator,
            pitch_energy_integrator: control_state.pitch_energy_integrator,
            rabbit_phase: request.schedule.rabbit_phase,
            rabbit_radius: request.schedule.rabbit_radius,
            rabbit_target_n: request.schedule.rabbit_target_n,
            curvature_y_ref: k_tg_y,
            curvature_y_estimate: omega_n[2] / inertial_speed.max(1.0),
            omega_world_z_ref,
            omega_world_z: omega_n[2],
            beta_ref: 0.0,
            roll_ref,
            roll_feedforward: raw_roll_breakdown.feedforward,
            roll_proportional: raw_roll_breakdown.proportional,
            roll_integrator: raw_roll_breakdown.integrator,
            pitch_ref,
            pitch_ref_proportional: pitch_ref_breakdown.proportional,
            pitch_ref_integrator: pitch_ref_breakdown.integrator,
            curvature_z_ref: k_tg_z,
            aileron_trim: aileron_breakdown.trim,
            aileron_roll_proportional: aileron_breakdown.roll_p,
            aileron_roll_derivative: aileron_breakdown.roll_d,
            rudder_trim: rudder_breakdown.trim,
            rudder_beta_proportional: rudder_breakdown.beta_p,
            rudder_rate_derivative: rudder_breakdown.rate_d,
            rudder_world_z_proportional: rudder_breakdown.world_z_p,
            elevator_trim: elevator_cmd_breakdown.trim,
            elevator_pitch_proportional: elevator_cmd_breakdown.pitch_p,
            elevator_pitch_derivative: elevator_cmd_breakdown.pitch_d,
            elevator_alpha_protection: elevator_cmd_breakdown.alpha_protection,
            motor_torque_trim: motor_torque_breakdown.bias,
            motor_torque_proportional: motor_torque_breakdown.proportional,
            motor_torque_integrator: motor_torque_breakdown.integrator,
        },
    }
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
    let is_free_flight = N_COMMON == 0 && N_UPPER == 0;

    let speed_targets: [f64; NK] = std::array::from_fn(|index| {
        if is_free_flight {
            free_flight_reference(
                index,
                time,
                state.initial_altitude[index],
                params.controller.speed_ref,
            )
            .speed_target
        } else {
            speed_integrator_target(
                phase_errors[index],
                params.controller.speed_ref,
                &params.controller.tuning,
            )
        }
    });
    let rabbit_distances: [f64; NK] = std::array::from_fn(|index| {
        scheduled_rabbit_distance(
            speed_targets[index],
            params.controller.rabbit_distance,
            &params.controller.tuning,
        )
    });

    // Swarm coordination is the only multi-aircraft step in this module.
    // Everything below receives one scheduled kite at a time.
    let schedules: [KiteSchedule; NK] = std::array::from_fn(|index| {
        let rabbit_phase = diag.kites[index].phase_angle
            + rabbit_distances[index] / params.controller.disk_radius.max(1.0e-6);
        let rabbit_radius = params.controller.disk_radius
            * (1.0
                + phase_errors[index] / std::f64::consts::PI
                    * params.controller.phase_lag_to_radius);
        KiteSchedule {
            phase_error: phase_errors[index],
            speed_target: speed_targets[index],
            rabbit_distance: rabbit_distances[index],
            rabbit_phase,
            rabbit_radius,
            rabbit_target_n: Vector3::new(
                params.controller.disk_center_n[0] + rabbit_radius * rabbit_phase.cos(),
                params.controller.disk_center_n[1] + rabbit_radius * rabbit_phase.sin(),
                params.controller.disk_center_n[2],
            ),
        }
    });

    let outputs: [KiteControlOutput; NK] = std::array::from_fn(|index| {
        let initial_altitude = state.initial_altitude[index];
        compute_kite_control(
            KiteControlRequest {
                index,
                schedule: schedules[index].clone(),
                control_state: &mut state.kites[index],
                plant_state,
                diag,
                params,
                initial_altitude,
                dt_control,
                longitudinal_mode,
                time,
            },
            free_flight_reference,
        )
    });

    (
        Controls {
            kites: std::array::from_fn(|index| outputs[index].controls.clone()),
        },
        ControllerTrace {
            kites: std::array::from_fn(|index| outputs[index].trace.clone()),
        },
    )
}

pub fn apply_trace<const NK: usize>(
    diagnostics: &mut Diagnostics<f64, NK>,
    trace: &ControllerTrace<NK>,
) {
    for index in 0..NK {
        let kite_trace = &trace.kites[index];
        diagnostics.kites[index].phase_error = kite_trace.phase_error;
        diagnostics.kites[index].speed_target = kite_trace.speed_target;
        diagnostics.kites[index].altitude = kite_trace.altitude;
        diagnostics.kites[index].altitude_ref = kite_trace.altitude_ref;
        diagnostics.kites[index].kinetic_energy_specific = kite_trace.kinetic_energy_specific;
        diagnostics.kites[index].kinetic_energy_ref_specific =
            kite_trace.kinetic_energy_ref_specific;
        diagnostics.kites[index].kinetic_energy_error_specific =
            kite_trace.kinetic_energy_error_specific;
        diagnostics.kites[index].potential_energy_specific = kite_trace.potential_energy_specific;
        diagnostics.kites[index].potential_energy_ref_specific =
            kite_trace.potential_energy_ref_specific;
        diagnostics.kites[index].potential_energy_error_specific =
            kite_trace.potential_energy_error_specific;
        diagnostics.kites[index].total_energy_error_specific =
            kite_trace.total_energy_error_specific;
        diagnostics.kites[index].energy_balance_error_specific =
            kite_trace.energy_balance_error_specific;
        diagnostics.kites[index].thrust_energy_integrator = kite_trace.thrust_energy_integrator;
        diagnostics.kites[index].pitch_energy_integrator = kite_trace.pitch_energy_integrator;
        diagnostics.kites[index].rabbit_phase = kite_trace.rabbit_phase;
        diagnostics.kites[index].rabbit_radius = kite_trace.rabbit_radius;
        diagnostics.kites[index].rabbit_target_n = kite_trace.rabbit_target_n;
        diagnostics.kites[index].curvature_y_ref = kite_trace.curvature_y_ref;
        diagnostics.kites[index].curvature_y_est = kite_trace.curvature_y_estimate;
        diagnostics.kites[index].omega_world_z_ref = kite_trace.omega_world_z_ref;
        diagnostics.kites[index].omega_world_z = kite_trace.omega_world_z;
        diagnostics.kites[index].beta_ref = kite_trace.beta_ref;
        diagnostics.kites[index].roll_ref = kite_trace.roll_ref;
        diagnostics.kites[index].roll_ff = kite_trace.roll_feedforward;
        diagnostics.kites[index].roll_p = kite_trace.roll_proportional;
        diagnostics.kites[index].roll_i = kite_trace.roll_integrator;
        diagnostics.kites[index].pitch_ref = kite_trace.pitch_ref;
        diagnostics.kites[index].pitch_ref_p = kite_trace.pitch_ref_proportional;
        diagnostics.kites[index].pitch_ref_i = kite_trace.pitch_ref_integrator;
        diagnostics.kites[index].curvature_z_ref = kite_trace.curvature_z_ref;
        diagnostics.kites[index].aileron_trim = kite_trace.aileron_trim;
        diagnostics.kites[index].aileron_roll_p = kite_trace.aileron_roll_proportional;
        diagnostics.kites[index].aileron_roll_d = kite_trace.aileron_roll_derivative;
        diagnostics.kites[index].rudder_trim = kite_trace.rudder_trim;
        diagnostics.kites[index].rudder_beta_p = kite_trace.rudder_beta_proportional;
        diagnostics.kites[index].rudder_rate_d = kite_trace.rudder_rate_derivative;
        diagnostics.kites[index].rudder_world_z_p = kite_trace.rudder_world_z_proportional;
        diagnostics.kites[index].elevator_trim = kite_trace.elevator_trim;
        diagnostics.kites[index].elevator_pitch_p = kite_trace.elevator_pitch_proportional;
        diagnostics.kites[index].elevator_pitch_d = kite_trace.elevator_pitch_derivative;
        diagnostics.kites[index].elevator_alpha_protection = kite_trace.elevator_alpha_protection;
        diagnostics.kites[index].motor_torque_trim = kite_trace.motor_torque_trim;
        diagnostics.kites[index].motor_torque_p = kite_trace.motor_torque_proportional;
        diagnostics.kites[index].motor_torque_i = kite_trace.motor_torque_integrator;
    }
}

#[cfg(test)]
mod tests {
    use super::lateral::{
        GUIDANCE_MODE_CURVATURE, GUIDANCE_MODE_RABBIT, GUIDANCE_MODE_SWITCH,
        direct_rabbit_bearing_y,
    };
    use super::*;
    use crate::types::ControllerTuning;

    #[test]
    fn tethered_rudder_feedback_damps_beta_and_body_z_rate() {
        let tuning = ControllerTuning::default();
        let beta_error = 5.0_f64.to_radians();
        assert!(tethered_rudder_breakdown(0.0, beta_error, 0.0, 0.0, 0.0, &tuning).total < 0.0);
        assert!(tethered_rudder_breakdown(0.0, -beta_error, 0.0, 0.0, 0.0, &tuning).total > 0.0);
        assert!(tethered_rudder_breakdown(0.0, 0.0, 0.5, 0.0, 0.0, &tuning).total > 0.0);
        assert!(tethered_rudder_breakdown(0.0, 0.0, -0.5, 0.0, 0.0, &tuning).total < 0.0);
    }

    #[test]
    fn tethered_aileron_feedback_closes_roll_error_and_damps_roll_rate() {
        let tuning = ControllerTuning::default();
        let roll_error = 5.0_f64.to_radians();
        assert!(tethered_aileron_breakdown(0.0, roll_error, 0.0, 0.0, &tuning).total < 0.0);
        assert!(tethered_aileron_breakdown(0.0, -roll_error, 0.0, 0.0, &tuning).total > 0.0);
        assert!(tethered_aileron_breakdown(0.0, 0.0, 0.0, 0.5, &tuning).total > 0.0);
        assert!(tethered_aileron_breakdown(0.0, 0.0, 0.0, -0.5, &tuning).total < 0.0);
    }

    #[test]
    fn guidance_mode_selects_direct_rabbit_or_curvature_conversion() {
        let mut tuning = ControllerTuning::default();
        let rabbit_vector_b = Vector3::new(40.0, 10.0, -5.0);

        tuning.guidance_mode = GUIDANCE_MODE_RABBIT as f64;
        let rabbit = lateral_guidance_curvatures(&rabbit_vector_b, 120.0, 40.0, &tuning);
        assert_eq!(rabbit, (0.0, 0.0));
        assert!(direct_rabbit_bearing_y(&rabbit_vector_b) > 0.0);

        tuning.guidance_mode = GUIDANCE_MODE_CURVATURE as f64;
        let curvature = lateral_guidance_curvatures(&rabbit_vector_b, 120.0, 40.0, &tuning);
        assert!((curvature.0 - 0.0125).abs() < 1.0e-12);
        assert!((curvature.1 + 0.00625).abs() < 1.0e-12);

        tuning.guidance_mode = GUIDANCE_MODE_SWITCH as f64;
        let ahead = lateral_guidance_curvatures(&rabbit_vector_b, 120.0, 40.0, &tuning);
        assert_eq!(ahead, (0.0, 0.0));
        let behind =
            lateral_guidance_curvatures(&Vector3::new(-2.0, 10.0, 0.0), 120.0, 40.0, &tuning);
        assert!(behind.0 > 0.0);
        assert!(behind.0 <= tuning.guidance_curvature_limit);
    }

    #[test]
    fn rabbit_distance_is_scheduled_from_speed_with_bounds() {
        let tuning = ControllerTuning::default();
        assert!((scheduled_rabbit_distance(6.25, 90.0, &tuning) - 22.5).abs() < 1.0e-12);
        assert_eq!(
            scheduled_rabbit_distance(1.0, 90.0, &tuning),
            tuning.rabbit_min_distance_m
        );
        assert_eq!(
            scheduled_rabbit_distance(100.0, 90.0, &tuning),
            tuning.rabbit_max_distance_m
        );
    }

    #[test]
    fn positive_phase_error_schedules_higher_speed() {
        let tuning = ControllerTuning::default();
        let speed_ref = 32.0;
        let phase_error = 0.1;
        assert!(speed_integrator_target(phase_error, speed_ref, &tuning) > speed_ref);
        assert!(speed_integrator_target(-phase_error, speed_ref, &tuning) < speed_ref);
    }

    #[test]
    fn rotor_speed_limiter_fades_torque_above_fit_range() {
        let tuning = ControllerTuning::default();
        let commanded = tuning.motor_torque_max_nm;
        assert_eq!(
            limit_motor_torque_for_rotor_speed(
                commanded,
                tuning.rotor_speed_soft_limit_radps - 1.0,
                &tuning
            ),
            commanded
        );
        assert!(
            limit_motor_torque_for_rotor_speed(commanded, 850.0, &tuning) < commanded,
            "torque should fade inside the overspeed band"
        );
        assert_eq!(
            limit_motor_torque_for_rotor_speed(
                commanded,
                tuning.rotor_speed_hard_limit_radps + 1.0,
                &tuning
            ),
            0.0
        );
    }

    #[test]
    fn tethered_tecs_clamps_potential_and_weights_kinetic_deficit() {
        let tuning = ControllerTuning::default();
        let tethered = tecs_terms(100.0, 200.0, 10.0, 20.0, 9.81, true, &tuning);
        assert!(
            (tethered.potential_energy_error - tuning.tethered_tecs_potential_error_limit).abs()
                < 1.0e-9
        );

        let free = tecs_terms(100.0, 200.0, 10.0, 20.0, 9.81, false, &tuning);
        assert!(free.potential_energy_error > tethered.potential_energy_error);
        assert!(tethered.energy_balance_error < free.energy_balance_error);
    }
}
