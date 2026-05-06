mod actuators;
mod lateral;
mod longitudinal;
mod state;

pub use state::{ControllerState, ControllerTrace, KiteControllerTrace};

use actuators::{elevator_breakdown, tethered_aileron_breakdown, tethered_rudder_breakdown};
use lateral::{
    RollReferenceBreakdown, TETHERED_CURVATURE_Y_INTEGRATOR_OUTPUT_LIMIT,
    TETHERED_CURVATURE_Z_INTEGRATOR_OUTPUT_LIMIT, direct_rabbit_bearing_y,
    direct_rabbit_roll_reference_breakdown, guidance_uses_direct_rabbit,
    lateral_guidance_curvatures, orbit_roll_feedforward, rate_limit, roll_integrator_output_limit,
    roll_integrator_with_reference_antiwindup, scheduled_rabbit_distance, speed_integrator_target,
};
pub(crate) use longitudinal::FreeFlightReference;
use longitudinal::{
    SaturatedPiBreakdown, SaturatedPiConfig, default_free_flight_reference,
    limit_motor_torque_for_rotor_speed, saturated_pi_breakdown, tecs_terms,
};
use state::KiteControllerState;

use crate::math::{
    circular_mean, clamp, pitch_angle_from_quat_n2b, roll_angle_from_quat_n2b, rotate_body_to_nav,
    rotate_nav_to_body, wrap_angle, yaw_angle_from_quat_n2b, yaw_quaternion_n2b,
};
use crate::types::{
    ControlSurfaces, Controls, Diagnostics, ForwardFrameMode, KiteControls, LateralOuterMode,
    LongitudinalMode, Params, PhaseMode, State,
};
use nalgebra::{Quaternion, Vector3};

fn pairwise_phase_errors<const NK: usize>(diag: &Diagnostics<f64, NK>) -> [f64; NK] {
    let slot_errors: [f64; NK] = std::array::from_fn(|index| {
        let desired_slot = 2.0 * std::f64::consts::PI * index as f64 / NK as f64;
        wrap_angle(diag.kites[index].phase_angle - desired_slot)
    });
    let mean_error = circular_mean(&slot_errors);
    std::array::from_fn(|index| wrap_angle(mean_error - slot_errors[index]))
}

fn lateral_rabbit_vector_n(
    rabbit_target_n: &Vector3<f64>,
    cad_position_n: &Vector3<f64>,
) -> Vector3<f64> {
    Vector3::new(
        rabbit_target_n[0] - cad_position_n[0],
        rabbit_target_n[1] - cad_position_n[1],
        0.0,
    )
}

fn lateral_rabbit_vector_yaw_b(
    quat_n2b: &Quaternion<f64>,
    rabbit_target_n: &Vector3<f64>,
    cad_position_n: &Vector3<f64>,
) -> Vector3<f64> {
    let rabbit_vector_n = lateral_rabbit_vector_n(rabbit_target_n, cad_position_n);
    let yaw_n2b = yaw_quaternion_n2b(yaw_angle_from_quat_n2b(quat_n2b));
    rotate_nav_to_body(&yaw_n2b, &rabbit_vector_n)
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

fn lateral_outer_mode_value(mode: LateralOuterMode) -> f64 {
    match mode {
        LateralOuterMode::Orbit => 0.0,
        LateralOuterMode::ForwardFormation => 1.0,
        LateralOuterMode::TimedTransition => 2.0,
    }
}

fn active_lateral_outer_mode(
    configured: LateralOuterMode,
    time: f64,
    transition_to_forward_s: f64,
    transition_to_orbit_s: Option<f64>,
) -> LateralOuterMode {
    match configured {
        LateralOuterMode::Orbit => LateralOuterMode::Orbit,
        LateralOuterMode::ForwardFormation => LateralOuterMode::ForwardFormation,
        LateralOuterMode::TimedTransition => {
            if time < transition_to_forward_s {
                LateralOuterMode::Orbit
            } else if transition_to_orbit_s.is_some_and(|transition| time >= transition) {
                LateralOuterMode::Orbit
            } else {
                LateralOuterMode::ForwardFormation
            }
        }
    }
}

fn clamped_base_speed_target<const NK: usize>(params: &Params<f64, NK>) -> f64 {
    let tuning = &params.controller.tuning;
    let min_speed = tuning.speed_min_mps.min(tuning.speed_max_mps);
    let max_speed = tuning.speed_min_mps.max(tuning.speed_max_mps);
    clamp(params.controller.speed_ref, min_speed, max_speed)
}

pub(crate) fn forward_formation_spacing<const NK: usize>(params: &Params<f64, NK>) -> f64 {
    let configured = params.controller.tuning.formation_spacing_m;
    if configured > 0.0 {
        configured
    } else {
        2.0 * params.controller.disk_radius / ((NK.saturating_sub(1)).max(1) as f64)
    }
}

fn forward_frame_heading<const NK: usize>(
    diag: &Diagnostics<f64, NK>,
    mode: ForwardFrameMode,
    fallback_heading_deg: f64,
) -> f64 {
    let fallback = fallback_heading_deg.to_radians();
    match mode {
        ForwardFrameMode::WorldFixed => fallback,
        ForwardFrameMode::MeanVelocity => {
            let mean_velocity = diag
                .kites
                .iter()
                .fold(Vector3::<f64>::zeros(), |acc, kite| {
                    acc + kite.cad_velocity_n
                })
                / NK.max(1) as f64;
            let horizontal_speed = mean_velocity[0].hypot(mean_velocity[1]);
            if horizontal_speed > 1.0e-3 {
                mean_velocity[1].atan2(mean_velocity[0])
            } else {
                fallback
            }
        }
    }
}

pub(crate) fn horizontal_axes_from_heading(heading_rad: f64) -> (Vector3<f64>, Vector3<f64>) {
    let f_n = Vector3::new(heading_rad.cos(), heading_rad.sin(), 0.0);
    let l_n = Vector3::new(-heading_rad.sin(), heading_rad.cos(), 0.0);
    (f_n, l_n)
}

pub(crate) fn lane_slot(index: usize, nk: usize, spacing: f64) -> f64 {
    (index as f64 - (nk as f64 - 1.0) * 0.5) * spacing
}

fn formation_error_from_neighbor_y(
    index: usize,
    nk: usize,
    prev_y_f: Option<f64>,
    next_y_f: Option<f64>,
    spacing: f64,
) -> f64 {
    let mut error = 0.0;
    if index > 0 {
        error += prev_y_f.unwrap_or(0.0) + spacing;
    }
    if index + 1 < nk {
        error += next_y_f.unwrap_or(0.0) - spacing;
    }
    error
}

fn formation_frame_lateral_component(
    from_n: &Vector3<f64>,
    to_n: &Vector3<f64>,
    forward_l_n: &Vector3<f64>,
) -> f64 {
    let vector_n = Vector3::new(to_n[0] - from_n[0], to_n[1] - from_n[1], 0.0);
    vector_n.dot(forward_l_n)
}

fn update_forward_lateral_offset(
    control_state: &mut KiteControllerState,
    lateral_error: f64,
    dt_control: f64,
    tuning: &crate::types::ControllerTuning<f64>,
) -> f64 {
    let offset_limit = tuning.formation_lateral_offset_limit_m.abs();
    if offset_limit <= 1.0e-9 {
        control_state.forward_lateral_offset_m = 0.0;
        return 0.0;
    }

    let error_limit = tuning.formation_lateral_error_limit_m.abs();
    let limited_error = if error_limit > 1.0e-9 {
        clamp(lateral_error, -error_limit, error_limit)
    } else {
        lateral_error
    };
    control_state.forward_lateral_offset_m = clamp(
        control_state.forward_lateral_offset_m
            - tuning.formation_lateral_offset_i_per_s * limited_error * dt_control.max(0.0),
        -offset_limit,
        offset_limit,
    );
    control_state.forward_lateral_offset_m
}

#[derive(Clone, Copy, Debug)]
struct ForwardScheduleTerms {
    lane_y: f64,
    cross_track_error: f64,
    prev_y_f: f64,
    next_y_f: f64,
    formation_error: f64,
    formation_spacing: f64,
    lateral_offset: f64,
    lane_point_n: Vector3<f64>,
    formation_error_tip_n: Vector3<f64>,
}

impl ForwardScheduleTerms {
    fn zero(spacing: f64) -> Self {
        Self {
            lane_y: 0.0,
            cross_track_error: 0.0,
            prev_y_f: 0.0,
            next_y_f: 0.0,
            formation_error: 0.0,
            formation_spacing: spacing,
            lateral_offset: 0.0,
            lane_point_n: Vector3::zeros(),
            formation_error_tip_n: Vector3::zeros(),
        }
    }
}

fn orbit_schedule_target<const NK: usize>(
    index: usize,
    diag: &Diagnostics<f64, NK>,
    params: &Params<f64, NK>,
    rabbit_distance: f64,
    phase_error: f64,
) -> (f64, f64, Vector3<f64>) {
    let rabbit_phase =
        diag.kites[index].phase_angle + rabbit_distance / params.controller.disk_radius.max(1.0e-6);
    let rabbit_radius = params.controller.disk_radius
        * (1.0 + phase_error / std::f64::consts::PI * params.controller.phase_lag_to_radius);
    let target_n = Vector3::new(
        params.controller.disk_center_n[0] + rabbit_radius * rabbit_phase.cos(),
        params.controller.disk_center_n[1] + rabbit_radius * rabbit_phase.sin(),
        params.controller.disk_center_n[2],
    );
    (rabbit_phase, rabbit_radius, target_n)
}

fn forward_schedule_target<const NK: usize>(
    index: usize,
    control_state: &mut KiteControllerState,
    diag: &Diagnostics<f64, NK>,
    params: &Params<f64, NK>,
    rabbit_distance: f64,
    spacing: f64,
    forward_f_n: &Vector3<f64>,
    forward_l_n: &Vector3<f64>,
    dt_control: f64,
) -> (f64, f64, Vector3<f64>, ForwardScheduleTerms) {
    let p_i = diag.kites[index].cad_position_n;
    let origin = params.controller.disk_center_n;
    let offset = p_i - origin;
    let x_i = offset.dot(forward_f_n);
    let y_i = offset.dot(forward_l_n);
    let lane_y = lane_slot(index, NK, spacing);

    let lateral_offset = update_forward_lateral_offset(
        control_state,
        y_i - lane_y,
        dt_control,
        &params.controller.tuning,
    );

    let lane_point_xy = origin + *forward_f_n * x_i + *forward_l_n * lane_y;
    let offset_lane_point_xy = lane_point_xy + *forward_l_n * lateral_offset;
    let lane_point_n = Vector3::new(lane_point_xy[0], lane_point_xy[1], origin[2]);
    let target_xy = offset_lane_point_xy + *forward_f_n * rabbit_distance;
    let target_n = Vector3::new(target_xy[0], target_xy[1], origin[2]);

    let prev_y_f = if index > 0 {
        formation_frame_lateral_component(&p_i, &diag.kites[index - 1].cad_position_n, forward_l_n)
    } else {
        0.0
    };
    let next_y_f = if index + 1 < NK {
        formation_frame_lateral_component(&p_i, &diag.kites[index + 1].cad_position_n, forward_l_n)
    } else {
        0.0
    };
    let neighbor_formation_error = formation_error_from_neighbor_y(
        index,
        NK,
        (index > 0).then_some(prev_y_f),
        (index + 1 < NK).then_some(next_y_f),
        spacing,
    );
    let formation_error_tip_n =
        Vector3::new(offset_lane_point_xy[0], offset_lane_point_xy[1], origin[2]);

    let phase_origin = target_n - origin;
    let terms = ForwardScheduleTerms {
        lane_y,
        cross_track_error: y_i - lane_y,
        prev_y_f,
        next_y_f,
        formation_error: neighbor_formation_error,
        formation_spacing: spacing,
        lateral_offset,
        lane_point_n,
        formation_error_tip_n,
    };
    (
        phase_origin[1].atan2(phase_origin[0]),
        phase_origin[0].hypot(phase_origin[1]).max(1.0e-6),
        target_n,
        terms,
    )
}

#[derive(Clone, Debug)]
struct KiteSchedule {
    lateral_outer_mode: LateralOuterMode,
    lateral_outer_mode_value: f64,
    phase_error: f64,
    speed_target: f64,
    rabbit_distance: f64,
    rabbit_phase: f64,
    rabbit_radius: f64,
    rabbit_target_n: Vector3<f64>,
    forward_frame_heading: f64,
    forward: ForwardScheduleTerms,
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

    let (
        altitude_ref_raw,
        raw_roll_breakdown,
        omega_world_z_ref,
        k_tg_y,
        k_tg_z,
        rabbit_vector_b,
        rabbit_bearing_y,
    ) = if is_free_flight {
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
            Vector3::<f64>::zeros(),
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
        let rabbit_vector_b = lateral_rabbit_vector_yaw_b(
            &request.plant_state.kites[index].body.quat_n2b,
            &request.schedule.rabbit_target_n,
            &kite_diag.cad_position_n,
        );
        let rabbit_bearing_y = direct_rabbit_bearing_y(&rabbit_vector_b);
        let (roll_breakdown, omega_world_z_ref, k_tg_y, k_tg_z) =
            if request.schedule.lateral_outer_mode == LateralOuterMode::ForwardFormation
                || guidance_uses_direct_rabbit(
                    &rabbit_vector_b,
                    request.schedule.rabbit_distance,
                    tuning,
                )
            {
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
                let curvature_roll_error = k_tg_y - curvature_y_est;
                let roll_proportional = tuning.roll_curvature_p * curvature_roll_error;
                let (roll_integrator, roll_ref) = roll_integrator_with_reference_antiwindup(
                    &mut control_state.curvature_to_roll_integrator,
                    curvature_roll_error,
                    request.dt_control,
                    tuning.roll_curvature_i,
                    roll_integrator_output_limit(
                        tuning.roll_curvature_i,
                        tuning.roll_curvature_integrator_limit,
                    ),
                    roll_feedforward + roll_proportional,
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
            rabbit_vector_b,
            rabbit_bearing_y,
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
            rabbit_distance: request.schedule.rabbit_distance,
            rabbit_target_distance: rabbit_vector_b.norm(),
            rabbit_bearing_y,
            rabbit_vector_b,
            rabbit_target_n: request.schedule.rabbit_target_n,
            lateral_outer_mode: request.schedule.lateral_outer_mode_value,
            forward_frame_heading: request.schedule.forward_frame_heading,
            forward_lane_y: request.schedule.forward.lane_y,
            forward_cross_track_error: request.schedule.forward.cross_track_error,
            forward_neighbor_prev_y_f: request.schedule.forward.prev_y_f,
            forward_neighbor_next_y_f: request.schedule.forward.next_y_f,
            forward_formation_error: request.schedule.forward.formation_error,
            forward_formation_spacing: request.schedule.forward.formation_spacing,
            forward_lateral_offset: request.schedule.forward.lateral_offset,
            forward_lane_point_n: request.schedule.forward.lane_point_n,
            forward_formation_error_tip_n: request.schedule.forward.formation_error_tip_n,
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
    lateral_outer_mode: LateralOuterMode,
    forward_frame_mode: ForwardFrameMode,
    transition_to_forward_s: f64,
    transition_to_orbit_s: Option<f64>,
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
        lateral_outer_mode,
        forward_frame_mode,
        transition_to_forward_s,
        transition_to_orbit_s,
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
    lateral_outer_mode: LateralOuterMode,
    forward_frame_mode: ForwardFrameMode,
    transition_to_forward_s: f64,
    transition_to_orbit_s: Option<f64>,
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
        lateral_outer_mode,
        forward_frame_mode,
        transition_to_forward_s,
        transition_to_orbit_s,
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
    lateral_outer_mode: LateralOuterMode,
    forward_frame_mode: ForwardFrameMode,
    transition_to_forward_s: f64,
    transition_to_orbit_s: Option<f64>,
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
    let active_lateral_mode = if is_free_flight {
        LateralOuterMode::Orbit
    } else {
        active_lateral_outer_mode(
            lateral_outer_mode,
            time,
            transition_to_forward_s,
            transition_to_orbit_s,
        )
    };
    let forward_spacing = forward_formation_spacing(params);
    let forward_heading = forward_frame_heading(
        diag,
        forward_frame_mode,
        params.controller.tuning.forward_heading_deg,
    );
    let (forward_f_n, forward_l_n) = horizontal_axes_from_heading(forward_heading);

    let speed_targets: [f64; NK] = std::array::from_fn(|index| {
        if is_free_flight {
            free_flight_reference(
                index,
                time,
                state.initial_altitude[index],
                params.controller.speed_ref,
            )
            .speed_target
        } else if active_lateral_mode == LateralOuterMode::ForwardFormation {
            clamped_base_speed_target(params)
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
        let phase_error = if active_lateral_mode == LateralOuterMode::ForwardFormation {
            0.0
        } else {
            phase_errors[index]
        };
        let (rabbit_phase, rabbit_radius, rabbit_target_n, forward) =
            if active_lateral_mode == LateralOuterMode::ForwardFormation {
                forward_schedule_target(
                    index,
                    &mut state.kites[index],
                    diag,
                    params,
                    rabbit_distances[index],
                    forward_spacing,
                    &forward_f_n,
                    &forward_l_n,
                    dt_control,
                )
            } else {
                let (rabbit_phase, rabbit_radius, rabbit_target_n) = orbit_schedule_target(
                    index,
                    diag,
                    params,
                    rabbit_distances[index],
                    phase_errors[index],
                );
                (
                    rabbit_phase,
                    rabbit_radius,
                    rabbit_target_n,
                    ForwardScheduleTerms::zero(forward_spacing),
                )
            };
        KiteSchedule {
            lateral_outer_mode: active_lateral_mode,
            lateral_outer_mode_value: lateral_outer_mode_value(active_lateral_mode),
            phase_error,
            speed_target: speed_targets[index],
            rabbit_distance: rabbit_distances[index],
            rabbit_phase,
            rabbit_radius,
            rabbit_target_n,
            forward_frame_heading: forward_heading,
            forward,
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
        diagnostics.kites[index].rabbit_distance = kite_trace.rabbit_distance;
        diagnostics.kites[index].rabbit_target_distance = kite_trace.rabbit_target_distance;
        diagnostics.kites[index].rabbit_bearing_y = kite_trace.rabbit_bearing_y;
        diagnostics.kites[index].rabbit_vector_b = kite_trace.rabbit_vector_b;
        diagnostics.kites[index].rabbit_target_n = kite_trace.rabbit_target_n;
        diagnostics.kites[index].lateral_outer_mode = kite_trace.lateral_outer_mode;
        diagnostics.kites[index].forward_frame_heading = kite_trace.forward_frame_heading;
        diagnostics.kites[index].forward_lane_y = kite_trace.forward_lane_y;
        diagnostics.kites[index].forward_cross_track_error = kite_trace.forward_cross_track_error;
        diagnostics.kites[index].forward_neighbor_prev_y_f = kite_trace.forward_neighbor_prev_y_f;
        diagnostics.kites[index].forward_neighbor_next_y_f = kite_trace.forward_neighbor_next_y_f;
        diagnostics.kites[index].forward_formation_error = kite_trace.forward_formation_error;
        diagnostics.kites[index].forward_formation_spacing = kite_trace.forward_formation_spacing;
        diagnostics.kites[index].forward_lateral_offset = kite_trace.forward_lateral_offset;
        diagnostics.kites[index].forward_lane_point_n = kite_trace.forward_lane_point_n;
        diagnostics.kites[index].forward_formation_error_tip_n =
            kite_trace.forward_formation_error_tip_n;
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
    use super::state::KiteControllerState;
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
    fn roll_reference_antiwindup_freezes_only_when_pushing_saturation() {
        let mut integrator_output_state = 0.0;
        let (integrator, total) = roll_integrator_with_reference_antiwindup(
            &mut integrator_output_state,
            1.0,
            0.1,
            1.0,
            1.0,
            2.0,
            1.0,
        );
        assert_eq!(integrator_output_state, 0.0);
        assert_eq!(integrator, 0.0);
        assert_eq!(total, 1.0);

        let (integrator, total) = roll_integrator_with_reference_antiwindup(
            &mut integrator_output_state,
            -1.0,
            0.1,
            1.0,
            1.0,
            2.0,
            1.0,
        );
        assert!((integrator_output_state + 0.1).abs() < 1.0e-12);
        assert_eq!(integrator, 0.0);
        assert_eq!(total, 1.0);

        let (integrator, total) = roll_integrator_with_reference_antiwindup(
            &mut integrator_output_state,
            1.0,
            0.1,
            1.0,
            1.0,
            0.0,
            1.0,
        );
        assert_eq!(integrator_output_state, 0.0);
        assert!((integrator + 0.1).abs() < 1.0e-12);
        assert!((total + 0.1).abs() < 1.0e-12);
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
    fn lateral_rabbit_vector_ignores_altitude_error() {
        let target = Vector3::new(120.0, -30.0, -300.0);
        let aircraft = Vector3::new(20.0, -80.0, -75.0);
        let vector = lateral_rabbit_vector_n(&target, &aircraft);
        assert_eq!(vector, Vector3::new(100.0, 50.0, 0.0));
    }

    #[test]
    fn lateral_rabbit_vector_rotates_with_yaw_only() {
        let target = Vector3::new(120.0, -30.0, -300.0);
        let aircraft = Vector3::new(20.0, -80.0, -75.0);
        let yaw = 0.4;
        let full_attitude =
            *nalgebra::UnitQuaternion::from_euler_angles(0.7, -0.3, yaw).quaternion();
        let yaw_only = yaw_quaternion_n2b(yaw);
        let expected = rotate_nav_to_body(&yaw_only, &Vector3::new(100.0, 50.0, 0.0));
        let actual = lateral_rabbit_vector_yaw_b(&full_attitude, &target, &aircraft);

        assert!((actual - expected).norm() < 1.0e-12);
        assert!(actual[2].abs() < 1.0e-12);
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
    fn timed_lateral_outer_mode_selects_orbit_forward_orbit() {
        assert_eq!(
            active_lateral_outer_mode(LateralOuterMode::Orbit, 100.0, 5.0, Some(20.0)),
            LateralOuterMode::Orbit
        );
        assert_eq!(
            active_lateral_outer_mode(LateralOuterMode::ForwardFormation, 0.0, 5.0, Some(20.0),),
            LateralOuterMode::ForwardFormation
        );
        assert_eq!(
            active_lateral_outer_mode(LateralOuterMode::TimedTransition, 4.9, 5.0, Some(20.0)),
            LateralOuterMode::Orbit
        );
        assert_eq!(
            active_lateral_outer_mode(LateralOuterMode::TimedTransition, 5.0, 5.0, Some(20.0)),
            LateralOuterMode::ForwardFormation
        );
        assert_eq!(
            active_lateral_outer_mode(LateralOuterMode::TimedTransition, 20.0, 5.0, Some(20.0)),
            LateralOuterMode::Orbit
        );
    }

    #[test]
    fn forward_formation_error_omits_missing_neighbors() {
        let spacing = 12.0;
        assert_eq!(
            formation_error_from_neighbor_y(0, 1, None, None, spacing),
            0.0
        );
        assert_eq!(
            formation_error_from_neighbor_y(1, 3, Some(-spacing), Some(spacing), spacing),
            0.0
        );
        assert_eq!(
            formation_error_from_neighbor_y(0, 3, None, Some(spacing), spacing),
            0.0
        );
        assert_eq!(
            formation_error_from_neighbor_y(2, 3, Some(-spacing), None, spacing),
            0.0
        );
    }

    #[test]
    fn forward_lateral_offset_integrator_targets_opposite_lane_error() {
        let mut state = KiteControllerState {
            thrust_energy_integrator: 0.0,
            pitch_energy_integrator: 0.0,
            curvature_to_roll_integrator: 0.0,
            rabbit_bearing_to_roll_integrator: 0.0,
            forward_lateral_offset_m: 0.0,
            roll_ref_command: 0.0,
            roll_ref_initialized: false,
            curvature_y_integrator: 0.0,
            curvature_z_integrator: 0.0,
        };
        let tuning = ControllerTuning {
            formation_lateral_offset_i_per_s: 0.5,
            formation_lateral_offset_limit_m: 2.0,
            formation_lateral_error_limit_m: 10.0,
            ..ControllerTuning::default()
        };

        let offset = update_forward_lateral_offset(&mut state, 4.0, 1.0, &tuning);
        assert_eq!(offset, -2.0);

        let offset = update_forward_lateral_offset(&mut state, -4.0, 1.0, &tuning);
        assert_eq!(offset, 0.0);
    }

    #[test]
    fn orbit_mode_keeps_phase_speed_scheduling_active() {
        let tuning = ControllerTuning::default();
        let speed_ref = 32.0;
        let phase_error = 0.1;
        assert_eq!(
            active_lateral_outer_mode(LateralOuterMode::Orbit, 10.0, 5.0, Some(20.0)),
            LateralOuterMode::Orbit
        );
        assert_ne!(
            speed_integrator_target(phase_error, speed_ref, &tuning),
            speed_ref
        );
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
