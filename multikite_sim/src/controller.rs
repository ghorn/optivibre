use crate::math::{circular_mean, clamp, rotate_body_to_nav, rotate_nav_to_body, sub, wrap_angle};
use crate::types::{
    ControlSurfaces, Controls, Diagnostics, KiteControls, Params, PhaseMode, State,
};
use nalgebra::{UnitQuaternion, Vector3};

const SPEED_INTEGRATOR_PHASE_GAIN: f64 = 100.0;
const SPEED_INTEGRATOR_MIN_MPS: f64 = 15.0;
const SPEED_INTEGRATOR_MAX_MPS: f64 = 35.0;
const SPEED_INTEGRATOR_STATE_MIN: f64 = 0.0;
const SPEED_INTEGRATOR_STATE_MAX: f64 = 4.0;
const ALPHA_BACKOFF_MAX_RAD: f64 = 0.15;
const ALPHA_INTEGRATOR_BACKOFF_GAIN: f64 = 0.5;
const ROLL_FROM_CURVATURE_P: f64 = 8.0;
const ROLL_FROM_CURVATURE_I: f64 = 2.0;
const ROLL_CURVATURE_INTEGRATOR_LIMIT: f64 = 0.08;
const ROLL_REF_LIMIT_RAD: f64 = 55.0_f64.to_radians();
const AILERON_ROLL_P: f64 = 0.7;
const AILERON_ROLL_D: f64 = 0.22;
const PITCH_FROM_CURVATURE_P: f64 = 3.5;
const PITCH_FROM_CURVATURE_I: f64 = 1.0;
const PITCH_CURVATURE_INTEGRATOR_LIMIT: f64 = 0.08;
const PITCH_REF_LIMIT_RAD: f64 = 25.0_f64.to_radians();
const ELEVATOR_PITCH_P: f64 = 0.6;
const ELEVATOR_PITCH_D: f64 = 0.18;
const ALPHA_PROTECTION_MIN_RAD: f64 = -10.0_f64.to_radians();
const ALPHA_PROTECTION_MAX_RAD: f64 = 12.0_f64.to_radians();
const ALPHA_TO_ELEVATOR: f64 = 2.0;
const ALPHA_TO_FLAP: f64 = 0.6;
const RUDDER_BETA_P: f64 = 0.35;
const RUDDER_OMEGA_WORLD_Z_P: f64 = 0.25;
const SURFACE_LIMIT_LATERAL_RAD: f64 = 15.0_f64.to_radians();
const SURFACE_LIMIT_ELEVATOR_RAD: f64 = 20.0_f64.to_radians();
const MOTOR_TORQUE_MAX_NM: f64 = 8.0;

#[derive(Clone, Debug)]
struct KiteControllerState {
    speed_integrator: f64,
    curvature_to_roll_integrator: f64,
    curvature_to_pitch_integrator: f64,
}

#[derive(Clone, Debug)]
pub struct ControllerState<const NK: usize> {
    kites: [KiteControllerState; NK],
    initial_phase: [f64; NK],
}

#[derive(Clone, Debug)]
pub struct ControllerTrace<const NK: usize> {
    pub phase_errors: [f64; NK],
    pub speed_targets: [f64; NK],
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
                speed_integrator: 0.0,
                curvature_to_roll_integrator: 0.0,
                curvature_to_pitch_integrator: 0.0,
            }),
            initial_phase: std::array::from_fn(|index| initial_diag.kites[index].phase_angle),
        }
    }
}

fn roll_angle_from_quat_n2b(quat_n2b: &nalgebra::Quaternion<f64>) -> f64 {
    UnitQuaternion::from_quaternion(*quat_n2b).euler_angles().0
}

fn pitch_angle_from_quat_n2b(quat_n2b: &nalgebra::Quaternion<f64>) -> f64 {
    UnitQuaternion::from_quaternion(*quat_n2b).euler_angles().1
}

fn pairwise_phase_errors<const NK: usize>(diag: &Diagnostics<f64, NK>) -> [f64; NK] {
    let slot_errors: [f64; NK] = std::array::from_fn(|index| {
        let desired_slot = 2.0 * std::f64::consts::PI * index as f64 / NK as f64;
        wrap_angle(diag.kites[index].phase_angle - desired_slot)
    });
    let mean_error = circular_mean(&slot_errors);
    std::array::from_fn(|index| wrap_angle(slot_errors[index] - mean_error))
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
        wrap_angle(diag.kites[index].phase_angle - desired_phase)
    })
}

fn speed_integrator_target(phase_error: f64, speed_ref: f64) -> f64 {
    clamp(
        speed_ref - SPEED_INTEGRATOR_PHASE_GAIN * phase_error,
        SPEED_INTEGRATOR_MIN_MPS,
        SPEED_INTEGRATOR_MAX_MPS,
    )
}

fn free_flight_roll_reference(time: f64) -> f64 {
    if time < 1.0 {
        0.0
    } else if time < 3.0 {
        20.0_f64.to_radians()
    } else if time < 5.0 {
        -20.0_f64.to_radians()
    } else {
        0.0
    }
}

fn free_flight_pitch_reference(time: f64) -> f64 {
    if time < 1.5 {
        0.0
    } else if time < 3.0 {
        -6.0_f64.to_radians()
    } else if time < 4.5 {
        6.0_f64.to_radians()
    } else {
        0.0
    }
}

fn free_flight_speed_reference(time: f64, speed_ref: f64) -> f64 {
    if time < 2.0 {
        speed_ref
    } else if time < 4.0 {
        speed_ref + 3.0
    } else {
        speed_ref - 3.0
    }
}

pub fn controller_step<const NK: usize, const N_COMMON: usize, const N_UPPER: usize>(
    state: &mut ControllerState<NK>,
    plant_state: &State<f64, NK, N_COMMON, N_UPPER>,
    diag: &Diagnostics<f64, NK>,
    params: &Params<f64, NK>,
    dt_control: f64,
    phase_mode: PhaseMode,
    time: f64,
) -> (Controls<f64, NK>, ControllerTrace<NK>) {
    let adaptive_phase_errors = pairwise_phase_errors(diag);
    let open_loop_phase_errors = open_loop_phase_errors(state, diag, params, time);
    let phase_errors = match phase_mode {
        PhaseMode::Adaptive => adaptive_phase_errors,
        PhaseMode::OpenLoop => open_loop_phase_errors,
    };

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
            let control_state = &mut state.kites[index];
            let inertial_speed = kite_diag.cad_velocity_n.norm();
            let airspeed = kite_diag.airspeed;
            let roll_angle = roll_angle_from_quat_n2b(&plant_state.kites[index].body.quat_n2b);
            let pitch_angle = pitch_angle_from_quat_n2b(&plant_state.kites[index].body.quat_n2b);
            let omega_n =
                rotate_body_to_nav(&plant_state.kites[index].body.quat_n2b, &kite_diag.omega_b);
            let alpha_exceeded = clamp(kite_diag.alpha, 0.0, ALPHA_BACKOFF_MAX_RAD);
            let alpha_protection = if kite_diag.alpha > ALPHA_PROTECTION_MAX_RAD {
                kite_diag.alpha - ALPHA_PROTECTION_MAX_RAD
            } else if kite_diag.alpha < ALPHA_PROTECTION_MIN_RAD {
                kite_diag.alpha - ALPHA_PROTECTION_MIN_RAD
            } else {
                0.0
            };

            let (speed_target, roll_ref, _roll_feedforward, omega_world_z_ref, pitch_ref, k_tg_y, k_tg_z) =
                if N_COMMON == 0 && N_UPPER == 0 {
                    let speed_target = free_flight_speed_reference(time, params.controller.speed_ref);
                    let roll_ref = free_flight_roll_reference(time);
                    let pitch_ref = free_flight_pitch_reference(time);
                    (
                        speed_target,
                        roll_ref,
                        0.0,
                        0.0,
                        pitch_ref,
                        0.0,
                        0.0,
                    )
                } else {
                    let speed_target =
                        speed_integrator_target(phase_errors[index], params.controller.speed_ref);
                    let rabbit_vector_n = sub(&rabbit_targets_n[index], &kite_diag.cad_position_n);
                    let rabbit_vector_b = rotate_nav_to_body(
                        &plant_state.kites[index].body.quat_n2b,
                        &rabbit_vector_n,
                    );
                    let x = rabbit_vector_b[0].abs().max(1.0);
                    let x2 = x * x;
                    let k_tg_y = 2.0 * rabbit_vector_b[1] / x2;
                    let k_tg_z = 2.0 * rabbit_vector_b[2] / x2;
                    let curvature_y_est = omega_n[2] / inertial_speed.max(1.0);
                    let roll_feedforward = clamp(
                        (inertial_speed * inertial_speed * k_tg_y
                            / params.environment.g.max(1.0e-6))
                            .atan(),
                        -ROLL_REF_LIMIT_RAD,
                        ROLL_REF_LIMIT_RAD,
                    );
                    control_state.curvature_to_roll_integrator = clamp(
                        control_state.curvature_to_roll_integrator
                            + (k_tg_y - curvature_y_est) * dt_control,
                        -ROLL_CURVATURE_INTEGRATOR_LIMIT,
                        ROLL_CURVATURE_INTEGRATOR_LIMIT,
                    );
                    control_state.curvature_to_pitch_integrator = clamp(
                        control_state.curvature_to_pitch_integrator
                            + dt_control * (k_tg_z - kite_diag.curvature_z_b)
                            - alpha_exceeded * ALPHA_INTEGRATOR_BACKOFF_GAIN,
                        -PITCH_CURVATURE_INTEGRATOR_LIMIT,
                        PITCH_CURVATURE_INTEGRATOR_LIMIT,
                    );
                    let roll_ref = clamp(
                        roll_feedforward
                            + ROLL_FROM_CURVATURE_P * (k_tg_y - curvature_y_est)
                            + ROLL_FROM_CURVATURE_I
                                * control_state.curvature_to_roll_integrator,
                        -ROLL_REF_LIMIT_RAD,
                        ROLL_REF_LIMIT_RAD,
                    );
                    let pitch_ref = clamp(
                        -PITCH_FROM_CURVATURE_P * (k_tg_z - kite_diag.curvature_z_b)
                            - PITCH_FROM_CURVATURE_I
                                * control_state.curvature_to_pitch_integrator,
                        -PITCH_REF_LIMIT_RAD,
                        PITCH_REF_LIMIT_RAD,
                    );
                    (
                        speed_target,
                        roll_ref,
                        roll_feedforward,
                        inertial_speed * k_tg_y,
                        pitch_ref,
                        k_tg_y,
                        k_tg_z,
                    )
                };

            let speed_error_for_integrator = if N_COMMON == 0 && N_UPPER == 0 {
                airspeed - speed_target
            } else {
                inertial_speed - speed_target
            };
            control_state.speed_integrator = clamp(
                control_state.speed_integrator + speed_error_for_integrator * dt_control,
                SPEED_INTEGRATOR_STATE_MIN,
                SPEED_INTEGRATOR_STATE_MAX,
            );

            let surfaces = ControlSurfaces {
                aileron: clamp(
                    params.controller.trim.surfaces.aileron
                        + AILERON_ROLL_P * (roll_ref - roll_angle)
                        - AILERON_ROLL_D * kite_diag.omega_b[0],
                    -SURFACE_LIMIT_LATERAL_RAD,
                    SURFACE_LIMIT_LATERAL_RAD,
                ),
                flap: clamp(
                    params.controller.trim.surfaces.flap + ALPHA_TO_FLAP * alpha_protection,
                    -SURFACE_LIMIT_LATERAL_RAD,
                    SURFACE_LIMIT_LATERAL_RAD,
                ),
                winglet: params.controller.trim.surfaces.winglet,
                elevator: clamp(
                    params.controller.trim.surfaces.elevator
                        - ELEVATOR_PITCH_P * (pitch_ref - pitch_angle)
                        + ELEVATOR_PITCH_D * kite_diag.omega_b[1]
                        + ALPHA_TO_ELEVATOR * alpha_protection,
                    -SURFACE_LIMIT_ELEVATOR_RAD,
                    SURFACE_LIMIT_ELEVATOR_RAD,
                ),
                rudder: clamp(
                    params.controller.trim.surfaces.rudder
                        - RUDDER_BETA_P * kite_diag.beta
                        + RUDDER_OMEGA_WORLD_Z_P * (omega_n[2] - omega_world_z_ref),
                    -SURFACE_LIMIT_LATERAL_RAD,
                    SURFACE_LIMIT_LATERAL_RAD,
                ),
            };

            let motor_speed_error = if N_COMMON == 0 && N_UPPER == 0 {
                airspeed - speed_target
            } else {
                inertial_speed - params.controller.speed_ref
            };

            let _ = (k_tg_y, k_tg_z);

            KiteControls {
                surfaces,
                motor_torque: clamp(
                    params.controller.trim.motor_torque
                        - params.controller.speed_to_torque_p * motor_speed_error
                        - params.controller.speed_to_torque_i * control_state.speed_integrator,
                    0.0,
                    MOTOR_TORQUE_MAX_NM,
                ),
            }
        }),
    };

    (
        controls,
        ControllerTrace {
            phase_errors,
            speed_targets: std::array::from_fn(|index| {
                if N_COMMON == 0 && N_UPPER == 0 {
                    free_flight_speed_reference(time, params.controller.speed_ref)
                } else {
                    speed_integrator_target(phase_errors[index], params.controller.speed_ref)
                }
            }),
            rabbit_phases,
            rabbit_radii,
            rabbit_targets_n,
            curvature_y_refs: std::array::from_fn(|index| {
                if N_COMMON == 0 && N_UPPER == 0 {
                    0.0
                } else {
                    let kite_diag = &diag.kites[index];
                    let rabbit_vector_n = sub(&rabbit_targets_n[index], &kite_diag.cad_position_n);
                    let rabbit_vector_b = rotate_nav_to_body(
                        &plant_state.kites[index].body.quat_n2b,
                        &rabbit_vector_n,
                    );
                    let x = rabbit_vector_b[0].abs().max(1.0);
                    let x2 = x * x;
                    2.0 * rabbit_vector_b[1] / x2
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
                    let rabbit_vector_n = sub(&rabbit_targets_n[index], &kite_diag.cad_position_n);
                    let rabbit_vector_b = rotate_nav_to_body(
                        &plant_state.kites[index].body.quat_n2b,
                        &rabbit_vector_n,
                    );
                    let x = rabbit_vector_b[0].abs().max(1.0);
                    let x2 = x * x;
                    let k_tg_y = 2.0 * rabbit_vector_b[1] / x2;
                    kite_diag.cad_velocity_n.norm() * k_tg_y
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
                if N_COMMON == 0 && N_UPPER == 0 {
                    free_flight_roll_reference(time)
                } else {
                    let kite_diag = &diag.kites[index];
                    let rabbit_vector_n = sub(&rabbit_targets_n[index], &kite_diag.cad_position_n);
                    let rabbit_vector_b = rotate_nav_to_body(
                        &plant_state.kites[index].body.quat_n2b,
                        &rabbit_vector_n,
                    );
                    let x = rabbit_vector_b[0].abs().max(1.0);
                    let x2 = x * x;
                    let k_tg_y = 2.0 * rabbit_vector_b[1] / x2;
                    let omega_n = rotate_body_to_nav(
                        &plant_state.kites[index].body.quat_n2b,
                        &kite_diag.omega_b,
                    );
                    let curvature_y_est = omega_n[2] / kite_diag.cad_velocity_n.norm().max(1.0);
                    let roll_feedforward = clamp(
                        (kite_diag.cad_velocity_n.norm() * kite_diag.cad_velocity_n.norm() * k_tg_y
                            / params.environment.g.max(1.0e-6))
                            .atan(),
                        -ROLL_REF_LIMIT_RAD,
                        ROLL_REF_LIMIT_RAD,
                    );
                    clamp(
                        roll_feedforward
                            + ROLL_FROM_CURVATURE_P * (k_tg_y - curvature_y_est)
                            + ROLL_FROM_CURVATURE_I
                                * state.kites[index].curvature_to_roll_integrator,
                        -ROLL_REF_LIMIT_RAD,
                        ROLL_REF_LIMIT_RAD,
                    )
                }
            }),
            roll_feedforwards: std::array::from_fn(|index| {
                if N_COMMON == 0 && N_UPPER == 0 {
                    0.0
                } else {
                    let kite_diag = &diag.kites[index];
                    let rabbit_vector_n = sub(&rabbit_targets_n[index], &kite_diag.cad_position_n);
                    let rabbit_vector_b = rotate_nav_to_body(
                        &plant_state.kites[index].body.quat_n2b,
                        &rabbit_vector_n,
                    );
                    let x = rabbit_vector_b[0].abs().max(1.0);
                    let x2 = x * x;
                    let k_tg_y = 2.0 * rabbit_vector_b[1] / x2;
                    clamp(
                        (kite_diag.cad_velocity_n.norm() * kite_diag.cad_velocity_n.norm() * k_tg_y
                            / params.environment.g.max(1.0e-6))
                            .atan(),
                        -ROLL_REF_LIMIT_RAD,
                        ROLL_REF_LIMIT_RAD,
                    )
                }
            }),
            pitch_refs: std::array::from_fn(|index| {
                if N_COMMON == 0 && N_UPPER == 0 {
                    free_flight_pitch_reference(time)
                } else {
                    let kite_diag = &diag.kites[index];
                    let rabbit_vector_n = sub(&rabbit_targets_n[index], &kite_diag.cad_position_n);
                    let rabbit_vector_b = rotate_nav_to_body(
                        &plant_state.kites[index].body.quat_n2b,
                        &rabbit_vector_n,
                    );
                    let x = rabbit_vector_b[0].abs().max(1.0);
                    let x2 = x * x;
                    let k_tg_z = 2.0 * rabbit_vector_b[2] / x2;
                    clamp(
                        -PITCH_FROM_CURVATURE_P * (k_tg_z - kite_diag.curvature_z_b)
                            - PITCH_FROM_CURVATURE_I
                                * state.kites[index].curvature_to_pitch_integrator,
                        -PITCH_REF_LIMIT_RAD,
                        PITCH_REF_LIMIT_RAD,
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
                    let x = rabbit_vector_b[0].abs().max(1.0);
                    let x2 = x * x;
                    2.0 * rabbit_vector_b[2] / x2
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
