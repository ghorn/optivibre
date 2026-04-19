use crate::math::{circular_mean, clamp, rotate_nav_to_body, scale, sub, wrap_angle};
use crate::types::{
    ControlSurfaces, Controls, Diagnostics, KiteControls, Params, PhaseMode, State,
};
use nalgebra::Vector3;

const SPEED_INTEGRATOR_BASE_TARGET_MPS: f64 = 28.0;
const SPEED_INTEGRATOR_PHASE_GAIN: f64 = 100.0;
const SPEED_INTEGRATOR_MIN_MPS: f64 = 15.0;
const SPEED_INTEGRATOR_MAX_MPS: f64 = 35.0;
const SPEED_INTEGRATOR_STATE_MIN: f64 = 0.0;
const SPEED_INTEGRATOR_STATE_MAX: f64 = 4.0;
const ALPHA_BACKOFF_MAX_RAD: f64 = 0.15;
const ALPHA_INTEGRATOR_BACKOFF_GAIN: f64 = 0.5;
const SURFACE_LIMIT_LATERAL_RAD: f64 = 15.0_f64.to_radians();
const SURFACE_LIMIT_ELEVATOR_RAD: f64 = 20.0_f64.to_radians();
const MOTOR_TORQUE_MAX_NM: f64 = 8.0;

#[derive(Clone, Debug)]
struct KiteControllerState {
    rabbit_lag_n: Vector3<f64>,
    rabbit_lag2_n: Vector3<f64>,
    speed_integrator: f64,
    int_k_y: f64,
    int_k_z: f64,
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
    pub curvature_z_refs: [f64; NK],
}

impl<const NK: usize> ControllerState<NK> {
    pub fn new(initial_diag: &Diagnostics<f64, NK>) -> Self {
        Self {
            kites: std::array::from_fn(|_| KiteControllerState {
                rabbit_lag_n: Vector3::zeros(),
                rabbit_lag2_n: Vector3::zeros(),
                speed_integrator: 0.0,
                int_k_y: 0.0,
                int_k_z: 0.0,
            }),
            initial_phase: std::array::from_fn(|index| initial_diag.kites[index].phase_angle),
        }
    }
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

fn speed_integrator_target(phase_error: f64) -> f64 {
    clamp(
        SPEED_INTEGRATOR_BASE_TARGET_MPS - SPEED_INTEGRATOR_PHASE_GAIN * phase_error,
        SPEED_INTEGRATOR_MIN_MPS,
        SPEED_INTEGRATOR_MAX_MPS,
    )
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
            let lag_alpha = clamp(dt_control / 0.1, 0.0, 1.0);
            let lag2_alpha = clamp(dt_control / 0.2, 0.0, 1.0);
            control_state.rabbit_lag_n = rabbit_targets_n[index] * lag_alpha
                + control_state.rabbit_lag_n * (1.0 - lag_alpha);
            control_state.rabbit_lag2_n = rabbit_targets_n[index] * lag2_alpha
                + control_state.rabbit_lag2_n * (1.0 - lag2_alpha);

            let speed_target = speed_integrator_target(phase_errors[index]);
            let inertial_speed = kite_diag.cad_velocity_n.norm();
            control_state.speed_integrator = clamp(
                control_state.speed_integrator + (inertial_speed - speed_target) * dt_control,
                SPEED_INTEGRATOR_STATE_MIN,
                SPEED_INTEGRATOR_STATE_MAX,
            );

            let rabbit_vector_n = sub(&rabbit_targets_n[index], &kite_diag.cad_position_n);
            let rabbit_vector_b =
                rotate_nav_to_body(&plant_state.kites[index].body.quat_n2b, &rabbit_vector_n);
            let x = rabbit_vector_b[0].abs().max(1.0);
            let x2 = x * x;
            let k_tg_y = 2.0 * rabbit_vector_b[1] / x2;
            let k_tg_z = 2.0 * rabbit_vector_b[2] / x2;

            let alpha_exceeded = clamp(kite_diag.alpha, 0.0, ALPHA_BACKOFF_MAX_RAD);
            control_state.int_k_y = clamp(
                control_state.int_k_y + dt_control * (kite_diag.curvature_y_b - k_tg_y)
                    - alpha_exceeded * ALPHA_INTEGRATOR_BACKOFF_GAIN,
                -0.1 / params.controller.gain_int_y.max(1.0e-6),
                0.1 / params.controller.gain_int_y.max(1.0e-6),
            );
            control_state.int_k_z = clamp(
                control_state.int_k_z + dt_control * (kite_diag.curvature_z_b - k_tg_z)
                    - alpha_exceeded * ALPHA_INTEGRATOR_BACKOFF_GAIN,
                -0.05 / params.controller.gain_int_z.max(1.0e-6),
                0.05 / params.controller.gain_int_z.max(1.0e-6),
            );

            let lag_chord = (rabbit_targets_n[index] - control_state.rabbit_lag2_n).norm() + 1.0e-9;
            let lag_center = (rabbit_targets_n[index] + control_state.rabbit_lag2_n) * 0.5
                - control_state.rabbit_lag_n;
            let rabbit_k_n = scale(&lag_center, lag_center.norm() / lag_chord.max(1.0e-6));
            let rabbit_k_b =
                rotate_nav_to_body(&plant_state.kites[index].body.quat_n2b, &rabbit_k_n);

            let surfaces = ControlSurfaces {
                aileron: clamp(
                    params.controller.trim.surfaces.aileron
                        - 2.0 * clamp(k_tg_y, -0.1, 0.1)
                        - 0.25 * rabbit_k_b[1]
                        + kite_diag.omega_b[0] * params.controller.wx_to_ail,
                    -SURFACE_LIMIT_LATERAL_RAD,
                    SURFACE_LIMIT_LATERAL_RAD,
                ),
                flap: clamp(
                    params.controller.trim.surfaces.flap + 0.3 * (kite_diag.curvature_z_b - k_tg_z)
                        - params.controller.gain_int_z * control_state.int_k_z,
                    -SURFACE_LIMIT_LATERAL_RAD,
                    SURFACE_LIMIT_LATERAL_RAD,
                ),
                winglet: params.controller.trim.surfaces.winglet,
                elevator: clamp(
                    params.controller.trim.surfaces.elevator
                        - (kite_diag.curvature_z_b - k_tg_z)
                        - params.controller.gain_int_z * control_state.int_k_z
                        + kite_diag.omega_b[1] * params.controller.wy_to_elev,
                    -SURFACE_LIMIT_ELEVATOR_RAD,
                    SURFACE_LIMIT_ELEVATOR_RAD,
                ),
                rudder: clamp(
                    params.controller.trim.surfaces.rudder
                        + 2.0 * (kite_diag.curvature_y_b - k_tg_y)
                        + params.controller.gain_int_y * control_state.int_k_y
                        + kite_diag.omega_b[2] * params.controller.wz_to_rudder,
                    -SURFACE_LIMIT_LATERAL_RAD,
                    SURFACE_LIMIT_LATERAL_RAD,
                ),
            };

            KiteControls {
                surfaces,
                motor_torque: clamp(
                    params.controller.trim.motor_torque
                        - params.controller.speed_to_torque_p
                            * (inertial_speed - params.controller.speed_ref)
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
                speed_integrator_target(phase_errors[index])
            }),
            rabbit_phases,
            rabbit_radii,
            rabbit_targets_n,
            curvature_y_refs: std::array::from_fn(|index| {
                let kite_diag = &diag.kites[index];
                let rabbit_vector_n = sub(&rabbit_targets_n[index], &kite_diag.cad_position_n);
                let rabbit_vector_b =
                    rotate_nav_to_body(&plant_state.kites[index].body.quat_n2b, &rabbit_vector_n);
                let x = rabbit_vector_b[0].abs().max(1.0);
                let x2 = x * x;
                2.0 * rabbit_vector_b[1] / x2
            }),
            curvature_z_refs: std::array::from_fn(|index| {
                let kite_diag = &diag.kites[index];
                let rabbit_vector_n = sub(&rabbit_targets_n[index], &kite_diag.cad_position_n);
                let rabbit_vector_b =
                    rotate_nav_to_body(&plant_state.kites[index].body.quat_n2b, &rabbit_vector_n);
                let x = rabbit_vector_b[0].abs().max(1.0);
                let x2 = x * x;
                2.0 * rabbit_vector_b[2] / x2
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
        diagnostics.kites[index].curvature_z_ref = trace.curvature_z_refs[index];
    }
}
