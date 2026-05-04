use crate::assets::{asset_manifest, reference_export};
use crate::controller::{ControllerState, apply_trace, controller_step};
use crate::math::{roll_yaw_quaternion_n2b, rotate_nav_to_body, yaw_quaternion_n2b, zero_if_nan};
use crate::model::{CompiledRhs, compute_bridle_node, compute_tether_link_tensions};
use crate::turbulence::DrydenField;
use crate::types::{
    AeroParams, BodyState, BridleParams, ControlSurfaces, ControllerGains, Controls,
    DEFAULT_SWARM_DISK_ALTITUDE_M, DEFAULT_SWARM_KITES, Diagnostics, Environment, InitRequest,
    KiteControls, KiteParams, KiteState, MAX_SWARM_KITES, MIN_SWARM_KITES, MassContactParams,
    Params, Preset, PresetInfo, RotorParams, RunResult, RunSummary, SimulationConfig,
    SimulationFailure, SimulationFrame, SimulationProgress, State, TetherNode, TetherParams,
};
use anyhow::{Result, bail};
use nalgebra::Vector3;
use optimization::{flatten_value, unflatten_value};
use std::array::from_fn;

const BASE_TETHER_NODES: usize = 5;
const TETHER_NODE_MULTIPLIER: usize = 3;
pub const COMMON_NODES: usize = BASE_TETHER_NODES * TETHER_NODE_MULTIPLIER;
pub const UPPER_NODES: usize = BASE_TETHER_NODES * TETHER_NODE_MULTIPLIER;
pub const FREE_COMMON_NODES: usize = 0;
pub const FREE_UPPER_NODES: usize = 0;
const SWARM_INITIAL_TENSION_N: f64 = 0.0;
const GROUND_CLAMP_CLEARANCE_M: f64 = 1.0e-3;

fn vec3(values: [f64; 3]) -> Vector3<f64> {
    Vector3::new(values[0], values[1], values[2])
}

fn base_params<const NK: usize>(init: &InitRequest) -> Result<Params<f64, NK>> {
    let _manifest = asset_manifest()?;
    let export = reference_export()?;
    let payload_mass = init.payload_mass_kg.unwrap_or(100.0);
    let wind_speed = init.wind_speed_mps.unwrap_or(export.environment.wind_n[0]);
    let upper = TetherParams {
        natural_length: export.tethers.upper.natural_length,
        total_mass: export.tethers.upper.total_mass,
        ea: export.tethers.upper.ea,
        viscous_damping_coeff: export.tethers.upper.viscous_damping_coeff,
        cd_phi: export.tethers.upper.cd_phi,
        diameter: export.tethers.upper.diameter,
        contact: MassContactParams {
            zeta: export.tethers.upper.contact.zeta,
            enable_length: export.tethers.upper.contact.enable_length,
            ground_altitude: export.tethers.upper.contact.ground_altitude,
        },
    };
    let common = TetherParams {
        natural_length: export.tethers.common.natural_length,
        total_mass: export.tethers.common.total_mass,
        ea: export.tethers.common.ea,
        viscous_damping_coeff: export.tethers.common.viscous_damping_coeff,
        cd_phi: export.tethers.common.cd_phi,
        diameter: export.tethers.common.diameter,
        contact: MassContactParams {
            zeta: export.tethers.common.contact.zeta,
            enable_length: export.tethers.common.contact.enable_length,
            ground_altitude: export.tethers.common.contact.ground_altitude,
        },
    };
    let trim = KiteControls {
        surfaces: ControlSurfaces {
            aileron: export.controller.trim_aileron,
            flap: export.controller.trim_flap,
            winglet: export.controller.trim_winglet,
            elevator: export.controller.trim_elevator,
            rudder: export.controller.trim_rudder,
        },
        motor_torque: export.controller.trim_motor_torque,
    };
    let speed_ref = export.controller.speed_ref;
    let kite = KiteParams {
        rigid_body: crate::types::RigidBodyParams {
            mass: export.rigid_body.mass,
            inertia_diagonal: vec3(export.rigid_body.inertia_diagonal),
            cad_offset_b: vec3(export.rigid_body.cad_offset_b),
        },
        aero: AeroParams {
            ref_area: export.aero.ref_area,
            ref_span: export.aero.ref_span,
            ref_chord: export.aero.ref_chord,
            cl0: export.aero.cl0,
            cl_alpha: export.aero.cl_alpha,
            cl_elevator: export.aero.cl_elevator,
            cl_flap: export.aero.cl_flap,
            cd0: export.aero.cd0,
            cd_induced: export.aero.cd_induced,
            cd_surface_abs: export.aero.cd_surface_abs,
            cy_beta: export.aero.cy_beta,
            cy_rudder: export.aero.cy_rudder,
            roll_beta: export.aero.roll_beta,
            roll_p: export.aero.roll_p,
            roll_r: export.aero.roll_r,
            roll_aileron: export.aero.roll_aileron,
            pitch0: export.aero.pitch0,
            pitch_alpha: export.aero.pitch_alpha,
            pitch_q: export.aero.pitch_q,
            pitch_elevator: export.aero.pitch_elevator,
            pitch_flap: export.aero.pitch_flap,
            yaw_beta: export.aero.yaw_beta,
            yaw_p: export.aero.yaw_p,
            yaw_r: export.aero.yaw_r,
            yaw_rudder: export.aero.yaw_rudder,
        },
        bridle: BridleParams {
            pivot_b: vec3(export.bridle.pivot_b),
            radius: export.bridle.radius,
        },
        tether: upper.clone(),
        rotor: RotorParams {
            axis_b: vec3(export.rotor.axis_b),
            position_b: vec3(export.rotor.position_b),
            radius: export.rotor.radius,
            inertia: export.rotor.inertia,
            sign: export.rotor.sign,
            initial_speed: export.rotor.initial_speed,
        },
    };
    let mut params = Params {
        kites: from_fn(|_| kite.clone()),
        common_tether: common.clone(),
        splitter_mass: 0.1,
        payload_mass,
        environment: Environment {
            rho: export.environment.rho,
            g: export.environment.g,
            wind_n: Vector3::new(wind_speed, 0.0, 0.0),
        },
        kite_gusts_n: from_fn(|_| Vector3::zeros()),
        controller: ControllerGains {
            trim,
            wx_to_ail: export.controller.wx_to_ail,
            wy_to_elev: export.controller.wy_to_elev,
            wz_to_rudder: export.controller.wz_to_rudder,
            speed_to_torque_p: export.controller.speed_to_torque_p,
            speed_to_torque_i: export.controller.speed_to_torque_i,
            rabbit_distance: export.controller.rabbit_distance,
            phase_lag_to_radius: export.controller.phase_lag_to_radius,
            vert_vel_to_rabbit_height: export.controller.vert_vel_to_rabbit_height,
            gain_int_y: export.controller.gain_int_y,
            gain_int_z: export.controller.gain_int_z,
            speed_ref,
            disk_center_n: vec3(export.controller.disk_center_n),
            disk_radius: export.controller.disk_radius,
            tuning: Default::default(),
        },
    };
    if is_swarm_preset(init.preset) {
        apply_swarm_controller_overrides(&mut params, init);
    }
    Ok(params)
}

fn apply_swarm_controller_overrides<const NK: usize>(
    params: &mut Params<f64, NK>,
    init: &InitRequest,
) {
    params.controller.rabbit_distance =
        env_tuning_value("MULTIKITE_SWARM_RABBIT_DISTANCE_M", 90.0).max(1.0);
    params.controller.phase_lag_to_radius =
        env_tuning_value("MULTIKITE_SWARM_PHASE_LAG_TO_RADIUS", -2.0);
    params.controller.speed_ref =
        env_tuning_value("MULTIKITE_SWARM_SPEED_REF_MPS", params.controller.speed_ref)
            .clamp(5.0, 80.0);
    apply_swarm_geometry_overrides(params, init);
}

fn apply_swarm_geometry_overrides<const NK: usize>(
    params: &mut Params<f64, NK>,
    init: &InitRequest,
) {
    let common_length = positive_request_value(
        init.swarm_common_tether_length_m,
        params.common_tether.natural_length,
    );
    let upper_length = positive_request_value(
        init.swarm_upper_tether_length_m,
        params.kites[0].tether.natural_length,
    );
    params.common_tether.natural_length = common_length;
    for kite in &mut params.kites {
        kite.tether.natural_length = upper_length;
    }

    let requested_disk_radius = init.swarm_disk_radius_m.and_then(finite_positive);
    let disk_radius = requested_disk_radius
        .unwrap_or_else(|| {
            env_tuning_value(
                "MULTIKITE_SWARM_DISK_RADIUS_M",
                params.controller.disk_radius,
            )
        })
        .max(1.0);
    let disk_altitude =
        nonnegative_request_value(init.swarm_disk_altitude_m, default_swarm_disk_altitude_m());
    let ground_altitude = params.kites[0].tether.contact.ground_altitude;

    params.controller.disk_radius = disk_radius;
    params.controller.disk_center_n[2] = -(ground_altitude + disk_altitude);
}

fn swarm_upper_vertical(upper_length: f64, disk_radius: f64) -> f64 {
    (upper_length * upper_length - disk_radius * disk_radius)
        .max(0.0)
        .sqrt()
}

fn default_swarm_disk_altitude_m() -> f64 {
    env_tuning_value(
        "MULTIKITE_SWARM_DISK_ALTITUDE_M",
        DEFAULT_SWARM_DISK_ALTITUDE_M,
    )
    .max(0.0)
}

fn swarm_payload_altitude_from_disk<const NK: usize>(
    params: &Params<f64, NK>,
    disk_altitude: f64,
) -> f64 {
    let disk_radius = params.controller.disk_radius;
    let upper_length = swarm_upper_initial_length(params);
    let upper_vertical = swarm_upper_vertical(upper_length, disk_radius);
    let payload_altitude = disk_altitude
        - params.common_tether.natural_length
        - upper_vertical
        - swarm_cad_altitude_offset_from_bridle(params, disk_radius);
    payload_altitude.max(GROUND_CLAMP_CLEARANCE_M)
}

fn swarm_initial_tension_n() -> f64 {
    std::env::var("MULTIKITE_SWARM_INITIAL_TENSION_N")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(SWARM_INITIAL_TENSION_N)
        .max(0.0)
}

fn env_tuning_value(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn finite_positive(value: f64) -> Option<f64> {
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

fn positive_request_value(value: Option<f64>, default: f64) -> f64 {
    value.and_then(finite_positive).unwrap_or(default)
}

fn nonnegative_request_value(value: Option<f64>, default: f64) -> f64 {
    value
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(default)
}

fn is_swarm_preset(preset: Preset) -> bool {
    matches!(preset, Preset::Swarm)
}

fn swarm_coordinated_roll<const NK: usize>(params: &Params<f64, NK>, turn_radius: f64) -> f64 {
    let speed = swarm_initial_speed_target(params);
    (speed * speed / (params.environment.g * turn_radius.max(1.0)))
        .atan()
        .clamp(-35.0_f64.to_radians(), 35.0_f64.to_radians())
}

fn swarm_initial_speed_target<const NK: usize>(params: &Params<f64, NK>) -> f64 {
    let min_speed = params.controller.tuning.speed_min_mps;
    let max_speed = params.controller.tuning.speed_max_mps;
    params
        .controller
        .speed_ref
        .clamp(min_speed.min(max_speed), min_speed.max(max_speed))
}

fn swarm_cad_altitude_offset_from_bridle<const NK: usize>(
    params: &Params<f64, NK>,
    turn_radius: f64,
) -> f64 {
    let quat_n2b = roll_yaw_quaternion_n2b(
        swarm_coordinated_roll(params, turn_radius),
        std::f64::consts::FRAC_PI_2,
    );
    let pivot_n = crate::math::rotate_body_to_nav(&quat_n2b, &params.kites[0].bridle.pivot_b);
    params.kites[0].bridle.radius + pivot_n[2]
}

fn swarm_upper_initial_length<const NK: usize>(params: &Params<f64, NK>) -> f64 {
    params.kites[0].tether.natural_length
        * (1.0 + swarm_initial_tension_n() / params.kites[0].tether.ea)
}

fn apply_simulation_config_to_params<const NK: usize>(
    params: &mut Params<f64, NK>,
    config: &SimulationConfig,
    init: &InitRequest,
) {
    if !config.bridle_enabled {
        for kite in &mut params.kites {
            kite.bridle.pivot_b = -kite.rigid_body.cad_offset_b;
            kite.bridle.radius = 0.0;
        }
    }
    if is_swarm_preset(init.preset) {
        apply_swarm_controller_overrides(params, init);
    }
    params.controller.tuning = config.controller_tuning.clone();
}

fn simple_tether_params(init: &InitRequest) -> Result<Params<f64, 0>> {
    let mut params = base_params::<0>(init)?;
    let wind_speed = init.wind_speed_mps.unwrap_or(0.0);
    params.splitter_mass = 0.0;
    params.environment.wind_n = Vector3::new(wind_speed, 0.0, 0.0);
    Ok(params)
}

fn interpolate_nodes<const N: usize>(
    bottom: &TetherNode<f64>,
    top: &TetherNode<f64>,
) -> [TetherNode<f64>; N] {
    from_fn(|index| {
        let frac = (index as f64 + 0.5) / N as f64;
        TetherNode {
            pos_n: bottom.pos_n * (1.0 - frac) + top.pos_n * frac,
            vel_n: bottom.vel_n * (1.0 - frac) + top.vel_n * frac,
        }
    })
}

fn clamp_node_above_ground(node: &mut TetherNode<f64>, ground_altitude: f64) {
    let ground_z = -ground_altitude - GROUND_CLAMP_CLEARANCE_M;
    if node.pos_n[2] > ground_z {
        node.pos_n[2] = ground_z;
    }
}

fn clamp_body_above_ground(body: &mut BodyState<f64>, ground_altitude: f64) {
    let ground_z = -ground_altitude - GROUND_CLAMP_CLEARANCE_M;
    if body.pos_n[2] > ground_z {
        body.pos_n[2] = ground_z;
    }
}

fn clamp_swarm_state_above_ground<const NK: usize, const N_COMMON: usize, const N_UPPER: usize>(
    state: &mut State<f64, NK, N_COMMON, N_UPPER>,
    ground_altitude: f64,
) {
    clamp_node_above_ground(&mut state.payload, ground_altitude);
    clamp_node_above_ground(&mut state.splitter, ground_altitude);
    for node in &mut state.common_tether {
        clamp_node_above_ground(node, ground_altitude);
    }
    for kite in &mut state.kites {
        clamp_body_above_ground(&mut kite.body, ground_altitude);
        for node in &mut kite.tether {
            clamp_node_above_ground(node, ground_altitude);
        }
    }
}

fn kite_with_consistent_tether<const N_UPPER: usize>(
    mut body: BodyState<f64>,
    splitter: &TetherNode<f64>,
    params: &KiteParams<f64>,
    initial_actuators: &KiteControls<f64>,
    top_guess: TetherNode<f64>,
    use_bridle_velocity: bool,
) -> KiteState<f64, N_UPPER> {
    let mut top = top_guess.clone();
    for _ in 0..8 {
        let kite = KiteState {
            body: body.clone(),
            rotor_speed: params.rotor.initial_speed,
            actuators: initial_actuators.clone(),
            tether: interpolate_nodes::<N_UPPER>(splitter, &top),
        };
        let bridle_node = compute_bridle_node(&kite, params);
        let position_error = top_guess.pos_n - bridle_node.pos_n;
        body.pos_n += position_error;
        if position_error.norm() < 1.0e-9 {
            break;
        }
    }

    let kite = KiteState {
        body: body.clone(),
        rotor_speed: params.rotor.initial_speed,
        actuators: initial_actuators.clone(),
        tether: interpolate_nodes::<N_UPPER>(splitter, &top),
    };
    let bridle_node = compute_bridle_node(&kite, params);
    top.vel_n = if use_bridle_velocity {
        bridle_node.vel_n
    } else {
        Vector3::zeros()
    };

    KiteState {
        body,
        rotor_speed: params.rotor.initial_speed,
        actuators: initial_actuators.clone(),
        tether: interpolate_nodes::<N_UPPER>(splitter, &top),
    }
}

pub fn swarm_configuration<const NK: usize, const N_COMMON: usize, const N_UPPER: usize>(
    params: &Params<f64, NK>,
    init: &InitRequest,
) -> State<f64, NK, N_COMMON, N_UPPER> {
    let ground_altitude = params.kites[0].tether.contact.ground_altitude;
    let disk_altitude = -params.controller.disk_center_n[2] - ground_altitude;
    let aircraft_altitude =
        nonnegative_request_value(init.swarm_aircraft_altitude_m, disk_altitude);
    swarm_payload_configuration(
        params,
        swarm_payload_altitude_from_disk(params, disk_altitude),
        1.0,
        aircraft_altitude,
    )
}

fn swarm_payload_configuration<const NK: usize, const N_COMMON: usize, const N_UPPER: usize>(
    params: &Params<f64, NK>,
    payload_altitude_above_ground: f64,
    common_length_fraction: f64,
    aircraft_altitude_above_ground: f64,
) -> State<f64, NK, N_COMMON, N_UPPER> {
    swarm_payload_configuration_with_kite_cad_altitude(
        params,
        payload_altitude_above_ground,
        common_length_fraction,
        aircraft_altitude_above_ground,
    )
}

fn swarm_payload_configuration_with_kite_cad_altitude<
    const NK: usize,
    const N_COMMON: usize,
    const N_UPPER: usize,
>(
    params: &Params<f64, NK>,
    payload_altitude_above_ground: f64,
    common_length_fraction: f64,
    target_cad_altitude: f64,
) -> State<f64, NK, N_COMMON, N_UPPER> {
    let ground_altitude = params.kites[0].tether.contact.ground_altitude;
    let payload_altitude = ground_altitude + payload_altitude_above_ground;
    let payload = TetherNode {
        pos_n: Vector3::new(0.0, 0.0, -payload_altitude),
        vel_n: Vector3::zeros(),
    };
    let common_length =
        params.common_tether.natural_length * common_length_fraction.clamp(0.0, 1.0);
    let splitter_altitude = payload_altitude + common_length;
    let splitter = TetherNode {
        pos_n: Vector3::new(0.0, 0.0, -splitter_altitude),
        vel_n: Vector3::zeros(),
    };

    let upper_length = swarm_upper_initial_length(params);
    let bridle_orbit_radius = params
        .controller
        .disk_radius
        .clamp(1.0, upper_length * 0.98);
    let upper_vertical = swarm_upper_vertical(upper_length, bridle_orbit_radius);
    let mut bridle_altitude = splitter_altitude + upper_vertical;
    let turn_radius = bridle_orbit_radius.max(1.0);
    let speed_ref = swarm_initial_speed_target(params);
    let coordinated_roll = swarm_coordinated_roll(params, turn_radius);
    let omega_world_z = speed_ref / turn_radius;

    for _ in 0..8 {
        let kite: KiteState<f64, N_UPPER> = swarm_kite_state_at_phase(
            params,
            &splitter,
            0,
            0.0,
            bridle_orbit_radius,
            bridle_altitude,
            speed_ref,
            omega_world_z,
            coordinated_roll,
        );
        let altitude_error = target_cad_altitude - swarm_kite_cad_altitude(&kite, &params.kites[0]);
        bridle_altitude += altitude_error;
        if altitude_error.abs() < 1.0e-10 {
            break;
        }
    }

    let kites = from_fn(|index| {
        let theta = 2.0 * std::f64::consts::PI * index as f64 / NK as f64;
        swarm_kite_state_at_phase(
            params,
            &splitter,
            index,
            theta,
            bridle_orbit_radius,
            bridle_altitude,
            speed_ref,
            omega_world_z,
            coordinated_roll,
        )
    });

    let mut state = State {
        kites,
        splitter: splitter.clone(),
        common_tether: interpolate_nodes(&payload, &splitter),
        payload,
        total_work: 0.0,
        total_dissipated_work: 0.0,
        mechanical_energy_reference: 0.0,
    };
    clamp_swarm_state_above_ground(&mut state, ground_altitude);
    state
}

#[allow(clippy::too_many_arguments)]
fn swarm_kite_state_at_phase<const NK: usize, const N_UPPER: usize>(
    params: &Params<f64, NK>,
    splitter: &TetherNode<f64>,
    index: usize,
    theta: f64,
    bridle_orbit_radius: f64,
    bridle_altitude: f64,
    speed_ref: f64,
    omega_world_z: f64,
    coordinated_roll: f64,
) -> KiteState<f64, N_UPPER> {
    let yaw = theta + std::f64::consts::FRAC_PI_2;
    let quat_n2b = roll_yaw_quaternion_n2b(coordinated_roll, yaw);
    let omega_b = rotate_nav_to_body(&quat_n2b, &Vector3::new(0.0, 0.0, omega_world_z));
    let body_vel_b = Vector3::new(speed_ref, 0.0, 0.0)
        + rotate_nav_to_body(&quat_n2b, &params.environment.wind_n)
        - omega_b.cross(&params.kites[index].rigid_body.cad_offset_b);
    let bridle_pos_n = Vector3::new(
        params.controller.disk_center_n[0] + bridle_orbit_radius * theta.cos(),
        params.controller.disk_center_n[1] + bridle_orbit_radius * theta.sin(),
        -bridle_altitude,
    );
    let bridle_to_body_b =
        -(params.kites[index].bridle.pivot_b + params.kites[index].rigid_body.cad_offset_b);
    let body_pos_n = bridle_pos_n
        + Vector3::new(0.0, 0.0, -params.kites[index].bridle.radius)
        + crate::math::rotate_body_to_nav(&quat_n2b, &bridle_to_body_b);
    let body = BodyState {
        pos_n: body_pos_n,
        vel_b: body_vel_b,
        quat_n2b,
        omega_b,
    };
    let top = TetherNode {
        pos_n: bridle_pos_n,
        vel_n: Vector3::zeros(),
    };
    kite_with_consistent_tether(
        body,
        splitter,
        &params.kites[index],
        &params.controller.trim,
        top,
        true,
    )
}

fn swarm_kite_cad_altitude<const N_UPPER: usize>(
    kite: &KiteState<f64, N_UPPER>,
    params: &KiteParams<f64>,
) -> f64 {
    let cad_pos_n = kite.body.pos_n
        + crate::math::rotate_body_to_nav(&kite.body.quat_n2b, &params.rigid_body.cad_offset_b);
    -cad_pos_n[2] - params.tether.contact.ground_altitude
}

pub fn free_flight_configuration<const N_COMMON: usize, const N_UPPER: usize>(
    params: &Params<f64, 1>,
) -> State<f64, 1, N_COMMON, N_UPPER> {
    let body = BodyState {
        pos_n: Vector3::new(0.0, 0.0, -120.0),
        vel_b: Vector3::new(
            params.controller.speed_ref + params.environment.wind_n[0],
            0.0,
            0.0,
        ),
        quat_n2b: yaw_quaternion_n2b(0.0),
        omega_b: Vector3::zeros(),
    };
    State {
        kites: [KiteState {
            body,
            rotor_speed: params.kites[0].rotor.initial_speed,
            actuators: params.controller.trim.clone(),
            tether: std::array::from_fn(|_| TetherNode {
                pos_n: Vector3::zeros(),
                vel_n: Vector3::zeros(),
            }),
        }],
        splitter: TetherNode {
            pos_n: Vector3::zeros(),
            vel_n: Vector3::zeros(),
        },
        common_tether: std::array::from_fn(|_| TetherNode {
            pos_n: Vector3::zeros(),
            vel_n: Vector3::zeros(),
        }),
        payload: TetherNode {
            pos_n: Vector3::zeros(),
            vel_n: Vector3::zeros(),
        },
        total_work: 0.0,
        total_dissipated_work: 0.0,
        mechanical_energy_reference: 0.0,
    }
}

pub fn simple_tether_configuration<const N_COMMON: usize, const N_UPPER: usize>(
    params: &Params<f64, 0>,
) -> State<f64, 0, N_COMMON, N_UPPER> {
    let anchor = TetherNode {
        pos_n: Vector3::zeros(),
        vel_n: Vector3::zeros(),
    };
    let payload = TetherNode {
        pos_n: Vector3::new(0.0, params.common_tether.natural_length, 0.0),
        vel_n: Vector3::zeros(),
    };
    State {
        kites: [],
        splitter: anchor.clone(),
        common_tether: interpolate_nodes(&payload, &anchor),
        payload,
        total_work: 0.0,
        total_dissipated_work: 0.0,
        mechanical_energy_reference: 0.0,
    }
}

fn combine_state<const NK: usize, const N_COMMON: usize, const N_UPPER: usize>(
    base: &State<f64, NK, N_COMMON, N_UPPER>,
    increments: &[(&State<f64, NK, N_COMMON, N_UPPER>, f64)],
) -> Result<State<f64, NK, N_COMMON, N_UPPER>> {
    let mut flat = flatten_value(base);
    for (state, coeff) in increments {
        let delta = flatten_value(*state);
        for (value, increment) in flat.iter_mut().zip(delta.iter()) {
            *value += coeff * increment;
        }
    }
    let mut out = unflatten_value::<State<f64, NK, N_COMMON, N_UPPER>, f64>(&flat)?;
    out.renormalize_attitudes();
    Ok(out)
}

fn error_norm(current: &[f64], proposal: &[f64], error: &[f64], abs_tol: f64, rel_tol: f64) -> f64 {
    let sum = current
        .iter()
        .zip(proposal.iter())
        .zip(error.iter())
        .map(|((current_value, proposed_value), error_value)| {
            let scale = abs_tol + rel_tol * current_value.abs().max(proposed_value.abs());
            let ratio = error_value / scale.max(1.0e-12);
            ratio * ratio
        })
        .sum::<f64>();
    (sum / error.len() as f64).sqrt()
}

fn integrate_interval<const NK: usize, const N_COMMON: usize, const N_UPPER: usize>(
    rhs: &CompiledRhs<NK, N_COMMON, N_UPPER>,
    start: &State<f64, NK, N_COMMON, N_UPPER>,
    controls: &Controls<f64, NK>,
    params: &Params<f64, NK>,
    dt: f64,
    config: &SimulationConfig,
) -> Result<(
    State<f64, NK, N_COMMON, N_UPPER>,
    Diagnostics<f64, NK>,
    usize,
    usize,
    usize,
)> {
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut state = start.clone();
    let mut h = (dt / 8.0).max(1.0e-4).min(config.dt_control.max(1.0e-3));
    let mut elapsed = 0.0;
    let mut substeps = 0usize;
    while elapsed < dt - 1.0e-12 {
        if substeps >= config.max_substeps {
            bail!("rk45 exceeded max_substeps={}", config.max_substeps);
        }
        substeps += 1;
        h = h.min(dt - elapsed);
        let (k1, _) = rhs.eval(&state, controls, params)?;
        let s2 = combine_state(&state, &[(&k1, h * (1.0 / 5.0))])?;
        let (k2, _) = rhs.eval(&s2, controls, params)?;
        let s3 = combine_state(&state, &[(&k1, h * (3.0 / 40.0)), (&k2, h * (9.0 / 40.0))])?;
        let (k3, _) = rhs.eval(&s3, controls, params)?;
        let s4 = combine_state(
            &state,
            &[
                (&k1, h * (44.0 / 45.0)),
                (&k2, h * (-56.0 / 15.0)),
                (&k3, h * (32.0 / 9.0)),
            ],
        )?;
        let (k4, _) = rhs.eval(&s4, controls, params)?;
        let s5 = combine_state(
            &state,
            &[
                (&k1, h * (19372.0 / 6561.0)),
                (&k2, h * (-25360.0 / 2187.0)),
                (&k3, h * (64448.0 / 6561.0)),
                (&k4, h * (-212.0 / 729.0)),
            ],
        )?;
        let (k5, _) = rhs.eval(&s5, controls, params)?;
        let s6 = combine_state(
            &state,
            &[
                (&k1, h * (9017.0 / 3168.0)),
                (&k2, h * (-355.0 / 33.0)),
                (&k3, h * (46732.0 / 5247.0)),
                (&k4, h * (49.0 / 176.0)),
                (&k5, h * (-5103.0 / 18656.0)),
            ],
        )?;
        let (k6, _) = rhs.eval(&s6, controls, params)?;
        let proposal = combine_state(
            &state,
            &[
                (&k1, h * (35.0 / 384.0)),
                (&k3, h * (500.0 / 1113.0)),
                (&k4, h * (125.0 / 192.0)),
                (&k5, h * (-2187.0 / 6784.0)),
                (&k6, h * (11.0 / 84.0)),
            ],
        )?;
        let (k7, _) = rhs.eval(&proposal, controls, params)?;
        let embedded = combine_state(
            &state,
            &[
                (&k1, h * (5179.0 / 57600.0)),
                (&k3, h * (7571.0 / 16695.0)),
                (&k4, h * (393.0 / 640.0)),
                (&k5, h * (-92097.0 / 339200.0)),
                (&k6, h * (187.0 / 2100.0)),
                (&k7, h * (1.0 / 40.0)),
            ],
        )?;
        let current_flat = flatten_value(&state);
        let proposal_flat = flatten_value(&proposal);
        let embedded_flat = flatten_value(&embedded);
        let error_vec = proposal_flat
            .iter()
            .zip(embedded_flat.iter())
            .map(|(high, low)| high - low)
            .collect::<Vec<_>>();
        let norm = error_norm(
            &current_flat,
            &proposal_flat,
            &error_vec,
            config.rk_abs_tol,
            config.rk_rel_tol,
        );
        if norm <= 1.0 {
            accepted += 1;
            state = proposal;
            elapsed += h;
            let growth = if norm < 1.0e-12 {
                2.0
            } else {
                (0.9 * norm.powf(-0.2)).clamp(0.2, 5.0)
            };
            h *= growth;
        } else {
            rejected += 1;
            h *= (0.9 * norm.powf(-0.25)).clamp(0.1, 0.5);
        }
    }
    let (_, diagnostics) = rhs.eval(&state, controls, params)?;
    Ok((state, diagnostics, accepted, rejected, substeps))
}

fn initial_controls<const NK: usize>(_params: &Params<f64, NK>) -> Controls<f64, NK> {
    Controls {
        kites: from_fn(|_| KiteControls::zero()),
    }
}

fn frame_tether_tensions<const NK: usize, const N_COMMON: usize, const N_UPPER: usize>(
    state: &State<f64, NK, N_COMMON, N_UPPER>,
    params: &Params<f64, NK>,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let common_tensions = compute_tether_link_tensions(
        &state.payload,
        &state.splitter,
        &params.common_tether,
        &state.common_tether,
    );
    let upper_tensions = state
        .kites
        .iter()
        .enumerate()
        .map(|(index, kite)| {
            let bridle_node = compute_bridle_node(kite, &params.kites[index]);
            compute_tether_link_tensions(
                &state.splitter,
                &bridle_node,
                &params.kites[index].tether,
                &kite.tether,
            )
        })
        .collect();
    (common_tensions, upper_tensions)
}

fn finalize_summary<const NK: usize, const N_COMMON: usize, const N_UPPER: usize>(
    duration: f64,
    accepted_steps: usize,
    rejected_steps: usize,
    frames: &[SimulationFrame<f64, NK, N_COMMON, N_UPPER>],
    failure: Option<SimulationFailure>,
) -> RunSummary {
    let final_frame = frames
        .last()
        .expect("frames should contain at least one sample");
    let max_phase_error = frames
        .iter()
        .flat_map(|frame| {
            frame
                .diagnostics
                .kites
                .iter()
                .map(|kite| kite.phase_error.abs())
        })
        .fold(0.0_f64, f64::max);
    RunSummary {
        duration,
        accepted_steps,
        rejected_steps,
        max_phase_error,
        final_total_work: final_frame.state.total_work,
        final_total_dissipated_work: final_frame.state.total_dissipated_work,
        final_total_kinetic_energy: final_frame.diagnostics.total_kinetic_energy,
        final_total_potential_energy: final_frame.diagnostics.total_potential_energy,
        final_total_tether_strain_energy: final_frame.diagnostics.total_tether_strain_energy,
        final_total_mechanical_energy: final_frame.diagnostics.total_mechanical_energy,
        final_energy_conservation_residual: final_frame.diagnostics.energy_conservation_residual,
        failure,
    }
}

fn detect_failure<const NK: usize>(
    diagnostics: &Diagnostics<f64, NK>,
    time: f64,
) -> Option<SimulationFailure> {
    const AOA_MIN_DEG: f64 = -15.0;
    const AOA_MAX_DEG: f64 = 20.0;
    const AOS_MIN_DEG: f64 = -30.0;
    const AOS_MAX_DEG: f64 = 30.0;

    diagnostics.kites.iter().enumerate().find_map(|(kite_index, kite)| {
        let alpha_deg = kite.alpha.to_degrees();
        let beta_deg = kite.beta.to_degrees();
        if !(AOA_MIN_DEG..=AOA_MAX_DEG).contains(&alpha_deg) {
            return Some(SimulationFailure {
                time,
                kite_index,
                quantity: "angle of attack".to_string(),
                value_deg: alpha_deg,
                lower_limit_deg: AOA_MIN_DEG,
                upper_limit_deg: AOA_MAX_DEG,
                alpha_deg,
                beta_deg,
                message: format!(
                    "Kite {} exceeded angle of attack: {:.2} deg outside [{:.1}, {:.1}] deg. Beta was {:.2} deg with limits [{:.1}, {:.1}] deg.",
                    kite_index + 1,
                    alpha_deg,
                    AOA_MIN_DEG,
                    AOA_MAX_DEG,
                    beta_deg,
                    AOS_MIN_DEG,
                    AOS_MAX_DEG,
                ),
            });
        }
        if !(AOS_MIN_DEG..=AOS_MAX_DEG).contains(&beta_deg) {
            return Some(SimulationFailure {
                time,
                kite_index,
                quantity: "angle of sideslip".to_string(),
                value_deg: beta_deg,
                lower_limit_deg: AOS_MIN_DEG,
                upper_limit_deg: AOS_MAX_DEG,
                alpha_deg,
                beta_deg,
                message: format!(
                    "Kite {} exceeded angle of sideslip: {:.2} deg outside [{:.1}, {:.1}] deg. Alpha was {:.2} deg with limits [{:.1}, {:.1}] deg.",
                    kite_index + 1,
                    beta_deg,
                    AOS_MIN_DEG,
                    AOS_MAX_DEG,
                    alpha_deg,
                    AOA_MIN_DEG,
                    AOA_MAX_DEG,
                ),
            });
        }
        None
    })
}

fn emit_progress<F: FnMut(SimulationProgress)>(
    progress_cb: &mut F,
    config: &SimulationConfig,
    iteration: usize,
    time: f64,
    interval_dt: f64,
    sample_count: usize,
    accepted_steps_total: usize,
    rejected_steps_total: usize,
    accepted_steps_interval: usize,
    rejected_steps_interval: usize,
    substeps_interval: usize,
) {
    progress_cb(SimulationProgress {
        iteration,
        time,
        duration: config.duration,
        interval_dt,
        sample_count,
        accepted_steps_total,
        rejected_steps_total,
        accepted_steps_interval,
        rejected_steps_interval,
        substeps_interval,
        substep_budget: config.max_substeps,
    });
}

fn simulate<
    const NK: usize,
    const N_COMMON: usize,
    const N_UPPER: usize,
    P: FnMut(SimulationProgress),
    G: FnMut(SimulationFrame<f64, NK, N_COMMON, N_UPPER>),
>(
    init: &InitRequest,
    config: &SimulationConfig,
    initializer: impl Fn(&Params<f64, NK>) -> State<f64, NK, N_COMMON, N_UPPER>,
    progress_cb: &mut P,
    frame_cb: &mut G,
) -> Result<RunResult<NK, N_COMMON, N_UPPER>> {
    let mut params = base_params::<NK>(init)?;
    apply_simulation_config_to_params(&mut params, config, init);
    let mut dryden_config = config
        .dryden
        .finite_or_default(&crate::types::DrydenConfig::default());
    dryden_config.seed ^= NK as u64;
    let mut dryden = DrydenField::<NK>::new(dryden_config);
    params.kite_gusts_n = if config.sim_noise_enabled {
        dryden.gusts_n()
    } else {
        from_fn(|_| Vector3::zeros())
    };
    let rhs = CompiledRhs::<NK, N_COMMON, N_UPPER>::shared()?;
    let mut state = initializer(&params);
    let mut controls = initial_controls(&params);
    let (_, initial_diag) = rhs.eval(&state, &controls, &params)?;
    state.mechanical_energy_reference = initial_diag.total_mechanical_energy;
    let mut controller_state = ControllerState::<NK>::new(&initial_diag);
    let mut frames = Vec::new();
    let mut time = 0.0_f64;
    let mut iteration = 0usize;
    let mut accepted_steps = 0usize;
    let mut rejected_steps = 0usize;
    let mut failure = None;
    let mut accepted_steps_interval = 0usize;
    let mut rejected_steps_interval = 0usize;
    let mut substeps_interval = 0usize;
    let mut interval_dt = 0.0_f64;

    loop {
        let (_, diagnostics_for_controller) = rhs.eval(&state, &controls, &params)?;
        let (next_controls, trace) = controller_step(
            &mut controller_state,
            &state,
            &diagnostics_for_controller,
            &params,
            config.dt_control,
            config.phase_mode,
            config.longitudinal_mode,
            time,
        );
        let (_, mut diagnostics) = rhs.eval(&state, &next_controls, &params)?;
        apply_trace(&mut diagnostics, &trace);
        let step_failure = detect_failure(&diagnostics, time);
        if iteration % config.sample_stride == 0 {
            let (common_tether_tensions, upper_tether_tensions) =
                frame_tether_tensions(&state, &params);
            let kite_bridle_positions_n = from_fn(|index| {
                compute_bridle_node(&state.kites[index], &params.kites[index]).pos_n
            });
            let frame = SimulationFrame {
                time,
                state: state.clone(),
                controls: next_controls.clone(),
                diagnostics: diagnostics.clone(),
                clean_wind_n: params.environment.wind_n,
                kite_gusts_n: params.kite_gusts_n,
                kite_ref_spans: from_fn(|index| params.kites[index].aero.ref_span),
                kite_ref_chords: from_fn(|index| params.kites[index].aero.ref_chord),
                kite_ref_areas: from_fn(|index| params.kites[index].aero.ref_area),
                kite_cad_offsets_b: from_fn(|index| params.kites[index].rigid_body.cad_offset_b),
                kite_bridle_pivots_b: from_fn(|index| params.kites[index].bridle.pivot_b),
                kite_bridle_radii: from_fn(|index| params.kites[index].bridle.radius),
                kite_bridle_positions_n,
                control_ring_center_n: params.controller.disk_center_n,
                control_ring_radius: params.controller.disk_radius,
                common_tether_tensions,
                upper_tether_tensions,
            };
            frame_cb(frame.clone());
            frames.push(frame);
        }
        emit_progress(
            progress_cb,
            config,
            iteration,
            time,
            interval_dt,
            frames.len(),
            accepted_steps,
            rejected_steps,
            accepted_steps_interval,
            rejected_steps_interval,
            substeps_interval,
        );
        if let Some(limit_failure) = step_failure {
            failure = Some(limit_failure);
            break;
        }
        if time >= config.duration - 1.0e-12 {
            break;
        }
        let step = (config.duration - time).min(config.dt_control);
        let (next_state, _, accepted, rejected, substeps) =
            integrate_interval(rhs.as_ref(), &state, &next_controls, &params, step, config)?;
        if config.sim_noise_enabled {
            dryden.advance(
                step,
                &diagnostics,
                &params.environment.wind_n,
                params.common_tether.contact.ground_altitude,
            );
            params.kite_gusts_n = dryden.gusts_n();
        } else {
            params.kite_gusts_n = from_fn(|_| Vector3::zeros());
        }
        state = next_state;
        controls = next_controls;
        time = zero_if_nan(time + step);
        iteration += 1;
        accepted_steps += accepted;
        rejected_steps += rejected;
        accepted_steps_interval = accepted;
        rejected_steps_interval = rejected;
        substeps_interval = substeps;
        interval_dt = step;
    }

    let summary = finalize_summary(time, accepted_steps, rejected_steps, &frames, failure);
    Ok(RunResult { frames, summary })
}

fn simulate_passive<
    const NK: usize,
    const N_COMMON: usize,
    const N_UPPER: usize,
    P: FnMut(SimulationProgress),
    G: FnMut(SimulationFrame<f64, NK, N_COMMON, N_UPPER>),
>(
    mut params: Params<f64, NK>,
    config: &SimulationConfig,
    initializer: fn(&Params<f64, NK>) -> State<f64, NK, N_COMMON, N_UPPER>,
    progress_cb: &mut P,
    frame_cb: &mut G,
) -> Result<RunResult<NK, N_COMMON, N_UPPER>> {
    let simple_init = InitRequest {
        preset: Preset::SimpleTether,
        ..InitRequest::default()
    };
    apply_simulation_config_to_params(&mut params, config, &simple_init);
    let rhs = CompiledRhs::<NK, N_COMMON, N_UPPER>::shared()?;
    let mut state = initializer(&params);
    let controls = initial_controls(&params);
    let (_, initial_diag) = rhs.eval(&state, &controls, &params)?;
    state.mechanical_energy_reference = initial_diag.total_mechanical_energy;
    let mut frames = Vec::new();
    let mut time = 0.0_f64;
    let mut iteration = 0usize;
    let mut accepted_steps = 0usize;
    let mut rejected_steps = 0usize;
    let mut failure = None;
    let mut accepted_steps_interval = 0usize;
    let mut rejected_steps_interval = 0usize;
    let mut substeps_interval = 0usize;
    let mut interval_dt = 0.0_f64;

    loop {
        let (_, diagnostics) = rhs.eval(&state, &controls, &params)?;
        let step_failure = detect_failure(&diagnostics, time);
        if iteration % config.sample_stride == 0 {
            let (common_tether_tensions, upper_tether_tensions) =
                frame_tether_tensions(&state, &params);
            let kite_bridle_positions_n = from_fn(|index| {
                compute_bridle_node(&state.kites[index], &params.kites[index]).pos_n
            });
            let frame = SimulationFrame {
                time,
                state: state.clone(),
                controls: controls.clone(),
                diagnostics,
                clean_wind_n: params.environment.wind_n,
                kite_gusts_n: params.kite_gusts_n,
                kite_ref_spans: from_fn(|index| params.kites[index].aero.ref_span),
                kite_ref_chords: from_fn(|index| params.kites[index].aero.ref_chord),
                kite_ref_areas: from_fn(|index| params.kites[index].aero.ref_area),
                kite_cad_offsets_b: from_fn(|index| params.kites[index].rigid_body.cad_offset_b),
                kite_bridle_pivots_b: from_fn(|index| params.kites[index].bridle.pivot_b),
                kite_bridle_radii: from_fn(|index| params.kites[index].bridle.radius),
                kite_bridle_positions_n,
                control_ring_center_n: params.controller.disk_center_n,
                control_ring_radius: params.controller.disk_radius,
                common_tether_tensions,
                upper_tether_tensions,
            };
            frame_cb(frame.clone());
            frames.push(frame);
        }
        emit_progress(
            progress_cb,
            config,
            iteration,
            time,
            interval_dt,
            frames.len(),
            accepted_steps,
            rejected_steps,
            accepted_steps_interval,
            rejected_steps_interval,
            substeps_interval,
        );
        if let Some(limit_failure) = step_failure {
            failure = Some(limit_failure);
            break;
        }
        if time >= config.duration - 1.0e-12 {
            break;
        }
        let step = (config.duration - time).min(config.dt_control);
        let (next_state, _, accepted, rejected, substeps) =
            integrate_interval(rhs.as_ref(), &state, &controls, &params, step, config)?;
        state = next_state;
        time = zero_if_nan(time + step);
        iteration += 1;
        accepted_steps += accepted;
        rejected_steps += rejected;
        accepted_steps_interval = accepted;
        rejected_steps_interval = rejected;
        substeps_interval = substeps;
        interval_dt = step;
    }

    let summary = finalize_summary(time, accepted_steps, rejected_steps, &frames, failure);
    Ok(RunResult { frames, summary })
}

pub fn available_presets() -> Vec<PresetInfo> {
    vec![
        PresetInfo {
            preset: Preset::Swarm,
            name: "Swarm",
            description: "Configurable equal-phase tethered swarm with explicit disk, aircraft, and tether geometry.",
            kites: DEFAULT_SWARM_KITES,
            common_nodes: COMMON_NODES,
            upper_nodes: UPPER_NODES,
        },
        PresetInfo {
            preset: Preset::FreeFlight1,
            name: "FreeFlight1",
            description: "Single-kite free-flight bring-up harness with direct roll, pitch, and airspeed references.",
            kites: 1,
            common_nodes: FREE_COMMON_NODES,
            upper_nodes: FREE_UPPER_NODES,
        },
        PresetInfo {
            preset: Preset::SimpleTether,
            name: "SimpleTether",
            description: "Pinned single tether with configured payload mass and zero-velocity horizontal start.",
            kites: 0,
            common_nodes: COMMON_NODES,
            upper_nodes: 0,
        },
    ]
}

pub fn simulate_swarm<const NK: usize>(
    init: &InitRequest,
    config: &SimulationConfig,
) -> Result<RunResult<NK, COMMON_NODES, UPPER_NODES>> {
    let mut progress_cb = |_| {};
    let mut frame_cb = |_| {};
    simulate_swarm_with_callbacks::<NK, _, _>(init, config, &mut progress_cb, &mut frame_cb)
}

pub fn simulate_swarm_with_progress<const NK: usize, F: FnMut(SimulationProgress)>(
    init: &InitRequest,
    config: &SimulationConfig,
    progress_cb: &mut F,
) -> Result<RunResult<NK, COMMON_NODES, UPPER_NODES>> {
    let mut frame_cb = |_| {};
    simulate_swarm_with_callbacks::<NK, _, _>(init, config, progress_cb, &mut frame_cb)
}

pub fn simulate_swarm_with_callbacks<
    const NK: usize,
    P: FnMut(SimulationProgress),
    G: FnMut(SimulationFrame<f64, NK, COMMON_NODES, UPPER_NODES>),
>(
    init: &InitRequest,
    config: &SimulationConfig,
    progress_cb: &mut P,
    frame_cb: &mut G,
) -> Result<RunResult<NK, COMMON_NODES, UPPER_NODES>> {
    if !(MIN_SWARM_KITES..=MAX_SWARM_KITES).contains(&NK) {
        bail!("swarm kites must be in {MIN_SWARM_KITES}..={MAX_SWARM_KITES}, got {NK}");
    }
    simulate::<NK, COMMON_NODES, UPPER_NODES, _, _>(
        init,
        config,
        |params| swarm_configuration::<NK, COMMON_NODES, UPPER_NODES>(params, init),
        progress_cb,
        frame_cb,
    )
}

pub fn simulate_free_flight1(
    init: &InitRequest,
    config: &SimulationConfig,
) -> Result<RunResult<1, FREE_COMMON_NODES, FREE_UPPER_NODES>> {
    let mut progress_cb = |_| {};
    let mut frame_cb = |_| {};
    simulate::<1, FREE_COMMON_NODES, FREE_UPPER_NODES, _, _>(
        init,
        config,
        free_flight_configuration::<FREE_COMMON_NODES, FREE_UPPER_NODES>,
        &mut progress_cb,
        &mut frame_cb,
    )
}

pub fn simulate_free_flight1_with_progress<F: FnMut(SimulationProgress)>(
    init: &InitRequest,
    config: &SimulationConfig,
    progress_cb: &mut F,
) -> Result<RunResult<1, FREE_COMMON_NODES, FREE_UPPER_NODES>> {
    let mut frame_cb = |_| {};
    simulate::<1, FREE_COMMON_NODES, FREE_UPPER_NODES, _, _>(
        init,
        config,
        free_flight_configuration::<FREE_COMMON_NODES, FREE_UPPER_NODES>,
        progress_cb,
        &mut frame_cb,
    )
}

pub fn simulate_free_flight1_with_callbacks<
    P: FnMut(SimulationProgress),
    G: FnMut(SimulationFrame<f64, 1, FREE_COMMON_NODES, FREE_UPPER_NODES>),
>(
    init: &InitRequest,
    config: &SimulationConfig,
    progress_cb: &mut P,
    frame_cb: &mut G,
) -> Result<RunResult<1, FREE_COMMON_NODES, FREE_UPPER_NODES>> {
    simulate::<1, FREE_COMMON_NODES, FREE_UPPER_NODES, _, _>(
        init,
        config,
        free_flight_configuration::<FREE_COMMON_NODES, FREE_UPPER_NODES>,
        progress_cb,
        frame_cb,
    )
}

pub fn simulate_simple_tether(
    init: &InitRequest,
    config: &SimulationConfig,
) -> Result<RunResult<0, COMMON_NODES, UPPER_NODES>> {
    let mut progress_cb = |_| {};
    let mut frame_cb = |_| {};
    simulate_passive::<0, COMMON_NODES, UPPER_NODES, _, _>(
        simple_tether_params(init)?,
        config,
        simple_tether_configuration::<COMMON_NODES, UPPER_NODES>,
        &mut progress_cb,
        &mut frame_cb,
    )
}

pub fn simulate_simple_tether_with_progress<F: FnMut(SimulationProgress)>(
    init: &InitRequest,
    config: &SimulationConfig,
    progress_cb: &mut F,
) -> Result<RunResult<0, COMMON_NODES, UPPER_NODES>> {
    let mut frame_cb = |_| {};
    simulate_passive::<0, COMMON_NODES, UPPER_NODES, _, _>(
        simple_tether_params(init)?,
        config,
        simple_tether_configuration::<COMMON_NODES, UPPER_NODES>,
        progress_cb,
        &mut frame_cb,
    )
}

pub fn simulate_simple_tether_with_callbacks<
    P: FnMut(SimulationProgress),
    G: FnMut(SimulationFrame<f64, 0, COMMON_NODES, UPPER_NODES>),
>(
    init: &InitRequest,
    config: &SimulationConfig,
    progress_cb: &mut P,
    frame_cb: &mut G,
) -> Result<RunResult<0, COMMON_NODES, UPPER_NODES>> {
    simulate_passive::<0, COMMON_NODES, UPPER_NODES, _, _>(
        simple_tether_params(init)?,
        config,
        simple_tether_configuration::<COMMON_NODES, UPPER_NODES>,
        progress_cb,
        frame_cb,
    )
}

#[cfg(test)]
mod tests {
    use crate::controller::{
        FreeFlightReference, controller_step, controller_step_with_free_flight_reference,
    };
    use crate::math::{pitch_angle_from_quat_n2b, roll_angle_from_quat_n2b};
    use crate::model::evaluate_rhs;
    use crate::types::{LongitudinalMode, PhaseMode};

    use super::*;
    use nalgebra::UnitQuaternion;

    const TEST_DT_CONTROL: f64 = 0.02;
    const ROLL_STEP_RAD: f64 = 20.0_f64.to_radians();

    fn constrain_free_flight_to_2d(state: &mut State<f64, 1, FREE_COMMON_NODES, FREE_UPPER_NODES>) {
        let body = &mut state.kites[0].body;
        body.pos_n[1] = 0.0;
        body.vel_b[1] = 0.0;
        body.omega_b[0] = 0.0;
        body.omega_b[2] = 0.0;

        let pitch = pitch_angle_from_quat_n2b(&body.quat_n2b);
        body.quat_n2b = *UnitQuaternion::from_euler_angles(0.0, pitch, 0.0).quaternion();
        state.renormalize_attitudes();
    }

    fn suppress_lateral_controls(controls: &mut Controls<f64, 1>, params: &Params<f64, 1>) {
        controls.kites[0].surfaces.aileron = params.controller.trim.surfaces.aileron;
        controls.kites[0].surfaces.rudder = params.controller.trim.surfaces.rudder;
        controls.kites[0].surfaces.winglet = params.controller.trim.surfaces.winglet;
    }

    fn roll_reversal_reference(time: f64) -> f64 {
        if time < 1.0 {
            0.0
        } else if time < 3.0 {
            ROLL_STEP_RAD
        } else if time < 5.0 {
            -ROLL_STEP_RAD
        } else if time < 7.0 {
            ROLL_STEP_RAD
        } else {
            0.0
        }
    }

    fn beta_after_euler_step(
        mut state: State<f64, 1, FREE_COMMON_NODES, FREE_UPPER_NODES>,
        mut controls: Controls<f64, 1>,
        params: &Params<f64, 1>,
        rudder: f64,
    ) -> f64 {
        controls.kites[0].surfaces.rudder = rudder;
        state.kites[0].actuators.surfaces.rudder = rudder;
        let (xdot, _) = evaluate_rhs(&state, &controls, params);
        let dt = 1.0e-3;
        state.kites[0].body.pos_n += xdot.kites[0].body.pos_n * dt;
        state.kites[0].body.vel_b += xdot.kites[0].body.vel_b * dt;
        state.kites[0].body.quat_n2b.coords += xdot.kites[0].body.quat_n2b.coords * dt;
        state.kites[0].body.omega_b += xdot.kites[0].body.omega_b * dt;
        state.kites[0].rotor_speed += xdot.kites[0].rotor_speed * dt;
        state.kites[0].actuators.surfaces.rudder += xdot.kites[0].actuators.surfaces.rudder * dt;
        state.renormalize_attitudes();
        let (_, diag) = evaluate_rhs(&state, &controls, params);
        diag.kites[0].beta
    }

    #[test]
    fn rudder_force_and_yaw_moment_have_opposite_beta_effects() {
        let init = InitRequest {
            preset: Preset::FreeFlight1,
            payload_mass_kg: None,
            wind_speed_mps: Some(0.0),
            ..InitRequest::default()
        };
        let params = base_params::<1>(&init).expect("free-flight params");
        let mut state = free_flight_configuration::<FREE_COMMON_NODES, FREE_UPPER_NODES>(&params);
        state.kites[0].body.vel_b = Vector3::new(25.0, 25.0 * 5.0_f64.to_radians().tan(), 0.0);
        state.kites[0].body.omega_b = Vector3::zeros();
        let controls = initial_controls(&params);
        let (_, initial_diag) = evaluate_rhs(&state, &controls, &params);
        let positive = beta_after_euler_step(
            state.clone(),
            controls.clone(),
            &params,
            10.0_f64.to_radians(),
        );
        let negative = beta_after_euler_step(
            state.clone(),
            controls.clone(),
            &params,
            -10.0_f64.to_radians(),
        );
        eprintln!(
            "beta response: initial={:.6} deg positive_rudder={:.6} deg negative_rudder={:.6} deg",
            initial_diag.kites[0].beta.to_degrees(),
            positive.to_degrees(),
            negative.to_degrees()
        );
        assert!(
            negative < positive,
            "negative rudder sideforce should reduce positive beta in one tiny free-flight step"
        );

        let mut positive_state = state.clone();
        let mut positive_controls = controls.clone();
        positive_controls.kites[0].surfaces.rudder = 10.0_f64.to_radians();
        positive_state.kites[0].actuators.surfaces.rudder = 10.0_f64.to_radians();
        let (_, positive_diag) = evaluate_rhs(&positive_state, &positive_controls, &params);
        let mut negative_state = state;
        let mut negative_controls = controls;
        negative_controls.kites[0].surfaces.rudder = -10.0_f64.to_radians();
        negative_state.kites[0].actuators.surfaces.rudder = -10.0_f64.to_radians();
        let (_, negative_diag) = evaluate_rhs(&negative_state, &negative_controls, &params);
        assert!(
            positive_diag.kites[0].rudder_moment_b[2] < negative_diag.kites[0].rudder_moment_b[2],
            "positive rudder should create the negative body-yaw moment needed by the swarm beta/yaw loop"
        );
    }

    #[test]
    fn swarm_initializes_on_controller_speed_and_altitude_refs() {
        let init = InitRequest {
            preset: Preset::Swarm,
            payload_mass_kg: None,
            wind_speed_mps: None,
            swarm_kites: 2,
            ..InitRequest::default()
        };
        let params = base_params::<2>(&init).expect("swarm params");
        let state = swarm_configuration::<2, COMMON_NODES, UPPER_NODES>(&params, &init);
        let controls = initial_controls(&params);
        let rhs =
            CompiledRhs::<2, COMMON_NODES, UPPER_NODES>::shared().expect("compiled swarm rhs");
        let (_, mut diagnostics) = rhs
            .eval(&state, &controls, &params)
            .expect("initial swarm diagnostics");
        let mut controller_state = ControllerState::<2>::new(&diagnostics);
        let (_, trace) = controller_step(
            &mut controller_state,
            &state,
            &diagnostics,
            &params,
            TEST_DT_CONTROL,
            PhaseMode::Adaptive,
            LongitudinalMode::TotalEnergy,
            0.0,
        );
        apply_trace(&mut diagnostics, &trace);

        let expected_altitude =
            -params.controller.disk_center_n[2] - params.kites[0].tether.contact.ground_altitude;
        let expected_speed = swarm_initial_speed_target(&params);
        let payload_altitude =
            -state.payload.pos_n[2] - params.kites[0].tether.contact.ground_altitude;
        let expected_payload_altitude =
            swarm_payload_altitude_from_disk(&params, expected_altitude);

        assert!(
            (payload_altitude - expected_payload_altitude).abs() < 1.0e-9,
            "swarm payload altitude should be {:.3} m, got {payload_altitude:.9} m",
            expected_payload_altitude,
        );

        for (index, kite_diag) in diagnostics.kites.iter().enumerate() {
            let altitude_error = (kite_diag.altitude - kite_diag.altitude_ref).abs();
            let speed_error = (kite_diag.airspeed - kite_diag.speed_target).abs();
            let reference_altitude_error = (kite_diag.altitude_ref - expected_altitude).abs();
            let reference_speed_error = (kite_diag.speed_target - expected_speed).abs();

            assert!(
                altitude_error < 1.0e-9,
                "kite {index} initial altitude is not on reference: altitude={:.9}, ref={:.9}, error={altitude_error:e}",
                kite_diag.altitude,
                kite_diag.altitude_ref,
            );
            assert!(
                speed_error < 1.0e-6,
                "kite {index} initial airspeed is not on reference: airspeed={:.9}, ref={:.9}, error={speed_error:e}",
                kite_diag.airspeed,
                kite_diag.speed_target,
            );
            assert!(
                reference_altitude_error < 1.0e-9,
                "kite {index} initial altitude reference is not the disk-center elevation: ref={:.9}, expected={expected_altitude:.9}, error={reference_altitude_error:e}",
                kite_diag.altitude_ref,
            );
            assert!(
                reference_speed_error < 1.0e-9,
                "kite {index} initial speed reference is not the effective swarm speed target: ref={:.9}, expected={expected_speed:.9}, error={reference_speed_error:e}",
                kite_diag.speed_target,
            );
            assert!(
                kite_diag.alpha.abs().to_degrees() < 1.0e-9,
                "kite {index} expected zero initial alpha, got {:.9} deg",
                kite_diag.alpha.to_degrees(),
            );
            assert!(
                kite_diag.beta.abs().to_degrees() < 1.0e-9,
                "kite {index} expected zero initial beta, got {:.9} deg",
                kite_diag.beta.to_degrees(),
            );
        }
    }

    #[test]
    fn swarm_geometry_request_sets_tether_lengths_and_altitudes() {
        let target_init = InitRequest {
            preset: Preset::Swarm,
            payload_mass_kg: None,
            wind_speed_mps: None,
            swarm_kites: 2,
            ..InitRequest::default()
        };
        let target_params = base_params::<2>(&target_init).expect("swarm params");
        let disk_altitude = -target_params.controller.disk_center_n[2]
            - target_params.kites[0].tether.contact.ground_altitude;
        let expected_disk_altitude = default_swarm_disk_altitude_m();
        assert!(
            (disk_altitude - expected_disk_altitude).abs() < 1.0e-9,
            "control disk altitude should use the configured disk altitude, got {disk_altitude:.3} m expected {expected_disk_altitude:.3} m",
        );

        let reference_state =
            swarm_configuration::<2, COMMON_NODES, UPPER_NODES>(&target_params, &target_init);

        let payload_altitude =
            |params: &Params<f64, 2>, state: &State<f64, 2, COMMON_NODES, UPPER_NODES>| {
                -state.payload.pos_n[2] - params.kites[0].tether.contact.ground_altitude
            };
        let kite_altitude =
            |params: &Params<f64, 2>, state: &State<f64, 2, COMMON_NODES, UPPER_NODES>| {
                let cad_pos_n = state.kites[0].body.pos_n
                    + crate::math::rotate_body_to_nav(
                        &state.kites[0].body.quat_n2b,
                        &params.kites[0].rigid_body.cad_offset_b,
                    );
                -cad_pos_n[2] - params.kites[0].tether.contact.ground_altitude
            };

        assert!(
            (payload_altitude(&target_params, &reference_state)
                - swarm_payload_altitude_from_disk(&target_params, disk_altitude))
            .abs()
                < 1.0e-9,
            "default swarm payload should be derived from disk altitude and tether geometry"
        );
        assert!(
            (kite_altitude(&target_params, &reference_state) - disk_altitude).abs() < 1.0e-9,
            "default swarm kites should start on the control disk"
        );

        let custom_init = InitRequest {
            preset: Preset::Swarm,
            swarm_kites: 2,
            swarm_common_tether_length_m: Some(80.0),
            swarm_upper_tether_length_m: Some(95.0),
            swarm_disk_radius_m: Some(60.0),
            swarm_disk_altitude_m: Some(150.0),
            swarm_aircraft_altitude_m: Some(160.0),
            ..InitRequest::default()
        };
        let custom_params = base_params::<2>(&custom_init).expect("custom swarm params");
        let custom_state =
            swarm_configuration::<2, COMMON_NODES, UPPER_NODES>(&custom_params, &custom_init);
        assert!((custom_params.common_tether.natural_length - 80.0).abs() < 1.0e-12);
        assert!((custom_params.kites[0].tether.natural_length - 95.0).abs() < 1.0e-12);
        assert!((custom_params.controller.disk_radius - 60.0).abs() < 1.0e-12);
        assert!(
            (-custom_params.controller.disk_center_n[2]
                - custom_params.kites[0].tether.contact.ground_altitude
                - 150.0)
                .abs()
                < 1.0e-12
        );
        assert!(
            (payload_altitude(&custom_params, &custom_state)
                - swarm_payload_altitude_from_disk(&custom_params, 150.0))
            .abs()
                < 1.0e-9,
            "payload altitude should be derived from disk altitude and tether geometry"
        );
        assert!(
            (kite_altitude(&custom_params, &custom_state) - 160.0).abs() < 1.0e-9,
            "explicit aircraft start altitude should be honored"
        );
    }

    #[test]
    fn swarm_initialization_is_wind_aware() {
        let init = InitRequest {
            preset: Preset::Swarm,
            payload_mass_kg: None,
            wind_speed_mps: Some(5.0),
            swarm_kites: 2,
            ..InitRequest::default()
        };
        let params = base_params::<2>(&init).expect("swarm params");
        let state = swarm_configuration::<2, COMMON_NODES, UPPER_NODES>(&params, &init);
        let controls = initial_controls(&params);
        let rhs =
            CompiledRhs::<2, COMMON_NODES, UPPER_NODES>::shared().expect("compiled swarm rhs");
        let (_, diagnostics) = rhs
            .eval(&state, &controls, &params)
            .expect("initial swarm diagnostics");

        for (index, kite_diag) in diagnostics.kites.iter().enumerate() {
            assert!(
                kite_diag.alpha.abs().to_degrees() < 1.0e-9,
                "kite {index} expected wind-aware zero initial alpha, got {:.9} deg",
                kite_diag.alpha.to_degrees(),
            );
            assert!(
                kite_diag.beta.abs().to_degrees() < 1.0e-9,
                "kite {index} expected wind-aware zero initial beta, got {:.9} deg",
                kite_diag.beta.to_degrees(),
            );
        }
    }

    #[test]
    fn max_throttle_altitude_pitch_mode_commands_motor_limit_for_swarm() {
        let init = InitRequest {
            preset: Preset::Swarm,
            payload_mass_kg: None,
            wind_speed_mps: None,
            swarm_kites: 2,
            ..InitRequest::default()
        };
        let params = base_params::<2>(&init).expect("swarm params");
        let state = swarm_configuration::<2, COMMON_NODES, UPPER_NODES>(&params, &init);
        let controls = initial_controls(&params);
        let rhs =
            CompiledRhs::<2, COMMON_NODES, UPPER_NODES>::shared().expect("compiled swarm rhs");
        let (_, diagnostics) = rhs
            .eval(&state, &controls, &params)
            .expect("initial swarm diagnostics");
        let mut controller_state = ControllerState::<2>::new(&diagnostics);

        let (next_controls, trace) = controller_step(
            &mut controller_state,
            &state,
            &diagnostics,
            &params,
            TEST_DT_CONTROL,
            PhaseMode::Adaptive,
            LongitudinalMode::MaxThrottleAltitudePitch,
            0.0,
        );

        for (index, control) in next_controls.kites.iter().enumerate() {
            assert!(
                (control.motor_torque - 45.6).abs() < 1.0e-12,
                "kite {index} did not command tethered motor limit: {}",
                control.motor_torque
            );
            assert!(
                trace.kites[index].thrust_energy_integrator >= 45.6 - 1.0e-12,
                "kite {index} thrust integrator should reflect max-throttle command"
            );
        }
    }

    #[test]
    fn total_energy_controller_tracks_2d_free_flight_climb_and_descent() {
        let init = InitRequest {
            preset: Preset::FreeFlight1,
            payload_mass_kg: None,
            wind_speed_mps: None,
            ..InitRequest::default()
        };
        let params = base_params::<1>(&init).expect("free-flight params");
        let config = SimulationConfig {
            duration: 12.0,
            dt_control: TEST_DT_CONTROL,
            sample_stride: 1,
            ..SimulationConfig::default()
        };
        let rhs = CompiledRhs::<1, FREE_COMMON_NODES, FREE_UPPER_NODES>::shared()
            .expect("compiled free-flight rhs");
        let mut state = free_flight_configuration::<FREE_COMMON_NODES, FREE_UPPER_NODES>(&params);
        constrain_free_flight_to_2d(&mut state);

        let mut controls = initial_controls(&params);
        suppress_lateral_controls(&mut controls, &params);
        let (_, initial_diag) = rhs
            .eval(&state, &controls, &params)
            .expect("initial diagnostics");
        let initial_altitude = (-initial_diag.kites[0].cad_position_n[2]
            - params.kites[0].tether.contact.ground_altitude)
            .max(0.0);
        let mut controller_state = ControllerState::<1>::new(&initial_diag);
        let reference = |index: usize, time: f64, initial_altitude: f64, speed_ref: f64| {
            assert_eq!(index, 0);
            FreeFlightReference {
                speed_target: if time < 2.0 {
                    speed_ref
                } else if time < 4.0 {
                    speed_ref + 3.0
                } else {
                    speed_ref - 3.0
                },
                altitude_ref_raw: if time < 2.0 {
                    initial_altitude
                } else if time < 7.0 {
                    initial_altitude + 12.0
                } else {
                    initial_altitude - 8.0
                },
                roll_ref: 0.0,
            }
        };

        let mut time = 0.0_f64;
        let mut min_airspeed = f64::INFINITY;
        let mut max_abs_alpha = 0.0_f64;
        let mut max_abs_beta = 0.0_f64;
        let mut max_abs_y = 0.0_f64;
        let mut max_abs_roll = 0.0_f64;
        let mut peak_altitude = initial_altitude;
        let mut final_altitude = initial_altitude;

        while time <= config.duration + 1.0e-12 {
            constrain_free_flight_to_2d(&mut state);
            let (_, mut diagnostics) = rhs
                .eval(&state, &controls, &params)
                .expect("diagnostics during 2D TECS test");
            let (mut next_controls, trace) = controller_step_with_free_flight_reference(
                &mut controller_state,
                &state,
                &diagnostics,
                &params,
                config.dt_control,
                config.phase_mode,
                config.longitudinal_mode,
                time,
                reference,
            );
            suppress_lateral_controls(&mut next_controls, &params);
            apply_trace(&mut diagnostics, &trace);

            let kite_diag = &diagnostics.kites[0];
            min_airspeed = min_airspeed.min(kite_diag.airspeed);
            max_abs_alpha = max_abs_alpha.max(kite_diag.alpha.abs());
            max_abs_beta = max_abs_beta.max(kite_diag.beta.abs());
            max_abs_y = max_abs_y.max(state.kites[0].body.pos_n[1].abs());
            let roll = roll_angle_from_quat_n2b(&state.kites[0].body.quat_n2b);
            max_abs_roll = max_abs_roll.max(roll.abs());

            final_altitude = kite_diag.altitude;
            if time >= 2.0 && time <= 7.0 {
                peak_altitude = peak_altitude.max(kite_diag.altitude);
            }

            if time >= config.duration - 1.0e-12 {
                break;
            }

            let step = (config.duration - time).min(config.dt_control);
            let (next_state, _, _, _, _) =
                integrate_interval(rhs.as_ref(), &state, &next_controls, &params, step, &config)
                    .expect("2D TECS integration interval");
            state = next_state;
            constrain_free_flight_to_2d(&mut state);
            controls = next_controls;
            time = zero_if_nan(time + step);
        }

        assert!(
            peak_altitude > initial_altitude + 2.0,
            "TECS did not climb enough: initial={initial_altitude:.2} m, peak={peak_altitude:.2} m"
        );
        assert!(
            final_altitude < peak_altitude - 8.0,
            "TECS did not descend after the descent command: peak={peak_altitude:.2} m, final={final_altitude:.2} m"
        );
        assert!(
            min_airspeed > 15.0,
            "TECS let airspeed get too low: min={min_airspeed:.2} m/s"
        );
        assert!(
            max_abs_alpha.to_degrees() < 15.0,
            "TECS exceeded alpha margin in 2D free flight: max_abs_alpha={:.2} deg",
            max_abs_alpha.to_degrees()
        );
        assert!(
            max_abs_beta.to_degrees() < 1.0,
            "2D constraint leaked sideslip: max_abs_beta={:.3} deg",
            max_abs_beta.to_degrees()
        );
        assert!(
            max_abs_y < 1.0e-9,
            "2D constraint leaked lateral position: max_abs_y={max_abs_y:e} m"
        );
        assert!(
            max_abs_roll.to_degrees() < 1.0e-9,
            "2D constraint leaked roll: max_abs_roll={:.3e} deg",
            max_abs_roll.to_degrees()
        );
    }

    #[test]
    fn roll_controller_tracks_3d_free_flight_reversals_with_fixed_tecs_refs() {
        let init = InitRequest {
            preset: Preset::FreeFlight1,
            payload_mass_kg: None,
            wind_speed_mps: None,
            ..InitRequest::default()
        };
        let params = base_params::<1>(&init).expect("free-flight params");
        let config = SimulationConfig {
            duration: 8.0,
            dt_control: TEST_DT_CONTROL,
            sample_stride: 1,
            ..SimulationConfig::default()
        };
        let rhs = CompiledRhs::<1, FREE_COMMON_NODES, FREE_UPPER_NODES>::shared()
            .expect("compiled free-flight rhs");
        let mut state = free_flight_configuration::<FREE_COMMON_NODES, FREE_UPPER_NODES>(&params);
        let mut controls = initial_controls(&params);
        let (_, initial_diag) = rhs
            .eval(&state, &controls, &params)
            .expect("initial diagnostics");
        let mut controller_state = ControllerState::<1>::new(&initial_diag);

        let fixed_altitude = (-initial_diag.kites[0].cad_position_n[2]
            - params.kites[0].tether.contact.ground_altitude)
            .max(0.0);
        let fixed_speed = params.controller.speed_ref;
        let reference = |index: usize, time: f64, _initial_altitude: f64, _speed_ref: f64| {
            assert_eq!(index, 0);
            FreeFlightReference {
                speed_target: fixed_speed,
                altitude_ref_raw: fixed_altitude,
                roll_ref: roll_reversal_reference(time),
            }
        };

        let mut time = 0.0_f64;
        let mut min_airspeed = f64::INFINITY;
        let mut max_abs_alpha = 0.0_f64;
        let mut max_abs_beta = 0.0_f64;
        let mut max_abs_altitude_error = 0.0_f64;
        let mut max_abs_roll = 0.0_f64;
        let mut max_settled_roll_error = 0.0_f64;
        let mut settled_samples = 0usize;

        while time <= config.duration + 1.0e-12 {
            let (_, mut diagnostics) = rhs
                .eval(&state, &controls, &params)
                .expect("diagnostics during roll regression");
            let (next_controls, trace) = controller_step_with_free_flight_reference(
                &mut controller_state,
                &state,
                &diagnostics,
                &params,
                config.dt_control,
                config.phase_mode,
                config.longitudinal_mode,
                time,
                reference,
            );
            apply_trace(&mut diagnostics, &trace);

            let kite_diag = &diagnostics.kites[0];
            let roll = roll_angle_from_quat_n2b(&state.kites[0].body.quat_n2b);
            let roll_ref = trace.kites[0].roll_ref;
            let settled = (2.2..3.0).contains(&time)
                || (4.2..5.0).contains(&time)
                || (6.2..7.0).contains(&time);

            min_airspeed = min_airspeed.min(kite_diag.airspeed);
            max_abs_alpha = max_abs_alpha.max(kite_diag.alpha.abs());
            max_abs_beta = max_abs_beta.max(kite_diag.beta.abs());
            max_abs_altitude_error =
                max_abs_altitude_error.max((kite_diag.altitude - fixed_altitude).abs());
            max_abs_roll = max_abs_roll.max(roll.abs());
            if settled {
                settled_samples += 1;
                max_settled_roll_error =
                    max_settled_roll_error.max(crate::math::wrap_angle(roll - roll_ref).abs());
            }

            if time >= config.duration - 1.0e-12 {
                break;
            }

            let step = (config.duration - time).min(config.dt_control);
            let (next_state, _, _, _, _) =
                integrate_interval(rhs.as_ref(), &state, &next_controls, &params, step, &config)
                    .expect("roll regression integration interval");
            state = next_state;
            state.renormalize_attitudes();
            controls = next_controls;
            time = zero_if_nan(time + step);
        }

        assert!(
            settled_samples > 30,
            "roll regression did not collect enough settled samples"
        );
        assert!(
            max_settled_roll_error.to_degrees() < 6.0,
            "roll controller did not track reversals: max settled error={:.2} deg",
            max_settled_roll_error.to_degrees()
        );
        assert!(
            max_abs_roll.to_degrees() < 32.0,
            "roll controller overshot too far: max_abs_roll={:.2} deg",
            max_abs_roll.to_degrees()
        );
        assert!(
            min_airspeed > 18.0,
            "roll reversals let airspeed get too low: min={min_airspeed:.2} m/s"
        );
        assert!(
            max_abs_alpha.to_degrees() < 12.0,
            "roll reversals exceeded alpha margin: max_abs_alpha={:.2} deg",
            max_abs_alpha.to_degrees()
        );
        assert!(
            max_abs_beta.to_degrees() < 12.0,
            "roll reversals exceeded beta margin: max_abs_beta={:.2} deg",
            max_abs_beta.to_degrees()
        );
        assert!(
            max_abs_altitude_error < 12.0,
            "fixed-altitude TECS drifted during roll test: max_abs_altitude_error={max_abs_altitude_error:.2} m"
        );
    }
}
