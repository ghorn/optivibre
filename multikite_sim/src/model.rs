use crate::math::{
    Scalar, add, cross, ddt_quat_n2b, dot, norm, norm_squared, normalize, rotate_body_to_nav,
    rotate_nav_to_body, scale, smooth_enable, square, sub,
};
use crate::types::{
    BodyState, Controls, Diagnostics, KiteControls, KiteDiagnostics, KiteParams, KiteState, Params,
    State, TetherNode, TetherParams,
};
use anyhow::Result;
use nalgebra::{Quaternion, Vector3};
use optimization::{
    FunctionCompileOptions, LlvmOptimizationLevel, flatten_value, symbolic_column, symbolic_value,
    unflatten_value,
};
use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Mutex;
use sx_codegen_llvm::{CompiledJitFunction, JitExecutionContext};
use sx_core::{NamedMatrix, SX, SXFunction};

#[derive(Clone, Debug)]
struct MassContactOutput<T> {
    total_force_n: Vector3<T>,
}

#[derive(Clone, Debug)]
struct TetherOutputs<T, const N: usize> {
    node_ddt: [TetherNode<T>; N],
    force_on_top_n: Vector3<T>,
    force_on_bottom_n: Vector3<T>,
    top_tension: T,
    strain_energy: T,
}

fn zero_vec<T: Scalar>() -> Vector3<T> {
    Vector3::new(T::zero(), T::zero(), T::zero())
}

fn blank_kite_diag<T: Scalar>() -> KiteDiagnostics<T> {
    KiteDiagnostics {
        cad_position_n: zero_vec(),
        cad_velocity_n: zero_vec(),
        body_accel_b: zero_vec(),
        body_accel_n: zero_vec(),
        omega_b: zero_vec(),
        airspeed: T::zero(),
        alpha: T::zero(),
        beta: T::zero(),
        top_tension: T::zero(),
        phase_angle: T::zero(),
        phase_error: T::zero(),
        speed_target: T::zero(),
        altitude: T::zero(),
        altitude_ref: T::zero(),
        kinetic_energy_specific: T::zero(),
        kinetic_energy_ref_specific: T::zero(),
        kinetic_energy_error_specific: T::zero(),
        potential_energy_specific: T::zero(),
        potential_energy_ref_specific: T::zero(),
        potential_energy_error_specific: T::zero(),
        total_energy_error_specific: T::zero(),
        energy_balance_error_specific: T::zero(),
        thrust_energy_integrator: T::zero(),
        pitch_energy_integrator: T::zero(),
        rabbit_phase: T::zero(),
        rabbit_radius: T::zero(),
        rabbit_target_n: zero_vec(),
        orbit_radius: T::zero(),
        curvature_y_b: T::zero(),
        curvature_y_ref: T::zero(),
        curvature_y_est: T::zero(),
        omega_world_z_ref: T::zero(),
        omega_world_z: T::zero(),
        beta_ref: T::zero(),
        roll_ref: T::zero(),
        roll_ff: T::zero(),
        pitch_ref: T::zero(),
        curvature_z_b: T::zero(),
        curvature_z_ref: T::zero(),
        motor_force: T::zero(),
        motor_power: T::zero(),
        total_force_b: zero_vec(),
        aero_force_b: zero_vec(),
        aero_force_drag_b: zero_vec(),
        aero_force_side_b: zero_vec(),
        aero_force_lift_b: zero_vec(),
        tether_force_b: zero_vec(),
        gravity_force_b: zero_vec(),
        motor_force_b: zero_vec(),
        total_moment_b: zero_vec(),
        aero_moment_b: zero_vec(),
        tether_moment_b: zero_vec(),
        cl_total: T::zero(),
        cl_0_term: T::zero(),
        cl_alpha_term: T::zero(),
        cl_elevator_term: T::zero(),
        cl_flap_term: T::zero(),
        cd_total: T::zero(),
        cd_0_term: T::zero(),
        cd_induced_term: T::zero(),
        cd_surface_term: T::zero(),
        cy_total: T::zero(),
        cy_beta_term: T::zero(),
        cy_rudder_term: T::zero(),
        roll_coeff_total: T::zero(),
        roll_beta_term: T::zero(),
        roll_p_term: T::zero(),
        roll_r_term: T::zero(),
        roll_aileron_term: T::zero(),
        pitch_coeff_total: T::zero(),
        pitch_0_term: T::zero(),
        pitch_alpha_term: T::zero(),
        pitch_q_term: T::zero(),
        pitch_elevator_term: T::zero(),
        pitch_flap_term: T::zero(),
        yaw_coeff_total: T::zero(),
        yaw_beta_term: T::zero(),
        yaw_p_term: T::zero(),
        yaw_r_term: T::zero(),
        yaw_rudder_term: T::zero(),
        kinetic_energy: T::zero(),
        potential_energy: T::zero(),
        tether_strain_energy: T::zero(),
    }
}

pub(crate) fn blank_diagnostics<T: Scalar, const NK: usize>() -> Diagnostics<T, NK> {
    Diagnostics {
        kites: std::array::from_fn(|_| blank_kite_diag()),
        payload_position_n: zero_vec(),
        splitter_position_n: zero_vec(),
        total_kinetic_energy: T::zero(),
        total_potential_energy: T::zero(),
        total_tether_strain_energy: T::zero(),
        total_motor_power: T::zero(),
        work_minus_potential: T::zero(),
    }
}

fn vec_add_assign<T: Scalar>(acc: &mut Vector3<T>, value: &Vector3<T>) {
    *acc = add(acc, value);
}

fn mass_contact_model<T: Scalar>(
    contact_mass: T,
    gravity: T,
    params: &crate::types::MassContactParams<T>,
    pos_n: &Vector3<T>,
    vel_n: &Vector3<T>,
) -> MassContactOutput<T> {
    let delta_pz = pos_n[2] + params.ground_altitude;
    let enable = smooth_enable(delta_pz / params.enable_length);
    let spring_constant = contact_mass * gravity / params.enable_length;
    let damping_constant = T::from_f64(2.0) * (spring_constant * contact_mass).sqrt() * params.zeta;
    let spring_force_z = enable * (-spring_constant * delta_pz);
    let damping_force_z = enable * (-damping_constant * vel_n[2]);
    MassContactOutput {
        total_force_n: Vector3::new(T::zero(), T::zero(), spring_force_z + damping_force_z),
    }
}

fn spring_energy<T: Scalar>(ea: T, natural_length: T, current_length: T) -> T {
    let strain = current_length / natural_length - T::one();
    T::from_f64(0.5) * ea * natural_length * strain * strain
}

fn compute_tether<T: Scalar, const N: usize>(
    bottom: &TetherNode<T>,
    top: &TetherNode<T>,
    params: &TetherParams<T>,
    wind_n: &Vector3<T>,
    rho: T,
    gravity: T,
    nodes: &[TetherNode<T>; N],
) -> TetherOutputs<T, N> {
    if N == 0 {
        return TetherOutputs {
            node_ddt: std::array::from_fn(|_| TetherNode {
                pos_n: zero_vec(),
                vel_n: zero_vec(),
            }),
            force_on_top_n: zero_vec(),
            force_on_bottom_n: zero_vec(),
            top_tension: T::zero(),
            strain_energy: T::zero(),
        };
    }

    let masses = params.total_mass / T::from_f64(N as f64);
    let segment_length = params.natural_length / T::from_f64(N as f64);
    let mut node_ddt = std::array::from_fn(|_| TetherNode {
        pos_n: zero_vec(),
        vel_n: zero_vec(),
    });
    let mut upper_spring_forces = [zero_vec(); N];
    let mut upper_damping_forces = [zero_vec(); N];
    let mut lower_spring_forces = [zero_vec(); N];
    let mut lower_damping_forces = [zero_vec(); N];
    let mut top_tension = T::zero();
    let mut strain_energy = T::zero();

    for index in 0..N {
        let current = &nodes[index];
        let lower = if index == 0 {
            bottom
        } else {
            &nodes[index - 1]
        };
        let upper = if index + 1 == N {
            top
        } else {
            &nodes[index + 1]
        };

        let lower_delta_pos = sub(&lower.pos_n, &current.pos_n);
        let upper_delta_pos = sub(&upper.pos_n, &current.pos_n);
        let lower_delta_vel = sub(&lower.vel_n, &current.vel_n);
        let upper_delta_vel = sub(&upper.vel_n, &current.vel_n);
        let lower_length = norm(&lower_delta_pos);
        let upper_length = norm(&upper_delta_pos);
        let lower_natural = if index == 0 {
            T::from_f64(0.5) * segment_length
        } else {
            segment_length
        };
        let upper_natural = if index + 1 == N {
            T::from_f64(0.5) * segment_length
        } else {
            segment_length
        };

        let lower_strain = lower_length / lower_natural - T::one();
        let upper_strain = upper_length / upper_natural - T::one();
        let lower_spring_tension = params.ea * lower_strain;
        let upper_spring_tension = params.ea * upper_strain;
        lower_spring_forces[index] = scale(&lower_delta_pos, lower_spring_tension / lower_length);
        upper_spring_forces[index] = scale(&upper_delta_pos, upper_spring_tension / upper_length);

        let lower_damping_tension =
            params.viscous_damping_coeff * dot(&lower_delta_pos, &lower_delta_vel);
        let upper_damping_tension =
            params.viscous_damping_coeff * dot(&upper_delta_pos, &upper_delta_vel);
        lower_damping_forces[index] = scale(&lower_delta_pos, lower_damping_tension / lower_length);
        upper_damping_forces[index] = scale(&upper_delta_pos, upper_damping_tension / upper_length);

        let tangent = sub(&upper.pos_n, &lower.pos_n);
        let tangent_hat = normalize(&tangent);
        let apparent_wind_n = sub(&current.vel_n, wind_n);
        let apparent_tangent = scale(&tangent_hat, dot(&apparent_wind_n, &tangent_hat));
        let apparent_normal = sub(&apparent_wind_n, &apparent_tangent);
        let tangent_length = norm(&tangent);
        let drag_scale = -T::from_f64(0.5)
            * rho
            * params.cd_phi
            * params.diameter
            * tangent_length
            * norm(&apparent_normal);
        let drag_force = scale(&apparent_normal, drag_scale);
        let gravity_force = Vector3::new(T::zero(), T::zero(), masses * gravity);
        let contact = mass_contact_model(
            masses,
            gravity,
            &params.contact,
            &current.pos_n,
            &current.vel_n,
        );

        let total_force = add(
            &add(&lower_spring_forces[index], &upper_spring_forces[index]),
            &add(&lower_damping_forces[index], &upper_damping_forces[index]),
        );
        let total_force = add(
            &add(&total_force, &drag_force),
            &add(&gravity_force, &contact.total_force_n),
        );
        node_ddt[index] = TetherNode {
            pos_n: current.vel_n,
            vel_n: scale(&total_force, T::one() / masses),
        };

        if index + 1 == N {
            top_tension = upper_spring_tension + upper_damping_tension;
        }
        if index == 0 {
            strain_energy = strain_energy + spring_energy(params.ea, lower_natural, lower_length);
        }
        strain_energy = strain_energy + spring_energy(params.ea, upper_natural, upper_length);
    }

    let force_on_top_n = upper_spring_forces
        .last()
        .zip(upper_damping_forces.last())
        .map(|(spring, damping)| scale(&add(spring, damping), -T::one()))
        .unwrap_or_else(zero_vec);
    let force_on_bottom_n = lower_spring_forces
        .first()
        .zip(lower_damping_forces.first())
        .map(|(spring, damping)| scale(&add(spring, damping), -T::one()))
        .unwrap_or_else(zero_vec);

    TetherOutputs {
        node_ddt,
        force_on_top_n,
        force_on_bottom_n,
        top_tension,
        strain_energy,
    }
}

pub(crate) fn compute_tether_link_tensions<const N: usize>(
    bottom: &TetherNode<f64>,
    top: &TetherNode<f64>,
    params: &TetherParams<f64>,
    nodes: &[TetherNode<f64>; N],
) -> Vec<f64> {
    if N == 0 {
        return Vec::new();
    }

    let segment_length = params.natural_length / N as f64;
    let mut segment_tensions = Vec::with_capacity(N + 1);
    let mut top_tension = 0.0;

    for index in 0..N {
        let current = &nodes[index];
        let lower = if index == 0 {
            bottom
        } else {
            &nodes[index - 1]
        };
        let upper = if index + 1 == N {
            top
        } else {
            &nodes[index + 1]
        };

        let lower_delta_pos = lower.pos_n - current.pos_n;
        let upper_delta_pos = upper.pos_n - current.pos_n;
        let lower_delta_vel = lower.vel_n - current.vel_n;
        let upper_delta_vel = upper.vel_n - current.vel_n;
        let lower_length = lower_delta_pos.norm();
        let upper_length = upper_delta_pos.norm();
        let lower_natural = if index == 0 {
            0.5 * segment_length
        } else {
            segment_length
        };
        let upper_natural = if index + 1 == N {
            0.5 * segment_length
        } else {
            segment_length
        };

        let lower_strain = lower_length / lower_natural - 1.0;
        let upper_strain = upper_length / upper_natural - 1.0;
        let lower_total = params.ea * lower_strain
            + params.viscous_damping_coeff * lower_delta_pos.dot(&lower_delta_vel);
        let upper_total = params.ea * upper_strain
            + params.viscous_damping_coeff * upper_delta_pos.dot(&upper_delta_vel);

        segment_tensions.push(lower_total);
        if index + 1 == N {
            top_tension = upper_total;
        }
    }

    segment_tensions.push(top_tension);
    segment_tensions
}

pub(crate) fn compute_bridle_node<const N_UPPER: usize>(
    kite: &KiteState<f64, N_UPPER>,
    params: &KiteParams<f64>,
) -> TetherNode<f64> {
    let fallback_node = TetherNode {
        pos_n: kite.body.pos_n,
        vel_n: rotate_body_to_nav(&kite.body.quat_n2b, &kite.body.vel_b),
    };
    let last_node = kite.tether.last().unwrap_or(&fallback_node);
    let (bridle_pos_n, bridle_vel_n, _) = bridle_geometry(
        &kite.body,
        &params.rigid_body.cad_offset_b,
        &params.bridle.pivot_b,
        params.bridle.radius,
        last_node,
    );
    TetherNode {
        pos_n: bridle_pos_n,
        vel_n: bridle_vel_n,
    }
}

fn bridle_geometry<T: Scalar>(
    body: &BodyState<T>,
    cad_offset_b: &Vector3<T>,
    bridle_pivot_b: &Vector3<T>,
    bridle_radius: T,
    last_tether_node: &TetherNode<T>,
) -> (Vector3<T>, Vector3<T>, Vector3<T>) {
    let cad_pos_n = add(
        &body.pos_n,
        &rotate_body_to_nav(&body.quat_n2b, cad_offset_b),
    );
    let _cad_vel_n = rotate_body_to_nav(
        &body.quat_n2b,
        &add(&body.vel_b, &cross(&body.omega_b, cad_offset_b)),
    );
    let pivot_pos_n = add(
        &cad_pos_n,
        &rotate_body_to_nav(&body.quat_n2b, bridle_pivot_b),
    );
    let pivot_vel_b = add(
        &body.vel_b,
        &cross(&body.omega_b, &add(cad_offset_b, bridle_pivot_b)),
    );
    let pivot_vel_n = rotate_body_to_nav(&body.quat_n2b, &pivot_vel_b);

    let tether_rel_n = sub(&last_tether_node.pos_n, &pivot_pos_n);
    let tether_rel_b = rotate_nav_to_body(&body.quat_n2b, &tether_rel_n);
    let tether_plane_b = Vector3::new(tether_rel_b[0], T::zero(), tether_rel_b[2]);
    let tether_plane_hat_b = normalize(&tether_plane_b);
    let bridle_offset_b = scale(&tether_plane_hat_b, bridle_radius);
    let bridle_pos_n = add(
        &pivot_pos_n,
        &rotate_body_to_nav(&body.quat_n2b, &bridle_offset_b),
    );

    let last_vel_rel_n = sub(&last_tether_node.vel_n, &pivot_vel_n);
    let last_vel_rel_b = rotate_nav_to_body(&body.quat_n2b, &last_vel_rel_n);
    let last_vel_plane_b = Vector3::new(last_vel_rel_b[0], T::zero(), last_vel_rel_b[2]);
    let radial_component = scale(
        &tether_plane_hat_b,
        dot(&tether_plane_hat_b, &last_vel_plane_b),
    );
    let tangential_component = sub(&last_vel_plane_b, &radial_component);
    let bridle_offset_speed_b = scale(&tangential_component, bridle_radius / norm(&tether_plane_b));
    let bridle_vel_n = add(
        &pivot_vel_n,
        &rotate_body_to_nav(&body.quat_n2b, &bridle_offset_speed_b),
    );

    (bridle_pos_n, bridle_vel_n, bridle_offset_b)
}

fn compute_kite<T: Scalar, const N_UPPER: usize>(
    kite: &KiteState<T, N_UPPER>,
    control: &KiteControls<T>,
    params: &KiteParams<T>,
    common_params: &Params<T, 1>,
    splitter: &TetherNode<T>,
) -> (KiteState<T, N_UPPER>, KiteDiagnostics<T>, Vector3<T>) {
    let last_node = kite.tether.last().unwrap_or(splitter);
    let (bridle_pos_n, bridle_vel_n, bridle_offset_b) = bridle_geometry(
        &kite.body,
        &params.rigid_body.cad_offset_b,
        &params.bridle.pivot_b,
        params.bridle.radius,
        last_node,
    );
    let bridle_node = TetherNode {
        pos_n: bridle_pos_n,
        vel_n: bridle_vel_n,
    };
    let upper_tether = compute_tether(
        splitter,
        &bridle_node,
        &params.tether,
        &common_params.environment.wind_n,
        common_params.environment.rho,
        common_params.environment.g,
        &kite.tether,
    );

    let effective_wind_n = add(
        &common_params.environment.wind_n,
        &common_params.kite_gusts_n[0],
    );
    let body_vel_n = rotate_body_to_nav(&kite.body.quat_n2b, &kite.body.vel_b);
    let apparent_air_n = sub(&body_vel_n, &effective_wind_n);
    let apparent_air_b = rotate_nav_to_body(&kite.body.quat_n2b, &apparent_air_n);
    let airspeed = norm(&apparent_air_b);
    let alpha = apparent_air_b[2].atan2(apparent_air_b[0]);
    let beta = apparent_air_b[1].atan2(
        (apparent_air_b[0] * apparent_air_b[0]
            + apparent_air_b[2] * apparent_air_b[2]
            + T::from_f64(1.0e-9))
        .sqrt(),
    );
    let qbar = T::from_f64(0.5) * common_params.environment.rho * airspeed * airspeed;
    let p_norm = kite.body.omega_b[0] * params.aero.ref_span / (T::from_f64(2.0) * airspeed);
    let q_norm = kite.body.omega_b[1] * params.aero.ref_chord / (T::from_f64(2.0) * airspeed);
    let r_norm = kite.body.omega_b[2] * params.aero.ref_span / (T::from_f64(2.0) * airspeed);
    let cl_0_term = params.aero.cl0;
    let cl_alpha_term = params.aero.cl_alpha * alpha;
    let cl_elevator_term = params.aero.cl_elevator * control.surfaces.elevator;
    let cl_flap_term = params.aero.cl_flap * control.surfaces.flap;
    let cl = cl_0_term + cl_alpha_term + cl_elevator_term + cl_flap_term;
    let cd_0_term = params.aero.cd0;
    let cd_induced_term = params.aero.cd_induced * cl * cl;
    let cd_surface_term = params.aero.cd_surface_abs
        * (control.surfaces.aileron.abs()
            + control.surfaces.flap.abs()
            + control.surfaces.elevator.abs()
            + control.surfaces.rudder.abs());
    let cd = cd_0_term + cd_induced_term + cd_surface_term;
    let cy_beta_term = params.aero.cy_beta * beta;
    let cy_rudder_term = params.aero.cy_rudder * control.surfaces.rudder;
    let cy = cy_beta_term + cy_rudder_term;
    let lift = qbar * params.aero.ref_area * cl;
    let drag = qbar * params.aero.ref_area * cd;
    let side_force = qbar * params.aero.ref_area * cy;
    let aero_force_drag_b = Vector3::new(-drag, T::zero(), T::zero());
    let aero_force_side_b = Vector3::new(T::zero(), side_force, T::zero());
    let aero_force_lift_b = Vector3::new(T::zero(), T::zero(), -lift);
    let aero_force_b = add(
        &add(&aero_force_drag_b, &aero_force_side_b),
        &aero_force_lift_b,
    );
    let roll_beta_term = params.aero.roll_beta * beta;
    let roll_p_term = params.aero.roll_p * p_norm;
    let roll_r_term = params.aero.roll_r * r_norm;
    let roll_aileron_term = params.aero.roll_aileron * control.surfaces.aileron;
    let roll_coeff = roll_beta_term + roll_p_term + roll_r_term + roll_aileron_term;
    let pitch_0_term = params.aero.pitch0;
    let pitch_alpha_term = params.aero.pitch_alpha * alpha;
    let pitch_q_term = params.aero.pitch_q * q_norm;
    let pitch_elevator_term = params.aero.pitch_elevator * control.surfaces.elevator;
    let pitch_flap_term = params.aero.pitch_flap * control.surfaces.flap;
    let pitch_coeff =
        pitch_0_term + pitch_alpha_term + pitch_q_term + pitch_elevator_term + pitch_flap_term;
    let yaw_beta_term = params.aero.yaw_beta * beta;
    let yaw_p_term = params.aero.yaw_p * p_norm;
    let yaw_r_term = params.aero.yaw_r * r_norm;
    let yaw_rudder_term = params.aero.yaw_rudder * control.surfaces.rudder;
    let yaw_coeff = yaw_beta_term + yaw_p_term + yaw_r_term + yaw_rudder_term;
    let aero_moment_b = Vector3::new(
        qbar * params.aero.ref_area * params.aero.ref_span * roll_coeff,
        qbar * params.aero.ref_area * params.aero.ref_chord * pitch_coeff,
        qbar * params.aero.ref_area * params.aero.ref_span * yaw_coeff,
    );

    let motor_force = control.motor_torque * params.rotor.torque_to_force;
    let motor_force_b = scale(&normalize(&params.rotor.axis_b), motor_force);
    let motor_power = motor_force.abs() * (airspeed + params.rotor.force_to_power);
    let gravity_n = Vector3::new(
        T::zero(),
        T::zero(),
        params.rigid_body.mass * common_params.environment.g,
    );
    let gravity_b = rotate_nav_to_body(&kite.body.quat_n2b, &gravity_n);
    let tether_force_b = rotate_nav_to_body(&kite.body.quat_n2b, &upper_tether.force_on_top_n);
    let cad_to_bridle_b = add(&params.bridle.pivot_b, &bridle_offset_b);
    let cg_to_bridle_b = add(&params.rigid_body.cad_offset_b, &cad_to_bridle_b);
    let tether_moment_b = cross(&cg_to_bridle_b, &tether_force_b);
    let total_force_b = add(
        &add(&aero_force_b, &motor_force_b),
        &add(&tether_force_b, &gravity_b),
    );
    let velocity_dot_b = sub(
        &scale(&total_force_b, T::one() / params.rigid_body.mass),
        &cross(&kite.body.omega_b, &kite.body.vel_b),
    );
    let inertial_accel_b = add(
        &velocity_dot_b,
        &cross(&kite.body.omega_b, &kite.body.vel_b),
    );
    let inertial_accel_n = rotate_body_to_nav(&kite.body.quat_n2b, &inertial_accel_b);
    let angular_momentum = Vector3::new(
        params.rigid_body.inertia_diagonal[0] * kite.body.omega_b[0],
        params.rigid_body.inertia_diagonal[1] * kite.body.omega_b[1],
        params.rigid_body.inertia_diagonal[2] * kite.body.omega_b[2],
    );
    let omega_cross_h = cross(&kite.body.omega_b, &angular_momentum);
    let total_moment_b = add(&aero_moment_b, &tether_moment_b);
    let omega_dot_b = Vector3::new(
        (total_moment_b[0] - omega_cross_h[0]) / params.rigid_body.inertia_diagonal[0],
        (total_moment_b[1] - omega_cross_h[1]) / params.rigid_body.inertia_diagonal[1],
        (total_moment_b[2] - omega_cross_h[2]) / params.rigid_body.inertia_diagonal[2],
    );
    let body_ddt = BodyState {
        pos_n: rotate_body_to_nav(&kite.body.quat_n2b, &kite.body.vel_b),
        vel_b: velocity_dot_b,
        quat_n2b: ddt_quat_n2b(&kite.body.quat_n2b, &kite.body.omega_b),
        omega_b: omega_dot_b,
    };
    let cad_position_n = add(
        &kite.body.pos_n,
        &rotate_body_to_nav(&kite.body.quat_n2b, &params.rigid_body.cad_offset_b),
    );
    let cad_velocity_b = add(
        &kite.body.vel_b,
        &cross(&kite.body.omega_b, &params.rigid_body.cad_offset_b),
    );
    let cad_velocity_n = rotate_body_to_nav(&kite.body.quat_n2b, &cad_velocity_b);
    let speed_sq = norm_squared(&cad_velocity_n) + T::from_f64(1.0e-9);
    let phase_origin = sub(&cad_position_n, &common_params.controller.disk_center_n);
    let phase = phase_origin[1].atan2(phase_origin[0]);
    let orbit_radius =
        (phase_origin[0] * phase_origin[0] + phase_origin[1] * phase_origin[1]).sqrt();

    let translational_ke =
        T::from_f64(0.5) * params.rigid_body.mass * norm_squared(&kite.body.vel_b);
    let rotational_ke = T::from_f64(0.5)
        * (params.rigid_body.inertia_diagonal[0] * square(kite.body.omega_b[0])
            + params.rigid_body.inertia_diagonal[1] * square(kite.body.omega_b[1])
            + params.rigid_body.inertia_diagonal[2] * square(kite.body.omega_b[2]));
    let potential_energy =
        -params.rigid_body.mass * common_params.environment.g * cad_position_n[2];

    (
        KiteState {
            body: body_ddt,
            tether: upper_tether.node_ddt,
        },
        KiteDiagnostics {
            cad_position_n,
            cad_velocity_n,
            body_accel_b: inertial_accel_b,
            body_accel_n: inertial_accel_n,
            omega_b: kite.body.omega_b,
            airspeed,
            alpha,
            beta,
            top_tension: upper_tether.top_tension,
            phase_angle: phase,
            phase_error: T::zero(),
            speed_target: common_params.controller.speed_ref,
            altitude: T::zero(),
            altitude_ref: T::zero(),
            kinetic_energy_specific: T::zero(),
            kinetic_energy_ref_specific: T::zero(),
            kinetic_energy_error_specific: T::zero(),
            potential_energy_specific: T::zero(),
            potential_energy_ref_specific: T::zero(),
            potential_energy_error_specific: T::zero(),
            total_energy_error_specific: T::zero(),
            energy_balance_error_specific: T::zero(),
            thrust_energy_integrator: T::zero(),
            pitch_energy_integrator: T::zero(),
            rabbit_phase: T::zero(),
            rabbit_radius: common_params.controller.disk_radius,
            rabbit_target_n: zero_vec(),
            orbit_radius,
            curvature_y_b: inertial_accel_b[1] / speed_sq,
            curvature_y_ref: T::zero(),
            curvature_y_est: T::zero(),
            omega_world_z_ref: T::zero(),
            omega_world_z: T::zero(),
            beta_ref: T::zero(),
            roll_ref: T::zero(),
            roll_ff: T::zero(),
            pitch_ref: T::zero(),
            curvature_z_b: inertial_accel_b[2] / speed_sq,
            curvature_z_ref: T::zero(),
            motor_force,
            motor_power,
            total_force_b,
            aero_force_b,
            aero_force_drag_b,
            aero_force_side_b,
            aero_force_lift_b,
            tether_force_b,
            gravity_force_b: gravity_b,
            motor_force_b,
            total_moment_b,
            aero_moment_b,
            tether_moment_b,
            cl_total: cl,
            cl_0_term,
            cl_alpha_term,
            cl_elevator_term,
            cl_flap_term,
            cd_total: cd,
            cd_0_term,
            cd_induced_term,
            cd_surface_term,
            cy_total: cy,
            cy_beta_term,
            cy_rudder_term,
            roll_coeff_total: roll_coeff,
            roll_beta_term,
            roll_p_term,
            roll_r_term,
            roll_aileron_term,
            pitch_coeff_total: pitch_coeff,
            pitch_0_term,
            pitch_alpha_term,
            pitch_q_term,
            pitch_elevator_term,
            pitch_flap_term,
            yaw_coeff_total: yaw_coeff,
            yaw_beta_term,
            yaw_p_term,
            yaw_r_term,
            yaw_rudder_term,
            kinetic_energy: translational_ke + rotational_ke,
            potential_energy,
            tether_strain_energy: upper_tether.strain_energy,
        },
        upper_tether.force_on_bottom_n,
    )
}

pub fn evaluate_rhs<T: Scalar, const NK: usize, const N_COMMON: usize, const N_UPPER: usize>(
    state: &State<T, NK, N_COMMON, N_UPPER>,
    controls: &Controls<T, NK>,
    params: &Params<T, NK>,
) -> (State<T, NK, N_COMMON, N_UPPER>, Diagnostics<T, NK>) {
    let mut diagnostics = blank_diagnostics();
    let mut kite_ddt = std::array::from_fn(|_| KiteState {
        body: BodyState {
            pos_n: zero_vec(),
            vel_b: zero_vec(),
            quat_n2b: Quaternion::new(T::one(), T::zero(), T::zero(), T::zero()),
            omega_b: zero_vec(),
        },
        tether: std::array::from_fn(|_| TetherNode {
            pos_n: zero_vec(),
            vel_n: zero_vec(),
        }),
    });
    let mut splitter_force_n = zero_vec();
    for index in 0..NK {
        let single_params = Params {
            kites: [params.kites[index].clone()],
            common_tether: params.common_tether.clone(),
            splitter_mass: params.splitter_mass,
            payload_mass: params.payload_mass,
            environment: params.environment.clone(),
            kite_gusts_n: [params.kite_gusts_n[index]],
            controller: params.controller.clone(),
        };
        let (kite_model, kite_diag, lower_force_n) = compute_kite(
            &state.kites[index],
            &controls.kites[index],
            &params.kites[index],
            &single_params,
            &state.splitter,
        );
        kite_ddt[index] = kite_model;
        diagnostics.kites[index] = kite_diag;
        vec_add_assign(&mut splitter_force_n, &lower_force_n);
    }

    let common_tether = compute_tether(
        &state.payload,
        &state.splitter,
        &params.common_tether,
        &params.environment.wind_n,
        params.environment.rho,
        params.environment.g,
        &state.common_tether,
    );
    let payload_contact = mass_contact_model(
        params.payload_mass,
        params.environment.g,
        &params.common_tether.contact,
        &state.payload.pos_n,
        &state.payload.vel_n,
    );
    let splitter_contact = mass_contact_model(
        params.splitter_mass,
        params.environment.g,
        &params.common_tether.contact,
        &state.splitter.pos_n,
        &state.splitter.vel_n,
    );
    let total_payload_force_n = add(
        &common_tether.force_on_bottom_n,
        &payload_contact.total_force_n,
    );
    let total_splitter_force_n = add(
        &add(
            &common_tether.force_on_top_n,
            &splitter_contact.total_force_n,
        ),
        &splitter_force_n,
    );

    let payload_ddt = if N_COMMON == 0 {
        TetherNode {
            pos_n: zero_vec(),
            vel_n: zero_vec(),
        }
    } else {
        TetherNode {
            pos_n: state.payload.vel_n,
            vel_n: add(
                &scale(&total_payload_force_n, T::one() / params.payload_mass),
                &Vector3::new(T::zero(), T::zero(), params.environment.g),
            ),
        }
    };
    let splitter_ddt = if NK == 0 || N_COMMON == 0 {
        TetherNode {
            pos_n: zero_vec(),
            vel_n: zero_vec(),
        }
    } else {
        TetherNode {
            pos_n: state.splitter.vel_n,
            vel_n: add(
                &scale(&total_splitter_force_n, T::one() / params.splitter_mass),
                &Vector3::new(T::zero(), T::zero(), params.environment.g),
            ),
        }
    };

    let total_motor_power = diagnostics
        .kites
        .iter()
        .fold(T::zero(), |acc, item| acc + item.motor_power);
    let total_kinetic_energy_kites = diagnostics
        .kites
        .iter()
        .fold(T::zero(), |acc, item| acc + item.kinetic_energy);
    let total_potential_energy_kites = diagnostics
        .kites
        .iter()
        .fold(T::zero(), |acc, item| acc + item.potential_energy);
    let total_tether_energy_kites = diagnostics
        .kites
        .iter()
        .fold(T::zero(), |acc, item| acc + item.tether_strain_energy);
    let payload_ke = T::from_f64(0.5) * params.payload_mass * norm_squared(&state.payload.vel_n);
    let splitter_ke = if NK == 0 {
        T::zero()
    } else {
        T::from_f64(0.5) * params.splitter_mass * norm_squared(&state.splitter.vel_n)
    };
    let payload_pe = -params.payload_mass * params.environment.g * state.payload.pos_n[2];
    let splitter_pe = if NK == 0 {
        T::zero()
    } else {
        -params.splitter_mass * params.environment.g * state.splitter.pos_n[2]
    };
    diagnostics.payload_position_n = state.payload.pos_n;
    diagnostics.splitter_position_n = state.splitter.pos_n;
    diagnostics.total_kinetic_energy = total_kinetic_energy_kites + payload_ke + splitter_ke;
    diagnostics.total_potential_energy = total_potential_energy_kites + payload_pe + splitter_pe;
    diagnostics.total_tether_strain_energy =
        total_tether_energy_kites + common_tether.strain_energy;
    diagnostics.total_motor_power = total_motor_power;
    diagnostics.work_minus_potential = state.total_work - diagnostics.total_potential_energy;

    (
        State {
            kites: kite_ddt,
            splitter: splitter_ddt,
            common_tether: common_tether.node_ddt,
            payload: payload_ddt,
            total_work: total_motor_power,
        },
        diagnostics,
    )
}

fn build_rhs_function<const NK: usize, const N_COMMON: usize, const N_UPPER: usize>()
-> Result<SXFunction> {
    let x = symbolic_value::<State<SX, NK, N_COMMON, N_UPPER>>("x")?;
    let u = symbolic_value::<Controls<SX, NK>>("u")?;
    let p = symbolic_value::<Params<SX, NK>>("p")?;
    let (xdot, diag) = evaluate_rhs(&x, &u, &p);
    Ok(SXFunction::new(
        format!("multikite_rhs_{NK}_{N_COMMON}_{N_UPPER}"),
        vec![
            NamedMatrix::new("x", symbolic_column(&x)?)?,
            NamedMatrix::new("u", symbolic_column(&u)?)?,
            NamedMatrix::new("p", symbolic_column(&p)?)?,
        ],
        vec![
            NamedMatrix::new("xdot", symbolic_column(&xdot)?)?,
            NamedMatrix::new("diag", symbolic_column(&diag)?)?,
        ],
    )?)
}

pub struct CompiledRhs<const NK: usize, const N_COMMON: usize, const N_UPPER: usize> {
    function: CompiledJitFunction,
    context: Mutex<JitExecutionContext>,
}

thread_local! {
    static COMPILED_RHS_CACHE: RefCell<HashMap<(usize, usize, usize), Box<dyn Any>>> =
        RefCell::new(HashMap::new());
}

impl<const NK: usize, const N_COMMON: usize, const N_UPPER: usize>
    CompiledRhs<NK, N_COMMON, N_UPPER>
{
    pub fn new() -> Result<Self> {
        let function = build_rhs_function::<NK, N_COMMON, N_UPPER>()?;
        let compiled = CompiledJitFunction::compile_function_with_options(
            &function,
            FunctionCompileOptions::from(LlvmOptimizationLevel::O3),
        )?;
        let context = Mutex::new(compiled.create_context());
        Ok(Self {
            function: compiled,
            context,
        })
    }

    pub fn shared() -> Result<Rc<Self>> {
        let key = (NK, N_COMMON, N_UPPER);
        if let Some(existing) = COMPILED_RHS_CACHE.with(|cache| {
            let cache = cache.borrow();
            cache
                .get(&key)
                .and_then(|entry| entry.downcast_ref::<Rc<Self>>())
                .cloned()
        }) {
            return Ok(existing);
        }

        let rhs = Rc::new(Self::new()?);
        COMPILED_RHS_CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            let existing = cache
                .entry(key)
                .or_insert_with(|| Box::new(rhs.clone()) as Box<dyn Any>);
            if let Some(cached) = existing.downcast_ref::<Rc<Self>>() {
                Ok(cached.clone())
            } else {
                unreachable!("compiled RHS cache type mismatch for key {:?}", key);
            }
        })
    }

    pub fn eval(
        &self,
        state: &State<f64, NK, N_COMMON, N_UPPER>,
        controls: &Controls<f64, NK>,
        params: &Params<f64, NK>,
    ) -> Result<(State<f64, NK, N_COMMON, N_UPPER>, Diagnostics<f64, NK>)> {
        let flat_x = flatten_value(state);
        let flat_u = flatten_value(controls);
        let flat_p = flatten_value(params);
        let mut context = match self.context.lock() {
            Ok(guard) => guard,
            Err(poison) => poison.into_inner(),
        };
        context.input_mut(0).copy_from_slice(&flat_x);
        context.input_mut(1).copy_from_slice(&flat_u);
        context.input_mut(2).copy_from_slice(&flat_p);
        self.function.eval(&mut context);
        Ok((
            unflatten_value::<State<f64, NK, N_COMMON, N_UPPER>, f64>(context.output(0))?,
            unflatten_value::<Diagnostics<f64, NK>, f64>(context.output(1))?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::CompiledRhs;
    use std::rc::Rc;

    #[test]
    fn compiled_rhs_shared_reuses_instance() {
        let first = CompiledRhs::<1, 0, 0>::shared().expect("first compiled rhs");
        let second = CompiledRhs::<1, 0, 0>::shared().expect("second compiled rhs");
        assert!(Rc::ptr_eq(&first, &second));
    }
}
