use crate::assets::{
    FlapPolynomialExport, ForceMomentFlapPolynomialExport, ForceMomentQuartic2Export,
    Quartic2Export, reference_avl_fit_ref, reference_rotor_fit_ref,
};
use crate::math::{
    Scalar, add, cross, ddt_quat_n2b, dot, norm, norm_exact, norm_squared, normalize_exact,
    rotate_body_to_nav, rotate_nav_to_body, scale, smooth_enable, square, sub,
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
    dissipated_power: T,
}

#[derive(Clone, Debug)]
struct TetherOutputs<T, const N: usize> {
    node_ddt: [TetherNode<T>; N],
    force_on_top_n: Vector3<T>,
    force_on_bottom_n: Vector3<T>,
    top_tension: T,
    strain_energy: T,
    dissipated_power: T,
}

#[derive(Clone, Debug)]
struct AeroCoeffEval<T> {
    total_force_w: Vector3<T>,
    nominal_force_w: Vector3<T>,
    pqr_force_w: Vector3<T>,
    surface_force_w: Vector3<T>,
    total_moment_c: Vector3<T>,
    nominal_moment_c: Vector3<T>,
    pqr_moment_c: Vector3<T>,
    surface_moment_c: Vector3<T>,
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
        aero_dissipated_power: T::zero(),
        tether_dissipated_power: T::zero(),
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
        motor_moment_b: zero_vec(),
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
        total_dissipated_power: T::zero(),
        total_mechanical_energy: T::zero(),
        energy_conservation_residual: T::zero(),
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
        dissipated_power: enable * damping_constant * square(vel_n[2]),
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
            dissipated_power: T::zero(),
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
    let mut dissipated_power = T::zero();

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
        let lower_length = norm_exact(&lower_delta_pos);
        let upper_length = norm_exact(&upper_delta_pos);
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
        let lower_damping_tension =
            params.viscous_damping_coeff * dot(&lower_delta_pos, &lower_delta_vel);
        let upper_damping_tension =
            params.viscous_damping_coeff * dot(&upper_delta_pos, &upper_delta_vel);
        let upper_tension = upper_spring_tension + upper_damping_tension;
        lower_spring_forces[index] = scale(&lower_delta_pos, lower_spring_tension / lower_length);
        upper_spring_forces[index] = scale(&upper_delta_pos, upper_spring_tension / upper_length);
        lower_damping_forces[index] = scale(&lower_delta_pos, lower_damping_tension / lower_length);
        upper_damping_forces[index] = scale(&upper_delta_pos, upper_damping_tension / upper_length);

        let tangent_before = if index == 0 {
            lower.pos_n
        } else {
            scale(&add(&lower.pos_n, &current.pos_n), T::from_f64(0.5))
        };
        let tangent_after = if index + 1 == N {
            upper.pos_n
        } else {
            scale(&add(&current.pos_n, &upper.pos_n), T::from_f64(0.5))
        };
        let tangent = sub(&tangent_after, &tangent_before);
        let tangent_hat = normalize_exact(&tangent);
        let apparent_wind_n = sub(&current.vel_n, wind_n);
        let apparent_tangent = scale(&tangent_hat, dot(&apparent_wind_n, &tangent_hat));
        let apparent_normal = sub(&apparent_wind_n, &apparent_tangent);
        let tangent_length = norm_exact(&tangent);
        let drag_scale = -T::from_f64(0.5)
            * rho
            * params.cd_phi
            * params.diameter
            * tangent_length
            * norm(&apparent_normal);
        let drag_force = scale(&apparent_normal, drag_scale);
        let drag_dissipated_power = -dot(&drag_force, &apparent_normal);
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
            top_tension = upper_tension;
            dissipated_power = dissipated_power
                + params.viscous_damping_coeff * square(dot(&upper_delta_pos, &upper_delta_vel))
                    / upper_length;
        }
        dissipated_power = dissipated_power
            + params.viscous_damping_coeff * square(dot(&lower_delta_pos, &lower_delta_vel))
                / lower_length
            + drag_dissipated_power
            + contact.dissipated_power;
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
        dissipated_power,
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
    let tether_plane_hat_b = normalize_exact(&tether_plane_b);
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
    let bridle_offset_speed_b = scale(
        &tangential_component,
        bridle_radius / norm_exact(&tether_plane_b),
    );
    let bridle_vel_n = add(
        &pivot_vel_n,
        &rotate_body_to_nav(&body.quat_n2b, &bridle_offset_speed_b),
    );

    (bridle_pos_n, bridle_vel_n, bridle_offset_b)
}

fn eval_quartic2<T: Scalar>(coeff: &Quartic2Export, x: T, y: T) -> T {
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;
    let y2 = y * y;
    let y3 = y2 * y;
    let y4 = y2 * y2;

    T::from_f64(coeff.p42x0y0)
        + T::from_f64(coeff.p42x1y0) * x
        + T::from_f64(coeff.p42x0y1) * y
        + T::from_f64(coeff.p42x0y2) * y2
        + T::from_f64(coeff.p42x1y1) * x * y
        + T::from_f64(coeff.p42x2y0) * x2
        + T::from_f64(coeff.p42x0y3) * y3
        + T::from_f64(coeff.p42x1y2) * x * y2
        + T::from_f64(coeff.p42x2y1) * x2 * y
        + T::from_f64(coeff.p42x3y0) * x3
        + T::from_f64(coeff.p42x0y4) * y4
        + T::from_f64(coeff.p42x1y3) * x * y3
        + T::from_f64(coeff.p42x2y2) * x2 * y2
        + T::from_f64(coeff.p42x3y1) * x3 * y
        + T::from_f64(coeff.p42x4y0) * x4
}

fn eval_flap_polynomial<T: Scalar>(coeff: &FlapPolynomialExport, alpha: T, beta: T, delta: T) -> T {
    let a2 = alpha * alpha;
    let a3 = a2 * alpha;
    let b2 = beta * beta;
    let b3 = b2 * beta;
    let d2 = delta * delta;
    let d3 = d2 * delta;
    let d4 = d2 * d2;

    T::from_f64(coeff.fpA0B0D1) * delta
        + T::from_f64(coeff.fpA0B0D2) * d2
        + T::from_f64(coeff.fpA1B0D1) * alpha * delta
        + T::from_f64(coeff.fpA0B1D1) * beta * delta
        + T::from_f64(coeff.fpA0B0D3) * d3
        + T::from_f64(coeff.fpA0B1D2) * beta * d2
        + T::from_f64(coeff.fpA1B0D2) * alpha * d2
        + T::from_f64(coeff.fpA1B1D1) * alpha * beta * delta
        + T::from_f64(coeff.fpA0B2D1) * b2 * delta
        + T::from_f64(coeff.fpA2B0D1) * a2 * delta
        + T::from_f64(coeff.fpA0B0D4) * d4
        + T::from_f64(coeff.fpA1B0D3) * alpha * d3
        + T::from_f64(coeff.fpA0B1D3) * beta * d3
        + T::from_f64(coeff.fpA2B0D2) * a2 * d2
        + T::from_f64(coeff.fpA0B2D2) * b2 * d2
        + T::from_f64(coeff.fpA1B1D2) * alpha * beta * d2
        + T::from_f64(coeff.fpA3B0D1) * a3 * delta
        + T::from_f64(coeff.fpA2B1D1) * a2 * beta * delta
        + T::from_f64(coeff.fpA1B2D1) * alpha * b2 * delta
        + T::from_f64(coeff.fpA0B3D1) * b3 * delta
}

fn eval_force_moment_quartic<T: Scalar>(
    coeffs: &ForceMomentQuartic2Export,
    alpha: T,
    beta: T,
) -> (Vector3<T>, Vector3<T>) {
    (
        Vector3::new(
            eval_quartic2(&coeffs.force[0], alpha, beta),
            eval_quartic2(&coeffs.force[1], alpha, beta),
            eval_quartic2(&coeffs.force[2], alpha, beta),
        ),
        Vector3::new(
            eval_quartic2(&coeffs.moment[0], alpha, beta),
            eval_quartic2(&coeffs.moment[1], alpha, beta),
            eval_quartic2(&coeffs.moment[2], alpha, beta),
        ),
    )
}

fn eval_force_moment_flap<T: Scalar>(
    coeffs: &ForceMomentFlapPolynomialExport,
    alpha: T,
    beta: T,
    delta: T,
) -> (Vector3<T>, Vector3<T>) {
    (
        Vector3::new(
            eval_flap_polynomial(&coeffs.force[0], alpha, beta, delta),
            eval_flap_polynomial(&coeffs.force[1], alpha, beta, delta),
            eval_flap_polynomial(&coeffs.force[2], alpha, beta, delta),
        ),
        Vector3::new(
            eval_flap_polynomial(&coeffs.moment[0], alpha, beta, delta),
            eval_flap_polynomial(&coeffs.moment[1], alpha, beta, delta),
            eval_flap_polynomial(&coeffs.moment[2], alpha, beta, delta),
        ),
    )
}

fn eval_aero_coeffs<T: Scalar>(
    alpha: T,
    beta: T,
    omega_b: &Vector3<T>,
    airspeed: T,
    params: &crate::types::AeroParams<T>,
    control: &KiteControls<T>,
) -> AeroCoeffEval<T> {
    let fit = reference_avl_fit_ref();
    let (nominal_force_w, nominal_moment_c) = eval_force_moment_quartic(&fit.nominal, alpha, beta);

    let rate_hat = Vector3::new(
        params.ref_span * omega_b[0] * T::from_f64(0.5) / airspeed,
        params.ref_chord * omega_b[1] * T::from_f64(0.5) / airspeed,
        params.ref_span * omega_b[2] * T::from_f64(0.5) / airspeed,
    );
    let mut pqr_force_w = zero_vec();
    let mut pqr_moment_c = zero_vec();
    for index in 0..3 {
        let (force_w, moment_c) = eval_force_moment_quartic(&fit.pqr[index], alpha, beta);
        pqr_force_w = add(&pqr_force_w, &scale(&force_w, rate_hat[index]));
        pqr_moment_c = add(&pqr_moment_c, &scale(&moment_c, rate_hat[index]));
    }

    let surface_inputs = [
        (&fit.flaps.r_aileron, control.surfaces.aileron),
        (&fit.flaps.flap, control.surfaces.flap),
        (&fit.flaps.winglet, control.surfaces.winglet),
        (&fit.flaps.elevator, control.surfaces.elevator),
        (&fit.flaps.rudder, control.surfaces.rudder),
    ];
    let mut surface_force_w = zero_vec();
    let mut surface_moment_c = zero_vec();
    for (surface_coeffs, delta) in surface_inputs {
        let (force_w, moment_c) = eval_force_moment_flap(surface_coeffs, alpha, beta, delta);
        surface_force_w = add(&surface_force_w, &force_w);
        surface_moment_c = add(&surface_moment_c, &moment_c);
    }

    AeroCoeffEval {
        total_force_w: add(&add(&nominal_force_w, &pqr_force_w), &surface_force_w),
        nominal_force_w,
        pqr_force_w,
        surface_force_w,
        total_moment_c: add(&add(&nominal_moment_c, &pqr_moment_c), &surface_moment_c),
        nominal_moment_c,
        pqr_moment_c,
        surface_moment_c,
    }
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
    let cad_position_n = add(
        &kite.body.pos_n,
        &rotate_body_to_nav(&kite.body.quat_n2b, &params.rigid_body.cad_offset_b),
    );
    let cad_velocity_b = add(
        &kite.body.vel_b,
        &cross(&kite.body.omega_b, &params.rigid_body.cad_offset_b),
    );
    let cad_velocity_n = rotate_body_to_nav(&kite.body.quat_n2b, &cad_velocity_b);
    let apparent_air_n = sub(&cad_velocity_n, &effective_wind_n);
    let apparent_air_b = rotate_nav_to_body(&kite.body.quat_n2b, &apparent_air_n);
    let airspeed = norm(&apparent_air_b);
    let alpha = apparent_air_b[2].atan2(apparent_air_b[0]);
    let beta = apparent_air_b[1].atan2(
        (apparent_air_b[0] * apparent_air_b[0] + apparent_air_b[2] * apparent_air_b[2]).sqrt(),
    );
    let qbar = T::from_f64(0.5) * common_params.environment.rho * airspeed * airspeed;
    let aero_coeffs = eval_aero_coeffs(
        alpha,
        beta,
        &kite.body.omega_b,
        airspeed,
        &params.aero,
        control,
    );
    let force_scale = qbar * params.aero.ref_area;
    let moment_scale = qbar * params.aero.ref_area;
    let airspeed_geometry = norm_exact(&apparent_air_b);
    let xz_norm =
        (apparent_air_b[0] * apparent_air_b[0] + apparent_air_b[2] * apparent_air_b[2]).sqrt();
    let wind_x_b = scale(&apparent_air_b, T::one() / airspeed_geometry);
    let wind_y_b = Vector3::new(
        -apparent_air_b[0] * apparent_air_b[1] / (xz_norm * airspeed_geometry),
        xz_norm / airspeed_geometry,
        -apparent_air_b[2] * apparent_air_b[1] / (xz_norm * airspeed_geometry),
    );
    let wind_z_b = Vector3::new(
        -apparent_air_b[2] / xz_norm,
        T::zero(),
        apparent_air_b[0] / xz_norm,
    );
    let aero_force_drag_b = scale(&wind_x_b, force_scale * aero_coeffs.total_force_w[0]);
    let aero_force_side_b = scale(&wind_y_b, force_scale * aero_coeffs.total_force_w[1]);
    let aero_force_lift_b = scale(&wind_z_b, force_scale * aero_coeffs.total_force_w[2]);
    let aero_force_b = add(
        &add(&aero_force_drag_b, &aero_force_side_b),
        &aero_force_lift_b,
    );
    let aero_moment_coeff_b = Vector3::new(
        moment_scale * params.aero.ref_span * aero_coeffs.total_moment_c[0],
        moment_scale * params.aero.ref_chord * aero_coeffs.total_moment_c[1],
        moment_scale * params.aero.ref_span * aero_coeffs.total_moment_c[2],
    );
    let aero_moment_b = add(
        &aero_moment_coeff_b,
        &cross(&params.rigid_body.cad_offset_b, &aero_force_b),
    );

    let rotor_fit = reference_rotor_fit_ref();
    let rotor_aero_thrust = eval_quartic2(&rotor_fit.aero_thrust, airspeed, kite.rotor_speed);
    let rotor_aero_torque = eval_quartic2(&rotor_fit.aero_torque, airspeed, kite.rotor_speed);
    let rotor_speed_dot = (control.motor_torque - rotor_aero_torque) / params.rotor.inertia;
    let rotor_axis_b = normalize_exact(&params.rotor.axis_b);
    let motor_force = rotor_aero_thrust;
    let motor_force_b = scale(&rotor_axis_b, rotor_aero_thrust);
    let motor_moment_b = add(
        &scale(&rotor_axis_b, -control.motor_torque * params.rotor.sign),
        &cross(&params.rotor.position_b, &motor_force_b),
    );
    let motor_power = control.motor_torque * kite.rotor_speed;
    let rotor_dissipated_power =
        rotor_aero_torque * kite.rotor_speed - dot(&motor_force_b, &kite.body.vel_b);
    let aero_dissipated_power =
        -force_scale * aero_coeffs.total_force_w[0] * airspeed + rotor_dissipated_power;
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
    let total_moment_b = add(&add(&aero_moment_b, &tether_moment_b), &motor_moment_b);
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
    let rotor_ke = T::from_f64(0.5) * params.rotor.inertia * square(kite.rotor_speed);
    let potential_energy =
        -params.rigid_body.mass * common_params.environment.g * kite.body.pos_n[2];
    let cl = -aero_coeffs.total_force_w[2];
    let cl_0_term = -aero_coeffs.nominal_force_w[2];
    let cl_alpha_term = -aero_coeffs.pqr_force_w[2];
    let cl_elevator_term = -aero_coeffs.surface_force_w[2];
    let cl_flap_term = T::zero();
    let cd = -aero_coeffs.total_force_w[0];
    let cd_0_term = -aero_coeffs.nominal_force_w[0];
    let cd_induced_term = -aero_coeffs.pqr_force_w[0];
    let cd_surface_term = -aero_coeffs.surface_force_w[0];
    let cy = aero_coeffs.total_force_w[1];
    let cy_beta_term = aero_coeffs.nominal_force_w[1] + aero_coeffs.pqr_force_w[1];
    let cy_rudder_term = aero_coeffs.surface_force_w[1];
    let roll_coeff = aero_coeffs.total_moment_c[0];
    let roll_beta_term = aero_coeffs.nominal_moment_c[0];
    let roll_p_term = aero_coeffs.pqr_moment_c[0];
    let roll_r_term = T::zero();
    let roll_aileron_term = aero_coeffs.surface_moment_c[0];
    let pitch_coeff = aero_coeffs.total_moment_c[1];
    let pitch_0_term = aero_coeffs.nominal_moment_c[1];
    let pitch_alpha_term = T::zero();
    let pitch_q_term = aero_coeffs.pqr_moment_c[1];
    let pitch_elevator_term = aero_coeffs.surface_moment_c[1];
    let pitch_flap_term = T::zero();
    let yaw_coeff = aero_coeffs.total_moment_c[2];
    let yaw_beta_term = aero_coeffs.nominal_moment_c[2];
    let yaw_p_term = T::zero();
    let yaw_r_term = aero_coeffs.pqr_moment_c[2];
    let yaw_rudder_term = aero_coeffs.surface_moment_c[2];

    (
        KiteState {
            body: body_ddt,
            rotor_speed: rotor_speed_dot,
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
            aero_dissipated_power,
            tether_dissipated_power: upper_tether.dissipated_power,
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
            motor_moment_b,
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
            kinetic_energy: translational_ke + rotational_ke + rotor_ke,
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
        rotor_speed: T::zero(),
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
    let kite_dissipated_power = diagnostics.kites.iter().fold(T::zero(), |acc, item| {
        acc + item.aero_dissipated_power + item.tether_dissipated_power
    });
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
    diagnostics.total_dissipated_power = kite_dissipated_power
        + common_tether.dissipated_power
        + payload_contact.dissipated_power
        + splitter_contact.dissipated_power;
    diagnostics.total_mechanical_energy = diagnostics.total_kinetic_energy
        + diagnostics.total_potential_energy
        + diagnostics.total_tether_strain_energy;
    diagnostics.energy_conservation_residual =
        diagnostics.total_mechanical_energy - state.mechanical_energy_reference - state.total_work
            + state.total_dissipated_work;
    diagnostics.work_minus_potential = state.total_work - diagnostics.total_potential_energy;

    (
        State {
            kites: kite_ddt,
            splitter: splitter_ddt,
            common_tether: common_tether.node_ddt,
            payload: payload_ddt,
            total_work: total_motor_power,
            total_dissipated_work: diagnostics.total_dissipated_power,
            mechanical_energy_reference: T::zero(),
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
    use super::{
        CompiledRhs, bridle_geometry, compute_kite, compute_tether, eval_aero_coeffs, eval_quartic2,
    };
    use crate::assets::reference_rotor_fit_ref;
    use crate::types::{
        AeroParams, BodyState, BridleParams, ControlSurfaces, ControllerGains, Environment,
        KiteControls, KiteParams, KiteState, MassContactParams, Params, RigidBodyParams,
        RotorParams, TetherNode, TetherParams,
    };
    use approx::assert_relative_eq;
    use nalgebra::{Quaternion, Vector3};
    use std::rc::Rc;

    fn assert_vec_close(actual: &Vector3<f64>, expected: [f64; 3], epsilon: f64) {
        for index in 0..3 {
            assert_relative_eq!(
                actual[index],
                expected[index],
                epsilon = epsilon,
                max_relative = epsilon
            );
        }
    }

    fn aero_parity_params() -> KiteParams<f64> {
        KiteParams {
            rigid_body: RigidBodyParams {
                mass: 20.0,
                inertia_diagonal: Vector3::new(4.647, 18.025, 22.743),
                cad_offset_b: Vector3::new(-0.02, 0.0, 0.0),
            },
            aero: AeroParams {
                ref_area: 1.6,
                ref_span: 3.1,
                ref_chord: 0.5167,
                cl0: 0.0,
                cl_alpha: 0.0,
                cl_elevator: 0.0,
                cl_flap: 0.0,
                cd0: 0.0,
                cd_induced: 0.0,
                cd_surface_abs: 0.0,
                cy_beta: 0.0,
                cy_rudder: 0.0,
                roll_beta: 0.0,
                roll_p: 0.0,
                roll_r: 0.0,
                roll_aileron: 0.0,
                pitch0: 0.0,
                pitch_alpha: 0.0,
                pitch_q: 0.0,
                pitch_elevator: 0.0,
                pitch_flap: 0.0,
                yaw_beta: 0.0,
                yaw_p: 0.0,
                yaw_r: 0.0,
                yaw_rudder: 0.0,
            },
            bridle: BridleParams {
                pivot_b: Vector3::new(0.05, 0.0, 0.2),
                radius: 0.05,
            },
            tether: TetherParams {
                natural_length: 120.0,
                total_mass: 0.5,
                ea: 157079.63267948964,
                viscous_damping_coeff: 3.0,
                cd_phi: 1.2,
                diameter: 0.002,
                contact: MassContactParams {
                    zeta: 0.7071067811865476,
                    enable_length: 0.2,
                    ground_altitude: 0.0,
                },
            },
            rotor: RotorParams {
                axis_b: Vector3::new(1.0, 0.0, 0.0),
                position_b: Vector3::new(1.0, 0.0, 0.0),
                radius: 0.07,
                inertia: 0.01,
                sign: 1.0,
                initial_speed: 335.0,
            },
        }
    }

    #[test]
    fn compiled_rhs_shared_reuses_instance() {
        let first = CompiledRhs::<1, 0, 0>::shared().expect("first compiled rhs");
        let second = CompiledRhs::<1, 0, 0>::shared().expect("second compiled rhs");
        assert!(Rc::ptr_eq(&first, &second));
    }

    #[test]
    fn tether_drag_uses_midpoint_tangent_length_like_reference() {
        let params = TetherParams {
            natural_length: 30.0,
            total_mass: 3.0,
            ea: 0.0,
            viscous_damping_coeff: 0.0,
            cd_phi: 1.0,
            diameter: 1.0,
            contact: MassContactParams {
                zeta: 0.0,
                enable_length: 1.0,
                ground_altitude: 1000.0,
            },
        };
        let bottom = TetherNode {
            pos_n: Vector3::new(0.0, 0.0, 0.0),
            vel_n: Vector3::zeros(),
        };
        let top = TetherNode {
            pos_n: Vector3::new(0.0, 30.0, 0.0),
            vel_n: Vector3::zeros(),
        };
        let nodes = [
            TetherNode {
                pos_n: Vector3::new(0.0, 5.0, 0.0),
                vel_n: Vector3::new(1.0, 0.0, 0.0),
            },
            TetherNode {
                pos_n: Vector3::new(0.0, 15.0, 0.0),
                vel_n: Vector3::new(1.0, 0.0, 0.0),
            },
            TetherNode {
                pos_n: Vector3::new(0.0, 25.0, 0.0),
                vel_n: Vector3::new(1.0, 0.0, 0.0),
            },
        ];

        let output = compute_tether(&bottom, &top, &params, &Vector3::zeros(), 2.0, 0.0, &nodes);

        assert!((output.node_ddt[1].vel_n[0] + 10.0).abs() < 1.0e-7);
    }

    #[test]
    fn haskell_tether_model_golden_matches_pre_removal_source() {
        // Source: reference_source@2052ae8e69af45be8a2aee4eee14edd9c88ff68f
        // sim/models/src/Kitty/Models/Tether.hs:tetherModel.
        // This pins the half-segment end lengths, spring/damping force signs,
        // midpoint drag tangent, top/bottom endpoint loads, and integrated
        // strain/dissipated energy formulas.
        let params = TetherParams {
            natural_length: 12.0,
            total_mass: 6.0,
            ea: 1200.0,
            viscous_damping_coeff: 0.7,
            cd_phi: 1.15,
            diameter: 0.018,
            contact: MassContactParams {
                zeta: 0.0,
                enable_length: 0.2,
                ground_altitude: 0.0,
            },
        };
        let bottom = TetherNode {
            pos_n: Vector3::new(0.3, -0.1, -2.0),
            vel_n: Vector3::new(0.2, -0.4, 0.1),
        };
        let top = TetherNode {
            pos_n: Vector3::new(8.0, 10.5, -3.2),
            vel_n: Vector3::new(-0.1, 0.25, -0.05),
        };
        let nodes = [
            TetherNode {
                pos_n: Vector3::new(1.4, 1.7, -2.1),
                vel_n: Vector3::new(0.5, -0.2, 0.0),
            },
            TetherNode {
                pos_n: Vector3::new(3.9, 5.1, -2.6),
                vel_n: Vector3::new(-0.3, 0.1, 0.2),
            },
            TetherNode {
                pos_n: Vector3::new(6.5, 8.4, -3.0),
                vel_n: Vector3::new(0.2, 0.3, -0.1),
            },
        ];

        let output = compute_tether(
            &bottom,
            &top,
            &params,
            &Vector3::new(0.4, -0.2, 0.1),
            1.18,
            9.81,
            &nodes,
        );

        assert_vec_close(&output.node_ddt[0].pos_n, [0.5, -0.2, 0.0], 1.0e-12);
        assert_vec_close(&output.node_ddt[1].pos_n, [-0.3, 0.1, 0.2], 1.0e-12);
        assert_vec_close(&output.node_ddt[2].pos_n, [0.2, 0.3, -0.1], 1.0e-12);
        assert_vec_close(
            &output.node_ddt[0].vel_n,
            [4.203263025128117, 0.8505319449131126, 7.048680906556031],
            1.0e-10,
        );
        assert_vec_close(
            &output.node_ddt[1].vel_n,
            [-1.0039651503622458, -3.2770555040420226, 10.971363920270305],
            1.0e-10,
        );
        assert_vec_close(
            &output.node_ddt[2].vel_n,
            [81.39351784563422, 116.65936286911791, -0.612693460671494],
            1.0e-10,
        );
        assert_vec_close(
            &output.force_on_top_n,
            [-204.37013917209009, -286.11819484092609, 27.24935188961204],
            1.0e-10,
        );
        assert_vec_close(
            &output.force_on_bottom_n,
            [35.21710846743428, 57.62799567398337, -3.201555315221301],
            1.0e-10,
        );
        assert_relative_eq!(
            output.top_tension,
            352.66599266537384,
            epsilon = 1.0e-10,
            max_relative = 1.0e-10
        );
        assert_relative_eq!(
            output.strain_energy,
            124.25703518371334,
            epsilon = 1.0e-10,
            max_relative = 1.0e-10
        );
        assert_relative_eq!(
            output.dissipated_power,
            1.1866216254572319,
            epsilon = 1.0e-12,
            max_relative = 1.0e-12
        );
    }

    #[test]
    fn haskell_bridle_geometry_golden_matches_pre_removal_source() {
        // Source: reference_source@2052ae8e69af45be8a2aee4eee14edd9c88ff68f
        // sim/models/src/Kitty/Models/Bridle.hs:bridleModel.
        // The bridle knot is constrained to the body X/Z plane about the pivot,
        // with velocity projected tangentially in that plane.
        let yaw = 0.7_f64;
        let body = BodyState {
            pos_n: Vector3::new(10.0, -5.0, -20.0),
            vel_b: Vector3::new(12.0, -0.8, 0.5),
            quat_n2b: Quaternion::new((0.5 * yaw).cos(), 0.0, 0.0, (0.5 * yaw).sin()),
            omega_b: Vector3::new(0.2, -0.3, 0.4),
        };
        let last_tether_node = TetherNode {
            pos_n: Vector3::new(12.0, -6.0, -21.0),
            vel_n: Vector3::new(3.0, 2.0, -1.0),
        };

        let (bridle_pos_n, bridle_vel_n, bridle_offset_b) = bridle_geometry(
            &body,
            &Vector3::new(-0.02, 0.0, 0.0),
            &Vector3::new(0.05, 0.0, 0.2),
            0.05,
            &last_tether_node,
        );

        assert_vec_close(
            &bridle_pos_n,
            [10.0563156182378, -5.047433990880292, -19.824421102424353],
            1.0e-12,
        );
        assert_vec_close(
            &bridle_vel_n,
            [8.548193283146091, -8.282620143405913, 0.3907803112404863],
            1.0e-12,
        );
        assert_vec_close(
            &bridle_offset_b,
            [0.04363037653263289, 0.0, -0.024421102424351707],
            1.0e-12,
        );
    }

    #[test]
    fn e189_aero_fit_evaluation_matches_haskell_coeff_model() {
        // Source: reference_source@e18990d54
        // kittybutt/core/src/Kitty/Models/Aero/AeroCoeffs.hs:evalAeroCoeffs.
        // This pins nominal quartic, p/q/r rate scaling, and direct surface
        // polynomial summation for the vendored Reference AVL fit.
        let params = aero_parity_params();
        let control = KiteControls {
            surfaces: ControlSurfaces {
                aileron: 0.04,
                flap: -0.015,
                winglet: -0.02,
                elevator: 0.03,
                rudder: -0.05,
            },
            motor_torque: 0.0,
        };

        let coeffs = eval_aero_coeffs(
            0.11,
            -0.07,
            &Vector3::new(0.6, -0.4, 0.2),
            23.5,
            &params.aero,
            &control,
        );

        assert_vec_close(
            &coeffs.nominal_force_w,
            [
                -0.0918263150852303,
                0.03724505132241283,
                -0.9197449154252236,
            ],
            1.0e-12,
        );
        assert_vec_close(
            &coeffs.pqr_force_w,
            [
                0.0037595310726404553,
                0.003222298128795805,
                0.04669063074638498,
            ],
            1.0e-12,
        );
        assert_vec_close(
            &coeffs.surface_force_w,
            [
                0.0064789288380503645,
                -0.015516955734825835,
                0.01882603192620346,
            ],
            1.0e-12,
        );
        assert_vec_close(
            &coeffs.total_force_w,
            [
                -0.08158785517453948,
                0.024950393716382798,
                -0.8542282527526351,
            ],
            1.0e-12,
        );
        assert_vec_close(
            &coeffs.nominal_moment_c,
            [
                0.01518248766259375,
                -0.13894895141993285,
                -0.008438372173576817,
            ],
            1.0e-12,
        );
        assert_vec_close(
            &coeffs.pqr_moment_c,
            [
                -0.02026412482377206,
                0.07947840238777976,
                -0.006584974874487046,
            ],
            1.0e-12,
        );
        assert_vec_close(
            &coeffs.surface_moment_c,
            [
                -0.02305955237904088,
                -0.05560220920883867,
                0.007984451211538448,
            ],
            1.0e-12,
        );
        assert_vec_close(
            &coeffs.total_moment_c,
            [
                -0.028141189540219186,
                -0.11507275824099177,
                -0.007038895836525415,
            ],
            1.0e-12,
        );
    }

    #[test]
    fn e189_aero_force_moment_path_matches_haskell_aircraft_model() {
        // Sources:
        // - reference_source@e18990d54 kittybutt/core/src/Kitty/Models/Aero/AeroFrames.hs
        // - reference_source@e18990d54 kittybutt/core/src/Kitty/Models/Aero/AeroCoeffs.hs
        // - reference_source@2052ae8e69af45be8a2aee4eee14edd9c88ff68f sim/models/src/Kitty/Models/Aircraft.hs
        //
        // This is the full non-rotor aero plant path used by compute_kite:
        // CAD apparent wind -> alpha/beta -> AVL fit coefficients -> wind-axis
        // force rotation -> qbar/Sref scaling -> reference-length moment scaling
        // -> body-origin moment shift from the CAD offset. Strip/crossfade and
        // blown lift are intentionally out of scope for this milestone path.
        let kite_params = aero_parity_params();
        let common_params = Params {
            kites: [kite_params.clone()],
            common_tether: kite_params.tether.clone(),
            splitter_mass: 0.0,
            payload_mass: 0.0,
            environment: Environment {
                rho: 1.225,
                g: 0.0,
                wind_n: Vector3::new(1.5, -0.4, 0.3),
            },
            kite_gusts_n: [Vector3::zeros()],
            controller: ControllerGains {
                trim: KiteControls::zero(),
                wx_to_ail: 0.0,
                wy_to_elev: 0.0,
                wz_to_rudder: 0.0,
                speed_to_torque_p: 0.0,
                speed_to_torque_i: 0.0,
                rabbit_distance: 0.0,
                phase_lag_to_radius: 0.0,
                vert_vel_to_rabbit_height: 0.0,
                gain_int_y: 0.0,
                gain_int_z: 0.0,
                speed_ref: 0.0,
                disk_center_n: Vector3::zeros(),
                disk_radius: 1.0,
            },
        };
        let state = KiteState::<f64, 0> {
            body: BodyState {
                pos_n: Vector3::new(2.0, -3.0, -40.0),
                vel_b: Vector3::new(21.0, -1.5, 2.8),
                quat_n2b: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                omega_b: Vector3::new(0.6, -0.4, 0.2),
            },
            rotor_speed: 335.0,
            tether: [],
        };
        let controls = KiteControls {
            surfaces: ControlSurfaces {
                aileron: 0.04,
                flap: -0.015,
                winglet: -0.02,
                elevator: 0.03,
                rudder: -0.05,
            },
            motor_torque: 0.0,
        };
        let splitter = TetherNode {
            pos_n: Vector3::new(10.0, 0.0, -100.0),
            vel_n: Vector3::zeros(),
        };

        let (_, diagnostics, _) =
            compute_kite(&state, &controls, &kite_params, &common_params, &splitter);

        assert_vec_close(&diagnostics.cad_velocity_n, [21.0, -1.504, 2.792], 1.0e-12);
        assert_relative_eq!(
            diagnostics.airspeed,
            19.689562717363735,
            epsilon = 1.0e-12,
            max_relative = 1.0e-12
        );
        assert_relative_eq!(
            diagnostics.alpha,
            0.12710591509909316,
            epsilon = 1.0e-14,
            max_relative = 1.0e-14
        );
        assert_relative_eq!(
            diagnostics.beta,
            -0.05609973719397039,
            epsilon = 1.0e-14,
            max_relative = 1.0e-14
        );
        assert_vec_close(
            &diagnostics.aero_force_drag_b,
            [-38.73217462888582, 2.192836963604613, -4.949773291035051],
            1.0e-10,
        );
        assert_vec_close(
            &diagnostics.aero_force_side_b,
            [0.39206329942488694, 7.038127452052814, 0.05010367908547786],
            1.0e-10,
        );
        assert_vec_close(
            &diagnostics.aero_force_lift_b,
            [44.88432670366575, -0.0, -351.22165759289015],
            1.0e-10,
        );
        assert_vec_close(
            &diagnostics.aero_force_b,
            [6.544215374204811, 9.230964415657427, -356.1213272048397],
            1.0e-10,
        );
        assert_vec_close(
            &diagnostics.aero_moment_b,
            [-40.840459744696574, -32.926015619392516, -7.491818909071911],
            1.0e-10,
        );
    }

    #[test]
    fn e189_rotor_fit_evaluation_matches_haskell_rotor_model() {
        // Source: reference_source@e18990d54
        // kittybutt/core/src/Kitty/Models/Rotors.hs:rotorModel.
        // The fitted XROTOR thrust is intentionally pinned with its sign.
        let fit = reference_rotor_fit_ref();
        let airspeed = 23.5;
        let motor_speed = 335.0;
        let motor_torque = 2.1;
        let rotor_aero_thrust = eval_quartic2(&fit.aero_thrust, airspeed, motor_speed);
        let rotor_aero_torque = eval_quartic2(&fit.aero_torque, airspeed, motor_speed);
        let rotor_speed_dot = (motor_torque - rotor_aero_torque) / 0.01;
        let motor_power = motor_torque * motor_speed;
        let dissipated_power = rotor_aero_torque * motor_speed - rotor_aero_thrust * airspeed;

        assert_relative_eq!(
            rotor_aero_thrust,
            -27.63000728138548,
            epsilon = 1.0e-10,
            max_relative = 1.0e-10
        );
        assert_relative_eq!(
            rotor_aero_torque,
            -0.4050971499927598,
            epsilon = 1.0e-12,
            max_relative = 1.0e-12
        );
        assert_relative_eq!(
            rotor_speed_dot,
            250.50971499927596,
            epsilon = 1.0e-10,
            max_relative = 1.0e-10
        );
        assert_relative_eq!(
            motor_power,
            703.5,
            epsilon = 1.0e-12,
            max_relative = 1.0e-12
        );
        assert_relative_eq!(
            dissipated_power,
            513.59762586498425,
            epsilon = 1.0e-10,
            max_relative = 1.0e-10
        );
    }
}
