use crate::common::{
    ControlSpec, FromMap, LatexSection, MetricKey, PlotMode, ProblemId, ProblemSpec, Scene2D,
    SceneAnimation, SceneFrame, ScenePath, SolveArtifact, SolveStreamEvent, SolverMethod,
    SolverReport, SqpConfig, TranscriptionConfig, TranscriptionMethod, chart,
    default_solver_method, default_sqp_config, default_transcription, deg_to_rad,
    emit_symbolic_setup_status, expect_finite, interval_arc_bound_series, interval_arc_series,
    metric_with_key, node_times, numeric_metric_with_key, rad_to_deg, sample_or_default,
    segmented_bound_series, segmented_series, solve_direct_collocation_problem,
    solve_direct_collocation_problem_with_progress, solve_multiple_shooting_problem,
    solve_multiple_shooting_problem_with_progress, solver_config_from_map, solver_controls,
    solver_method_from_map, transcription_controls, transcription_from_map, transcription_metrics,
};
use anyhow::Result;
use optimal_control::{
    Bounds1D, DirectCollocation, DirectCollocationInitialGuess, DirectCollocationRuntimeValues,
    DirectCollocationTimeGrid, DirectCollocationTrajectories, InterpolatedTrajectory, IntervalArc,
    MultipleShooting, MultipleShootingInitialGuess, MultipleShootingRuntimeValues,
    MultipleShootingTrajectories, Ocp, direct_collocation_root_arcs,
    direct_collocation_state_like_arcs,
};
use serde::Serialize;
use std::collections::BTreeMap;
use sx_core::SX;
const DEFAULT_INTERVALS: usize = 30;
const DEFAULT_COLLOCATION_DEGREE: usize = 3;
const SUPPORTED_INTERVALS: [usize; 1] = [DEFAULT_INTERVALS];
const SUPPORTED_DEGREES: [usize; 1] = [DEFAULT_COLLOCATION_DEGREE];
const G: f64 = 9.81;
const CART_MASS_KG: f64 = 8.0;
const LOAD_MASS_KG: f64 = 1.6;
const CART_FRICTION_NSPM: f64 = 3.0;
const PIVOT_FRICTION_NMS: f64 = 0.9;
const FORCE_WEIGHT: f64 = 2.0e-4;
#[derive(Clone, Debug, PartialEq, Serialize, optimization::Vectorize)]
pub struct State<T> {
    pub x: T,
    pub v: T,
    pub theta: T,
    pub omega: T,
}
#[derive(Clone, Debug, PartialEq, Serialize, optimization::Vectorize)]
pub struct Control<T> {
    pub force: T,
}
#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct Path<T> {
    x: T,
    theta: T,
    force: T,
    force_rate: T,
}
#[derive(Clone, Debug, PartialEq, optimization::Vectorize)]
struct Boundary<T> {
    x0: T,
    v0: T,
    theta0: T,
    omega0: T,
    force0: T,
    x_t: T,
    v_t: T,
    theta_t: T,
    omega_t: T,
    force_t: T,
}
#[derive(Clone, Debug)]
pub struct Params {
    pub target_x_m: f64,
    pub tf_s: f64,
    pub rope_length_m: f64,
    pub force_limit_n: f64,
    pub force_rate_limit_nps: f64,
    pub force_rate_regularization: f64,
    pub swing_limit_deg: f64,
    pub solver_method: SolverMethod,
    pub solver: SqpConfig,
    pub transcription: TranscriptionConfig,
}
impl Default for Params {
    fn default() -> Self {
        Self {
            target_x_m: 10.0,
            tf_s: 7.5,
            rope_length_m: 3.0,
            force_limit_n: 28.0,
            force_rate_limit_nps: 75.0,
            force_rate_regularization: 5.0e-4,
            swing_limit_deg: 15.0,
            solver_method: default_solver_method(),
            solver: default_sqp_config(),
            transcription: default_transcription(DEFAULT_INTERVALS),
        }
    }
}
impl FromMap for Params {
    fn from_map(values: &BTreeMap<String, f64>) -> Result<Self> {
        let defaults = Self::default();
        Ok(Self {
            target_x_m: expect_finite(
                sample_or_default(values, "target_x_m", defaults.target_x_m),
                "target_x_m",
            )?,
            tf_s: expect_finite(sample_or_default(values, "tf_s", defaults.tf_s), "tf_s")?,
            rope_length_m: expect_finite(
                sample_or_default(values, "rope_length_m", defaults.rope_length_m),
                "rope_length_m",
            )?,
            force_limit_n: expect_finite(
                sample_or_default(values, "force_limit_n", defaults.force_limit_n),
                "force_limit_n",
            )?,
            force_rate_limit_nps: expect_finite(
                sample_or_default(
                    values,
                    "force_rate_limit_nps",
                    defaults.force_rate_limit_nps,
                ),
                "force_rate_limit_nps",
            )?,
            force_rate_regularization: expect_finite(
                sample_or_default(
                    values,
                    "force_rate_regularization",
                    defaults.force_rate_regularization,
                ),
                "force_rate_regularization",
            )?,
            swing_limit_deg: expect_finite(
                sample_or_default(values, "swing_limit_deg", defaults.swing_limit_deg),
                "swing_limit_deg",
            )?,
            solver_method: solver_method_from_map(values, defaults.solver_method)?,
            solver: solver_config_from_map(values, defaults.solver)?,
            transcription: transcription_from_map(
                values,
                defaults.transcription,
                &SUPPORTED_INTERVALS,
                &SUPPORTED_DEGREES,
            )?,
        })
    }
}
pub fn spec() -> ProblemSpec {
    let mut controls = transcription_controls(
        Params::default().transcription,
        &SUPPORTED_INTERVALS,
        &SUPPORTED_DEGREES,
    );
    controls.extend(solver_controls(
        Params::default().solver_method,
        Params::default().solver,
    ));
    controls.extend(vec![
        ControlSpec {
            id: "target_x_m".to_string(),
            label: "Target X".to_string(),
            min: 4.0,
            max: 16.0,
            step: 0.25,
            default: Params::default().target_x_m,
            unit: "m".to_string(),
            help: "Trolley travel distance.".to_string(),
            choices: Vec::new(),
            ..Default::default()
        },
        ControlSpec {
            id: "tf_s".to_string(),
            label: "Transfer Time".to_string(),
            min: 3.0,
            max: 10.0,
            step: 0.25,
            default: Params::default().tf_s,
            unit: "s".to_string(),
            help: "Fixed time allocated for the move.".to_string(),
            choices: Vec::new(),
            ..Default::default()
        },
        ControlSpec {
            id: "rope_length_m".to_string(),
            label: "Rope Length".to_string(),
            min: 1.0,
            max: 6.0,
            step: 0.1,
            default: Params::default().rope_length_m,
            unit: "m".to_string(),
            help: "Distance from trolley hook to the payload.".to_string(),
            choices: Vec::new(),
            ..Default::default()
        },
        ControlSpec {
            id: "force_limit_n".to_string(),
            label: "Force Limit".to_string(),
            min: 5.0,
            max: 80.0,
            step: 1.0,
            default: Params::default().force_limit_n,
            unit: "N".to_string(),
            help: "Absolute bound on commanded cart force.".to_string(),
            choices: Vec::new(),
            ..Default::default()
        },
        ControlSpec {
            id: "force_rate_limit_nps".to_string(),
            label: "Force-Rate Limit".to_string(),
            min: 5.0,
            max: 200.0,
            step: 1.0,
            default: Params::default().force_rate_limit_nps,
            unit: "N/s".to_string(),
            help: "Absolute bound on the rate of change of cart force.".to_string(),
            choices: Vec::new(),
            ..Default::default()
        },
        ControlSpec {
            id: "force_rate_regularization".to_string(),
            label: "Force-Rate Weight".to_string(),
            min: 0.0,
            max: 5.0e-3,
            step: 5.0e-5,
            default: Params::default().force_rate_regularization,
            unit: "".to_string(),
            help: "Quadratic stage-cost weight on force-rate effort.".to_string(),
            choices: Vec::new(),
            ..Default::default()
        },
        ControlSpec {
            id: "swing_limit_deg".to_string(),
            label: "Swing Limit".to_string(),
            min: 3.0,
            max: 25.0,
            step: 0.5,
            default: Params::default().swing_limit_deg,
            unit: "deg".to_string(),
            help: "Maximum payload swing angle.".to_string(),
            choices: Vec::new(),
            ..Default::default()
        },
    ]);
    ProblemSpec {
        id: ProblemId::CraneTransfer,
        name: "Crane Load Transfer".to_string(),
        description: "A trolley drives a suspended load with a bounded cart force and bounded force-rate while respecting the fully coupled cart-pole dynamics.".to_string(),
        controls,
        math_sections: vec![
            LatexSection {
                title: "Physical State".to_string(),
                entries: vec![r"\mathbf{x} = \begin{bmatrix} x & v & \theta & \omega \end{bmatrix}^{\mathsf T}".to_string()],
            },
            LatexSection {
                title: "Control-State".to_string(),
                entries: vec![
                    r"\mathbf{u} = \begin{bmatrix} F \end{bmatrix}".to_string(),
                    r"\dot{\mathbf{u}} = \begin{bmatrix} \dot{F} \end{bmatrix}".to_string(),
                ],
            },
            LatexSection {
                title: "Objective".to_string(),
                entries: vec![
                    r"J = \int_0^T \left(1.6\,\theta^2 + w_{\dot{F}}\,\dot{F}^2 + 2\cdot 10^{-4} F^2\right) \, dt".to_string(),
                ],
            },
            LatexSection {
                title: "Differential Equations".to_string(),
                entries: vec![
                    r"\dot{x} = v".to_string(),
                    r"\dot{\theta} = \omega".to_string(),
                    r"\dot{v} = \frac{F - b\,v + m\ell\,\omega^2\sin\theta + m g \sin\theta \cos\theta + \frac{c}{\ell}\,\omega\cos\theta}{M + m\sin^2\theta}".to_string(),
                    r"\dot{\omega} = \frac{-F\cos\theta + b\,v\cos\theta - m\ell\,\omega^2\sin\theta\cos\theta - (M+m)g\sin\theta - \frac{(M+m)c}{m\ell}\,\omega}{\ell\,(M + m\sin^2\theta)}".to_string(),
                    r"\dot{F} = \dot{F}_{\mathrm{cmd}}".to_string(),
                ],
            },
        ],
        notes: vec![
            format!(
                "The cart-pole model uses M = {:.1} kg, m = {:.1} kg, cart friction b = {:.1} N·s/m, and pivot damping c = {:.1} N·m·s.",
                CART_MASS_KG, LOAD_MASS_KG, CART_FRICTION_NSPM, PIVOT_FRICTION_NMS
            ),
            "This example supports both multiple shooting and direct collocation with the same coupled cart-pole dynamics.".to_string(),
        ],
    }
}
fn model_ms<const N: usize>(
    params: &Params,
) -> Ocp<State<SX>, Control<SX>, (), Path<SX>, Boundary<SX>, (), MultipleShooting<N, 2>> {
    let rope_length = params.rope_length_m;
    let force_rate_weight = params.force_rate_regularization;
    Ocp::new("crane_transfer", MultipleShooting::<N, 2>)
        .objective_lagrange(
            move |state: &State<SX>, control: &Control<SX>, force_rate: &Control<SX>, _: &()| {
                1.6 * state.theta.sqr()
                    + force_rate_weight * force_rate.force.sqr()
                    + FORCE_WEIGHT * control.force.sqr()
            },
        )
        .objective_mayer(
            |_: &State<SX>, _: &Control<SX>, _: &State<SX>, _: &Control<SX>, _: &(), _: &SX| {
                SX::zero()
            },
        )
        .ode(move |state: &State<SX>, control: &Control<SX>, _: &()| {
            let (x_ddot, theta_ddot) = cart_pole_accelerations_sx(state, control, rope_length);
            State {
                x: state.v,
                v: x_ddot,
                theta: state.omega,
                omega: theta_ddot,
            }
        })
        .path_constraints(
            |state: &State<SX>, control: &Control<SX>, force_rate: &Control<SX>, _: &()| Path {
                x: state.x,
                theta: state.theta,
                force: control.force,
                force_rate: force_rate.force,
            },
        )
        .boundary_equalities(
            |initial: &State<SX>,
             initial_control: &Control<SX>,
             terminal: &State<SX>,
             terminal_control: &Control<SX>,
             _: &(),
             _: &SX| Boundary {
                x0: initial.x,
                v0: initial.v,
                theta0: initial.theta,
                omega0: initial.omega,
                force0: initial_control.force,
                x_t: terminal.x,
                v_t: terminal.v,
                theta_t: terminal.theta,
                omega_t: terminal.omega,
                force_t: terminal_control.force,
            },
        )
        .boundary_inequalities(
            |_: &State<SX>, _: &Control<SX>, _: &State<SX>, _: &Control<SX>, _: &(), _: &SX| (),
        )
        .build()
        .expect("crane model should build")
}
fn model_dc<const N: usize, const K: usize>(
    params: &Params,
) -> Ocp<State<SX>, Control<SX>, (), Path<SX>, Boundary<SX>, (), DirectCollocation<N, K>> {
    let rope_length = params.rope_length_m;
    let force_rate_weight = params.force_rate_regularization;
    Ocp::new(
        "crane_transfer",
        DirectCollocation::<N, K> {
            family: params.transcription.collocation_family,
        },
    )
    .objective_lagrange(
        move |state: &State<SX>, control: &Control<SX>, force_rate: &Control<SX>, _: &()| {
            1.6 * state.theta.sqr()
                + force_rate_weight * force_rate.force.sqr()
                + FORCE_WEIGHT * control.force.sqr()
        },
    )
    .objective_mayer(
        |_: &State<SX>, _: &Control<SX>, _: &State<SX>, _: &Control<SX>, _: &(), _: &SX| SX::zero(),
    )
    .ode(move |state: &State<SX>, control: &Control<SX>, _: &()| {
        let (x_ddot, theta_ddot) = cart_pole_accelerations_sx(state, control, rope_length);
        State {
            x: state.v,
            v: x_ddot,
            theta: state.omega,
            omega: theta_ddot,
        }
    })
    .path_constraints(
        |state: &State<SX>, control: &Control<SX>, force_rate: &Control<SX>, _: &()| Path {
            x: state.x,
            theta: state.theta,
            force: control.force,
            force_rate: force_rate.force,
        },
    )
    .boundary_equalities(
        |initial: &State<SX>,
         initial_control: &Control<SX>,
         terminal: &State<SX>,
         terminal_control: &Control<SX>,
         _: &(),
         _: &SX| Boundary {
            x0: initial.x,
            v0: initial.v,
            theta0: initial.theta,
            omega0: initial.omega,
            force0: initial_control.force,
            x_t: terminal.x,
            v_t: terminal.v,
            theta_t: terminal.theta,
            omega_t: terminal.omega,
            force_t: terminal_control.force,
        },
    )
    .boundary_inequalities(
        |_: &State<SX>, _: &Control<SX>, _: &State<SX>, _: &Control<SX>, _: &(), _: &SX| (),
    )
    .build()
    .expect("crane model should build")
}
fn smoothstep(s: f64) -> f64 {
    let clipped = s.clamp(0.0, 1.0);
    clipped * clipped * (3.0 - 2.0 * clipped)
}

fn cart_pole_accelerations_sx(
    state: &State<SX>,
    control: &Control<SX>,
    rope_length: f64,
) -> (SX, SX) {
    let sin_theta = state.theta.sin();
    let cos_theta = state.theta.cos();
    let denominator = CART_MASS_KG + LOAD_MASS_KG * sin_theta.sqr();
    let x_ddot = (control.force - CART_FRICTION_NSPM * state.v
        + (LOAD_MASS_KG * rope_length) * state.omega.sqr() * sin_theta
        + (LOAD_MASS_KG * G) * sin_theta * cos_theta
        + (PIVOT_FRICTION_NMS / rope_length) * state.omega * cos_theta)
        / denominator;
    let theta_ddot = (-control.force * cos_theta + CART_FRICTION_NSPM * state.v * cos_theta
        - (LOAD_MASS_KG * rope_length) * state.omega.sqr() * sin_theta * cos_theta
        - ((CART_MASS_KG + LOAD_MASS_KG) * G) * sin_theta
        - ((CART_MASS_KG + LOAD_MASS_KG) * PIVOT_FRICTION_NMS / (LOAD_MASS_KG * rope_length))
            * state.omega)
        / (rope_length * denominator);
    (x_ddot, theta_ddot)
}

fn finite_difference(values: &[f64], dt: f64) -> Vec<f64> {
    (0..values.len())
        .map(|index| {
            if index == 0 {
                (values[1] - values[0]) / dt
            } else if index + 1 == values.len() {
                (values[index] - values[index - 1]) / dt
            } else {
                (values[index + 1] - values[index - 1]) / (2.0 * dt)
            }
        })
        .collect()
}
fn guess<const N: usize>(params: &Params) -> InterpolatedTrajectory<State<f64>, Control<f64>> {
    let sample_count = 2 * N + 1;
    let dt = params.tf_s / (sample_count as f64 - 1.0);
    let times = (0..sample_count)
        .map(|index| index as f64 * dt)
        .collect::<Vec<_>>();
    let x = times
        .iter()
        .map(|time| params.target_x_m * smoothstep(time / params.tf_s))
        .collect::<Vec<_>>();
    let v = finite_difference(&x, dt);
    let accel = finite_difference(&v, dt);
    let force = accel
        .iter()
        .zip(v.iter())
        .map(|(accel, v)| CART_MASS_KG * accel + CART_FRICTION_NSPM * v)
        .collect::<Vec<_>>();
    let force_rate = finite_difference(&force, dt);
    InterpolatedTrajectory {
        sample_times: times,
        x_samples: x
            .iter()
            .zip(v.iter())
            .map(|(x, v)| State {
                x: *x,
                v: *v,
                theta: 0.0,
                omega: 0.0,
            })
            .collect(),
        u_samples: force
            .iter()
            .map(|force| Control { force: *force })
            .collect(),
        dudt_samples: force_rate
            .iter()
            .map(|force_rate| Control { force: *force_rate })
            .collect(),
        tf: params.tf_s,
    }
}
fn load_position(state: &State<f64>, rope_length: f64) -> (f64, f64) {
    (
        state.x + rope_length * state.theta.sin(),
        -rope_length * state.theta.cos(),
    )
}
fn frame_for_state(state: &State<f64>, rope_length: f64) -> SceneFrame {
    let trolley = [state.x, 0.0];
    let (load_x, load_y) = load_position(state, rope_length);
    let left_wheel = [state.x - 0.45, 0.0];
    let right_wheel = [state.x + 0.45, 0.0];
    let mut points = BTreeMap::new();
    points.insert("trolley".to_string(), trolley);
    points.insert("load".to_string(), [load_x, load_y]);
    SceneFrame {
        points,
        segments: vec![
            (left_wheel, right_wheel),
            (trolley, [load_x, load_y]),
            ([load_x - 0.25, load_y], [load_x + 0.25, load_y]),
        ],
    }
}
pub fn solve(params: &Params) -> Result<SolveArtifact> {
    match params.transcription.method {
        TranscriptionMethod::MultipleShooting => {
            solve_multiple_shooting::<DEFAULT_INTERVALS>(params)
        }
        TranscriptionMethod::DirectCollocation => {
            solve_direct_collocation::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE>(params)
        }
    }
}
pub fn solve_with_progress<F>(params: &Params, emit: F) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    match params.transcription.method {
        TranscriptionMethod::MultipleShooting => {
            solve_multiple_shooting_with_progress::<DEFAULT_INTERVALS, F>(params, emit)
        }
        TranscriptionMethod::DirectCollocation => {
            solve_direct_collocation_with_progress::<DEFAULT_INTERVALS, DEFAULT_COLLOCATION_DEGREE, F>(
                params, emit,
            )
        }
    }
}
fn ms_runtime<const N: usize>(
    params: &Params,
) -> MultipleShootingRuntimeValues<(), Path<Bounds1D>, Boundary<f64>, (), State<f64>, Control<f64>, N>
{
    MultipleShootingRuntimeValues {
        parameters: (),
        beq: Boundary {
            x0: 0.0,
            v0: 0.0,
            theta0: 0.0,
            omega0: 0.0,
            force0: 0.0,
            x_t: params.target_x_m,
            v_t: 0.0,
            theta_t: 0.0,
            omega_t: 0.0,
            force_t: 0.0,
        },
        bineq_bounds: (),
        path_bounds: Path {
            x: Bounds1D {
                lower: Some(0.0),
                upper: Some(params.target_x_m),
            },
            theta: Bounds1D {
                lower: Some(-deg_to_rad(params.swing_limit_deg)),
                upper: Some(deg_to_rad(params.swing_limit_deg)),
            },
            force: Bounds1D {
                lower: Some(-params.force_limit_n),
                upper: Some(params.force_limit_n),
            },
            force_rate: Bounds1D {
                lower: Some(-params.force_rate_limit_nps),
                upper: Some(params.force_rate_limit_nps),
            },
        },
        tf_bounds: Bounds1D {
            lower: Some(params.tf_s),
            upper: Some(params.tf_s),
        },
        initial_guess: MultipleShootingInitialGuess::Interpolated(guess::<N>(params)),
    }
}
fn dc_runtime<const N: usize, const K: usize>(
    params: &Params,
) -> DirectCollocationRuntimeValues<
    (),
    Path<Bounds1D>,
    Boundary<f64>,
    (),
    State<f64>,
    Control<f64>,
    N,
    K,
> {
    DirectCollocationRuntimeValues {
        parameters: (),
        beq: Boundary {
            x0: 0.0,
            v0: 0.0,
            theta0: 0.0,
            omega0: 0.0,
            force0: 0.0,
            x_t: params.target_x_m,
            v_t: 0.0,
            theta_t: 0.0,
            omega_t: 0.0,
            force_t: 0.0,
        },
        bineq_bounds: (),
        path_bounds: Path {
            x: Bounds1D {
                lower: Some(0.0),
                upper: Some(params.target_x_m),
            },
            theta: Bounds1D {
                lower: Some(-deg_to_rad(params.swing_limit_deg)),
                upper: Some(deg_to_rad(params.swing_limit_deg)),
            },
            force: Bounds1D {
                lower: Some(-params.force_limit_n),
                upper: Some(params.force_limit_n),
            },
            force_rate: Bounds1D {
                lower: Some(-params.force_rate_limit_nps),
                upper: Some(params.force_rate_limit_nps),
            },
        },
        tf_bounds: Bounds1D {
            lower: Some(params.tf_s),
            upper: Some(params.tf_s),
        },
        initial_guess: DirectCollocationInitialGuess::Interpolated(guess::<N>(params)),
    }
}
fn solve_multiple_shooting<const N: usize>(params: &Params) -> Result<SolveArtifact> {
    let compiled = model_ms::<N>(params).compile_jit()?;
    let runtime = ms_runtime::<N>(params);
    solve_multiple_shooting_problem(
        &compiled,
        &runtime,
        params.solver_method,
        &params.solver,
        |trajectories, x_arcs, u_arcs| {
            artifact_from_ms_trajectories(params, trajectories, x_arcs, u_arcs)
        },
    )
}
fn solve_direct_collocation<const N: usize, const K: usize>(
    params: &Params,
) -> Result<SolveArtifact> {
    let compiled = model_dc::<N, K>(params).compile_jit()?;
    let runtime = dc_runtime::<N, K>(params);
    solve_direct_collocation_problem(
        &compiled,
        &runtime,
        params.solver_method,
        &params.solver,
        |trajectories, time_grid| artifact_from_dc_trajectories(params, trajectories, time_grid),
    )
}
fn solve_multiple_shooting_with_progress<const N: usize, F>(
    params: &Params,
    mut emit: F,
) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    emit_symbolic_setup_status(&mut emit);
    let compiled = model_ms::<N>(params).compile_jit()?;
    let runtime = ms_runtime::<N>(params);
    solve_multiple_shooting_problem_with_progress(
        &compiled,
        &runtime,
        params.solver_method,
        &params.solver,
        emit,
        |trajectories, x_arcs, u_arcs| {
            artifact_from_ms_trajectories(params, trajectories, x_arcs, u_arcs)
        },
    )
}
fn solve_direct_collocation_with_progress<const N: usize, const K: usize, F>(
    params: &Params,
    mut emit: F,
) -> Result<SolveArtifact>
where
    F: FnMut(SolveStreamEvent) + Send,
{
    emit_symbolic_setup_status(&mut emit);
    let compiled = model_dc::<N, K>(params).compile_jit()?;
    let runtime = dc_runtime::<N, K>(params);
    solve_direct_collocation_problem_with_progress(
        &compiled,
        &runtime,
        params.solver_method,
        &params.solver,
        emit,
        |trajectories, time_grid| artifact_from_dc_trajectories(params, trajectories, time_grid),
    )
}
fn artifact_summary(
    params: &Params,
    final_x: f64,
    max_swing: f64,
    max_force: f64,
    max_force_rate: f64,
    tf: f64,
) -> Vec<crate::common::Metric> {
    let mut summary = transcription_metrics(&params.transcription).to_vec();
    summary.extend([
        metric_with_key(
            MetricKey::TargetX,
            "Target",
            format!("{:.2} m", params.target_x_m),
        ),
        numeric_metric_with_key(
            MetricKey::FinalX,
            "Final X",
            final_x,
            format!("{final_x:.2} m"),
        ),
        numeric_metric_with_key(
            MetricKey::MaxSwing,
            "Max Swing",
            max_swing,
            format!("{max_swing:.2} deg"),
        ),
        numeric_metric_with_key(
            MetricKey::MaxAccel,
            "Max Force",
            max_force,
            format!("{max_force:.2} N"),
        ),
        numeric_metric_with_key(
            MetricKey::MaxJerk,
            "Max Force Rate",
            max_force_rate,
            format!("{max_force_rate:.2} N/s"),
        ),
        numeric_metric_with_key(MetricKey::Duration, "Duration", tf, format!("{tf:.2} s")),
    ]);
    summary
}
fn artifact_from_ms_trajectories<const N: usize>(
    params: &Params,
    trajectories: &MultipleShootingTrajectories<State<f64>, Control<f64>, N>,
    x_arcs: &[IntervalArc<State<f64>>],
    u_arcs: &[IntervalArc<Control<f64>>],
) -> SolveArtifact {
    let node_times = node_times::<N>(trajectories.tf);
    let mut trolley_x = Vec::with_capacity(N + 1);
    let mut trolley_v = Vec::with_capacity(N + 1);
    let mut swing_deg = Vec::with_capacity(N + 1);
    let mut swing_rate_deg = Vec::with_capacity(N + 1);
    let mut force = Vec::with_capacity(N + 1);
    let mut force_rate = Vec::with_capacity(N + 1);
    let mut load_x = Vec::with_capacity(N + 1);
    let mut load_y = Vec::with_capacity(N + 1);
    let mut load_path_x = Vec::with_capacity(N + 1);
    let mut load_path_y = Vec::with_capacity(N + 1);
    let mut frames = Vec::with_capacity(N + 1);
    for index in 0..N {
        let state = &trajectories.x.nodes[index];
        let control = &trajectories.u.nodes[index];
        let rate = &trajectories.dudt[index];
        let (load_px, load_py) = load_position(state, params.rope_length_m);
        trolley_x.push(state.x);
        trolley_v.push(state.v);
        swing_deg.push(rad_to_deg(state.theta));
        swing_rate_deg.push(rad_to_deg(state.omega));
        force.push(control.force);
        force_rate.push(rate.force);
        load_x.push(load_px);
        load_y.push(load_py);
        load_path_x.push(load_px);
        load_path_y.push(load_py);
        frames.push(frame_for_state(state, params.rope_length_m));
    }
    let terminal = &trajectories.x.terminal;
    let (load_px, load_py) = load_position(terminal, params.rope_length_m);
    trolley_x.push(terminal.x);
    trolley_v.push(terminal.v);
    swing_deg.push(rad_to_deg(terminal.theta));
    swing_rate_deg.push(rad_to_deg(terminal.omega));
    force.push(trajectories.u.terminal.force);
    force_rate.push(*force_rate.last().unwrap_or(&0.0));
    load_x.push(load_px);
    load_y.push(load_py);
    load_path_x.push(load_px);
    load_path_y.push(load_py);
    frames.push(frame_for_state(terminal, params.rope_length_m));
    let max_swing = swing_deg
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let max_force = force
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let max_force_rate = force_rate
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    SolveArtifact {
        title: "Crane Load Transfer".to_string(),
        summary: artifact_summary(
            params,
            terminal.x,
            max_swing,
            max_force,
            max_force_rate,
            trajectories.tf,
        ),
        solver: SolverReport::placeholder(),
        constraint_panels: Default::default(),
        charts: vec![
            chart(
                "Trolley Position",
                "x (m)",
                {
                    let mut series_out = interval_arc_series(
                        "Trolley x (m)",
                        x_arcs,
                        PlotMode::LinesMarkers,
                        |state| state.x,
                    );
                    series_out.extend(interval_arc_bound_series(
                        x_arcs,
                        Some(0.0),
                        Some(params.target_x_m),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
            chart(
                "Trolley Velocity",
                "v (m/s)",
                interval_arc_series("Trolley v (m/s)", x_arcs, PlotMode::LinesMarkers, |state| state.v),
            ),
            chart(
                "Swing Angle",
                "theta (deg)",
                {
                    let mut series_out = interval_arc_series(
                        "Swing (deg)",
                        x_arcs,
                        PlotMode::LinesMarkers,
                        |state| rad_to_deg(state.theta),
                    );
                    series_out.extend(interval_arc_bound_series(
                        x_arcs,
                        Some(-params.swing_limit_deg),
                        Some(params.swing_limit_deg),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
            chart(
                "Swing Rate",
                "omega (deg/s)",
                interval_arc_series(
                    "Swing Rate (deg/s)",
                    x_arcs,
                    PlotMode::LinesMarkers,
                    |state| rad_to_deg(state.omega),
                ),
            ),
            chart(
                "Cart Force",
                "Force (N)",
                {
                    let mut series_out = interval_arc_series(
                        "Force (N)",
                        u_arcs,
                        PlotMode::LinesMarkers,
                        |control| control.force,
                    );
                    series_out.extend(interval_arc_bound_series(
                        u_arcs,
                        Some(-params.force_limit_n),
                        Some(params.force_limit_n),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
            chart(
                "Force Rate",
                "Force Rate (N/s)",
                {
                    let mut series_out = segmented_series(
                        "Force Rate (N/s)",
                        x_arcs.iter().enumerate().map(|(interval, arc)| {
                            (
                                arc.times.clone(),
                                vec![trajectories.dudt[interval].force; arc.times.len()],
                            )
                        }),
                        PlotMode::LinesMarkers,
                    );
                    series_out.extend(segmented_bound_series(
                        x_arcs.iter().map(|arc| arc.times.clone()),
                        Some(-params.force_rate_limit_nps),
                        Some(params.force_rate_limit_nps),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
        ],
        scene: Scene2D {
            title: "Crane Geometry".to_string(),
            x_label: "x (m)".to_string(),
            y_label: "height (m)".to_string(),
            paths: vec![
                ScenePath {
                    name: "Trolley".to_string(),
                    x: trolley_x,
                    y: vec![0.0; N + 1],
                },
                ScenePath {
                    name: "Load".to_string(),
                    x: load_path_x,
                    y: load_path_y,
                },
            ],
            circles: Vec::new(),
            arrows: Vec::new(),
            animation: Some(SceneAnimation {
                times: node_times,
                frames,
            }),
        },
        notes: vec![
            "Cart force is the control-state and force rate is the true decision variable.".to_string(),
            "The multiple-shooting charts show RK4 interval arcs reconstructed from each mesh node with the same substepping used by the transcription.".to_string(),
        ],
    }
}
fn artifact_from_dc_trajectories<const N: usize, const K: usize>(
    params: &Params,
    trajectories: &DirectCollocationTrajectories<State<f64>, Control<f64>, N, K>,
    time_grid: &DirectCollocationTimeGrid<N, K>,
) -> SolveArtifact {
    let node_times = (0..N)
        .map(|index| time_grid.nodes.nodes[index])
        .chain(std::iter::once(time_grid.nodes.terminal))
        .collect::<Vec<_>>();
    let x_arcs =
        direct_collocation_state_like_arcs(&trajectories.x, &trajectories.root_x, time_grid)
            .expect("collocation state arcs should match trajectory layout");
    let u_arcs =
        direct_collocation_state_like_arcs(&trajectories.u, &trajectories.root_u, time_grid)
            .expect("collocation control-state arcs should match trajectory layout");
    let dudt_arcs = direct_collocation_root_arcs(&trajectories.root_dudt, time_grid);
    let mut trolley_x = Vec::with_capacity(N + 1);
    let mut trolley_v = Vec::with_capacity(N + 1);
    let mut swing_deg = Vec::with_capacity(N + 1);
    let mut swing_rate_deg = Vec::with_capacity(N + 1);
    let mut load_path_x = Vec::with_capacity(N + 1);
    let mut load_path_y = Vec::with_capacity(N + 1);
    let mut frames = Vec::with_capacity(N + 1);
    let mut max_swing = 0.0_f64;
    let mut max_force = 0.0_f64;
    let mut max_force_rate = 0.0_f64;
    for index in 0..N {
        let state = &trajectories.x.nodes[index];
        let (load_px, load_py) = load_position(state, params.rope_length_m);
        trolley_x.push(state.x);
        trolley_v.push(state.v);
        swing_deg.push(rad_to_deg(state.theta));
        swing_rate_deg.push(rad_to_deg(state.omega));
        load_path_x.push(load_px);
        load_path_y.push(load_py);
        frames.push(frame_for_state(state, params.rope_length_m));
        max_swing = max_swing.max(rad_to_deg(state.theta).abs());
    }
    let terminal = &trajectories.x.terminal;
    let (load_px, load_py) = load_position(terminal, params.rope_length_m);
    trolley_x.push(terminal.x);
    trolley_v.push(terminal.v);
    swing_deg.push(rad_to_deg(terminal.theta));
    swing_rate_deg.push(rad_to_deg(terminal.omega));
    load_path_x.push(load_px);
    load_path_y.push(load_py);
    frames.push(frame_for_state(terminal, params.rope_length_m));
    max_swing = max_swing.max(rad_to_deg(terminal.theta).abs());
    for interval in 0..N {
        for root in 0..K {
            max_swing = max_swing
                .max(rad_to_deg(trajectories.root_x.intervals[interval][root].theta).abs());
            max_force = max_force.max(trajectories.root_u.intervals[interval][root].force.abs());
            max_force_rate =
                max_force_rate.max(trajectories.root_dudt.intervals[interval][root].force.abs());
        }
    }
    SolveArtifact {
        title: "Crane Load Transfer".to_string(),
        summary: artifact_summary(
            params,
            terminal.x,
            max_swing,
            max_force,
            max_force_rate,
            trajectories.tf,
        ),
        solver: SolverReport::placeholder(),
        constraint_panels: Default::default(),
        charts: vec![
            chart(
                "Trolley Position",
                "x (m)",
                {
                    let mut series_out =
                        interval_arc_series("Trolley x (m)", &x_arcs, PlotMode::LinesMarkers, |state| {
                            state.x
                        });
                    series_out.extend(interval_arc_bound_series(
                        &x_arcs,
                        Some(0.0),
                        Some(params.target_x_m),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
            chart(
                "Trolley Velocity",
                "v (m/s)",
                interval_arc_series("Trolley v (m/s)", &x_arcs, PlotMode::LinesMarkers, |state| state.v),
            ),
            chart(
                "Swing Angle",
                "theta (deg)",
                {
                    let mut series_out = interval_arc_series(
                        "Swing (deg)",
                        &x_arcs,
                        PlotMode::LinesMarkers,
                        |state| rad_to_deg(state.theta),
                    );
                    series_out.extend(interval_arc_bound_series(
                        &x_arcs,
                        Some(-params.swing_limit_deg),
                        Some(params.swing_limit_deg),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
            chart(
                "Swing Rate",
                "omega (deg/s)",
                interval_arc_series(
                    "Swing Rate (deg/s)",
                    &x_arcs,
                    PlotMode::LinesMarkers,
                    |state| rad_to_deg(state.omega),
                ),
            ),
            chart(
                "Cart Force",
                "Force (N)",
                {
                    let mut series_out = interval_arc_series(
                        "Force (N)",
                        &u_arcs,
                        PlotMode::LinesMarkers,
                        |control| control.force,
                    );
                    series_out.extend(interval_arc_bound_series(
                        &u_arcs,
                        Some(-params.force_limit_n),
                        Some(params.force_limit_n),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
            chart(
                "Force Rate",
                "Force Rate (N/s)",
                {
                    let mut series_out = interval_arc_series(
                        "Force Rate (N/s)",
                        &dudt_arcs,
                        PlotMode::LinesMarkers,
                        |force_rate| force_rate.force,
                    );
                    series_out.extend(interval_arc_bound_series(
                        &dudt_arcs,
                        Some(-params.force_rate_limit_nps),
                        Some(params.force_rate_limit_nps),
                        PlotMode::Lines,
                    ));
                    series_out
                },
            ),
        ],
        scene: Scene2D {
            title: "Crane Geometry".to_string(),
            x_label: "x (m)".to_string(),
            y_label: "height (m)".to_string(),
            paths: vec![
                ScenePath {
                    name: "Trolley".to_string(),
                    x: trolley_x,
                    y: vec![0.0; N + 1],
                },
                ScenePath {
                    name: "Load".to_string(),
                    x: load_path_x,
                    y: load_path_y,
                },
            ],
            circles: Vec::new(),
            arrows: Vec::new(),
            animation: Some(SceneAnimation {
                times: node_times,
                frames,
            }),
        },
        notes: vec![
            "Cart force is the control-state and force rate is the true decision variable.".to_string(),
            "The collocation charts are interval-local: state-like quantities include the extrapolated interval endpoint, while force rate is only shown at collocation nodes.".to_string(),
        ],
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn crane_reaches_target_with_bounded_swing() {
        let artifact = solve(&Params::default()).expect("crane solve should succeed");
        let final_x = crate::find_metric(&artifact.summary, crate::MetricKey::FinalX)
            .and_then(|metric| metric.numeric_value)
            .expect("final x should exist");
        let max_swing = crate::find_metric(&artifact.summary, crate::MetricKey::MaxSwing)
            .and_then(|metric| metric.numeric_value)
            .expect("max swing should exist");
        assert!((final_x - Params::default().target_x_m).abs() < 0.2);
        assert!(max_swing < 10.0, "swing should remain controlled");
    }
}
