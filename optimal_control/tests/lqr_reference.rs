use approx::assert_abs_diff_eq;
use lqr_solvers::{
    FiniteHorizonLqrProblem, FiniteHorizonLqrSolution, simulate_time_varying_closed_loop,
    solve_finite_horizon, steady_state_gain,
};
use nalgebra::{DMatrix, DVector};
use optimal_control::runtime as ocp_runtime;
use optimal_control::runtime::{
    DirectCollocation, DirectCollocationInitialGuess, MultipleShooting,
    MultipleShootingInitialGuess,
};
use optimal_control::{Bounds1D, CollocationFamily, FinalTime, Ocp};
use optimization::{ClarabelSqpOptions, Vectorize};
use sx_core::SX;

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct State<T> {
    position: T,
    velocity: T,
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct Control<T> {
    acceleration: T,
}

type Boundary<T> = (State<T>, Control<T>);
type Mesh<T> = ocp_runtime::Mesh<T>;
type IntervalGrid<T> = ocp_runtime::IntervalGrid<T>;
type MultipleShootingRuntimeValues<P, C, Beq, Bineq, X, U> =
    ocp_runtime::MultipleShootingRuntimeValues<
        P,
        C,
        Beq,
        Bineq,
        X,
        U,
        FinalTime<f64>,
        FinalTime<Bounds1D>,
        (),
        (),
    >;
type DirectCollocationRuntimeValues<P, C, Beq, Bineq, X, U> =
    ocp_runtime::DirectCollocationRuntimeValues<
        P,
        C,
        Beq,
        Bineq,
        X,
        U,
        FinalTime<f64>,
        FinalTime<Bounds1D>,
        (),
        (),
    >;
type MultipleShootingTrajectories<X, U> = ocp_runtime::MultipleShootingTrajectories<X, U>;
type DirectCollocationTrajectories<X, U> = ocp_runtime::DirectCollocationTrajectories<X, U>;

const Q_POSITION: f64 = 10.0;
const Q_VELOCITY: f64 = 1.0;
const Q_CONTROL: f64 = 0.5;
const R_DUDT: f64 = 0.1;
const QF_POSITION: f64 = 5.0;
const QF_VELOCITY: f64 = 1.0;
const QF_CONTROL: f64 = 0.5;
const TF: f64 = 2.0;
const REFERENCE_STEPS: usize = 400;

fn augmented_a() -> DMatrix<f64> {
    DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
}

fn augmented_b() -> DMatrix<f64> {
    DMatrix::from_column_slice(3, 1, &[0.0, 0.0, 1.0])
}

fn q_matrix() -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_vec(vec![Q_POSITION, Q_VELOCITY, Q_CONTROL]))
}

fn qf_matrix() -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_vec(vec![
        QF_POSITION,
        QF_VELOCITY,
        QF_CONTROL,
    ]))
}

fn r_matrix() -> DMatrix<f64> {
    DMatrix::from_element(1, 1, R_DUDT)
}

fn finite_horizon_solution() -> FiniteHorizonLqrSolution {
    solve_finite_horizon(&FiniteHorizonLqrProblem {
        a: augmented_a(),
        b: augmented_b(),
        q: q_matrix(),
        r: r_matrix(),
        qf: qf_matrix(),
        tf: TF,
        steps: REFERENCE_STEPS,
    })
    .expect("finite-horizon LQR should solve")
}

fn ocp_ms<const N: usize, const RK4_SUBSTEPS: usize>()
-> Ocp<State<SX>, Control<SX>, (), (), Boundary<SX>, (), MultipleShooting> {
    Ocp::new(
        "augmented_lqr_ms",
        MultipleShooting {
            intervals: N,
            rk4_substeps: RK4_SUBSTEPS,
        },
    )
    .objective_lagrange(
        |x: &State<SX>, u: &Control<SX>, dudt: &Control<SX>, _: &()| {
            Q_POSITION * x.position.sqr()
                + Q_VELOCITY * x.velocity.sqr()
                + Q_CONTROL * u.acceleration.sqr()
                + R_DUDT * dudt.acceleration.sqr()
        },
    )
    .objective_mayer(
        |_: &State<SX>, _: &Control<SX>, x_t: &State<SX>, u_t: &Control<SX>, _: &(), _: &SX| {
            QF_POSITION * x_t.position.sqr()
                + QF_VELOCITY * x_t.velocity.sqr()
                + QF_CONTROL * u_t.acceleration.sqr()
        },
    )
    .ode(|x: &State<SX>, u: &Control<SX>, _: &()| State {
        position: x.velocity,
        velocity: u.acceleration,
    })
    .path_constraints(|_: &State<SX>, _: &Control<SX>, _: &Control<SX>, _: &()| ())
    .boundary_equalities(
        |x0: &State<SX>, u0: &Control<SX>, _: &State<SX>, _: &Control<SX>, _: &(), _: &SX| {
            (x0.clone(), u0.clone())
        },
    )
    .boundary_inequalities(
        |_: &State<SX>, _: &Control<SX>, _: &State<SX>, _: &Control<SX>, _: &(), _: &SX| (),
    )
    .build()
    .expect("builder should succeed")
}

fn ocp_dc<const N: usize, const K: usize>()
-> Ocp<State<SX>, Control<SX>, (), (), Boundary<SX>, (), DirectCollocation> {
    Ocp::new(
        "augmented_lqr_dc",
        DirectCollocation {
            intervals: N,
            order: K,
            family: CollocationFamily::RadauIIA,
            time_grid: Default::default(),
        },
    )
    .objective_lagrange(
        |x: &State<SX>, u: &Control<SX>, dudt: &Control<SX>, _: &()| {
            Q_POSITION * x.position.sqr()
                + Q_VELOCITY * x.velocity.sqr()
                + Q_CONTROL * u.acceleration.sqr()
                + R_DUDT * dudt.acceleration.sqr()
        },
    )
    .objective_mayer(
        |_: &State<SX>, _: &Control<SX>, x_t: &State<SX>, u_t: &Control<SX>, _: &(), _: &SX| {
            QF_POSITION * x_t.position.sqr()
                + QF_VELOCITY * x_t.velocity.sqr()
                + QF_CONTROL * u_t.acceleration.sqr()
        },
    )
    .ode(|x: &State<SX>, u: &Control<SX>, _: &()| State {
        position: x.velocity,
        velocity: u.acceleration,
    })
    .path_constraints(|_: &State<SX>, _: &Control<SX>, _: &Control<SX>, _: &()| ())
    .boundary_equalities(
        |x0: &State<SX>, u0: &Control<SX>, _: &State<SX>, _: &Control<SX>, _: &(), _: &SX| {
            (x0.clone(), u0.clone())
        },
    )
    .boundary_inequalities(
        |_: &State<SX>, _: &Control<SX>, _: &State<SX>, _: &Control<SX>, _: &(), _: &SX| (),
    )
    .build()
    .expect("builder should succeed")
}

fn to_state(z: &DVector<f64>) -> State<f64> {
    State {
        position: z[0],
        velocity: z[1],
    }
}

fn to_control_state(z: &DVector<f64>) -> Control<f64> {
    Control { acceleration: z[2] }
}

fn to_control_rate(v: &DVector<f64>) -> Control<f64> {
    Control { acceleration: v[0] }
}

fn solve_reference_at(times: &[f64]) -> Vec<(State<f64>, Control<f64>, Control<f64>)> {
    let solution = finite_horizon_solution();
    let z0 = DVector::from_vec(vec![1.0, -0.25, 0.1]);
    let trajectory = simulate_time_varying_closed_loop(
        &augmented_a(),
        &augmented_b(),
        &solution,
        &z0,
        times,
        16,
    )
    .expect("reference rollout should succeed");
    trajectory
        .points
        .iter()
        .map(|point| {
            (
                to_state(&point.state),
                to_control_state(&point.state),
                to_control_rate(&point.control),
            )
        })
        .collect()
}

fn node_times(tf: f64, intervals: usize) -> Vec<f64> {
    let step = tf / intervals as f64;
    (0..intervals).map(|index| index as f64 * step).collect()
}

fn terminal_time(tf: f64) -> f64 {
    tf
}

fn ms_interpolated_guess<const N: usize>()
-> optimal_control::InterpolatedTrajectory<State<f64>, Control<f64>> {
    let dense_times = (0..=REFERENCE_STEPS)
        .map(|index| index as f64 * TF / REFERENCE_STEPS as f64)
        .collect::<Vec<_>>();
    let dense_samples = solve_reference_at(&dense_times);
    optimal_control::InterpolatedTrajectory {
        sample_times: dense_times,
        x_samples: dense_samples
            .iter()
            .map(|sample| sample.0.clone())
            .collect(),
        u_samples: dense_samples
            .iter()
            .map(|sample| sample.1.clone())
            .collect(),
        dudt_samples: dense_samples
            .iter()
            .map(|sample| sample.2.clone())
            .collect(),
        global: FinalTime { tf: TF },
        tf: TF,
    }
}

fn dc_explicit_guess<const N: usize, const K: usize>()
-> DirectCollocationTrajectories<State<f64>, Control<f64>> {
    let step = TF / N as f64;
    let node_query_times = {
        let mut times = node_times(TF, N);
        times.push(terminal_time(TF));
        times
    };
    let node_samples = solve_reference_at(&node_query_times);
    let root_times: Vec<Vec<f64>> = (0..N)
        .map(|interval| {
            (0..K)
                .map(|root| {
                    let root_offset = match root {
                        0 => 1.0 / 3.0,
                        1 => 1.0,
                        _ => unreachable!("test only supports Radau IIA with K=2"),
                    };
                    (interval as f64 + root_offset) * step
                })
                .collect()
        })
        .collect();
    let root_query_times = root_times
        .iter()
        .flat_map(|interval| interval.iter().copied())
        .collect::<Vec<_>>();
    let root_samples = solve_reference_at(&root_query_times);

    DirectCollocationTrajectories {
        x: Mesh {
            nodes: (0..N).map(|index| node_samples[index].0.clone()).collect(),
            terminal: node_samples[N].0.clone(),
        },
        u: Mesh {
            nodes: (0..N).map(|index| node_samples[index].1.clone()).collect(),
            terminal: node_samples[N].1.clone(),
        },
        root_x: IntervalGrid {
            intervals: (0..N)
                .map(|interval| {
                    (0..K)
                        .map(|root| root_samples[interval * K + root].0.clone())
                        .collect()
                })
                .collect(),
        },
        root_u: IntervalGrid {
            intervals: (0..N)
                .map(|interval| {
                    (0..K)
                        .map(|root| root_samples[interval * K + root].1.clone())
                        .collect()
                })
                .collect(),
        },
        root_dudt: IntervalGrid {
            intervals: (0..N)
                .map(|interval| {
                    (0..K)
                        .map(|root| root_samples[interval * K + root].2.clone())
                        .collect()
                })
                .collect(),
        },
        global: FinalTime { tf: TF },
        tf: TF,
    }
}

fn compare_ms<const N: usize>(
    solved: &MultipleShootingTrajectories<State<f64>, Control<f64>>,
    tolerance: f64,
) {
    assert_abs_diff_eq!(solved.tf, TF, epsilon = 1e-9);
    let mut query_times = node_times(TF, N);
    query_times.push(TF);
    let reference = solve_reference_at(&query_times);
    for (index, sample) in reference.iter().take(N).enumerate() {
        assert_abs_diff_eq!(
            solved.x.nodes[index].position,
            sample.0.position,
            epsilon = tolerance
        );
        assert_abs_diff_eq!(
            solved.x.nodes[index].velocity,
            sample.0.velocity,
            epsilon = tolerance
        );
        assert_abs_diff_eq!(
            solved.u.nodes[index].acceleration,
            sample.1.acceleration,
            epsilon = tolerance
        );
        assert_abs_diff_eq!(
            solved.dudt[index].acceleration,
            sample.2.acceleration,
            epsilon = 20.0 * tolerance
        );
    }
    assert_abs_diff_eq!(
        solved.x.terminal.position,
        reference[N].0.position,
        epsilon = tolerance
    );
    assert_abs_diff_eq!(
        solved.x.terminal.velocity,
        reference[N].0.velocity,
        epsilon = tolerance
    );
    assert_abs_diff_eq!(
        solved.u.terminal.acceleration,
        reference[N].1.acceleration,
        epsilon = tolerance
    );
}

fn compare_dc<const N: usize, const K: usize>(
    solved: &DirectCollocationTrajectories<State<f64>, Control<f64>>,
    tolerance: f64,
) {
    assert_abs_diff_eq!(solved.tf, TF, epsilon = 1e-9);
    let mut query_times = node_times(TF, N);
    query_times.push(TF);
    let reference = solve_reference_at(&query_times);
    for (index, sample) in reference.iter().take(N).enumerate() {
        assert_abs_diff_eq!(
            solved.x.nodes[index].position,
            sample.0.position,
            epsilon = tolerance
        );
        assert_abs_diff_eq!(
            solved.x.nodes[index].velocity,
            sample.0.velocity,
            epsilon = tolerance
        );
        assert_abs_diff_eq!(
            solved.u.nodes[index].acceleration,
            sample.1.acceleration,
            epsilon = tolerance
        );
    }
    assert_abs_diff_eq!(
        solved.x.terminal.position,
        reference[N].0.position,
        epsilon = tolerance
    );
    assert_abs_diff_eq!(
        solved.x.terminal.velocity,
        reference[N].0.velocity,
        epsilon = tolerance
    );
    assert_abs_diff_eq!(
        solved.u.terminal.acceleration,
        reference[N].1.acceleration,
        epsilon = tolerance
    );
}

fn sqp_options() -> ClarabelSqpOptions {
    ClarabelSqpOptions {
        verbose: false,
        max_iters: 160,
        dual_tol: 5.0e-2,
        constraint_tol: 1.0e-8,
        complementarity_tol: 1.0e-6,
        overall_tol: 5.0e-2,
        ..ClarabelSqpOptions::default()
    }
}

#[test]
fn multiple_shooting_tracks_finite_horizon_lqr_reference_and_emits_structured_callbacks() {
    const N: usize = 16;
    const RK4_SUBSTEPS: usize = 4;

    let compiled = ocp_ms::<N, RK4_SUBSTEPS>()
        .compile_jit()
        .expect("multiple shooting OCP should compile");
    let beq = (
        State {
            position: 1.0,
            velocity: -0.25,
        },
        Control { acceleration: 0.1 },
    );
    let runtime = MultipleShootingRuntimeValues {
        parameters: (),
        beq: beq.clone(),
        bineq_bounds: (),
        path_bounds: (),
        global_bounds: FinalTime {
            tf: Bounds1D {
                lower: Some(TF),
                upper: Some(TF),
            },
        },
        initial_guess: MultipleShootingInitialGuess::Interpolated(ms_interpolated_guess::<N>()),
        scaling: None,
    };
    let options = sqp_options();
    let mut callback_count = 0usize;
    let mut saw_terminal_time = 0.0;
    let result = compiled
        .solve_sqp_with_callback(&runtime, &options, |snapshot| {
            callback_count += 1;
            saw_terminal_time = snapshot.time_grid.nodes.terminal;
            assert_abs_diff_eq!(snapshot.time_grid.nodes.nodes[0], 0.0, epsilon = 1e-12);
        })
        .expect("multiple shooting SQP solve should succeed");

    assert!(callback_count > 0);
    assert_abs_diff_eq!(saw_terminal_time, TF, epsilon = 1e-9);
    compare_ms::<N>(&result.trajectories, 2.5e-1);
}

#[test]
fn direct_collocation_tracks_finite_horizon_lqr_reference() {
    const N: usize = 10;
    const K: usize = 2;

    let compiled = ocp_dc::<N, K>()
        .compile_jit()
        .expect("direct collocation OCP should compile");
    let runtime = DirectCollocationRuntimeValues {
        parameters: (),
        beq: (
            State {
                position: 1.0,
                velocity: -0.25,
            },
            Control { acceleration: 0.1 },
        ),
        bineq_bounds: (),
        path_bounds: (),
        global_bounds: FinalTime {
            tf: Bounds1D {
                lower: Some(TF),
                upper: Some(TF),
            },
        },
        initial_guess: DirectCollocationInitialGuess::Explicit(dc_explicit_guess::<N, K>()),
        scaling: None,
    };
    let options = sqp_options();
    let result = compiled
        .solve_sqp(&runtime, &options)
        .expect("direct collocation SQP solve should succeed");

    compare_dc::<N, K>(&result.trajectories, 1.2e-1);
}

#[test]
fn rollout_initial_guess_from_steady_state_lqr_gain_converges() {
    const N: usize = 12;
    const RK4_SUBSTEPS: usize = 4;

    let compiled = ocp_ms::<N, RK4_SUBSTEPS>()
        .compile_jit()
        .expect("multiple shooting OCP should compile");
    let gain = steady_state_gain(
        &augmented_a(),
        &augmented_b(),
        &q_matrix(),
        &r_matrix(),
        20.0,
        400,
    )
    .expect("steady-state gain should solve");
    let runtime = MultipleShootingRuntimeValues {
        parameters: (),
        beq: (
            State {
                position: 1.0,
                velocity: -0.25,
            },
            Control { acceleration: 0.1 },
        ),
        bineq_bounds: (),
        path_bounds: (),
        global_bounds: FinalTime {
            tf: Bounds1D {
                lower: Some(TF),
                upper: Some(TF),
            },
        },
        initial_guess: MultipleShootingInitialGuess::Rollout {
            x0: State {
                position: 1.0,
                velocity: -0.25,
            },
            u0: Control { acceleration: 0.1 },
            tf: TF,
            controller: Box::new(move |_, x, u, _| Control {
                acceleration: -(gain[(0, 0)] * x.position
                    + gain[(0, 1)] * x.velocity
                    + gain[(0, 2)] * u.acceleration),
            }),
        },
        scaling: None,
    };
    let options = sqp_options();
    let result = compiled
        .solve_sqp(&runtime, &options)
        .expect("multiple shooting SQP solve should succeed from rollout guess");

    compare_ms::<N>(&result.trajectories, 2.5e-1);
}
