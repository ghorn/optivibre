use approx::assert_abs_diff_eq;
use optimal_control::{Bounds1D, Ocp};
use optimization::{ClarabelSqpOptions, Vectorize};
use sx_core::SX;

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct State<T> {
    x: T,
}

#[derive(Clone, Debug, PartialEq, Vectorize)]
struct Control<T> {
    u: T,
}

fn sqp_options() -> ClarabelSqpOptions {
    ClarabelSqpOptions {
        verbose: false,
        max_iters: 120,
        dual_tol: 1.0e-1,
        constraint_tol: 1.0e-8,
        complementarity_tol: 1.0e-6,
        overall_tol: 1.0e-1,
        ..ClarabelSqpOptions::default()
    }
}

fn runtime_ms_ocp(
    intervals: usize,
    rk4_substeps: usize,
) -> Ocp<State<SX>, Control<SX>, (), (), State<SX>, (), optimal_control::runtime::MultipleShooting>
{
    Ocp::new(
        "runtime_scalar_ms",
        optimal_control::runtime::MultipleShooting {
            intervals,
            rk4_substeps,
        },
    )
    .objective_lagrange(
        |x: &State<SX>, u: &Control<SX>, dudt: &Control<SX>, _: &()| {
            x.x.sqr() + 0.1 * u.u.sqr() + 1.0e-3 * dudt.u.sqr()
        },
    )
    .objective_mayer(
        |_: &State<SX>, _: &Control<SX>, xf: &State<SX>, _: &Control<SX>, _: &(), _: &SX| {
            10.0 * xf.x.sqr()
        },
    )
    .ode(|_: &State<SX>, u: &Control<SX>, _: &()| State { x: u.u })
    .path_constraints(|_: &State<SX>, _: &Control<SX>, _: &Control<SX>, _: &()| ())
    .boundary_equalities(
        |x0: &State<SX>, _: &Control<SX>, _: &State<SX>, _: &Control<SX>, _: &(), _: &SX| {
            x0.clone()
        },
    )
    .boundary_inequalities(
        |_: &State<SX>, _: &Control<SX>, _: &State<SX>, _: &Control<SX>, _: &(), _: &SX| (),
    )
    .build()
    .expect("runtime multiple-shooting builder should succeed")
}

fn runtime_dc_ocp(
    intervals: usize,
    order: usize,
    time_grid: optimal_control::runtime::TimeGrid,
) -> Ocp<State<SX>, Control<SX>, (), (), State<SX>, (), optimal_control::runtime::DirectCollocation>
{
    Ocp::new(
        "runtime_scalar_dc",
        optimal_control::runtime::DirectCollocation {
            intervals,
            order,
            family: optimal_control::CollocationFamily::RadauIIA,
            time_grid,
        },
    )
    .objective_lagrange(
        |x: &State<SX>, u: &Control<SX>, dudt: &Control<SX>, _: &()| {
            x.x.sqr() + 0.1 * u.u.sqr() + 1.0e-3 * dudt.u.sqr()
        },
    )
    .objective_mayer(
        |_: &State<SX>, _: &Control<SX>, xf: &State<SX>, _: &Control<SX>, _: &(), _: &SX| {
            10.0 * xf.x.sqr()
        },
    )
    .ode(|_: &State<SX>, u: &Control<SX>, _: &()| State { x: u.u })
    .path_constraints(|_: &State<SX>, _: &Control<SX>, _: &Control<SX>, _: &()| ())
    .boundary_equalities(
        |x0: &State<SX>, _: &Control<SX>, _: &State<SX>, _: &Control<SX>, _: &(), _: &SX| {
            x0.clone()
        },
    )
    .boundary_inequalities(
        |_: &State<SX>, _: &Control<SX>, _: &State<SX>, _: &Control<SX>, _: &(), _: &SX| (),
    )
    .build()
    .expect("runtime direct-collocation builder should succeed")
}

#[test]
fn runtime_multiple_shooting_compiles_solves_and_projects() {
    let intervals = 8;
    let compiled = runtime_ms_ocp(intervals, 4)
        .compile_jit()
        .expect("runtime multiple-shooting OCP should compile");
    let runtime = optimal_control::runtime::MultipleShootingRuntimeValues {
        parameters: (),
        beq: State { x: 1.0 },
        bineq_bounds: (),
        path_bounds: (),
        tf_bounds: Bounds1D {
            lower: Some(1.0),
            upper: Some(1.0),
        },
        initial_guess: optimal_control::runtime::MultipleShootingInitialGuess::Constant {
            x: State { x: 1.0 },
            u: Control { u: 0.0 },
            dudt: Control { u: 0.0 },
            tf: 1.0,
        },
        scaling: None,
    };
    let mut callback_count = 0usize;
    let result = compiled
        .solve_sqp_with_callback(&runtime, &sqp_options(), |_| {
            callback_count += 1;
        })
        .expect("runtime multiple-shooting solve should succeed");

    assert!(callback_count > 0);
    assert_eq!(result.trajectories.x.nodes.len(), intervals);
    assert_eq!(result.trajectories.u.nodes.len(), intervals);
    assert_eq!(result.trajectories.dudt.len(), intervals);
    assert_abs_diff_eq!(result.trajectories.tf, 1.0, epsilon = 1.0e-9);
    assert!(result.trajectories.x.terminal.x.abs() < 0.5);
    assert_abs_diff_eq!(result.time_grid.nodes.nodes[0], 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(result.time_grid.nodes.terminal, 1.0, epsilon = 1.0e-9);
}

#[test]
fn runtime_direct_collocation_compiles_solves_and_projects() {
    let intervals = 6;
    let order = 2;
    let compiled = runtime_dc_ocp(
        intervals,
        order,
        optimal_control::runtime::TimeGrid::Cosine { strength: 0.5 },
    )
    .compile_jit()
    .expect("runtime direct-collocation OCP should compile");
    let runtime = optimal_control::runtime::DirectCollocationRuntimeValues {
        parameters: (),
        beq: State { x: 1.0 },
        bineq_bounds: (),
        path_bounds: (),
        tf_bounds: Bounds1D {
            lower: Some(1.0),
            upper: Some(1.0),
        },
        initial_guess: optimal_control::runtime::DirectCollocationInitialGuess::Constant {
            x: State { x: 1.0 },
            u: Control { u: 0.0 },
            dudt: Control { u: 0.0 },
            tf: 1.0,
        },
        scaling: None,
    };
    let mut callback_count = 0usize;
    let mut saw_warped_callback_grid = false;
    let result = compiled
        .solve_sqp_with_callback(&runtime, &sqp_options(), |snapshot| {
            callback_count += 1;
            let first_step = snapshot.time_grid.nodes.nodes[1] - snapshot.time_grid.nodes.nodes[0];
            let second_step = snapshot.time_grid.nodes.nodes[2] - snapshot.time_grid.nodes.nodes[1];
            saw_warped_callback_grid |= first_step < second_step;
        })
        .expect("runtime direct-collocation solve should succeed");

    assert!(callback_count > 0);
    assert!(saw_warped_callback_grid);
    assert_eq!(result.trajectories.x.nodes.len(), intervals);
    assert_eq!(result.trajectories.root_x.intervals.len(), intervals);
    assert_eq!(result.trajectories.root_x.intervals[0].len(), order);
    assert_eq!(result.time_grid.roots.intervals.len(), intervals);
    assert_eq!(result.time_grid.roots.intervals[0].len(), order);
    assert!(
        result.time_grid.nodes.nodes[1] - result.time_grid.nodes.nodes[0]
            < result.time_grid.nodes.nodes[2] - result.time_grid.nodes.nodes[1]
    );
    assert_abs_diff_eq!(result.trajectories.tf, 1.0, epsilon = 1.0e-9);
    assert!(result.trajectories.x.terminal.x.abs() < 0.5);
}
