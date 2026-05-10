use super::*;
use optimization::{
    DynamicCompiledJitNlp, RuntimeNlpBounds, RuntimeNlpScaling, symbolic_nlp_dynamic,
};
use std::marker::PhantomData;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultipleShooting {
    pub intervals: usize,
    pub rk4_substeps: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum TimeGrid {
    #[default]
    Uniform,
    Cosine {
        strength: f64,
    },
    Tanh {
        strength: f64,
    },
    Geometric {
        strength: f64,
        bias: TimeGridBias,
    },
    Focus {
        center: f64,
        width: f64,
        strength: f64,
    },
    Piecewise {
        breakpoint: f64,
        first_interval_fraction: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TimeGridBias {
    Start,
    End,
}

impl TimeGrid {
    pub const fn uniform() -> Self {
        Self::Uniform
    }

    pub const fn cosine(strength: f64) -> Self {
        Self::Cosine { strength }
    }

    pub const fn tanh(strength: f64) -> Self {
        Self::Tanh { strength }
    }

    pub const fn geometric_start(strength: f64) -> Self {
        Self::Geometric {
            strength,
            bias: TimeGridBias::Start,
        }
    }

    pub const fn geometric_end(strength: f64) -> Self {
        Self::Geometric {
            strength,
            bias: TimeGridBias::End,
        }
    }

    pub const fn focus(center: f64, width: f64, strength: f64) -> Self {
        Self::Focus {
            center,
            width,
            strength,
        }
    }

    pub const fn piecewise(breakpoint: f64, first_interval_fraction: f64) -> Self {
        Self::Piecewise {
            breakpoint,
            first_interval_fraction,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectCollocation {
    pub intervals: usize,
    pub order: usize,
    pub family: CollocationFamily,
    pub time_grid: TimeGrid,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mesh<T> {
    pub nodes: Vec<T>,
    pub terminal: T,
}

impl<T> Mesh<T> {
    pub fn interval_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len() + 1
    }

    pub fn states(&self) -> impl Iterator<Item = &T> {
        self.nodes.iter().chain(std::iter::once(&self.terminal))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IntervalGrid<T> {
    pub intervals: Vec<Vec<T>>,
}

impl<T> IntervalGrid<T> {
    pub fn interval_count(&self) -> usize {
        self.intervals.len()
    }

    pub fn order(&self) -> usize {
        self.intervals.first().map_or(0, Vec::len)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultipleShootingTrajectories<X, U, G = FinalTime<f64>> {
    pub x: Mesh<X>,
    pub u: Mesh<U>,
    pub dudt: Vec<U>,
    pub global: G,
    pub tf: f64,
}

impl<X, U, G> MultipleShootingTrajectories<X, U, G> {
    pub fn interval_count(&self) -> usize {
        self.dudt.len()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectCollocationTrajectories<X, U, G = FinalTime<f64>> {
    pub x: Mesh<X>,
    pub u: Mesh<U>,
    pub root_x: IntervalGrid<X>,
    pub root_u: IntervalGrid<U>,
    pub root_dudt: IntervalGrid<U>,
    pub global: G,
    pub tf: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultipleShootingTimeGrid {
    pub nodes: Mesh<f64>,
}

impl MultipleShootingTimeGrid {
    pub fn times(&self) -> impl Iterator<Item = f64> + '_ {
        self.nodes
            .nodes
            .iter()
            .copied()
            .chain(std::iter::once(self.nodes.terminal))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectCollocationTimeGrid {
    pub nodes: Mesh<f64>,
    pub roots: IntervalGrid<f64>,
}

impl DirectCollocationTimeGrid {
    pub fn interval_times(&self) -> Vec<Vec<f64>> {
        (0..self.nodes.nodes.len())
            .map(|interval| {
                let mut times = Vec::with_capacity(self.roots.order() + 2);
                times.push(self.nodes.nodes[interval]);
                times.extend(self.roots.intervals[interval].iter().copied());
                let end_time = if interval + 1 < self.nodes.nodes.len() {
                    self.nodes.nodes[interval + 1]
                } else {
                    self.nodes.terminal
                };
                times.push(end_time);
                times
            })
            .collect()
    }

    pub fn root_times(&self) -> Vec<Vec<f64>> {
        self.roots.intervals.clone()
    }
}

pub fn direct_collocation_extrapolated_end<T>(
    start: &T,
    roots: &[T],
    time_grid: &DirectCollocationTimeGrid,
    interval: usize,
) -> Result<T, optimization::VectorizeLayoutError>
where
    T: Vectorize<f64, Rebind<f64> = T> + Clone,
{
    let start_time = time_grid.nodes.nodes[interval];
    let end_time = if interval + 1 < time_grid.nodes.nodes.len() {
        time_grid.nodes.nodes[interval + 1]
    } else {
        time_grid.nodes.terminal
    };
    let step = end_time - start_time;
    let nodes = roots
        .iter()
        .enumerate()
        .map(|(root, _)| {
            if step.abs() <= 1.0e-12 {
                0.0
            } else {
                (time_grid.roots.intervals[interval][root] - start_time) / step
            }
        })
        .collect::<Vec<_>>();
    let mut basis_nodes = Vec::with_capacity(roots.len() + 1);
    basis_nodes.push(0.0);
    basis_nodes.extend(nodes.iter().copied());
    let weights = basis_nodes
        .iter()
        .enumerate()
        .map(|(basis_index, &basis_node)| {
            lagrange_basis_polynomial(&basis_nodes, basis_index, basis_node)
                .map(|poly| evaluate_polynomial(&poly, 1.0))
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("collocation nodes derived from the time grid should define a valid basis");

    let mut flat = vec![0.0; T::LEN];
    let start_flat = flatten_value(start);
    for (index, value) in start_flat.iter().enumerate() {
        flat[index] += weights[0] * value;
    }
    for (root, root_value) in roots.iter().enumerate() {
        let root_flat = flatten_value(root_value);
        for (index, value) in root_flat.iter().enumerate() {
            flat[index] += weights[root + 1] * value;
        }
    }
    unflatten_value::<T, f64>(&flat)
}

pub fn direct_collocation_state_like_arcs<T>(
    mesh: &Mesh<T>,
    roots: &IntervalGrid<T>,
    time_grid: &DirectCollocationTimeGrid,
) -> Result<Vec<IntervalArc<T>>, optimization::VectorizeLayoutError>
where
    T: Vectorize<f64, Rebind<f64> = T> + Clone,
{
    let interval_times = time_grid.interval_times();
    (0..mesh.nodes.len())
        .map(|interval| {
            let mut values = Vec::with_capacity(roots.order() + 2);
            values.push(mesh.nodes[interval].clone());
            values.extend(roots.intervals[interval].iter().cloned());
            values.push(direct_collocation_extrapolated_end(
                &mesh.nodes[interval],
                &roots.intervals[interval],
                time_grid,
                interval,
            )?);
            Ok(IntervalArc {
                times: interval_times[interval].clone(),
                values,
            })
        })
        .collect()
}

pub fn direct_collocation_root_arcs<T>(
    roots: &IntervalGrid<T>,
    time_grid: &DirectCollocationTimeGrid,
) -> Vec<IntervalArc<T>>
where
    T: Clone,
{
    let root_times = time_grid.root_times();
    (0..roots.intervals.len())
        .map(|interval| IntervalArc {
            times: root_times[interval].clone(),
            values: roots.intervals[interval].clone(),
        })
        .collect()
}

pub enum MultipleShootingInitialGuess<X, U, P, G = FinalTime<f64>> {
    Explicit(MultipleShootingTrajectories<X, U, G>),
    Constant {
        x: X,
        u: U,
        dudt: U,
        tf: f64,
    },
    ConstantGlobal {
        x: X,
        u: U,
        dudt: U,
        global: G,
    },
    Interpolated(InterpolatedTrajectory<X, U, G>),
    Rollout {
        x0: X,
        u0: U,
        tf: f64,
        controller: Box<ControllerFn<X, U, P>>,
    },
    RolloutGlobal {
        x0: X,
        u0: U,
        global: G,
        controller: Box<ControllerFn<X, U, P>>,
    },
}

pub enum DirectCollocationInitialGuess<X, U, P, G = FinalTime<f64>> {
    Explicit(DirectCollocationTrajectories<X, U, G>),
    Constant {
        x: X,
        u: U,
        dudt: U,
        tf: f64,
    },
    ConstantGlobal {
        x: X,
        u: U,
        dudt: U,
        global: G,
    },
    Interpolated(InterpolatedTrajectory<X, U, G>),
    Rollout {
        x0: X,
        u0: U,
        tf: f64,
        controller: Box<ControllerFn<X, U, P>>,
    },
    RolloutGlobal {
        x0: X,
        u0: U,
        global: G,
        controller: Box<ControllerFn<X, U, P>>,
    },
}

pub struct MultipleShootingRuntimeValues<
    P,
    C,
    Beq,
    Bineq,
    X,
    U,
    G = FinalTime<f64>,
    GBounds = FinalTime<Bounds1D>,
> {
    pub parameters: P,
    pub beq: Beq,
    pub bineq_bounds: Bineq,
    pub path_bounds: C,
    pub global_bounds: GBounds,
    pub initial_guess: MultipleShootingInitialGuess<X, U, P, G>,
    pub scaling: Option<OcpScaling<P, X, U, G>>,
}

pub struct DirectCollocationRuntimeValues<
    P,
    C,
    Beq,
    Bineq,
    X,
    U,
    G = FinalTime<f64>,
    GBounds = FinalTime<Bounds1D>,
> {
    pub parameters: P,
    pub beq: Beq,
    pub bineq_bounds: Bineq,
    pub path_bounds: C,
    pub global_bounds: GBounds,
    pub initial_guess: DirectCollocationInitialGuess<X, U, P, G>,
    pub scaling: Option<OcpScaling<P, X, U, G>>,
}

#[derive(Clone, Debug)]
pub struct MultipleShootingSqpSnapshot<X, U, G = FinalTime<f64>> {
    pub trajectories: MultipleShootingTrajectories<X, U, G>,
    pub time_grid: MultipleShootingTimeGrid,
    pub solver: SqpIterationSnapshot,
}

#[derive(Clone, Debug)]
pub struct DirectCollocationSqpSnapshot<X, U, G = FinalTime<f64>> {
    pub trajectories: DirectCollocationTrajectories<X, U, G>,
    pub time_grid: DirectCollocationTimeGrid,
    pub solver: SqpIterationSnapshot,
}

#[derive(Clone, Debug)]
pub struct MultipleShootingInteriorPointSnapshot<X, U, G = FinalTime<f64>> {
    pub trajectories: MultipleShootingTrajectories<X, U, G>,
    pub time_grid: MultipleShootingTimeGrid,
    pub solver: InteriorPointIterationSnapshot,
}

#[derive(Clone, Debug)]
pub struct DirectCollocationInteriorPointSnapshot<X, U, G = FinalTime<f64>> {
    pub trajectories: DirectCollocationTrajectories<X, U, G>,
    pub time_grid: DirectCollocationTimeGrid,
    pub solver: InteriorPointIterationSnapshot,
}

#[derive(Clone, Debug)]
pub struct MultipleShootingSqpSolveResult<X, U, G = FinalTime<f64>> {
    pub trajectories: MultipleShootingTrajectories<X, U, G>,
    pub time_grid: MultipleShootingTimeGrid,
    pub solver: ClarabelSqpSummary,
    pub setup_timing: OcpSolveSetupTiming,
}

#[derive(Clone, Debug)]
pub struct DirectCollocationSqpSolveResult<X, U, G = FinalTime<f64>> {
    pub trajectories: DirectCollocationTrajectories<X, U, G>,
    pub time_grid: DirectCollocationTimeGrid,
    pub solver: ClarabelSqpSummary,
    pub setup_timing: OcpSolveSetupTiming,
}

#[derive(Clone, Debug)]
pub struct MultipleShootingInteriorPointSolveResult<X, U, G = FinalTime<f64>> {
    pub trajectories: MultipleShootingTrajectories<X, U, G>,
    pub time_grid: MultipleShootingTimeGrid,
    pub solver: InteriorPointSummary,
    pub setup_timing: OcpSolveSetupTiming,
}

#[derive(Clone, Debug)]
pub struct DirectCollocationInteriorPointSolveResult<X, U, G = FinalTime<f64>> {
    pub trajectories: DirectCollocationTrajectories<X, U, G>,
    pub time_grid: DirectCollocationTimeGrid,
    pub solver: InteriorPointSummary,
    pub setup_timing: OcpSolveSetupTiming,
}

#[cfg(feature = "ipopt")]
#[derive(Clone, Debug)]
pub struct MultipleShootingIpoptSolveResult<X, U, G = FinalTime<f64>> {
    pub trajectories: MultipleShootingTrajectories<X, U, G>,
    pub time_grid: MultipleShootingTimeGrid,
    pub solver: IpoptSummary,
    pub setup_timing: OcpSolveSetupTiming,
}

#[cfg(feature = "ipopt")]
#[derive(Clone, Debug)]
pub struct DirectCollocationIpoptSolveResult<X, U, G = FinalTime<f64>> {
    pub trajectories: DirectCollocationTrajectories<X, U, G>,
    pub time_grid: DirectCollocationTimeGrid,
    pub solver: IpoptSummary,
    pub setup_timing: OcpSolveSetupTiming,
}

#[cfg(feature = "ipopt")]
#[derive(Clone, Debug)]
pub struct MultipleShootingIpoptSnapshot<X, U, G = FinalTime<f64>> {
    pub trajectories: MultipleShootingTrajectories<X, U, G>,
    pub time_grid: MultipleShootingTimeGrid,
    pub solver: IpoptIterationSnapshot,
}

#[cfg(feature = "ipopt")]
#[derive(Clone, Debug)]
pub struct DirectCollocationIpoptSnapshot<X, U, G = FinalTime<f64>> {
    pub trajectories: DirectCollocationTrajectories<X, U, G>,
    pub time_grid: DirectCollocationTimeGrid,
    pub solver: IpoptIterationSnapshot,
}

struct CompiledMultipleShootingArcDyn<X, U, P> {
    function: CompiledJitFunction,
    context: Mutex<JitExecutionContext>,
    _marker: PhantomData<fn() -> (X, U, P)>,
}

struct MsTranscription {
    objective: SX,
    equalities: Vec<SX>,
    boundary_eq_residual: Vec<SX>,
    boundary_ineq: Vec<SX>,
    path: Vec<SX>,
}

struct DcTranscription {
    objective: SX,
    equalities: Vec<SX>,
    boundary_eq_residual: Vec<SX>,
    boundary_ineq: Vec<SX>,
    path: Vec<SX>,
}

pub struct CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, G = FinalTime<SX>> {
    compiled: DynamicCompiledJitNlp,
    scheme: MultipleShooting,
    promotion_plan: PromotionPlan,
    promotion_offsets: PromotionOffsets<OcpParameters<P, Beq>>,
    xdot_helper: CompiledXdot<X, U, P>,
    rk4_arc_helper: CompiledMultipleShootingArcDyn<X, U, P>,
    helper_compile_stats: OcpHelperCompileStats,
    _marker: PhantomData<fn() -> (C, Bineq, G)>,
}

pub struct CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, G = FinalTime<SX>> {
    compiled: DynamicCompiledJitNlp,
    scheme: DirectCollocation,
    promotion_plan: PromotionPlan,
    promotion_offsets: PromotionOffsets<OcpParameters<P, Beq>>,
    xdot_helper: CompiledXdot<X, U, P>,
    coefficients: CollocationCoefficients,
    helper_compile_stats: OcpHelperCompileStats,
    _marker: PhantomData<fn() -> (C, Bineq, G)>,
}

fn ms_variable_count<X, U, G>(intervals: usize) -> usize
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
    G: Vectorize<SX>,
{
    (intervals + 1) * (X::LEN + U::LEN) + intervals * U::LEN + G::LEN
}

fn ms_equality_count<X, U>(intervals: usize) -> usize
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
{
    intervals * (X::LEN + U::LEN)
}

fn ms_inequality_count<C, Beq, Bineq>(intervals: usize) -> usize
where
    C: Vectorize<SX>,
    Beq: Vectorize<SX>,
    Bineq: Vectorize<SX>,
{
    Beq::LEN + Bineq::LEN + intervals * C::LEN
}

fn dc_variable_count<X, U, G>(intervals: usize, order: usize) -> usize
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
    G: Vectorize<SX>,
{
    (intervals + 1) * (X::LEN + U::LEN) + intervals * order * (X::LEN + 2 * U::LEN) + G::LEN
}

fn dc_equality_count<X, U>(intervals: usize, order: usize) -> usize
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
{
    intervals * order * (X::LEN + U::LEN) + intervals * (X::LEN + U::LEN)
}

fn dc_inequality_count<C, Beq, Bineq>(intervals: usize, order: usize) -> usize
where
    C: Vectorize<SX>,
    Beq: Vectorize<SX>,
    Bineq: Vectorize<SX>,
{
    Beq::LEN + Bineq::LEN + intervals * order * C::LEN
}

fn symbolic_dense_vector(prefix: &str, len: usize) -> Vec<SX> {
    (0..len)
        .map(|index| SX::sym(format!("{prefix}_{index}")))
        .collect()
}

fn symbolic_zero_value<T>() -> Result<T, SxError>
where
    T: Vectorize<SX, Rebind<SX> = T>,
{
    unflatten_value::<T, SX>(&[]).map_err(|err| SxError::Graph(err.to_string()))
}

fn take_numeric_chunk<'a>(
    values: &'a [f64],
    index: &mut usize,
    len: usize,
) -> Result<&'a [f64], VectorizeLayoutError> {
    let start = *index;
    let end = start + len;
    if end > values.len() {
        return Err(VectorizeLayoutError::LengthMismatch {
            expected: end,
            got: values.len(),
        });
    }
    *index = end;
    Ok(&values[start..end])
}

fn take_symbolic_chunk<'a>(
    values: &'a [SX],
    index: &mut usize,
    len: usize,
) -> Result<&'a [SX], SxError> {
    let start = *index;
    let end = start + len;
    if end > values.len() {
        return Err(SxError::Graph(format!(
            "symbolic runtime layout mismatch: expected at least {end} entries, got {}",
            values.len()
        )));
    }
    *index = end;
    Ok(&values[start..end])
}

fn take_numeric_value<V>(
    values: &[f64],
    index: &mut usize,
) -> Result<Numeric<V>, VectorizeLayoutError>
where
    V: Vectorize<SX>,
    Numeric<V>: Vectorize<f64, Rebind<f64> = Numeric<V>>,
{
    unflatten_value::<Numeric<V>, f64>(take_numeric_chunk(values, index, V::LEN)?)
}

fn take_symbolic_value<V>(values: &[SX], index: &mut usize) -> Result<V, SxError>
where
    V: Vectorize<SX, Rebind<SX> = V>,
{
    unflatten_value::<V, SX>(take_symbolic_chunk(values, index, V::LEN)?)
        .map_err(|err| SxError::Graph(err.to_string()))
}

fn take_numeric_mesh<V>(
    values: &[f64],
    index: &mut usize,
    intervals: usize,
) -> Result<Mesh<Numeric<V>>, VectorizeLayoutError>
where
    V: Vectorize<SX>,
    Numeric<V>: Vectorize<f64, Rebind<f64> = Numeric<V>>,
{
    let mut nodes = Vec::with_capacity(intervals);
    for _ in 0..intervals {
        nodes.push(take_numeric_value::<V>(values, index)?);
    }
    let terminal = take_numeric_value::<V>(values, index)?;
    Ok(Mesh { nodes, terminal })
}

fn take_symbolic_mesh<V>(
    values: &[SX],
    index: &mut usize,
    intervals: usize,
) -> Result<Mesh<V>, SxError>
where
    V: Vectorize<SX, Rebind<SX> = V>,
{
    let mut nodes = Vec::with_capacity(intervals);
    for _ in 0..intervals {
        nodes.push(take_symbolic_value::<V>(values, index)?);
    }
    let terminal = take_symbolic_value::<V>(values, index)?;
    Ok(Mesh { nodes, terminal })
}

fn take_numeric_interval_grid<V>(
    values: &[f64],
    index: &mut usize,
    intervals: usize,
    order: usize,
) -> Result<IntervalGrid<Numeric<V>>, VectorizeLayoutError>
where
    V: Vectorize<SX>,
    Numeric<V>: Vectorize<f64, Rebind<f64> = Numeric<V>>,
{
    let mut out = Vec::with_capacity(intervals);
    for _ in 0..intervals {
        let mut interval = Vec::with_capacity(order);
        for _ in 0..order {
            interval.push(take_numeric_value::<V>(values, index)?);
        }
        out.push(interval);
    }
    Ok(IntervalGrid { intervals: out })
}

fn take_symbolic_interval_grid<V>(
    values: &[SX],
    index: &mut usize,
    intervals: usize,
    order: usize,
) -> Result<IntervalGrid<V>, SxError>
where
    V: Vectorize<SX, Rebind<SX> = V>,
{
    let mut out = Vec::with_capacity(intervals);
    for _ in 0..intervals {
        let mut interval = Vec::with_capacity(order);
        for _ in 0..order {
            interval.push(take_symbolic_value::<V>(values, index)?);
        }
        out.push(interval);
    }
    Ok(IntervalGrid { intervals: out })
}

fn flatten_mesh<T, S>(mesh: &Mesh<T>) -> Vec<S>
where
    T: Vectorize<S>,
    S: ScalarLeaf,
{
    let mut out = Vec::with_capacity((mesh.nodes.len() + 1) * T::LEN);
    for node in &mesh.nodes {
        out.extend(node.flatten_cloned());
    }
    out.extend(mesh.terminal.flatten_cloned());
    out
}

fn flatten_dynamic_values<T, S>(values: &[T]) -> Vec<S>
where
    T: Vectorize<S>,
    S: ScalarLeaf,
{
    let mut out = Vec::with_capacity(values.len() * T::LEN);
    for value in values {
        out.extend(value.flatten_cloned());
    }
    out
}

fn flatten_interval_grid<T, S>(grid: &IntervalGrid<T>) -> Vec<S>
where
    T: Vectorize<S>,
    S: ScalarLeaf,
{
    let mut out = Vec::new();
    for interval in &grid.intervals {
        out.extend(flatten_dynamic_values(interval));
    }
    out
}

fn validate_time_grid(time_grid: TimeGrid) -> Result<(), OcpCompileError> {
    fn require_unit_interval(value: f64, label: &str) -> Result<(), OcpCompileError> {
        if !value.is_finite() {
            return Err(OcpCompileError::InvalidConfiguration(format!(
                "{label} must be finite"
            )));
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(OcpCompileError::InvalidConfiguration(format!(
                "{label} must be in [0, 1]"
            )));
        }
        Ok(())
    }

    fn require_open_unit_interval(value: f64, label: &str) -> Result<(), OcpCompileError> {
        if !value.is_finite() {
            return Err(OcpCompileError::InvalidConfiguration(format!(
                "{label} must be finite"
            )));
        }
        if !(0.0..1.0).contains(&value) {
            return Err(OcpCompileError::InvalidConfiguration(format!(
                "{label} must be in (0, 1)"
            )));
        }
        Ok(())
    }

    match time_grid {
        TimeGrid::Uniform => Ok(()),
        TimeGrid::Cosine { strength }
        | TimeGrid::Tanh { strength }
        | TimeGrid::Geometric { strength, .. } => {
            require_unit_interval(strength, "time-grid strength")?;
            Ok(())
        }
        TimeGrid::Focus {
            center,
            width,
            strength,
        } => {
            require_unit_interval(center, "focus time-grid center")?;
            require_open_unit_interval(width, "focus time-grid width")?;
            require_unit_interval(strength, "time-grid strength")?;
            Ok(())
        }
        TimeGrid::Piecewise {
            breakpoint,
            first_interval_fraction,
        } => {
            require_open_unit_interval(breakpoint, "piecewise time-grid breakpoint")?;
            require_open_unit_interval(
                first_interval_fraction,
                "piecewise time-grid first interval fraction",
            )?;
            Ok(())
        }
    }
}

fn normalized_nodes_from_interval_weights(weights: &[f64]) -> Vec<f64> {
    let total = weights.iter().sum::<f64>();
    let mut nodes = Vec::with_capacity(weights.len() + 1);
    nodes.push(0.0);
    let mut cumulative = 0.0;
    for weight in weights {
        cumulative += weight / total;
        nodes.push(cumulative);
    }
    if let Some(first) = nodes.first_mut() {
        *first = 0.0;
    }
    if let Some(last) = nodes.last_mut() {
        *last = 1.0;
    }
    nodes
}

fn normalized_time_grid_nodes(intervals: usize, time_grid: TimeGrid) -> Vec<f64> {
    if intervals == 0 {
        return vec![0.0];
    }
    match time_grid {
        TimeGrid::Geometric { strength, bias } => {
            if intervals == 1 {
                return vec![0.0, 1.0];
            }
            let ratio = 1.0 + 19.0 * strength;
            let denom = (intervals - 1) as f64;
            let weights = (0..intervals)
                .map(|interval| {
                    let exponent = interval as f64 / denom;
                    match bias {
                        TimeGridBias::Start => ratio.powf(exponent),
                        TimeGridBias::End => ratio.powf(1.0 - exponent),
                    }
                })
                .collect::<Vec<_>>();
            return normalized_nodes_from_interval_weights(&weights);
        }
        TimeGrid::Focus {
            center,
            width,
            strength,
        } => {
            let focus_gain = 19.0 * strength;
            let weights = (0..intervals)
                .map(|interval| {
                    let midpoint = (interval as f64 + 0.5) / intervals as f64;
                    let z = (midpoint - center) / width;
                    let focus = (-0.5 * z * z).exp();
                    1.0 / (1.0 + focus_gain * focus)
                })
                .collect::<Vec<_>>();
            return normalized_nodes_from_interval_weights(&weights);
        }
        TimeGrid::Piecewise {
            breakpoint,
            first_interval_fraction,
        } => {
            if intervals == 1 {
                return vec![0.0, 1.0];
            }
            let split_index = ((first_interval_fraction * intervals as f64).round() as usize)
                .clamp(1, intervals - 1);
            let split = split_index as f64 / intervals as f64;
            let mut nodes = (0..=intervals)
                .map(|index| {
                    let s = index as f64 / intervals as f64;
                    if index <= split_index {
                        breakpoint * s / split
                    } else {
                        breakpoint + (1.0 - breakpoint) * (s - split) / (1.0 - split)
                    }
                })
                .collect::<Vec<_>>();
            nodes[0] = 0.0;
            nodes[split_index] = breakpoint;
            nodes[intervals] = 1.0;
            return nodes;
        }
        TimeGrid::Uniform | TimeGrid::Cosine { .. } | TimeGrid::Tanh { .. } => {}
    }

    let mut nodes = (0..=intervals)
        .map(|index| {
            let s = index as f64 / intervals as f64;
            match time_grid {
                TimeGrid::Uniform => s,
                TimeGrid::Cosine { strength } => {
                    let cosine = 0.5 * (1.0 - (std::f64::consts::PI * s).cos());
                    (1.0 - strength) * s + strength * cosine
                }
                TimeGrid::Tanh { strength } => {
                    if strength == 0.0 {
                        s
                    } else {
                        let beta = 6.0 * strength;
                        0.5 * (1.0 + (beta * (2.0 * s - 1.0)).tanh() / beta.tanh())
                    }
                }
                TimeGrid::Geometric { .. }
                | TimeGrid::Focus { .. }
                | TimeGrid::Piecewise { .. } => unreachable!("handled above"),
            }
        })
        .collect::<Vec<_>>();
    nodes[0] = 0.0;
    nodes[intervals] = 1.0;
    nodes
}

fn time_grid_interval_fractions(intervals: usize, time_grid: TimeGrid) -> Vec<f64> {
    normalized_time_grid_nodes(intervals, time_grid)
        .windows(2)
        .map(|window| window[1] - window[0])
        .collect()
}

fn time_grid_mesh_unchecked(tf: f64, intervals: usize, time_grid: TimeGrid) -> Mesh<f64> {
    let normalized = normalized_time_grid_nodes(intervals, time_grid);
    Mesh {
        nodes: normalized
            .iter()
            .take(intervals)
            .map(|fraction| fraction * tf)
            .collect(),
        terminal: tf,
    }
}

fn mesh_times(tf: f64, intervals: usize) -> Mesh<f64> {
    time_grid_mesh_unchecked(tf, intervals, TimeGrid::Uniform)
}

pub fn time_grid_mesh(
    tf: f64,
    intervals: usize,
    time_grid: TimeGrid,
) -> Result<Mesh<f64>, OcpCompileError> {
    if intervals == 0 {
        return Err(OcpCompileError::InvalidConfiguration(
            "time grid requires at least one interval".to_string(),
        ));
    }
    validate_time_grid(time_grid)?;
    Ok(time_grid_mesh_unchecked(tf, intervals, time_grid))
}

pub fn time_grid_mesh_from_interval_weights(
    tf: f64,
    weights: &[f64],
) -> Result<Mesh<f64>, OcpCompileError> {
    if weights.is_empty() {
        return Err(OcpCompileError::InvalidConfiguration(
            "time grid interval weights cannot be empty".to_string(),
        ));
    }
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight <= 0.0)
    {
        return Err(OcpCompileError::InvalidConfiguration(
            "time grid interval weights must be finite and positive".to_string(),
        ));
    }
    let normalized = normalized_nodes_from_interval_weights(weights);
    Ok(Mesh {
        nodes: normalized
            .iter()
            .take(weights.len())
            .map(|fraction| fraction * tf)
            .collect(),
        terminal: tf,
    })
}

pub fn adaptive_time_grid_mesh(
    tf: f64,
    indicators: &[f64],
    strength: f64,
) -> Result<Mesh<f64>, OcpCompileError> {
    validate_time_grid(TimeGrid::Cosine { strength })?;
    if indicators.is_empty() {
        return Err(OcpCompileError::InvalidConfiguration(
            "adaptive time-grid indicators cannot be empty".to_string(),
        ));
    }
    if indicators
        .iter()
        .any(|indicator| !indicator.is_finite() || *indicator < 0.0)
    {
        return Err(OcpCompileError::InvalidConfiguration(
            "adaptive time-grid indicators must be finite and nonnegative".to_string(),
        ));
    }
    let max_indicator = indicators.iter().copied().fold(0.0, f64::max);
    if max_indicator == 0.0 || strength == 0.0 {
        return time_grid_mesh(tf, indicators.len(), TimeGrid::Uniform);
    }
    let weights = indicators
        .iter()
        .map(|indicator| {
            let normalized = indicator / max_indicator;
            1.0 / (1.0 + 19.0 * strength * normalized)
        })
        .collect::<Vec<_>>();
    time_grid_mesh_from_interval_weights(tf, &weights)
}

fn direct_collocation_times(
    tf: f64,
    coeffs: &CollocationCoefficients,
    intervals: usize,
    time_grid: TimeGrid,
) -> DirectCollocationTimeGrid {
    let nodes = time_grid_mesh_unchecked(tf, intervals, time_grid);
    DirectCollocationTimeGrid {
        nodes: nodes.clone(),
        roots: IntervalGrid {
            intervals: (0..intervals)
                .map(|interval| {
                    let start = nodes.nodes[interval];
                    let end = if interval + 1 < intervals {
                        nodes.nodes[interval + 1]
                    } else {
                        nodes.terminal
                    };
                    let step = end - start;
                    (0..coeffs.nodes.len())
                        .map(|root| start + coeffs.nodes[root] * step)
                        .collect()
                })
                .collect(),
        },
    }
}

pub fn direct_collocation_time_grid_from_mesh(
    nodes: Mesh<f64>,
    family: CollocationFamily,
    order: usize,
) -> Result<DirectCollocationTimeGrid, OcpCompileError> {
    if nodes.nodes.is_empty() {
        return Err(OcpCompileError::InvalidConfiguration(
            "direct collocation time grid requires at least one interval".to_string(),
        ));
    }
    if order == 0 {
        return Err(OcpCompileError::InvalidConfiguration(
            "direct collocation time grid requires at least one collocation root".to_string(),
        ));
    }
    let mut all_nodes = nodes.nodes.clone();
    all_nodes.push(nodes.terminal);
    if all_nodes
        .windows(2)
        .any(|window| !window[0].is_finite() || !window[1].is_finite() || window[1] <= window[0])
    {
        return Err(OcpCompileError::InvalidConfiguration(
            "direct collocation time-grid nodes must be finite and strictly increasing".to_string(),
        ));
    }
    let coeffs = collocation_coefficients(family, order)?;
    Ok(DirectCollocationTimeGrid {
        nodes,
        roots: IntervalGrid {
            intervals: all_nodes
                .windows(2)
                .map(|window| {
                    let start = window[0];
                    let step = window[1] - start;
                    coeffs
                        .nodes
                        .iter()
                        .map(|root| start + root * step)
                        .collect()
                })
                .collect(),
        },
    })
}

pub fn direct_collocation_time_grid(
    tf: f64,
    scheme: DirectCollocation,
) -> Result<DirectCollocationTimeGrid, OcpCompileError> {
    if scheme.intervals == 0 {
        return Err(OcpCompileError::InvalidConfiguration(
            "direct collocation time grid requires at least one interval".to_string(),
        ));
    }
    if scheme.order == 0 {
        return Err(OcpCompileError::InvalidConfiguration(
            "direct collocation time grid requires at least one collocation root".to_string(),
        ));
    }
    validate_time_grid(scheme.time_grid)?;
    let coeffs = collocation_coefficients(scheme.family, scheme.order)?;
    Ok(direct_collocation_times(
        tf,
        &coeffs,
        scheme.intervals,
        scheme.time_grid,
    ))
}

pub fn direct_collocation_root_time_grid(
    tf: f64,
    scheme: DirectCollocation,
) -> Result<IntervalGrid<f64>, OcpCompileError> {
    Ok(direct_collocation_time_grid(tf, scheme)?.roots)
}

fn build_promotion_plan_flat(
    decision_symbols: &[SX],
    boundary_eq: &[SX],
    boundary_ineq: &[SX],
    path: &[SX],
) -> PromotionPlan {
    let decision_map = decision_symbols
        .iter()
        .copied()
        .enumerate()
        .map(|(index, symbol)| (symbol, index))
        .collect::<HashMap<_, _>>();
    let decision_set = decision_symbols.iter().copied().collect::<HashSet<_>>();
    let mut affine_memo = HashMap::new();
    let mut rows = Vec::with_capacity(boundary_eq.len() + boundary_ineq.len() + path.len());
    for expr in boundary_eq.iter().copied() {
        rows.push(RawInequalityRow {
            kind: RawInequalityKind::BoundaryEquality,
            promotion: classify_affine_row(
                expr,
                &decision_map,
                &decision_set,
                &mut affine_memo,
                rows.len(),
            ),
        });
    }
    for expr in boundary_ineq.iter().copied() {
        rows.push(RawInequalityRow {
            kind: RawInequalityKind::BoundaryInequality,
            promotion: classify_affine_row(
                expr,
                &decision_map,
                &decision_set,
                &mut affine_memo,
                rows.len(),
            ),
        });
    }
    for expr in path.iter().copied() {
        rows.push(RawInequalityRow {
            kind: RawInequalityKind::Path,
            promotion: classify_affine_row(
                expr,
                &decision_map,
                &decision_set,
                &mut affine_memo,
                rows.len(),
            ),
        });
    }
    PromotionPlan { rows }
}

fn compile_multiple_shooting_arc_helper_runtime<X, U, P>(
    ode: &OdeFn<X, U, P>,
    rk4_substeps: usize,
    options: FunctionCompileOptions,
) -> Result<CompiledMultipleShootingArcDyn<X, U, P>, OcpCompileError>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>>,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>>,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>>,
{
    let helper_options = FunctionCompileOptions {
        opt_level: match options.opt_level {
            LlvmOptimizationLevel::O3 => LlvmOptimizationLevel::O0,
            other => other,
        },
        ..options
    };
    let x = symbolic_value::<X>("x")?;
    let u = symbolic_value::<U>("u")?;
    let dudt = symbolic_value::<U>("dudt")?;
    let p = symbolic_value::<P>("p")?;
    let dt = SX::sym("dt");
    let outputs =
        rk4_integrate_symbolic_state_only(&x, &u, &dudt, dt, rk4_substeps, |x_eval, u_eval| {
            Ok(ode(x_eval, u_eval, &p))
        })?;
    let mut inputs = vec![
        NamedMatrix::new("x", symbolic_column(&x)?)?,
        NamedMatrix::new("u", symbolic_column(&u)?)?,
        NamedMatrix::new("dudt", symbolic_column(&dudt)?)?,
    ];
    if P::LEN > 0 {
        inputs.push(NamedMatrix::new("p", symbolic_column(&p)?)?);
    }
    inputs.push(NamedMatrix::new("dt", SXMatrix::dense_column(vec![dt])?)?);
    let function = SXFunction::new(
        "ocp_runtime_multiple_shooting_arc",
        inputs,
        vec![
            NamedMatrix::new("x_next", symbolic_column(&outputs.0)?)?,
            NamedMatrix::new("u_next", symbolic_column(&outputs.1)?)?,
        ],
    )?;
    let compiled = CompiledJitFunction::compile_function_with_options(&function, helper_options)?;
    Ok(CompiledMultipleShootingArcDyn {
        context: Mutex::new(compiled.create_context()),
        function: compiled,
        _marker: PhantomData,
    })
}

impl<X, U, P> CompiledMultipleShootingArcDyn<X, U, P>
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
    P: Vectorize<SX>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>>,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>>,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>>,
{
    fn eval(
        &self,
        x: &Numeric<X>,
        u: &Numeric<U>,
        dudt: &Numeric<U>,
        parameters: &Numeric<P>,
        dt: f64,
    ) -> AnyResult<(Numeric<X>, Numeric<U>)> {
        let flat_x = flatten_value(x);
        let flat_u = flatten_value(u);
        let flat_dudt = flatten_value(dudt);
        let flat_parameters = flatten_value(parameters);
        let mut context = lock_mutex(&self.context);
        context.input_mut(0).copy_from_slice(&flat_x);
        context.input_mut(1).copy_from_slice(&flat_u);
        context.input_mut(2).copy_from_slice(&flat_dudt);
        if P::LEN > 0 {
            context.input_mut(3).copy_from_slice(&flat_parameters);
        }
        context.input_mut(3 + usize::from(P::LEN > 0))[0] = dt;
        self.function.eval(&mut context);
        Ok((
            unflatten_value::<Numeric<X>, f64>(context.output(0))?,
            unflatten_value::<Numeric<U>, f64>(context.output(1))?,
        ))
    }
}

impl<X, U, P, C, Beq, Bineq, G> Ocp<X, U, P, C, Beq, Bineq, MultipleShooting, G>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    G: OcpGlobalDesign<SX> + Vectorize<SX, Rebind<SX> = G>,
    C: Vectorize<SX, Rebind<SX> = C>,
    Beq: Vectorize<SX, Rebind<SX> = Beq>,
    Bineq: Vectorize<SX, Rebind<SX> = Bineq>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>> + Clone,
    Numeric<Beq>: Vectorize<f64, Rebind<f64> = Numeric<Beq>> + Clone,
{
    fn build_runtime_multiple_shooting_symbolic_function_library(
        &self,
        options: OcpSymbolicFunctionOptions,
    ) -> Result<OcpSymbolicFunctionLibrary, SxError> {
        let mut library = self.build_symbolic_function_library(options)?;
        library.multiple_shooting_integrator =
            self.configured_symbolic_function(options.multiple_shooting_integrator, || {
                let x = symbolic_value::<X>("x")?;
                let u = symbolic_value::<U>("u")?;
                let dudt = symbolic_value::<U>("dudt")?;
                let p = symbolic_value::<P>("p")?;
                let dt = SX::sym("dt");
                let (x_next, u_next, objective) = rk4_integrate_symbolic(
                    &x,
                    &u,
                    &dudt,
                    dt,
                    self.scheme.rk4_substeps,
                    |x_eval, u_eval| self.eval_ode_symbolic(&library, x_eval, u_eval, &p),
                    |x_eval, u_eval| {
                        self.eval_objective_lagrange_symbolic(&library, x_eval, u_eval, &dudt, &p)
                    },
                )?;
                SXFunction::new(
                    format!("{}_runtime_multiple_shooting_integrator", self.name),
                    vec![
                        NamedMatrix::new("x", symbolic_column(&x)?)?,
                        NamedMatrix::new("u", symbolic_column(&u)?)?,
                        NamedMatrix::new("dudt", symbolic_column(&dudt)?)?,
                        NamedMatrix::new("p", symbolic_column(&p)?)?,
                        NamedMatrix::new("dt", SXMatrix::dense_column(vec![dt])?)?,
                    ],
                    vec![
                        NamedMatrix::new("x_next", symbolic_column(&x_next)?)?,
                        NamedMatrix::new("u_next", symbolic_column(&u_next)?)?,
                        NamedMatrix::new("objective", SXMatrix::scalar(objective))?,
                    ],
                )
            })?;
        Ok(library)
    }

    fn eval_runtime_multiple_shooting_integrator_symbolic(
        &self,
        library: &OcpSymbolicFunctionLibrary,
        x: &X,
        u: &U,
        dudt: &U,
        parameters: &P,
        dt: SX,
    ) -> Result<(X, U, SX), SxError> {
        match &library.multiple_shooting_integrator {
            Some(function) => {
                let outputs = function.call(&[
                    symbolic_column(x)?,
                    symbolic_column(u)?,
                    symbolic_column(dudt)?,
                    symbolic_column(parameters)?,
                    SXMatrix::dense_column(vec![dt])?,
                ])?;
                Ok((
                    unflatten_typed_output::<X>(&outputs[0])?,
                    unflatten_typed_output::<U>(&outputs[1])?,
                    outputs[2].scalar_expr()?,
                ))
            }
            None => rk4_integrate_symbolic(
                x,
                u,
                dudt,
                dt,
                self.scheme.rk4_substeps,
                |x_eval, u_eval| self.eval_ode_symbolic(library, x_eval, u_eval, parameters),
                |x_eval, u_eval| {
                    self.eval_objective_lagrange_symbolic(library, x_eval, u_eval, dudt, parameters)
                },
            ),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_symbolic_function_library(
        &self,
        options: OcpSymbolicFunctionOptions,
    ) -> Result<OcpSymbolicFunctionLibrary, SxError> {
        self.build_runtime_multiple_shooting_symbolic_function_library(options)
    }

    fn transcribe_runtime_multiple_shooting(
        &self,
        x_mesh: &Mesh<X>,
        u_mesh: &Mesh<U>,
        dudt: &[U],
        global: &G,
        parameters: &P,
        beq: &Beq,
        symbolic_library: &OcpSymbolicFunctionLibrary,
    ) -> Result<MsTranscription, SxError> {
        let tf = global.final_time();
        let step = tf / self.scheme.intervals as f64;
        let mut objective = SX::zero();
        let mut equalities = Vec::with_capacity(ms_equality_count::<X, U>(self.scheme.intervals));
        let mut path = Vec::with_capacity(self.scheme.intervals * C::LEN);

        for (interval, ((x_start, u_start), dudt_interval)) in x_mesh
            .nodes
            .iter()
            .zip(u_mesh.nodes.iter())
            .zip(dudt.iter())
            .enumerate()
        {
            let path_value = self.eval_path_constraints_symbolic(
                symbolic_library,
                x_start,
                u_start,
                dudt_interval,
                parameters,
            )?;
            path.extend(path_value.flatten_cloned());
            let (x_end, u_end, q_end) = self.eval_runtime_multiple_shooting_integrator_symbolic(
                symbolic_library,
                x_start,
                u_start,
                dudt_interval,
                parameters,
                step,
            )?;
            objective += q_end;
            let x_next = if interval + 1 < self.scheme.intervals {
                x_mesh.nodes[interval + 1].clone()
            } else {
                x_mesh.terminal.clone()
            };
            let u_next = if interval + 1 < self.scheme.intervals {
                u_mesh.nodes[interval + 1].clone()
            } else {
                u_mesh.terminal.clone()
            };
            equalities.extend(subtract_vectorized(&x_next, &x_end)?.flatten_cloned());
            equalities.extend(subtract_vectorized(&u_next, &u_end)?.flatten_cloned());
        }

        let boundary_eq_values = self.eval_boundary_equalities_symbolic(
            symbolic_library,
            &x_mesh.nodes[0],
            &u_mesh.nodes[0],
            &x_mesh.terminal,
            &u_mesh.terminal,
            parameters,
            global,
        )?;
        let boundary_eq_residual = subtract_vectorized(&boundary_eq_values, beq)?.flatten_cloned();
        let boundary_ineq = self
            .eval_boundary_inequalities_symbolic(
                symbolic_library,
                &x_mesh.nodes[0],
                &u_mesh.nodes[0],
                &x_mesh.terminal,
                &u_mesh.terminal,
                parameters,
                global,
            )?
            .flatten_cloned();
        objective += self.eval_objective_mayer_symbolic(
            symbolic_library,
            &x_mesh.nodes[0],
            &u_mesh.nodes[0],
            &x_mesh.terminal,
            &u_mesh.terminal,
            parameters,
            global,
        )?;

        Ok(MsTranscription {
            objective,
            equalities,
            boundary_eq_residual,
            boundary_ineq,
            path,
        })
    }

    pub fn compile_jit(
        &self,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, G>, OcpCompileError> {
        self.compile_jit_with_options(FunctionCompileOptions::from(LlvmOptimizationLevel::O3))
    }

    pub fn compile_jit_with_options(
        &self,
        options: FunctionCompileOptions,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, G>, OcpCompileError> {
        self.compile_jit_with_ocp_options(OcpCompileOptions::for_multiple_shooting(options))
    }

    pub fn compile_jit_with_ocp_options(
        &self,
        options: OcpCompileOptions,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, G>, OcpCompileError> {
        self.compile_jit_with_ocp_options_and_progress_callback(options, |_| {})
    }

    pub fn compile_jit_with_progress_callback<CB>(
        &self,
        callback: CB,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, G>, OcpCompileError>
    where
        CB: FnMut(OcpCompileProgress),
    {
        self.compile_jit_with_ocp_options_and_progress_callback(
            OcpCompileOptions::for_multiple_shooting(FunctionCompileOptions::from(
                LlvmOptimizationLevel::O3,
            )),
            callback,
        )
    }

    pub fn compile_jit_with_ocp_options_and_progress_callback<CB>(
        &self,
        options: OcpCompileOptions,
        mut on_progress: CB,
    ) -> Result<CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, G>, OcpCompileError>
    where
        CB: FnMut(OcpCompileProgress),
    {
        if self.scheme.intervals == 0 {
            return Err(OcpCompileError::InvalidConfiguration(
                "multiple shooting requires at least one interval".to_string(),
            ));
        }
        if self.scheme.rk4_substeps == 0 {
            return Err(OcpCompileError::InvalidConfiguration(
                "multiple shooting requires at least one RK4 substep".to_string(),
            ));
        }

        let symbolic_library = self.build_runtime_multiple_shooting_symbolic_function_library(
            options.symbolic_functions,
        )?;
        let decision_symbols =
            symbolic_dense_vector("w", ms_variable_count::<X, U, G>(self.scheme.intervals));
        let decision_matrix = SXMatrix::dense_column(decision_symbols.clone())?;
        let mut decision_index = 0usize;
        let x_mesh = take_symbolic_mesh::<X>(
            &decision_symbols,
            &mut decision_index,
            self.scheme.intervals,
        )?;
        let u_mesh = take_symbolic_mesh::<U>(
            &decision_symbols,
            &mut decision_index,
            self.scheme.intervals,
        )?;
        let mut dudt = Vec::with_capacity(self.scheme.intervals);
        for _ in 0..self.scheme.intervals {
            dudt.push(take_symbolic_value::<U>(
                &decision_symbols,
                &mut decision_index,
            )?);
        }
        let global = take_symbolic_value::<G>(&decision_symbols, &mut decision_index)?;

        let runtime_param_len = P::LEN + Beq::LEN;
        let parameter_symbols = symbolic_dense_vector("p", runtime_param_len);
        let parameter_matrix = (runtime_param_len > 0)
            .then(|| SXMatrix::dense_column(parameter_symbols.clone()))
            .transpose()?;
        let (parameters, beq) = if runtime_param_len > 0 {
            let mut param_index = 0usize;
            let parameters = take_symbolic_value::<P>(&parameter_symbols, &mut param_index)?;
            let beq = take_symbolic_value::<Beq>(&parameter_symbols, &mut param_index)?;
            (parameters, beq)
        } else {
            (symbolic_zero_value::<P>()?, symbolic_zero_value::<Beq>()?)
        };

        let outputs = self.transcribe_runtime_multiple_shooting(
            &x_mesh,
            &u_mesh,
            &dudt,
            &global,
            &parameters,
            &beq,
            &symbolic_library,
        )?;
        let inequalities = outputs
            .boundary_eq_residual
            .iter()
            .chain(outputs.boundary_ineq.iter())
            .chain(outputs.path.iter())
            .copied()
            .collect::<Vec<_>>();
        let promotion_plan = build_promotion_plan_flat(
            &decision_symbols,
            &outputs.boundary_eq_residual,
            &outputs.boundary_ineq,
            &outputs.path,
        );
        let promotion_offsets = compile_promotion_offsets(
            &promotion_plan,
            &(parameters, beq),
            options.function_options,
        )?;
        let mut helper_compile_stats = OcpHelperCompileStats::default();
        if let Some(function) = &promotion_offsets.function {
            helper_compile_stats.record_compile_report(function.function.compile_report());
        }

        let symbolic = symbolic_nlp_dynamic(
            self.name.clone(),
            decision_matrix,
            parameter_matrix,
            outputs.objective,
            (!outputs.equalities.is_empty())
                .then(|| SXMatrix::dense_column(outputs.equalities.clone()))
                .transpose()?,
            (!inequalities.is_empty())
                .then(|| SXMatrix::dense_column(inequalities))
                .transpose()?,
        )?;
        let compiled = symbolic.compile_jit_with_compile_options_and_symbolic_progress_callback(
            SymbolicNlpCompileOptions {
                function_options: options.function_options,
                hessian_strategy: options.hessian_strategy,
            },
            |progress| match progress {
                SymbolicCompileProgress::Stage(stage) => {
                    on_progress(OcpCompileProgress::SymbolicStage(stage));
                }
                SymbolicCompileProgress::Ready(metadata) => {
                    on_progress(OcpCompileProgress::SymbolicReady(metadata));
                }
            },
        )?;
        let nlp_compile_report = compiled.backend_compile_report();
        let nlp_jit_elapsed = nlp_compile_report
            .setup_profile
            .lowering
            .unwrap_or_default()
            + nlp_compile_report
                .setup_profile
                .llvm_jit
                .unwrap_or_default();
        on_progress(OcpCompileProgress::NlpKernelCompiled {
            elapsed: nlp_jit_elapsed,
            root_instructions: nlp_compile_report.stats.llvm_root_instructions_emitted,
            total_instructions: nlp_compile_report.stats.llvm_total_instructions_emitted,
        });

        let xdot_started = Instant::now();
        let xdot_helper = compile_xdot_helper::<X, U, P>(
            &*self.ode,
            symbolic_library.ode.as_ref(),
            options.function_options,
        )?;
        let xdot_helper_time = xdot_started.elapsed();
        let xdot_root_instructions = xdot_helper
            .function
            .compile_report()
            .stats
            .llvm_root_instructions_emitted;
        let xdot_total_instructions = xdot_helper
            .function
            .compile_report()
            .stats
            .llvm_total_instructions_emitted;
        helper_compile_stats.record_compile_report(xdot_helper.function.compile_report());
        on_progress(OcpCompileProgress::HelperCompiled {
            helper: OcpCompileHelperKind::Xdot,
            elapsed: xdot_helper_time,
            root_instructions: xdot_root_instructions,
            total_instructions: xdot_total_instructions,
        });

        let arc_started = Instant::now();
        let rk4_arc_helper = compile_multiple_shooting_arc_helper_runtime::<X, U, P>(
            &*self.ode,
            self.scheme.rk4_substeps,
            options.function_options,
        )?;
        let arc_time = arc_started.elapsed();
        let arc_root_instructions = rk4_arc_helper
            .function
            .compile_report()
            .stats
            .llvm_root_instructions_emitted;
        let arc_total_instructions = rk4_arc_helper
            .function
            .compile_report()
            .stats
            .llvm_total_instructions_emitted;
        helper_compile_stats.record_compile_report(rk4_arc_helper.function.compile_report());
        on_progress(OcpCompileProgress::HelperCompiled {
            helper: OcpCompileHelperKind::MultipleShootingArc,
            elapsed: arc_time,
            root_instructions: arc_root_instructions,
            total_instructions: arc_total_instructions,
        });

        Ok(CompiledMultipleShootingOcp {
            compiled,
            scheme: self.scheme,
            promotion_plan,
            promotion_offsets,
            xdot_helper,
            rk4_arc_helper,
            helper_compile_stats: OcpHelperCompileStats {
                xdot_helper_time: Some(xdot_helper_time),
                multiple_shooting_arc_helper_time: Some(arc_time),
                xdot_helper_root_instructions: Some(xdot_root_instructions),
                xdot_helper_total_instructions: Some(xdot_total_instructions),
                multiple_shooting_arc_helper_root_instructions: Some(arc_root_instructions),
                multiple_shooting_arc_helper_total_instructions: Some(arc_total_instructions),
                llvm_cache_hits: helper_compile_stats.llvm_cache_hits,
                llvm_cache_misses: helper_compile_stats.llvm_cache_misses,
                llvm_cache_load_time: helper_compile_stats.llvm_cache_load_time,
            },
            _marker: PhantomData,
        })
    }
}

fn build_multiple_shooting_interpolated_guess<X, U, G>(
    samples: &InterpolatedTrajectory<X, U, G>,
    intervals: usize,
) -> Result<MultipleShootingTrajectories<X, U, G>, GuessError>
where
    X: Clone,
    U: Clone,
    G: OcpGlobalDesign<f64> + Clone,
{
    validate_interpolation_samples(samples)?;
    let tf = samples.global.final_time();
    let times = mesh_times(tf, intervals);
    Ok(MultipleShootingTrajectories {
        x: Mesh {
            nodes: times
                .nodes
                .iter()
                .map(|&time| interpolate_at(&samples.sample_times, &samples.x_samples, time))
                .collect(),
            terminal: interpolate_at(&samples.sample_times, &samples.x_samples, times.terminal),
        },
        u: Mesh {
            nodes: times
                .nodes
                .iter()
                .map(|&time| interpolate_at(&samples.sample_times, &samples.u_samples, time))
                .collect(),
            terminal: interpolate_at(&samples.sample_times, &samples.u_samples, times.terminal),
        },
        dudt: times
            .nodes
            .iter()
            .map(|&time| interpolate_at(&samples.sample_times, &samples.dudt_samples, time))
            .collect(),
        global: samples.global.clone(),
        tf,
    })
}

fn build_multiple_shooting_rollout_guess<X, U, P, G>(
    xdot: &CompiledXdot<X, U, P>,
    x0: &Numeric<X>,
    u0: &Numeric<U>,
    global: Numeric<G>,
    parameters: &Numeric<P>,
    controller: &ControllerFn<Numeric<X>, Numeric<U>, Numeric<P>>,
    intervals: usize,
    rk4_substeps: usize,
) -> Result<MultipleShootingTrajectories<Numeric<X>, Numeric<U>, Numeric<G>>, GuessError>
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
    P: Vectorize<SX>,
    G: Vectorize<SX>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>>,
    Numeric<G>: OcpGlobalDesign<f64> + Clone,
{
    let tf = global.final_time();
    let h = tf / intervals as f64;
    let mut x = x0.clone();
    let mut u = u0.clone();
    let mut x_nodes = Vec::with_capacity(intervals);
    let mut u_nodes = Vec::with_capacity(intervals);
    let mut rates = Vec::with_capacity(intervals);
    for interval in 0..intervals {
        x_nodes.push(x.clone());
        u_nodes.push(u.clone());
        let t = interval as f64 * h;
        let dudt = controller(t, &x, &u, parameters);
        rates.push(dudt.clone());
        let (x_next, u_next) =
            rk4_rollout_numeric::<X, U, P>(xdot, &x, &u, &dudt, parameters, h, rk4_substeps)?;
        x = x_next;
        u = u_next;
    }
    Ok(MultipleShootingTrajectories {
        x: Mesh {
            nodes: x_nodes,
            terminal: x,
        },
        u: Mesh {
            nodes: u_nodes,
            terminal: u,
        },
        dudt: rates,
        global,
        tf,
    })
}

fn project_multiple_shooting<X, U, G>(
    values: &[f64],
    intervals: usize,
) -> Result<MultipleShootingTrajectories<Numeric<X>, Numeric<U>, Numeric<G>>, VectorizeLayoutError>
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
    G: Vectorize<SX>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>>,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>>,
    Numeric<G>: OcpGlobalDesign<f64> + Vectorize<f64, Rebind<f64> = Numeric<G>>,
{
    let mut index = 0usize;
    let x = take_numeric_mesh::<X>(values, &mut index, intervals)?;
    let u = take_numeric_mesh::<U>(values, &mut index, intervals)?;
    let mut dudt = Vec::with_capacity(intervals);
    for _ in 0..intervals {
        dudt.push(take_numeric_value::<U>(values, &mut index)?);
    }
    let global = take_numeric_value::<G>(values, &mut index)?;
    let tf = global.final_time();
    if index != values.len() {
        return Err(VectorizeLayoutError::LengthMismatch {
            expected: index,
            got: values.len(),
        });
    }
    Ok(MultipleShootingTrajectories {
        x,
        u,
        dudt,
        global,
        tf,
    })
}

impl<X, U, P, C, Beq, Bineq, G> CompiledMultipleShootingOcp<X, U, P, C, Beq, Bineq, G>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    G: OcpGlobalDesign<SX> + Vectorize<SX, Rebind<SX> = G>,
    C: Vectorize<SX, Rebind<SX> = C>,
    Beq: Vectorize<SX, Rebind<SX> = Beq>,
    Bineq: Vectorize<SX, Rebind<SX> = Bineq>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>> + Clone,
    Numeric<G>: OcpGlobalDesign<f64> + Vectorize<f64, Rebind<f64> = Numeric<G>> + Clone,
    Numeric<Beq>: Vectorize<f64, Rebind<f64> = Numeric<Beq>> + Clone,
    BoundTemplate<G>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<G>> + Clone,
    BoundTemplate<C>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<C>>,
    BoundTemplate<Bineq>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<Bineq>>,
{
    pub fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.compiled.backend_timing_metadata()
    }

    pub fn nlp_compile_stats(&self) -> NlpCompileStats {
        self.compiled.compile_stats()
    }

    pub fn helper_compile_stats(&self) -> OcpHelperCompileStats {
        self.helper_compile_stats
    }

    pub const fn helper_kernel_count(&self) -> usize {
        2
    }

    pub fn backend_compile_report(&self) -> &BackendCompileReport {
        self.compiled.backend_compile_report()
    }

    #[cfg(test)]
    pub(crate) fn test_dynamic_nlp(&self) -> &DynamicCompiledJitNlp {
        &self.compiled
    }

    #[cfg(test)]
    pub(crate) fn test_initial_guess_flat(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
    ) -> AnyResult<Vec<f64>> {
        Ok(self.build_initial_guess(values)?)
    }

    #[cfg(test)]
    pub(crate) fn test_runtime_bounds(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
    ) -> AnyResult<(RuntimeNlpBounds, Option<RuntimeNlpScaling>)> {
        Ok(self.build_runtime_bounds(values)?)
    }

    #[doc(hidden)]
    pub fn debug_lagrangian_hessian_lowered(&self) -> &sx_codegen::LoweredFunction {
        self.compiled.debug_lagrangian_hessian_lowered()
    }

    pub fn benchmark_nlp_evaluations(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: NlpEvaluationBenchmarkOptions,
    ) -> AnyResult<NlpEvaluationBenchmark> {
        self.benchmark_nlp_evaluations_with_progress(values, options, |_| {})
    }

    pub fn benchmark_nlp_evaluations_with_progress<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> AnyResult<NlpEvaluationBenchmark>
    where
        CB: FnMut(NlpEvaluationKernelKind),
    {
        let x0 = self.build_initial_guess(values)?;
        let (bounds, scaling) = self.build_runtime_bounds(values)?;
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        self.compiled.benchmark_bounded_evaluations_with_progress(
            &x0,
            runtime_params.as_deref(),
            &bounds,
            scaling.as_ref(),
            options,
            on_progress,
        )
    }

    pub fn validate_nlp_derivatives(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        options: FiniteDifferenceValidationOptions,
    ) -> AnyResult<NlpDerivativeValidationReport> {
        let x0 = self.build_initial_guess(values)?;
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        self.compiled.validate_derivatives_flat_values(
            &x0,
            runtime_params.as_deref(),
            equality_multipliers,
            inequality_multipliers,
            options,
        )
    }

    pub fn solve_sqp(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: &ClarabelSqpOptions,
    ) -> Result<MultipleShootingSqpSolveResult<Numeric<X>, Numeric<U>, Numeric<G>>, ClarabelSqpError>
    {
        let initial_guess_started = Instant::now();
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let initial_guess_time = initial_guess_started.elapsed();
        let runtime_bounds_started = Instant::now();
        let (bounds, scaling) = self
            .build_runtime_bounds(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let runtime_bounds_time = runtime_bounds_started.elapsed();
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        let summary = self.compiled.solve_sqp(
            &x0,
            runtime_params.as_deref(),
            &bounds,
            scaling.as_ref(),
            options,
        )?;
        let trajectories = project_multiple_shooting::<X, U, G>(&summary.x, self.scheme.intervals)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        Ok(MultipleShootingSqpSolveResult {
            time_grid: MultipleShootingTimeGrid {
                nodes: mesh_times(trajectories.tf, self.scheme.intervals),
            },
            trajectories,
            solver: summary,
            setup_timing: OcpSolveSetupTiming {
                initial_guess: initial_guess_time,
                runtime_bounds: runtime_bounds_time,
            },
        })
    }

    pub fn solve_sqp_with_callback<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: &ClarabelSqpOptions,
        mut callback: CB,
    ) -> Result<MultipleShootingSqpSolveResult<Numeric<X>, Numeric<U>, Numeric<G>>, ClarabelSqpError>
    where
        CB: FnMut(&MultipleShootingSqpSnapshot<Numeric<X>, Numeric<U>, Numeric<G>>),
    {
        let initial_guess_started = Instant::now();
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let initial_guess_time = initial_guess_started.elapsed();
        let runtime_bounds_started = Instant::now();
        let (bounds, scaling) = self
            .build_runtime_bounds(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let runtime_bounds_time = runtime_bounds_started.elapsed();
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        let summary = self.compiled.solve_sqp_with_callback(
            &x0,
            runtime_params.as_deref(),
            &bounds,
            scaling.as_ref(),
            options,
            |snapshot| {
                let trajectories =
                    project_multiple_shooting::<X, U, G>(&snapshot.x, self.scheme.intervals)
                        .expect("solver iterate should match runtime OCP layout");
                callback(&MultipleShootingSqpSnapshot {
                    time_grid: MultipleShootingTimeGrid {
                        nodes: mesh_times(trajectories.tf, self.scheme.intervals),
                    },
                    trajectories,
                    solver: snapshot.clone(),
                });
            },
        )?;
        let trajectories = project_multiple_shooting::<X, U, G>(&summary.x, self.scheme.intervals)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        Ok(MultipleShootingSqpSolveResult {
            time_grid: MultipleShootingTimeGrid {
                nodes: mesh_times(trajectories.tf, self.scheme.intervals),
            },
            trajectories,
            solver: summary,
            setup_timing: OcpSolveSetupTiming {
                initial_guess: initial_guess_time,
                runtime_bounds: runtime_bounds_time,
            },
        })
    }

    pub fn solve_interior_point(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: &InteriorPointOptions,
    ) -> Result<
        MultipleShootingInteriorPointSolveResult<Numeric<X>, Numeric<U>, Numeric<G>>,
        InteriorPointSolveError,
    > {
        let initial_guess_started = Instant::now();
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let initial_guess_time = initial_guess_started.elapsed();
        let runtime_bounds_started = Instant::now();
        let (bounds, scaling) = self
            .build_runtime_bounds(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let runtime_bounds_time = runtime_bounds_started.elapsed();
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        let summary = self.compiled.solve_interior_point(
            &x0,
            runtime_params.as_deref(),
            &bounds,
            scaling.as_ref(),
            options,
        )?;
        let trajectories = project_multiple_shooting::<X, U, G>(&summary.x, self.scheme.intervals)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        Ok(MultipleShootingInteriorPointSolveResult {
            time_grid: MultipleShootingTimeGrid {
                nodes: mesh_times(trajectories.tf, self.scheme.intervals),
            },
            trajectories,
            solver: summary,
            setup_timing: OcpSolveSetupTiming {
                initial_guess: initial_guess_time,
                runtime_bounds: runtime_bounds_time,
            },
        })
    }

    pub fn solve_interior_point_with_callback<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: &InteriorPointOptions,
        mut callback: CB,
    ) -> Result<
        MultipleShootingInteriorPointSolveResult<Numeric<X>, Numeric<U>, Numeric<G>>,
        InteriorPointSolveError,
    >
    where
        CB: FnMut(&MultipleShootingInteriorPointSnapshot<Numeric<X>, Numeric<U>, Numeric<G>>),
    {
        let initial_guess_started = Instant::now();
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let initial_guess_time = initial_guess_started.elapsed();
        let runtime_bounds_started = Instant::now();
        let (bounds, scaling) = self
            .build_runtime_bounds(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let runtime_bounds_time = runtime_bounds_started.elapsed();
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        let summary = self.compiled.solve_interior_point_with_callback(
            &x0,
            runtime_params.as_deref(),
            &bounds,
            scaling.as_ref(),
            options,
            |snapshot| {
                let trajectories =
                    project_multiple_shooting::<X, U, G>(&snapshot.x, self.scheme.intervals)
                        .expect("solver iterate should match runtime OCP layout");
                callback(&MultipleShootingInteriorPointSnapshot {
                    time_grid: MultipleShootingTimeGrid {
                        nodes: mesh_times(trajectories.tf, self.scheme.intervals),
                    },
                    trajectories,
                    solver: snapshot.clone(),
                });
            },
        )?;
        let trajectories = project_multiple_shooting::<X, U, G>(&summary.x, self.scheme.intervals)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        Ok(MultipleShootingInteriorPointSolveResult {
            time_grid: MultipleShootingTimeGrid {
                nodes: mesh_times(trajectories.tf, self.scheme.intervals),
            },
            trajectories,
            solver: summary,
            setup_timing: OcpSolveSetupTiming {
                initial_guess: initial_guess_time,
                runtime_bounds: runtime_bounds_time,
            },
        })
    }

    #[cfg(feature = "ipopt")]
    pub fn solve_ipopt(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: &IpoptOptions,
    ) -> Result<MultipleShootingIpoptSolveResult<Numeric<X>, Numeric<U>, Numeric<G>>, IpoptSolveError>
    {
        let initial_guess_started = Instant::now();
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let initial_guess_time = initial_guess_started.elapsed();
        let runtime_bounds_started = Instant::now();
        let (bounds, scaling) = self
            .build_runtime_bounds(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let runtime_bounds_time = runtime_bounds_started.elapsed();
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        let summary = self.compiled.solve_ipopt(
            &x0,
            runtime_params.as_deref(),
            &bounds,
            scaling.as_ref(),
            options,
        )?;
        let trajectories = project_multiple_shooting::<X, U, G>(&summary.x, self.scheme.intervals)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        Ok(MultipleShootingIpoptSolveResult {
            time_grid: MultipleShootingTimeGrid {
                nodes: mesh_times(trajectories.tf, self.scheme.intervals),
            },
            trajectories,
            solver: summary,
            setup_timing: OcpSolveSetupTiming {
                initial_guess: initial_guess_time,
                runtime_bounds: runtime_bounds_time,
            },
        })
    }

    #[cfg(feature = "ipopt")]
    pub fn solve_ipopt_with_callback<CB>(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: &IpoptOptions,
        mut callback: CB,
    ) -> Result<MultipleShootingIpoptSolveResult<Numeric<X>, Numeric<U>, Numeric<G>>, IpoptSolveError>
    where
        CB: FnMut(&MultipleShootingIpoptSnapshot<Numeric<X>, Numeric<U>, Numeric<G>>),
    {
        let initial_guess_started = Instant::now();
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let initial_guess_time = initial_guess_started.elapsed();
        let runtime_bounds_started = Instant::now();
        let (bounds, scaling) = self
            .build_runtime_bounds(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let runtime_bounds_time = runtime_bounds_started.elapsed();
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        let summary = self.compiled.solve_ipopt_with_callback(
            &x0,
            runtime_params.as_deref(),
            &bounds,
            scaling.as_ref(),
            options,
            |snapshot| {
                if let Ok(trajectories) =
                    project_multiple_shooting::<X, U, G>(&snapshot.x, self.scheme.intervals)
                {
                    callback(&MultipleShootingIpoptSnapshot {
                        time_grid: MultipleShootingTimeGrid {
                            nodes: mesh_times(trajectories.tf, self.scheme.intervals),
                        },
                        trajectories,
                        solver: snapshot.clone(),
                    });
                }
            },
        )?;
        let trajectories = project_multiple_shooting::<X, U, G>(&summary.x, self.scheme.intervals)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        Ok(MultipleShootingIpoptSolveResult {
            time_grid: MultipleShootingTimeGrid {
                nodes: mesh_times(trajectories.tf, self.scheme.intervals),
            },
            trajectories,
            solver: summary,
            setup_timing: OcpSolveSetupTiming {
                initial_guess: initial_guess_time,
                runtime_bounds: runtime_bounds_time,
            },
        })
    }

    pub fn interval_arcs(
        &self,
        trajectories: &MultipleShootingTrajectories<Numeric<X>, Numeric<U>, Numeric<G>>,
        parameters: &Numeric<P>,
    ) -> AnyResult<(Vec<IntervalArc<Numeric<X>>>, Vec<IntervalArc<Numeric<U>>>)> {
        let step = trajectories.tf / self.scheme.intervals as f64;
        let time_grid = MultipleShootingTimeGrid {
            nodes: mesh_times(trajectories.tf, self.scheme.intervals),
        };
        let mut x_arcs = Vec::with_capacity(self.scheme.intervals);
        let mut u_arcs = Vec::with_capacity(self.scheme.intervals);
        for interval in 0..self.scheme.intervals {
            let start_x = &trajectories.x.nodes[interval];
            let start_u = &trajectories.u.nodes[interval];
            let rate = &trajectories.dudt[interval];
            let start_time = time_grid.nodes.nodes[interval];
            let mut times = Vec::with_capacity(MULTIPLE_SHOOTING_ARC_SAMPLES + 1);
            times.push(start_time);
            times.extend((0..MULTIPLE_SHOOTING_ARC_SAMPLES).map(|sample| {
                start_time + ((sample + 1) as f64 / MULTIPLE_SHOOTING_ARC_SAMPLES as f64) * step
            }));
            let mut x_values = Vec::with_capacity(MULTIPLE_SHOOTING_ARC_SAMPLES + 1);
            x_values.push(start_x.clone());
            let mut u_values = Vec::with_capacity(MULTIPLE_SHOOTING_ARC_SAMPLES + 1);
            u_values.push(start_u.clone());
            for sample in 0..MULTIPLE_SHOOTING_ARC_SAMPLES {
                let fraction = (sample + 1) as f64 / MULTIPLE_SHOOTING_ARC_SAMPLES as f64;
                let (x_sample, u_sample) = self.rk4_arc_helper.eval(
                    start_x,
                    start_u,
                    rate,
                    parameters,
                    fraction * step,
                )?;
                x_values.push(x_sample);
                u_values.push(u_sample);
            }
            x_arcs.push(IntervalArc {
                times: times.clone(),
                values: x_values,
            });
            u_arcs.push(IntervalArc {
                times,
                values: u_values,
            });
        }
        Ok((x_arcs, u_arcs))
    }

    pub fn rank_constraint_violations(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        trajectories: &MultipleShootingTrajectories<Numeric<X>, Numeric<U>, Numeric<G>>,
        tolerance: f64,
    ) -> Result<OcpConstraintViolationReport, VectorizeLayoutError> {
        let decision = self.flatten_decision(trajectories);
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        let equality_values = self
            .compiled
            .evaluate_equalities_flat(&decision, runtime_params.as_deref())
            .map_err(|err| VectorizeLayoutError::LengthMismatch {
                expected: 0,
                got: err.to_string().len(),
            })?;
        let inequality_values = self
            .compiled
            .evaluate_inequalities_flat(&decision, runtime_params.as_deref())
            .map_err(|err| VectorizeLayoutError::LengthMismatch {
                expected: 0,
                got: err.to_string().len(),
            })?;

        let mut equality_groups = HashMap::new();
        let mut inequality_groups = HashMap::new();
        let continuity_x_labels = prefixed_leaf_names::<X>("continuity.x");
        let continuity_u_labels = prefixed_leaf_names::<U>("continuity.u");
        let boundary_eq_labels = prefixed_leaf_names::<Beq>("boundary_eq");
        let boundary_ineq_labels = prefixed_leaf_names::<Bineq>("boundary_ineq");
        let path_labels = prefixed_leaf_names::<C>("path");

        let continuity_x_count = self.scheme.intervals * X::LEN;
        let continuity_u_count = self.scheme.intervals * U::LEN;
        let boundary_eq_count = Beq::LEN;
        let boundary_ineq_count = Bineq::LEN;
        add_repeated_equalities(
            &mut equality_groups,
            &equality_values[..continuity_x_count],
            &continuity_x_labels,
            OcpConstraintCategory::ContinuityState,
            tolerance,
        );
        add_repeated_equalities(
            &mut equality_groups,
            &equality_values[continuity_x_count..continuity_x_count + continuity_u_count],
            &continuity_u_labels,
            OcpConstraintCategory::ContinuityControl,
            tolerance,
        );
        for (value, label) in inequality_values[..boundary_eq_count]
            .iter()
            .zip(boundary_eq_labels.iter())
        {
            accumulate_equality_group(
                &mut equality_groups,
                label,
                OcpConstraintCategory::BoundaryEquality,
                *value,
                tolerance,
            );
        }

        let boundary_ineq_values =
            &inequality_values[boundary_eq_count..boundary_eq_count + boundary_ineq_count];
        let path_values = &inequality_values[boundary_eq_count + boundary_ineq_count..];
        add_repeated_inequalities(
            &mut inequality_groups,
            boundary_ineq_values,
            &boundary_ineq_labels,
            &flatten_bounds(&values.bineq_bounds),
            OcpConstraintCategory::BoundaryInequality,
            tolerance,
        );
        add_repeated_inequalities(
            &mut inequality_groups,
            path_values,
            &path_labels,
            &flatten_bounds(&values.path_bounds),
            OcpConstraintCategory::Path,
            tolerance,
        );
        add_repeated_inequalities(
            &mut inequality_groups,
            &flatten_value(&trajectories.global),
            &prefixed_leaf_names::<G>("g"),
            &flatten_bounds(&values.global_bounds),
            OcpConstraintCategory::FinalTime,
            tolerance,
        );
        let mut report = OcpConstraintViolationReport {
            equalities: equality_groups_from_map(equality_groups, tolerance),
            inequalities: inequality_groups_from_map(inequality_groups, tolerance),
        };
        sort_ocp_constraint_report(&mut report);
        Ok(report)
    }

    fn runtime_parameters(&self, parameters: &Numeric<P>, beq: &Numeric<Beq>) -> Option<Vec<f64>> {
        let flat = flatten_value(&(parameters.clone(), beq.clone()));
        (!flat.is_empty()).then_some(flat)
    }

    fn flatten_decision(
        &self,
        trajectories: &MultipleShootingTrajectories<Numeric<X>, Numeric<U>, Numeric<G>>,
    ) -> Vec<f64> {
        let mut flat = Vec::with_capacity(ms_variable_count::<X, U, G>(self.scheme.intervals));
        flat.extend(flatten_mesh::<Numeric<X>, f64>(&trajectories.x));
        flat.extend(flatten_mesh::<Numeric<U>, f64>(&trajectories.u));
        flat.extend(flatten_dynamic_values::<Numeric<U>, f64>(
            &trajectories.dudt,
        ));
        flat.extend(flatten_value(&trajectories.global));
        flat
    }

    fn build_initial_guess(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
    ) -> Result<Vec<f64>, GuessError> {
        let trajectories = match &values.initial_guess {
            MultipleShootingInitialGuess::Explicit(guess) => guess.clone(),
            MultipleShootingInitialGuess::Constant { x, u, dudt, tf } => {
                MultipleShootingTrajectories {
                    x: Mesh {
                        nodes: vec![x.clone(); self.scheme.intervals],
                        terminal: x.clone(),
                    },
                    u: Mesh {
                        nodes: vec![u.clone(); self.scheme.intervals],
                        terminal: u.clone(),
                    },
                    dudt: vec![dudt.clone(); self.scheme.intervals],
                    global: Numeric::<G>::from_final_time(*tf),
                    tf: *tf,
                }
            }
            MultipleShootingInitialGuess::ConstantGlobal { x, u, dudt, global } => {
                MultipleShootingTrajectories {
                    x: Mesh {
                        nodes: vec![x.clone(); self.scheme.intervals],
                        terminal: x.clone(),
                    },
                    u: Mesh {
                        nodes: vec![u.clone(); self.scheme.intervals],
                        terminal: u.clone(),
                    },
                    dudt: vec![dudt.clone(); self.scheme.intervals],
                    global: global.clone(),
                    tf: global.final_time(),
                }
            }
            MultipleShootingInitialGuess::Interpolated(samples) => {
                build_multiple_shooting_interpolated_guess(samples, self.scheme.intervals)?
            }
            MultipleShootingInitialGuess::Rollout {
                x0,
                u0,
                tf,
                controller,
            } => build_multiple_shooting_rollout_guess::<X, U, P, G>(
                &self.xdot_helper,
                x0,
                u0,
                Numeric::<G>::from_final_time(*tf),
                &values.parameters,
                controller.as_ref(),
                self.scheme.intervals,
                self.scheme.rk4_substeps,
            )?,
            MultipleShootingInitialGuess::RolloutGlobal {
                x0,
                u0,
                global,
                controller,
            } => build_multiple_shooting_rollout_guess::<X, U, P, G>(
                &self.xdot_helper,
                x0,
                u0,
                global.clone(),
                &values.parameters,
                controller.as_ref(),
                self.scheme.intervals,
                self.scheme.rk4_substeps,
            )?,
        };
        Ok(self.flatten_decision(&trajectories))
    }

    fn build_runtime_bounds(
        &self,
        values: &MultipleShootingRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
    ) -> Result<(RuntimeNlpBounds, Option<RuntimeNlpScaling>), GuessError> {
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let offsets = self.promotion_offsets.eval(&runtime_params)?;
        let (variable_lower, variable_upper) = build_raw_bounds::<C, Beq, Bineq, G>(
            &self.promotion_plan,
            &offsets,
            &values.path_bounds,
            &values.bineq_bounds,
            &values.global_bounds,
            ms_variable_count::<X, U, G>(self.scheme.intervals),
        )?;
        let scaling = values
            .scaling
            .as_ref()
            .map(|scaling| self.build_nlp_scaling(scaling))
            .transpose()?;
        Ok((
            RuntimeNlpBounds {
                variables: optimization::ConstraintBounds {
                    lower: Some(variable_lower),
                    upper: Some(variable_upper),
                },
                inequalities: optimization::ConstraintBounds {
                    lower: Some(build_inequality_lower::<C, Beq, Bineq>(
                        &self.promotion_plan,
                        &offsets,
                        &values.path_bounds,
                        &values.bineq_bounds,
                    )?),
                    upper: Some(build_inequality_upper::<C, Beq, Bineq>(
                        &self.promotion_plan,
                        &offsets,
                        &values.path_bounds,
                        &values.bineq_bounds,
                    )?),
                },
            },
            scaling,
        ))
    }

    fn build_nlp_scaling(
        &self,
        scaling: &OcpScaling<Numeric<P>, Numeric<X>, Numeric<U>, Numeric<G>>,
    ) -> Result<RuntimeNlpScaling, GuessError> {
        if scaling.path.len() != C::LEN {
            return Err(GuessError::Invalid(format!(
                "path scaling length mismatch: expected {}, got {}",
                C::LEN,
                scaling.path.len()
            )));
        }
        if scaling.boundary_equalities.len() != Beq::LEN {
            return Err(GuessError::Invalid(format!(
                "boundary equality scaling length mismatch: expected {}, got {}",
                Beq::LEN,
                scaling.boundary_equalities.len()
            )));
        }
        if scaling.boundary_inequalities.len() != Bineq::LEN {
            return Err(GuessError::Invalid(format!(
                "boundary inequality scaling length mismatch: expected {}, got {}",
                Bineq::LEN,
                scaling.boundary_inequalities.len()
            )));
        }

        let mut variables = Vec::with_capacity(ms_variable_count::<X, U, G>(self.scheme.intervals));
        for _ in 0..=self.scheme.intervals {
            variables.extend(flatten_value(&scaling.state));
        }
        for _ in 0..=self.scheme.intervals {
            variables.extend(flatten_value(&scaling.control));
        }
        for _ in 0..self.scheme.intervals {
            variables.extend(flatten_value(&scaling.control_rate));
        }
        variables.extend(flatten_value(&scaling.global));

        let state_scale = flatten_value(&scaling.state);
        let control_scale = flatten_value(&scaling.control);
        let mut constraints = Vec::with_capacity(
            ms_equality_count::<X, U>(self.scheme.intervals)
                + ms_inequality_count::<C, Beq, Bineq>(self.scheme.intervals),
        );
        for _ in 0..self.scheme.intervals {
            constraints.extend_from_slice(&state_scale);
        }
        for _ in 0..self.scheme.intervals {
            constraints.extend_from_slice(&control_scale);
        }
        constraints.extend_from_slice(&scaling.boundary_equalities);
        constraints.extend_from_slice(&scaling.boundary_inequalities);
        for _ in 0..self.scheme.intervals {
            constraints.extend_from_slice(&scaling.path);
        }
        let _ = &scaling.parameters;
        Ok(RuntimeNlpScaling {
            variables,
            constraints,
            objective: scaling.objective,
        })
    }
}

fn build_direct_collocation_interpolated_guess<X, U, G>(
    samples: &InterpolatedTrajectory<X, U, G>,
    coeffs: &CollocationCoefficients,
    intervals: usize,
    time_grid: TimeGrid,
) -> Result<DirectCollocationTrajectories<X, U, G>, GuessError>
where
    X: Clone,
    U: Clone,
    G: OcpGlobalDesign<f64> + Clone,
{
    validate_interpolation_samples(samples)?;
    let tf = samples.global.final_time();
    let times = direct_collocation_times(tf, coeffs, intervals, time_grid);
    Ok(DirectCollocationTrajectories {
        x: Mesh {
            nodes: times
                .nodes
                .nodes
                .iter()
                .map(|&time| interpolate_at(&samples.sample_times, &samples.x_samples, time))
                .collect(),
            terminal: interpolate_at(
                &samples.sample_times,
                &samples.x_samples,
                times.nodes.terminal,
            ),
        },
        u: Mesh {
            nodes: times
                .nodes
                .nodes
                .iter()
                .map(|&time| interpolate_at(&samples.sample_times, &samples.u_samples, time))
                .collect(),
            terminal: interpolate_at(
                &samples.sample_times,
                &samples.u_samples,
                times.nodes.terminal,
            ),
        },
        root_x: IntervalGrid {
            intervals: times
                .roots
                .intervals
                .iter()
                .map(|interval| {
                    interval
                        .iter()
                        .map(|&time| {
                            interpolate_at(&samples.sample_times, &samples.x_samples, time)
                        })
                        .collect()
                })
                .collect(),
        },
        root_u: IntervalGrid {
            intervals: times
                .roots
                .intervals
                .iter()
                .map(|interval| {
                    interval
                        .iter()
                        .map(|&time| {
                            interpolate_at(&samples.sample_times, &samples.u_samples, time)
                        })
                        .collect()
                })
                .collect(),
        },
        root_dudt: IntervalGrid {
            intervals: times
                .roots
                .intervals
                .iter()
                .map(|interval| {
                    interval
                        .iter()
                        .map(|&time| {
                            interpolate_at(&samples.sample_times, &samples.dudt_samples, time)
                        })
                        .collect()
                })
                .collect(),
        },
        global: samples.global.clone(),
        tf,
    })
}

fn build_direct_collocation_rollout_guess<X, U, P, G>(
    xdot: &CompiledXdot<X, U, P>,
    x0: &Numeric<X>,
    u0: &Numeric<U>,
    global: Numeric<G>,
    parameters: &Numeric<P>,
    controller: &ControllerFn<Numeric<X>, Numeric<U>, Numeric<P>>,
    coeffs: &CollocationCoefficients,
    intervals: usize,
    time_grid: TimeGrid,
) -> Result<DirectCollocationTrajectories<Numeric<X>, Numeric<U>, Numeric<G>>, GuessError>
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
    P: Vectorize<SX>,
    G: Vectorize<SX>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>>,
    Numeric<G>: OcpGlobalDesign<f64> + Clone,
{
    let tf = global.final_time();
    let mesh_times = time_grid_mesh_unchecked(tf, intervals, time_grid);
    let mut x = x0.clone();
    let mut u = u0.clone();
    let mut x_nodes = Vec::with_capacity(intervals);
    let mut u_nodes = Vec::with_capacity(intervals);
    let mut root_x = Vec::with_capacity(intervals);
    let mut root_u = Vec::with_capacity(intervals);
    let mut root_dudt = Vec::with_capacity(intervals);
    for interval in 0..intervals {
        x_nodes.push(x.clone());
        u_nodes.push(u.clone());
        let t = mesh_times.nodes[interval];
        let interval_end = if interval + 1 < intervals {
            mesh_times.nodes[interval + 1]
        } else {
            mesh_times.terminal
        };
        let h = interval_end - t;
        let dudt = controller(t, &x, &u, parameters);
        let mut interval_x = Vec::with_capacity(coeffs.nodes.len());
        let mut interval_u = Vec::with_capacity(coeffs.nodes.len());
        let mut interval_dudt = Vec::with_capacity(coeffs.nodes.len());
        for &node in &coeffs.nodes {
            let duration = node * h;
            let (x_root, u_root) =
                rk4_rollout_numeric::<X, U, P>(xdot, &x, &u, &dudt, parameters, duration, 8)?;
            interval_x.push(x_root);
            interval_u.push(u_root);
            interval_dudt.push(dudt.clone());
        }
        root_x.push(interval_x);
        root_u.push(interval_u);
        root_dudt.push(interval_dudt);
        let (x_next, u_next) =
            rk4_rollout_numeric::<X, U, P>(xdot, &x, &u, &dudt, parameters, h, 8)?;
        x = x_next;
        u = u_next;
    }
    Ok(DirectCollocationTrajectories {
        x: Mesh {
            nodes: x_nodes,
            terminal: x,
        },
        u: Mesh {
            nodes: u_nodes,
            terminal: u,
        },
        root_x: IntervalGrid { intervals: root_x },
        root_u: IntervalGrid { intervals: root_u },
        root_dudt: IntervalGrid {
            intervals: root_dudt,
        },
        global,
        tf,
    })
}

fn project_direct_collocation<X, U, G>(
    values: &[f64],
    intervals: usize,
    order: usize,
) -> Result<DirectCollocationTrajectories<Numeric<X>, Numeric<U>, Numeric<G>>, VectorizeLayoutError>
where
    X: Vectorize<SX>,
    U: Vectorize<SX>,
    G: Vectorize<SX>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>>,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>>,
    Numeric<G>: OcpGlobalDesign<f64> + Vectorize<f64, Rebind<f64> = Numeric<G>>,
{
    let mut index = 0usize;
    let x = take_numeric_mesh::<X>(values, &mut index, intervals)?;
    let u = take_numeric_mesh::<U>(values, &mut index, intervals)?;
    let root_x = take_numeric_interval_grid::<X>(values, &mut index, intervals, order)?;
    let root_u = take_numeric_interval_grid::<U>(values, &mut index, intervals, order)?;
    let root_dudt = take_numeric_interval_grid::<U>(values, &mut index, intervals, order)?;
    let global = take_numeric_value::<G>(values, &mut index)?;
    let tf = global.final_time();
    if index != values.len() {
        return Err(VectorizeLayoutError::LengthMismatch {
            expected: index,
            got: values.len(),
        });
    }
    Ok(DirectCollocationTrajectories {
        x,
        u,
        root_x,
        root_u,
        root_dudt,
        global,
        tf,
    })
}

impl<X, U, P, C, Beq, Bineq, G> Ocp<X, U, P, C, Beq, Bineq, DirectCollocation, G>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    G: OcpGlobalDesign<SX> + Vectorize<SX, Rebind<SX> = G>,
    C: Vectorize<SX, Rebind<SX> = C>,
    Beq: Vectorize<SX, Rebind<SX> = Beq>,
    Bineq: Vectorize<SX, Rebind<SX> = Bineq>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>> + Clone,
    Numeric<Beq>: Vectorize<f64, Rebind<f64> = Numeric<Beq>> + Clone,
{
    fn transcribe_runtime_direct_collocation(
        &self,
        x_mesh: &Mesh<X>,
        u_mesh: &Mesh<U>,
        root_x: &IntervalGrid<X>,
        root_u: &IntervalGrid<U>,
        root_dudt: &IntervalGrid<U>,
        global: &G,
        parameters: &P,
        beq: &Beq,
        coeffs: &CollocationCoefficients,
        symbolic_library: &OcpSymbolicFunctionLibrary,
    ) -> Result<DcTranscription, SxError> {
        let tf = global.final_time();
        let interval_fractions =
            time_grid_interval_fractions(self.scheme.intervals, self.scheme.time_grid);
        let mut collocation_x =
            Vec::with_capacity(self.scheme.intervals * self.scheme.order * X::LEN);
        let mut collocation_u =
            Vec::with_capacity(self.scheme.intervals * self.scheme.order * U::LEN);
        let mut continuity_x = Vec::with_capacity(self.scheme.intervals * X::LEN);
        let mut continuity_u = Vec::with_capacity(self.scheme.intervals * U::LEN);
        let mut path = Vec::with_capacity(self.scheme.intervals * self.scheme.order * C::LEN);
        let mut objective = SX::zero();

        for (interval, interval_fraction) in interval_fractions
            .iter()
            .copied()
            .enumerate()
            .take(self.scheme.intervals)
        {
            let step = tf * interval_fraction;
            let x_start = x_mesh.nodes[interval].clone();
            let u_start = u_mesh.nodes[interval].clone();
            let mut basis_x = Vec::with_capacity(self.scheme.order + 1);
            basis_x.push(x_start.clone());
            basis_x.extend(root_x.intervals[interval].iter().cloned());
            let mut basis_u = Vec::with_capacity(self.scheme.order + 1);
            basis_u.push(u_start.clone());
            basis_u.extend(root_u.intervals[interval].iter().cloned());
            for root in 0..self.scheme.order {
                let path_value = self.eval_path_constraints_symbolic(
                    symbolic_library,
                    &root_x.intervals[interval][root],
                    &root_u.intervals[interval][root],
                    &root_dudt.intervals[interval][root],
                    parameters,
                )?;
                path.extend(path_value.flatten_cloned());
                let xdot = self.eval_ode_symbolic(
                    symbolic_library,
                    &root_x.intervals[interval][root],
                    &root_u.intervals[interval][root],
                    parameters,
                )?;
                collocation_x.extend(
                    subtract_vectorized(
                        &scale_vectorized(&xdot, step)?,
                        &weighted_sum_vectorized(&basis_x, &coeffs.c_matrix[root])?,
                    )?
                    .flatten_cloned(),
                );
                collocation_u.extend(
                    subtract_vectorized(
                        &scale_vectorized(&root_dudt.intervals[interval][root], step)?,
                        &weighted_sum_vectorized(&basis_u, &coeffs.c_matrix[root])?,
                    )?
                    .flatten_cloned(),
                );
                objective += step
                    * coeffs.b_vector[root + 1]
                    * self.eval_objective_lagrange_symbolic(
                        symbolic_library,
                        &root_x.intervals[interval][root],
                        &root_u.intervals[interval][root],
                        &root_dudt.intervals[interval][root],
                        parameters,
                    )?;
            }
            let x_end = weighted_sum_vectorized(&basis_x, &coeffs.d_vector)?;
            let u_end = weighted_sum_vectorized(&basis_u, &coeffs.d_vector)?;
            let x_next = if interval + 1 < self.scheme.intervals {
                x_mesh.nodes[interval + 1].clone()
            } else {
                x_mesh.terminal.clone()
            };
            let u_next = if interval + 1 < self.scheme.intervals {
                u_mesh.nodes[interval + 1].clone()
            } else {
                u_mesh.terminal.clone()
            };
            continuity_x.extend(subtract_vectorized(&x_next, &x_end)?.flatten_cloned());
            continuity_u.extend(subtract_vectorized(&u_next, &u_end)?.flatten_cloned());
        }

        let boundary_eq_values = self.eval_boundary_equalities_symbolic(
            symbolic_library,
            &x_mesh.nodes[0],
            &u_mesh.nodes[0],
            &x_mesh.terminal,
            &u_mesh.terminal,
            parameters,
            global,
        )?;
        let boundary_eq_residual = subtract_vectorized(&boundary_eq_values, beq)?.flatten_cloned();
        let boundary_ineq = self
            .eval_boundary_inequalities_symbolic(
                symbolic_library,
                &x_mesh.nodes[0],
                &u_mesh.nodes[0],
                &x_mesh.terminal,
                &u_mesh.terminal,
                parameters,
                global,
            )?
            .flatten_cloned();
        objective += self.eval_objective_mayer_symbolic(
            symbolic_library,
            &x_mesh.nodes[0],
            &u_mesh.nodes[0],
            &x_mesh.terminal,
            &u_mesh.terminal,
            parameters,
            global,
        )?;

        let mut equalities = Vec::with_capacity(dc_equality_count::<X, U>(
            self.scheme.intervals,
            self.scheme.order,
        ));
        equalities.extend(collocation_x);
        equalities.extend(collocation_u);
        equalities.extend(continuity_x);
        equalities.extend(continuity_u);

        Ok(DcTranscription {
            objective,
            equalities,
            boundary_eq_residual,
            boundary_ineq,
            path,
        })
    }

    pub fn compile_jit(
        &self,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, G>, OcpCompileError> {
        self.compile_jit_with_options(FunctionCompileOptions::from(LlvmOptimizationLevel::O3))
    }

    pub fn compile_jit_with_options(
        &self,
        options: FunctionCompileOptions,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, G>, OcpCompileError> {
        self.compile_jit_with_ocp_options(OcpCompileOptions::for_direct_collocation(options))
    }

    pub fn compile_jit_with_ocp_options(
        &self,
        options: OcpCompileOptions,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, G>, OcpCompileError> {
        self.compile_jit_with_ocp_options_and_progress_callback(options, |_| {})
    }

    pub fn compile_jit_with_ocp_options_and_progress_callback<CB>(
        &self,
        options: OcpCompileOptions,
        mut on_progress: CB,
    ) -> Result<CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, G>, OcpCompileError>
    where
        CB: FnMut(OcpCompileProgress),
    {
        if self.scheme.intervals == 0 {
            return Err(OcpCompileError::InvalidConfiguration(
                "direct collocation requires at least one interval".to_string(),
            ));
        }
        if self.scheme.order == 0 {
            return Err(OcpCompileError::InvalidConfiguration(
                "direct collocation requires at least one collocation root".to_string(),
            ));
        }
        validate_time_grid(self.scheme.time_grid)?;
        let coefficients = collocation_coefficients(self.scheme.family, self.scheme.order)?;
        let symbolic_library = self.build_symbolic_function_library(options.symbolic_functions)?;
        let decision_symbols = symbolic_dense_vector(
            "w",
            dc_variable_count::<X, U, G>(self.scheme.intervals, self.scheme.order),
        );
        let decision_matrix = SXMatrix::dense_column(decision_symbols.clone())?;
        let mut decision_index = 0usize;
        let x_mesh = take_symbolic_mesh::<X>(
            &decision_symbols,
            &mut decision_index,
            self.scheme.intervals,
        )?;
        let u_mesh = take_symbolic_mesh::<U>(
            &decision_symbols,
            &mut decision_index,
            self.scheme.intervals,
        )?;
        let root_x = take_symbolic_interval_grid::<X>(
            &decision_symbols,
            &mut decision_index,
            self.scheme.intervals,
            self.scheme.order,
        )?;
        let root_u = take_symbolic_interval_grid::<U>(
            &decision_symbols,
            &mut decision_index,
            self.scheme.intervals,
            self.scheme.order,
        )?;
        let root_dudt = take_symbolic_interval_grid::<U>(
            &decision_symbols,
            &mut decision_index,
            self.scheme.intervals,
            self.scheme.order,
        )?;
        let global = take_symbolic_value::<G>(&decision_symbols, &mut decision_index)?;

        let runtime_param_len = P::LEN + Beq::LEN;
        let parameter_symbols = symbolic_dense_vector("p", runtime_param_len);
        let parameter_matrix = (runtime_param_len > 0)
            .then(|| SXMatrix::dense_column(parameter_symbols.clone()))
            .transpose()?;
        let (parameters, beq) = if runtime_param_len > 0 {
            let mut param_index = 0usize;
            (
                take_symbolic_value::<P>(&parameter_symbols, &mut param_index)?,
                take_symbolic_value::<Beq>(&parameter_symbols, &mut param_index)?,
            )
        } else {
            (symbolic_zero_value::<P>()?, symbolic_zero_value::<Beq>()?)
        };

        let outputs = self.transcribe_runtime_direct_collocation(
            &x_mesh,
            &u_mesh,
            &root_x,
            &root_u,
            &root_dudt,
            &global,
            &parameters,
            &beq,
            &coefficients,
            &symbolic_library,
        )?;
        let inequalities = outputs
            .boundary_eq_residual
            .iter()
            .chain(outputs.boundary_ineq.iter())
            .chain(outputs.path.iter())
            .copied()
            .collect::<Vec<_>>();
        let promotion_plan = build_promotion_plan_flat(
            &decision_symbols,
            &outputs.boundary_eq_residual,
            &outputs.boundary_ineq,
            &outputs.path,
        );
        let promotion_offsets = compile_promotion_offsets(
            &promotion_plan,
            &(parameters, beq),
            options.function_options,
        )?;
        let mut helper_compile_stats = OcpHelperCompileStats::default();
        if let Some(function) = &promotion_offsets.function {
            helper_compile_stats.record_compile_report(function.function.compile_report());
        }
        let symbolic = symbolic_nlp_dynamic(
            self.name.clone(),
            decision_matrix,
            parameter_matrix,
            outputs.objective,
            (!outputs.equalities.is_empty())
                .then(|| SXMatrix::dense_column(outputs.equalities))
                .transpose()?,
            (!inequalities.is_empty())
                .then(|| SXMatrix::dense_column(inequalities))
                .transpose()?,
        )?;
        let compiled = symbolic.compile_jit_with_compile_options_and_symbolic_progress_callback(
            SymbolicNlpCompileOptions {
                function_options: options.function_options,
                hessian_strategy: options.hessian_strategy,
            },
            |progress| match progress {
                SymbolicCompileProgress::Stage(stage) => {
                    on_progress(OcpCompileProgress::SymbolicStage(stage));
                }
                SymbolicCompileProgress::Ready(metadata) => {
                    on_progress(OcpCompileProgress::SymbolicReady(metadata));
                }
            },
        )?;
        let nlp_compile_report = compiled.backend_compile_report();
        let nlp_jit_elapsed = nlp_compile_report
            .setup_profile
            .lowering
            .unwrap_or_default()
            + nlp_compile_report
                .setup_profile
                .llvm_jit
                .unwrap_or_default();
        on_progress(OcpCompileProgress::NlpKernelCompiled {
            elapsed: nlp_jit_elapsed,
            root_instructions: nlp_compile_report.stats.llvm_root_instructions_emitted,
            total_instructions: nlp_compile_report.stats.llvm_total_instructions_emitted,
        });
        let xdot_started = Instant::now();
        let xdot_helper = compile_xdot_helper::<X, U, P>(
            &*self.ode,
            symbolic_library.ode.as_ref(),
            options.function_options,
        )?;
        let xdot_helper_time = xdot_started.elapsed();
        let xdot_root_instructions = xdot_helper
            .function
            .compile_report()
            .stats
            .llvm_root_instructions_emitted;
        let xdot_total_instructions = xdot_helper
            .function
            .compile_report()
            .stats
            .llvm_total_instructions_emitted;
        helper_compile_stats.record_compile_report(xdot_helper.function.compile_report());
        on_progress(OcpCompileProgress::HelperCompiled {
            helper: OcpCompileHelperKind::Xdot,
            elapsed: xdot_helper_time,
            root_instructions: xdot_root_instructions,
            total_instructions: xdot_total_instructions,
        });
        Ok(CompiledDirectCollocationOcp {
            compiled,
            scheme: self.scheme,
            promotion_plan,
            promotion_offsets,
            xdot_helper,
            coefficients,
            helper_compile_stats: OcpHelperCompileStats {
                xdot_helper_time: Some(xdot_helper_time),
                multiple_shooting_arc_helper_time: None,
                xdot_helper_root_instructions: Some(xdot_root_instructions),
                xdot_helper_total_instructions: Some(xdot_total_instructions),
                multiple_shooting_arc_helper_root_instructions: None,
                multiple_shooting_arc_helper_total_instructions: None,
                llvm_cache_hits: helper_compile_stats.llvm_cache_hits,
                llvm_cache_misses: helper_compile_stats.llvm_cache_misses,
                llvm_cache_load_time: helper_compile_stats.llvm_cache_load_time,
            },
            _marker: PhantomData,
        })
    }
}

impl<X, U, P, C, Beq, Bineq, G> CompiledDirectCollocationOcp<X, U, P, C, Beq, Bineq, G>
where
    X: Vectorize<SX, Rebind<SX> = X> + Clone,
    U: Vectorize<SX, Rebind<SX> = U> + Clone,
    P: Vectorize<SX, Rebind<SX> = P>,
    G: OcpGlobalDesign<SX> + Vectorize<SX, Rebind<SX> = G>,
    C: Vectorize<SX, Rebind<SX> = C>,
    Beq: Vectorize<SX, Rebind<SX> = Beq>,
    Bineq: Vectorize<SX, Rebind<SX> = Bineq>,
    Numeric<X>: Vectorize<f64, Rebind<f64> = Numeric<X>> + Clone,
    Numeric<U>: Vectorize<f64, Rebind<f64> = Numeric<U>> + Clone,
    Numeric<P>: Vectorize<f64, Rebind<f64> = Numeric<P>> + Clone,
    Numeric<G>: OcpGlobalDesign<f64> + Vectorize<f64, Rebind<f64> = Numeric<G>> + Clone,
    Numeric<Beq>: Vectorize<f64, Rebind<f64> = Numeric<Beq>> + Clone,
    BoundTemplate<G>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<G>> + Clone,
    BoundTemplate<C>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<C>>,
    BoundTemplate<Bineq>: Vectorize<Bounds1D, Rebind<Bounds1D> = BoundTemplate<Bineq>>,
{
    pub fn backend_timing_metadata(&self) -> BackendTimingMetadata {
        self.compiled.backend_timing_metadata()
    }

    pub fn nlp_compile_stats(&self) -> NlpCompileStats {
        self.compiled.compile_stats()
    }

    pub fn helper_compile_stats(&self) -> OcpHelperCompileStats {
        self.helper_compile_stats
    }

    pub const fn helper_kernel_count(&self) -> usize {
        1
    }

    pub fn backend_compile_report(&self) -> &BackendCompileReport {
        self.compiled.backend_compile_report()
    }

    #[cfg(test)]
    pub(crate) fn test_dynamic_nlp(&self) -> &DynamicCompiledJitNlp {
        &self.compiled
    }

    #[cfg(test)]
    pub(crate) fn test_initial_guess_flat(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
    ) -> AnyResult<Vec<f64>> {
        Ok(self.build_initial_guess(values)?)
    }

    #[cfg(test)]
    pub(crate) fn test_runtime_bounds(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
    ) -> AnyResult<(RuntimeNlpBounds, Option<RuntimeNlpScaling>)> {
        Ok(self.build_runtime_bounds(values)?)
    }

    pub fn benchmark_nlp_evaluations(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: NlpEvaluationBenchmarkOptions,
    ) -> AnyResult<NlpEvaluationBenchmark> {
        self.benchmark_nlp_evaluations_with_progress(values, options, |_| {})
    }

    pub fn benchmark_nlp_evaluations_with_progress<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: NlpEvaluationBenchmarkOptions,
        on_progress: CB,
    ) -> AnyResult<NlpEvaluationBenchmark>
    where
        CB: FnMut(NlpEvaluationKernelKind),
    {
        let x0 = self.build_initial_guess(values)?;
        let (bounds, scaling) = self.build_runtime_bounds(values)?;
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        self.compiled.benchmark_bounded_evaluations_with_progress(
            &x0,
            runtime_params.as_deref(),
            &bounds,
            scaling.as_ref(),
            options,
            on_progress,
        )
    }

    pub fn validate_nlp_derivatives(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        equality_multipliers: &[f64],
        inequality_multipliers: &[f64],
        options: FiniteDifferenceValidationOptions,
    ) -> AnyResult<NlpDerivativeValidationReport> {
        let x0 = self.build_initial_guess(values)?;
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        self.compiled.validate_derivatives_flat_values(
            &x0,
            runtime_params.as_deref(),
            equality_multipliers,
            inequality_multipliers,
            options,
        )
    }

    fn time_grid_for_tf(&self, tf: f64) -> DirectCollocationTimeGrid {
        direct_collocation_times(
            tf,
            &self.coefficients,
            self.scheme.intervals,
            self.scheme.time_grid,
        )
    }

    pub fn solve_sqp(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: &ClarabelSqpOptions,
    ) -> Result<DirectCollocationSqpSolveResult<Numeric<X>, Numeric<U>, Numeric<G>>, ClarabelSqpError>
    {
        let initial_guess_started = Instant::now();
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let initial_guess_time = initial_guess_started.elapsed();
        let runtime_bounds_started = Instant::now();
        let (bounds, scaling) = self
            .build_runtime_bounds(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let runtime_bounds_time = runtime_bounds_started.elapsed();
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        let summary = self.compiled.solve_sqp(
            &x0,
            runtime_params.as_deref(),
            &bounds,
            scaling.as_ref(),
            options,
        )?;
        let trajectories = project_direct_collocation::<X, U, G>(
            &summary.x,
            self.scheme.intervals,
            self.scheme.order,
        )
        .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        Ok(DirectCollocationSqpSolveResult {
            time_grid: self.time_grid_for_tf(trajectories.tf),
            trajectories,
            solver: summary,
            setup_timing: OcpSolveSetupTiming {
                initial_guess: initial_guess_time,
                runtime_bounds: runtime_bounds_time,
            },
        })
    }

    pub fn solve_sqp_with_callback<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: &ClarabelSqpOptions,
        mut callback: CB,
    ) -> Result<DirectCollocationSqpSolveResult<Numeric<X>, Numeric<U>, Numeric<G>>, ClarabelSqpError>
    where
        CB: FnMut(&DirectCollocationSqpSnapshot<Numeric<X>, Numeric<U>, Numeric<G>>),
    {
        let initial_guess_started = Instant::now();
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let initial_guess_time = initial_guess_started.elapsed();
        let runtime_bounds_started = Instant::now();
        let (bounds, scaling) = self
            .build_runtime_bounds(values)
            .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        let runtime_bounds_time = runtime_bounds_started.elapsed();
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        let summary = self.compiled.solve_sqp_with_callback(
            &x0,
            runtime_params.as_deref(),
            &bounds,
            scaling.as_ref(),
            options,
            |snapshot| {
                let trajectories = project_direct_collocation::<X, U, G>(
                    &snapshot.x,
                    self.scheme.intervals,
                    self.scheme.order,
                )
                .expect("solver iterate should match runtime OCP layout");
                callback(&DirectCollocationSqpSnapshot {
                    time_grid: self.time_grid_for_tf(trajectories.tf),
                    trajectories,
                    solver: snapshot.clone(),
                });
            },
        )?;
        let trajectories = project_direct_collocation::<X, U, G>(
            &summary.x,
            self.scheme.intervals,
            self.scheme.order,
        )
        .map_err(|err| ClarabelSqpError::InvalidInput(err.to_string()))?;
        Ok(DirectCollocationSqpSolveResult {
            time_grid: self.time_grid_for_tf(trajectories.tf),
            trajectories,
            solver: summary,
            setup_timing: OcpSolveSetupTiming {
                initial_guess: initial_guess_time,
                runtime_bounds: runtime_bounds_time,
            },
        })
    }

    pub fn solve_interior_point(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: &InteriorPointOptions,
    ) -> Result<
        DirectCollocationInteriorPointSolveResult<Numeric<X>, Numeric<U>, Numeric<G>>,
        InteriorPointSolveError,
    > {
        let initial_guess_started = Instant::now();
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let initial_guess_time = initial_guess_started.elapsed();
        let runtime_bounds_started = Instant::now();
        let (bounds, scaling) = self
            .build_runtime_bounds(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let runtime_bounds_time = runtime_bounds_started.elapsed();
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        let summary = self.compiled.solve_interior_point(
            &x0,
            runtime_params.as_deref(),
            &bounds,
            scaling.as_ref(),
            options,
        )?;
        let trajectories = project_direct_collocation::<X, U, G>(
            &summary.x,
            self.scheme.intervals,
            self.scheme.order,
        )
        .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        Ok(DirectCollocationInteriorPointSolveResult {
            time_grid: self.time_grid_for_tf(trajectories.tf),
            trajectories,
            solver: summary,
            setup_timing: OcpSolveSetupTiming {
                initial_guess: initial_guess_time,
                runtime_bounds: runtime_bounds_time,
            },
        })
    }

    pub fn solve_interior_point_with_callback<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: &InteriorPointOptions,
        mut callback: CB,
    ) -> Result<
        DirectCollocationInteriorPointSolveResult<Numeric<X>, Numeric<U>, Numeric<G>>,
        InteriorPointSolveError,
    >
    where
        CB: FnMut(&DirectCollocationInteriorPointSnapshot<Numeric<X>, Numeric<U>, Numeric<G>>),
    {
        let initial_guess_started = Instant::now();
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let initial_guess_time = initial_guess_started.elapsed();
        let runtime_bounds_started = Instant::now();
        let (bounds, scaling) = self
            .build_runtime_bounds(values)
            .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        let runtime_bounds_time = runtime_bounds_started.elapsed();
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        let summary = self.compiled.solve_interior_point_with_callback(
            &x0,
            runtime_params.as_deref(),
            &bounds,
            scaling.as_ref(),
            options,
            |snapshot| {
                let trajectories = project_direct_collocation::<X, U, G>(
                    &snapshot.x,
                    self.scheme.intervals,
                    self.scheme.order,
                )
                .expect("solver iterate should match runtime OCP layout");
                callback(&DirectCollocationInteriorPointSnapshot {
                    time_grid: self.time_grid_for_tf(trajectories.tf),
                    trajectories,
                    solver: snapshot.clone(),
                });
            },
        )?;
        let trajectories = project_direct_collocation::<X, U, G>(
            &summary.x,
            self.scheme.intervals,
            self.scheme.order,
        )
        .map_err(|err| InteriorPointSolveError::InvalidInput(err.to_string()))?;
        Ok(DirectCollocationInteriorPointSolveResult {
            time_grid: self.time_grid_for_tf(trajectories.tf),
            trajectories,
            solver: summary,
            setup_timing: OcpSolveSetupTiming {
                initial_guess: initial_guess_time,
                runtime_bounds: runtime_bounds_time,
            },
        })
    }

    #[cfg(feature = "ipopt")]
    pub fn solve_ipopt(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: &IpoptOptions,
    ) -> Result<
        DirectCollocationIpoptSolveResult<Numeric<X>, Numeric<U>, Numeric<G>>,
        IpoptSolveError,
    > {
        let initial_guess_started = Instant::now();
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let initial_guess_time = initial_guess_started.elapsed();
        let runtime_bounds_started = Instant::now();
        let (bounds, scaling) = self
            .build_runtime_bounds(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let runtime_bounds_time = runtime_bounds_started.elapsed();
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        let summary = self.compiled.solve_ipopt(
            &x0,
            runtime_params.as_deref(),
            &bounds,
            scaling.as_ref(),
            options,
        )?;
        let trajectories = project_direct_collocation::<X, U, G>(
            &summary.x,
            self.scheme.intervals,
            self.scheme.order,
        )
        .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        Ok(DirectCollocationIpoptSolveResult {
            time_grid: self.time_grid_for_tf(trajectories.tf),
            trajectories,
            solver: summary,
            setup_timing: OcpSolveSetupTiming {
                initial_guess: initial_guess_time,
                runtime_bounds: runtime_bounds_time,
            },
        })
    }

    #[cfg(feature = "ipopt")]
    pub fn solve_ipopt_with_callback<CB>(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        options: &IpoptOptions,
        mut callback: CB,
    ) -> Result<
        DirectCollocationIpoptSolveResult<Numeric<X>, Numeric<U>, Numeric<G>>,
        IpoptSolveError,
    >
    where
        CB: FnMut(&DirectCollocationIpoptSnapshot<Numeric<X>, Numeric<U>, Numeric<G>>),
    {
        let initial_guess_started = Instant::now();
        let x0 = self
            .build_initial_guess(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let initial_guess_time = initial_guess_started.elapsed();
        let runtime_bounds_started = Instant::now();
        let (bounds, scaling) = self
            .build_runtime_bounds(values)
            .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        let runtime_bounds_time = runtime_bounds_started.elapsed();
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        let summary = self.compiled.solve_ipopt_with_callback(
            &x0,
            runtime_params.as_deref(),
            &bounds,
            scaling.as_ref(),
            options,
            |snapshot| {
                if let Ok(trajectories) = project_direct_collocation::<X, U, G>(
                    &snapshot.x,
                    self.scheme.intervals,
                    self.scheme.order,
                ) {
                    callback(&DirectCollocationIpoptSnapshot {
                        time_grid: self.time_grid_for_tf(trajectories.tf),
                        trajectories,
                        solver: snapshot.clone(),
                    });
                }
            },
        )?;
        let trajectories = project_direct_collocation::<X, U, G>(
            &summary.x,
            self.scheme.intervals,
            self.scheme.order,
        )
        .map_err(|err| IpoptSolveError::InvalidInput(err.to_string()))?;
        Ok(DirectCollocationIpoptSolveResult {
            time_grid: self.time_grid_for_tf(trajectories.tf),
            trajectories,
            solver: summary,
            setup_timing: OcpSolveSetupTiming {
                initial_guess: initial_guess_time,
                runtime_bounds: runtime_bounds_time,
            },
        })
    }

    pub fn rank_constraint_violations(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
        trajectories: &DirectCollocationTrajectories<Numeric<X>, Numeric<U>, Numeric<G>>,
        tolerance: f64,
    ) -> Result<OcpConstraintViolationReport, VectorizeLayoutError> {
        let decision = self.flatten_decision(trajectories);
        let runtime_params = self.runtime_parameters(&values.parameters, &values.beq);
        let equality_values = self
            .compiled
            .evaluate_equalities_flat(&decision, runtime_params.as_deref())
            .map_err(|err| VectorizeLayoutError::LengthMismatch {
                expected: 0,
                got: err.to_string().len(),
            })?;
        let inequality_values = self
            .compiled
            .evaluate_inequalities_flat(&decision, runtime_params.as_deref())
            .map_err(|err| VectorizeLayoutError::LengthMismatch {
                expected: 0,
                got: err.to_string().len(),
            })?;

        let mut equality_groups = HashMap::new();
        let mut inequality_groups = HashMap::new();

        let collocation_x_labels = prefixed_leaf_names::<X>("collocation.x");
        let collocation_u_labels = prefixed_leaf_names::<U>("collocation.u");
        let continuity_x_labels = prefixed_leaf_names::<X>("continuity.x");
        let continuity_u_labels = prefixed_leaf_names::<U>("continuity.u");
        let boundary_eq_labels = prefixed_leaf_names::<Beq>("boundary_eq");
        let boundary_ineq_labels = prefixed_leaf_names::<Bineq>("boundary_ineq");
        let path_labels = prefixed_leaf_names::<C>("path");

        let collocation_x_count = self.scheme.intervals * self.scheme.order * X::LEN;
        let collocation_u_count = self.scheme.intervals * self.scheme.order * U::LEN;
        let continuity_x_count = self.scheme.intervals * X::LEN;
        let continuity_u_count = self.scheme.intervals * U::LEN;
        let boundary_eq_count = Beq::LEN;
        let boundary_ineq_count = Bineq::LEN;
        let collocation_x_values = &equality_values[..collocation_x_count];
        let collocation_u_values =
            &equality_values[collocation_x_count..collocation_x_count + collocation_u_count];
        let continuity_x_values = &equality_values[collocation_x_count + collocation_u_count
            ..collocation_x_count + collocation_u_count + continuity_x_count];
        let continuity_u_values = &equality_values[collocation_x_count
            + collocation_u_count
            + continuity_x_count
            ..collocation_x_count + collocation_u_count + continuity_x_count + continuity_u_count];
        let boundary_eq_values = &inequality_values[..boundary_eq_count];
        let boundary_ineq_values =
            &inequality_values[boundary_eq_count..boundary_eq_count + boundary_ineq_count];
        let path_values = &inequality_values[boundary_eq_count + boundary_ineq_count..];

        add_repeated_equalities(
            &mut equality_groups,
            collocation_x_values,
            &collocation_x_labels,
            OcpConstraintCategory::CollocationState,
            tolerance,
        );
        add_repeated_equalities(
            &mut equality_groups,
            collocation_u_values,
            &collocation_u_labels,
            OcpConstraintCategory::CollocationControl,
            tolerance,
        );
        add_repeated_equalities(
            &mut equality_groups,
            continuity_x_values,
            &continuity_x_labels,
            OcpConstraintCategory::ContinuityState,
            tolerance,
        );
        add_repeated_equalities(
            &mut equality_groups,
            continuity_u_values,
            &continuity_u_labels,
            OcpConstraintCategory::ContinuityControl,
            tolerance,
        );
        for (value, label) in boundary_eq_values.iter().zip(boundary_eq_labels.iter()) {
            accumulate_equality_group(
                &mut equality_groups,
                label,
                OcpConstraintCategory::BoundaryEquality,
                *value,
                tolerance,
            );
        }

        add_repeated_inequalities(
            &mut inequality_groups,
            boundary_ineq_values,
            &boundary_ineq_labels,
            &flatten_bounds(&values.bineq_bounds),
            OcpConstraintCategory::BoundaryInequality,
            tolerance,
        );
        add_repeated_inequalities(
            &mut inequality_groups,
            path_values,
            &path_labels,
            &flatten_bounds(&values.path_bounds),
            OcpConstraintCategory::Path,
            tolerance,
        );
        add_repeated_inequalities(
            &mut inequality_groups,
            &flatten_value(&trajectories.global),
            &prefixed_leaf_names::<G>("g"),
            &flatten_bounds(&values.global_bounds),
            OcpConstraintCategory::FinalTime,
            tolerance,
        );

        let mut report = OcpConstraintViolationReport {
            equalities: equality_groups_from_map(equality_groups, tolerance),
            inequalities: inequality_groups_from_map(inequality_groups, tolerance),
        };
        sort_ocp_constraint_report(&mut report);
        Ok(report)
    }

    fn runtime_parameters(&self, parameters: &Numeric<P>, beq: &Numeric<Beq>) -> Option<Vec<f64>> {
        let flat = flatten_value(&(parameters.clone(), beq.clone()));
        (!flat.is_empty()).then_some(flat)
    }

    fn flatten_decision(
        &self,
        trajectories: &DirectCollocationTrajectories<Numeric<X>, Numeric<U>, Numeric<G>>,
    ) -> Vec<f64> {
        let mut flat = Vec::with_capacity(dc_variable_count::<X, U, G>(
            self.scheme.intervals,
            self.scheme.order,
        ));
        flat.extend(flatten_mesh::<Numeric<X>, f64>(&trajectories.x));
        flat.extend(flatten_mesh::<Numeric<U>, f64>(&trajectories.u));
        flat.extend(flatten_interval_grid::<Numeric<X>, f64>(
            &trajectories.root_x,
        ));
        flat.extend(flatten_interval_grid::<Numeric<U>, f64>(
            &trajectories.root_u,
        ));
        flat.extend(flatten_interval_grid::<Numeric<U>, f64>(
            &trajectories.root_dudt,
        ));
        flat.extend(flatten_value(&trajectories.global));
        flat
    }

    fn build_initial_guess(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
    ) -> Result<Vec<f64>, GuessError> {
        let trajectories = match &values.initial_guess {
            DirectCollocationInitialGuess::Explicit(guess) => guess.clone(),
            DirectCollocationInitialGuess::Constant { x, u, dudt, tf } => {
                DirectCollocationTrajectories {
                    x: Mesh {
                        nodes: vec![x.clone(); self.scheme.intervals],
                        terminal: x.clone(),
                    },
                    u: Mesh {
                        nodes: vec![u.clone(); self.scheme.intervals],
                        terminal: u.clone(),
                    },
                    root_x: IntervalGrid {
                        intervals: vec![vec![x.clone(); self.scheme.order]; self.scheme.intervals],
                    },
                    root_u: IntervalGrid {
                        intervals: vec![vec![u.clone(); self.scheme.order]; self.scheme.intervals],
                    },
                    root_dudt: IntervalGrid {
                        intervals: vec![
                            vec![dudt.clone(); self.scheme.order];
                            self.scheme.intervals
                        ],
                    },
                    global: Numeric::<G>::from_final_time(*tf),
                    tf: *tf,
                }
            }
            DirectCollocationInitialGuess::ConstantGlobal { x, u, dudt, global } => {
                DirectCollocationTrajectories {
                    x: Mesh {
                        nodes: vec![x.clone(); self.scheme.intervals],
                        terminal: x.clone(),
                    },
                    u: Mesh {
                        nodes: vec![u.clone(); self.scheme.intervals],
                        terminal: u.clone(),
                    },
                    root_x: IntervalGrid {
                        intervals: vec![vec![x.clone(); self.scheme.order]; self.scheme.intervals],
                    },
                    root_u: IntervalGrid {
                        intervals: vec![vec![u.clone(); self.scheme.order]; self.scheme.intervals],
                    },
                    root_dudt: IntervalGrid {
                        intervals: vec![
                            vec![dudt.clone(); self.scheme.order];
                            self.scheme.intervals
                        ],
                    },
                    global: global.clone(),
                    tf: global.final_time(),
                }
            }
            DirectCollocationInitialGuess::Interpolated(samples) => {
                build_direct_collocation_interpolated_guess(
                    samples,
                    &self.coefficients,
                    self.scheme.intervals,
                    self.scheme.time_grid,
                )?
            }
            DirectCollocationInitialGuess::Rollout {
                x0,
                u0,
                tf,
                controller,
            } => build_direct_collocation_rollout_guess::<X, U, P, G>(
                &self.xdot_helper,
                x0,
                u0,
                Numeric::<G>::from_final_time(*tf),
                &values.parameters,
                controller.as_ref(),
                &self.coefficients,
                self.scheme.intervals,
                self.scheme.time_grid,
            )?,
            DirectCollocationInitialGuess::RolloutGlobal {
                x0,
                u0,
                global,
                controller,
            } => build_direct_collocation_rollout_guess::<X, U, P, G>(
                &self.xdot_helper,
                x0,
                u0,
                global.clone(),
                &values.parameters,
                controller.as_ref(),
                &self.coefficients,
                self.scheme.intervals,
                self.scheme.time_grid,
            )?,
        };
        Ok(self.flatten_decision(&trajectories))
    }

    fn build_runtime_bounds(
        &self,
        values: &DirectCollocationRuntimeValues<
            Numeric<P>,
            BoundTemplate<C>,
            Numeric<Beq>,
            BoundTemplate<Bineq>,
            Numeric<X>,
            Numeric<U>,
            Numeric<G>,
            BoundTemplate<G>,
        >,
    ) -> Result<(RuntimeNlpBounds, Option<RuntimeNlpScaling>), GuessError> {
        let runtime_params: OcpParametersNum<P, Beq> =
            (values.parameters.clone(), values.beq.clone());
        let offsets = self.promotion_offsets.eval(&runtime_params)?;
        let (variable_lower, variable_upper) = build_raw_bounds::<C, Beq, Bineq, G>(
            &self.promotion_plan,
            &offsets,
            &values.path_bounds,
            &values.bineq_bounds,
            &values.global_bounds,
            dc_variable_count::<X, U, G>(self.scheme.intervals, self.scheme.order),
        )?;
        let scaling = values
            .scaling
            .as_ref()
            .map(|scaling| self.build_nlp_scaling(scaling))
            .transpose()?;
        Ok((
            RuntimeNlpBounds {
                variables: optimization::ConstraintBounds {
                    lower: Some(variable_lower),
                    upper: Some(variable_upper),
                },
                inequalities: optimization::ConstraintBounds {
                    lower: Some(build_inequality_lower::<C, Beq, Bineq>(
                        &self.promotion_plan,
                        &offsets,
                        &values.path_bounds,
                        &values.bineq_bounds,
                    )?),
                    upper: Some(build_inequality_upper::<C, Beq, Bineq>(
                        &self.promotion_plan,
                        &offsets,
                        &values.path_bounds,
                        &values.bineq_bounds,
                    )?),
                },
            },
            scaling,
        ))
    }

    fn build_nlp_scaling(
        &self,
        scaling: &OcpScaling<Numeric<P>, Numeric<X>, Numeric<U>, Numeric<G>>,
    ) -> Result<RuntimeNlpScaling, GuessError> {
        if scaling.path.len() != C::LEN {
            return Err(GuessError::Invalid(format!(
                "path scaling length mismatch: expected {}, got {}",
                C::LEN,
                scaling.path.len()
            )));
        }
        if scaling.boundary_equalities.len() != Beq::LEN {
            return Err(GuessError::Invalid(format!(
                "boundary equality scaling length mismatch: expected {}, got {}",
                Beq::LEN,
                scaling.boundary_equalities.len()
            )));
        }
        if scaling.boundary_inequalities.len() != Bineq::LEN {
            return Err(GuessError::Invalid(format!(
                "boundary inequality scaling length mismatch: expected {}, got {}",
                Bineq::LEN,
                scaling.boundary_inequalities.len()
            )));
        }

        let mut variables = Vec::with_capacity(dc_variable_count::<X, U, G>(
            self.scheme.intervals,
            self.scheme.order,
        ));
        for _ in 0..=self.scheme.intervals {
            variables.extend(flatten_value(&scaling.state));
        }
        for _ in 0..=self.scheme.intervals {
            variables.extend(flatten_value(&scaling.control));
        }
        for _ in 0..(self.scheme.intervals * self.scheme.order) {
            variables.extend(flatten_value(&scaling.state));
        }
        for _ in 0..(self.scheme.intervals * self.scheme.order) {
            variables.extend(flatten_value(&scaling.control));
        }
        for _ in 0..(self.scheme.intervals * self.scheme.order) {
            variables.extend(flatten_value(&scaling.control_rate));
        }
        variables.extend(flatten_value(&scaling.global));

        let state_scale = flatten_value(&scaling.state);
        let control_scale = flatten_value(&scaling.control);
        let mut constraints = Vec::with_capacity(
            dc_equality_count::<X, U>(self.scheme.intervals, self.scheme.order)
                + dc_inequality_count::<C, Beq, Bineq>(self.scheme.intervals, self.scheme.order),
        );
        for _ in 0..(self.scheme.intervals * self.scheme.order) {
            constraints.extend_from_slice(&state_scale);
        }
        for _ in 0..(self.scheme.intervals * self.scheme.order) {
            constraints.extend_from_slice(&control_scale);
        }
        for _ in 0..self.scheme.intervals {
            constraints.extend_from_slice(&state_scale);
        }
        for _ in 0..self.scheme.intervals {
            constraints.extend_from_slice(&control_scale);
        }
        constraints.extend_from_slice(&scaling.boundary_equalities);
        constraints.extend_from_slice(&scaling.boundary_inequalities);
        for _ in 0..(self.scheme.intervals * self.scheme.order) {
            constraints.extend_from_slice(&scaling.path);
        }
        let _ = &scaling.parameters;
        Ok(RuntimeNlpScaling {
            variables,
            constraints,
            objective: scaling.objective,
        })
    }
}
