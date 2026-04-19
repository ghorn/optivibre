use std::collections::{BTreeSet, VecDeque};

use amd::Control;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsrGraph {
    offsets: Vec<usize>,
    neighbors: Vec<usize>,
}

impl CsrGraph {
    pub fn new(offsets: Vec<usize>, neighbors: Vec<usize>) -> Result<Self, OrderingError> {
        if offsets.is_empty() {
            return Err(OrderingError::InvalidGraph(
                "CSR offsets must contain at least one entry".into(),
            ));
        }
        if offsets[0] != 0 {
            return Err(OrderingError::InvalidGraph(
                "CSR offsets must start at zero".into(),
            ));
        }
        if offsets[offsets.len() - 1] != neighbors.len() {
            return Err(OrderingError::InvalidGraph(
                "final CSR offset must equal neighbor length".into(),
            ));
        }
        if offsets.windows(2).any(|window| window[0] > window[1]) {
            return Err(OrderingError::InvalidGraph(
                "CSR offsets must be monotone".into(),
            ));
        }
        let vertex_count = offsets.len() - 1;
        for vertex in 0..vertex_count {
            let start = offsets[vertex];
            let end = offsets[vertex + 1];
            let slice = &neighbors[start..end];
            let mut previous = None;
            for &neighbor in slice {
                if neighbor >= vertex_count {
                    return Err(OrderingError::InvalidGraph(format!(
                        "neighbor index {neighbor} out of bounds for {vertex_count} vertices"
                    )));
                }
                if neighbor == vertex {
                    return Err(OrderingError::InvalidGraph(format!(
                        "self-loop at vertex {vertex} is not supported"
                    )));
                }
                if let Some(prev_neighbor) = previous
                    && prev_neighbor >= neighbor
                {
                    return Err(OrderingError::InvalidGraph(
                        "adjacency lists must be strictly increasing".into(),
                    ));
                }
                previous = Some(neighbor);
            }
        }
        Ok(Self { offsets, neighbors })
    }

    pub fn from_edges(
        vertex_count: usize,
        edges: &[(usize, usize)],
    ) -> Result<Self, OrderingError> {
        let mut adjacency = vec![Vec::new(); vertex_count];
        for &(lhs, rhs) in edges {
            if lhs >= vertex_count || rhs >= vertex_count {
                return Err(OrderingError::InvalidGraph(format!(
                    "edge ({lhs}, {rhs}) is out of bounds for {vertex_count} vertices"
                )));
            }
            if lhs == rhs {
                continue;
            }
            adjacency[lhs].push(rhs);
            adjacency[rhs].push(lhs);
        }
        Self::from_adjacency(adjacency)
    }

    pub fn from_symmetric_csc(
        dimension: usize,
        col_ptrs: &[usize],
        row_indices: &[usize],
    ) -> Result<Self, OrderingError> {
        if col_ptrs.len() != dimension + 1 {
            return Err(OrderingError::InvalidGraph(format!(
                "expected {} column pointers, got {}",
                dimension + 1,
                col_ptrs.len()
            )));
        }
        if col_ptrs.first().copied().unwrap_or_default() != 0 {
            return Err(OrderingError::InvalidGraph(
                "CSC column pointers must start at zero".into(),
            ));
        }
        if col_ptrs.last().copied().unwrap_or_default() != row_indices.len() {
            return Err(OrderingError::InvalidGraph(
                "final CSC column pointer must equal row index length".into(),
            ));
        }
        if col_ptrs.windows(2).any(|window| window[0] > window[1]) {
            return Err(OrderingError::InvalidGraph(
                "CSC column pointers must be monotone".into(),
            ));
        }
        let mut adjacency = vec![Vec::new(); dimension];
        for col in 0..dimension {
            for &row in &row_indices[col_ptrs[col]..col_ptrs[col + 1]] {
                if row >= dimension {
                    return Err(OrderingError::InvalidGraph(format!(
                        "row index {row} out of bounds for {dimension}x{dimension} matrix"
                    )));
                }
                if row == col {
                    continue;
                }
                adjacency[row].push(col);
                adjacency[col].push(row);
            }
        }
        Self::from_adjacency(adjacency)
    }

    pub fn vertex_count(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn edge_count(&self) -> usize {
        self.neighbors.len() / 2
    }

    pub fn degree(&self, vertex: usize) -> usize {
        self.offsets[vertex + 1] - self.offsets[vertex]
    }

    pub fn neighbors(&self, vertex: usize) -> &[usize] {
        &self.neighbors[self.offsets[vertex]..self.offsets[vertex + 1]]
    }

    fn from_adjacency(mut adjacency: Vec<Vec<usize>>) -> Result<Self, OrderingError> {
        let vertex_count = adjacency.len();
        let mut offsets = Vec::with_capacity(vertex_count + 1);
        let mut neighbors = Vec::new();
        offsets.push(0);
        for (vertex, vertex_neighbors) in adjacency.iter_mut().enumerate() {
            vertex_neighbors.sort_unstable();
            vertex_neighbors.dedup();
            if vertex_neighbors.binary_search(&vertex).is_ok() {
                return Err(OrderingError::InvalidGraph(format!(
                    "self-loop at vertex {vertex} is not supported"
                )));
            }
            neighbors.extend(vertex_neighbors.iter().copied());
            offsets.push(neighbors.len());
        }
        Self::new(offsets, neighbors)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Permutation {
    perm: Vec<usize>,
    inverse: Vec<usize>,
}

impl Permutation {
    pub fn new(perm: Vec<usize>) -> Result<Self, OrderingError> {
        let mut inverse = vec![usize::MAX; perm.len()];
        for (ordered, &original) in perm.iter().enumerate() {
            if original >= perm.len() {
                return Err(OrderingError::InvalidPermutation(format!(
                    "permutation entry {original} out of bounds for size {}",
                    perm.len()
                )));
            }
            if inverse[original] != usize::MAX {
                return Err(OrderingError::InvalidPermutation(format!(
                    "duplicate permutation entry {original}"
                )));
            }
            inverse[original] = ordered;
        }
        Ok(Self { perm, inverse })
    }

    pub fn identity(size: usize) -> Self {
        let perm = (0..size).collect::<Vec<_>>();
        Self {
            perm: perm.clone(),
            inverse: perm,
        }
    }

    pub fn len(&self) -> usize {
        self.perm.len()
    }

    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }

    pub fn perm(&self) -> &[usize] {
        &self.perm
    }

    pub fn inverse(&self) -> &[usize] {
        &self.inverse
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NestedDissectionOptions {
    pub leaf_size: usize,
    pub max_separator_fraction: f64,
    pub max_cross_edge_repairs: usize,
}

impl Default for NestedDissectionOptions {
    fn default() -> Self {
        Self {
            leaf_size: 32,
            max_separator_fraction: 0.35,
            max_cross_edge_repairs: 128,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderingStats {
    pub connected_components: usize,
    pub separator_calls: usize,
    pub leaf_calls: usize,
    pub separator_vertices: usize,
    pub max_separator_size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderingSummary {
    pub permutation: Permutation,
    pub stats: OrderingStats,
}

#[derive(Debug, Error)]
pub enum OrderingError {
    #[error("invalid graph: {0}")]
    InvalidGraph(String),
    #[error("invalid permutation: {0}")]
    InvalidPermutation(String),
    #[error("invalid nested-dissection options: {0}")]
    InvalidOptions(String),
    #[error("ordering algorithm failed: {0}")]
    Algorithm(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum VertexPartition {
    Left,
    Right,
    Separator,
    Unassigned,
}

struct SeparatorSplit {
    left: Vec<usize>,
    right: Vec<usize>,
    separator: Vec<usize>,
}

pub fn nested_dissection_order(
    graph: &CsrGraph,
    options: &NestedDissectionOptions,
) -> Result<OrderingSummary, OrderingError> {
    if options.leaf_size == 0 {
        return Err(OrderingError::InvalidOptions(
            "leaf size must be at least one".into(),
        ));
    }
    if !(0.0..1.0).contains(&options.max_separator_fraction) {
        return Err(OrderingError::InvalidOptions(
            "separator fraction must lie in [0, 1)".into(),
        ));
    }

    let mut order = Vec::with_capacity(graph.vertex_count());
    let mut stats = OrderingStats {
        connected_components: 0,
        separator_calls: 0,
        leaf_calls: 0,
        separator_vertices: 0,
        max_separator_size: 0,
    };
    let vertices = (0..graph.vertex_count()).collect::<Vec<_>>();
    order_recursive(graph, &vertices, options, &mut order, &mut stats);
    let permutation = Permutation::new(order)?;
    Ok(OrderingSummary { permutation, stats })
}

pub fn approximate_minimum_degree_order(
    graph: &CsrGraph,
) -> Result<OrderingSummary, OrderingError> {
    let adjacency = (0..graph.vertex_count())
        .map(|vertex| graph.neighbors(vertex).to_vec())
        .collect::<Vec<_>>();
    let permutation = Permutation::new(approximate_minimum_degree_local_order(&adjacency)?)?;
    let component_count =
        connected_components(graph, &(0..graph.vertex_count()).collect::<Vec<_>>());
    Ok(OrderingSummary {
        permutation,
        stats: OrderingStats {
            connected_components: component_count.len().saturating_sub(1),
            separator_calls: 0,
            leaf_calls: usize::from(graph.vertex_count() > 0),
            separator_vertices: 0,
            max_separator_size: 0,
        },
    })
}

fn order_recursive(
    graph: &CsrGraph,
    vertices: &[usize],
    options: &NestedDissectionOptions,
    order: &mut Vec<usize>,
    stats: &mut OrderingStats,
) {
    if vertices.is_empty() {
        return;
    }
    if vertices.len() <= options.leaf_size {
        stats.leaf_calls += 1;
        order.extend(leaf_minimum_degree_order(graph, vertices));
        return;
    }

    let components = connected_components(graph, vertices);
    if components.len() > 1 {
        stats.connected_components += components.len() - 1;
        for component in components {
            order_recursive(graph, &component, options, order, stats);
        }
        return;
    }

    stats.separator_calls += 1;
    if let Some(split) = find_separator_split(graph, vertices, options) {
        if split.left.is_empty() || split.right.is_empty() || split.separator.is_empty() {
            stats.leaf_calls += 1;
            order.extend(leaf_minimum_degree_order(graph, vertices));
            return;
        }
        stats.separator_vertices += split.separator.len();
        stats.max_separator_size = stats.max_separator_size.max(split.separator.len());
        order_recursive(graph, &split.left, options, order, stats);
        order_recursive(graph, &split.right, options, order, stats);
        order_recursive(graph, &split.separator, options, order, stats);
    } else {
        stats.leaf_calls += 1;
        order.extend(leaf_minimum_degree_order(graph, vertices));
    }
}

fn connected_components(graph: &CsrGraph, vertices: &[usize]) -> Vec<Vec<usize>> {
    let mut subset_mask = vec![false; graph.vertex_count()];
    for &vertex in vertices {
        subset_mask[vertex] = true;
    }
    let mut visited = vec![false; graph.vertex_count()];
    let mut components = Vec::new();
    for &start in vertices {
        if visited[start] {
            continue;
        }
        let mut queue = VecDeque::from([start]);
        let mut component = Vec::new();
        visited[start] = true;
        while let Some(vertex) = queue.pop_front() {
            component.push(vertex);
            for &neighbor in graph.neighbors(vertex) {
                if subset_mask[neighbor] && !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components.sort_by_key(|component| component[0]);
    components
}

fn leaf_minimum_degree_order(graph: &CsrGraph, vertices: &[usize]) -> Vec<usize> {
    let adjacency = induced_local_adjacency(graph, vertices);
    match approximate_minimum_degree_local_order(&adjacency) {
        Ok(local_order) => local_order
            .into_iter()
            .map(|local| vertices[local])
            .collect(),
        Err(_) => exact_minimum_degree_local_order(&adjacency)
            .into_iter()
            .map(|local| vertices[local])
            .collect(),
    }
}

fn induced_local_adjacency(graph: &CsrGraph, vertices: &[usize]) -> Vec<Vec<usize>> {
    let vertex_count = vertices.len();
    let mut local_index = vec![usize::MAX; graph.vertex_count()];
    for (local, &global) in vertices.iter().enumerate() {
        local_index[global] = local;
    }
    let mut adjacency = vec![Vec::new(); vertex_count];
    for (local, &global) in vertices.iter().enumerate() {
        adjacency[local] = graph
            .neighbors(global)
            .iter()
            .copied()
            .filter_map(|neighbor| {
                let mapped = local_index[neighbor];
                (mapped != usize::MAX).then_some(mapped)
            })
            .collect::<Vec<_>>();
        adjacency[local].sort_unstable();
        adjacency[local].dedup();
    }
    adjacency
}

fn approximate_minimum_degree_local_order(
    adjacency: &[Vec<usize>],
) -> Result<Vec<usize>, OrderingError> {
    let (col_ptrs, row_indices) = lower_csc_from_local_adjacency(adjacency);
    let n = isize::try_from(adjacency.len()).map_err(|_| {
        OrderingError::Algorithm("graph is too large for amd::order index width".into())
    })?;
    let (perm, _, _) = amd::order(n, &col_ptrs, &row_indices, &Control::default())
        .map_err(|status| OrderingError::Algorithm(format!("amd::order failed with {status:?}")))?;
    perm.into_iter()
        .map(|entry| {
            usize::try_from(entry).map_err(|_| {
                OrderingError::Algorithm(format!("invalid amd permutation entry {entry}"))
            })
        })
        .collect()
}

fn lower_csc_from_local_adjacency(adjacency: &[Vec<usize>]) -> (Vec<isize>, Vec<isize>) {
    let mut col_ptrs = Vec::with_capacity(adjacency.len() + 1);
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for (column, neighbors) in adjacency.iter().enumerate() {
        row_indices.push(column as isize);
        row_indices.extend(
            neighbors
                .iter()
                .copied()
                .filter(|&neighbor| neighbor > column)
                .map(|neighbor| neighbor as isize),
        );
        col_ptrs.push(row_indices.len() as isize);
    }
    (col_ptrs, row_indices)
}

fn exact_minimum_degree_local_order(adjacency: &[Vec<usize>]) -> Vec<usize> {
    let vertex_count = adjacency.len();
    let mut adjacency = adjacency.to_vec();
    let mut alive = vec![true; vertex_count];
    let mut order = Vec::with_capacity(vertex_count);
    for _ in 0..vertex_count {
        let pivot = (0..vertex_count)
            .filter(|&local| alive[local])
            .min_by_key(|&local| {
                let live_degree = adjacency[local]
                    .iter()
                    .filter(|&&neighbor| alive[neighbor])
                    .count();
                (live_degree, local)
            })
            .expect("at least one live vertex remains");
        let live_neighbors = adjacency[pivot]
            .iter()
            .copied()
            .filter(|&neighbor| alive[neighbor])
            .collect::<Vec<_>>();
        for idx in 0..live_neighbors.len() {
            for jdx in (idx + 1)..live_neighbors.len() {
                let lhs = live_neighbors[idx];
                let rhs = live_neighbors[jdx];
                if !adjacency[lhs].contains(&rhs) {
                    adjacency[lhs].push(rhs);
                    adjacency[lhs].sort_unstable();
                }
                if !adjacency[rhs].contains(&lhs) {
                    adjacency[rhs].push(lhs);
                    adjacency[rhs].sort_unstable();
                }
            }
        }
        alive[pivot] = false;
        order.push(pivot);
    }
    order
}

fn find_separator_split(
    graph: &CsrGraph,
    vertices: &[usize],
    options: &NestedDissectionOptions,
) -> Option<SeparatorSplit> {
    let mut subset_mask = vec![false; graph.vertex_count()];
    for &vertex in vertices {
        subset_mask[vertex] = true;
    }
    let mut candidates = Vec::new();
    let mut seen_pairs = BTreeSet::new();
    for start in candidate_seed_vertices(graph, vertices, &subset_mask) {
        let (_, endpoint_a) = bfs_distances(graph, &subset_mask, start);
        let (dist_a, endpoint_b) = bfs_distances(graph, &subset_mask, endpoint_a);
        let (dist_b, _) = bfs_distances(graph, &subset_mask, endpoint_b);
        let pair = endpoint_a.min(endpoint_b)..=endpoint_a.max(endpoint_b);
        let canonical_pair = (*pair.start(), *pair.end());
        if !seen_pairs.insert(canonical_pair) {
            continue;
        }
        if let Some(split) =
            best_separator_candidate(graph, vertices, &subset_mask, &dist_a, &dist_b, options)
        {
            candidates.push(split);
        }
    }
    candidates
        .into_iter()
        .min_by_key(|split| separator_score(split, vertices.len()))
}

fn candidate_seed_vertices(
    graph: &CsrGraph,
    vertices: &[usize],
    subset_mask: &[bool],
) -> Vec<usize> {
    let mut seeds = vec![vertices[0], vertices[vertices.len() / 2]];
    let min_degree = vertices
        .iter()
        .copied()
        .min_by_key(|&vertex| (subset_degree(graph, subset_mask, vertex), vertex))
        .expect("vertices is nonempty");
    let max_degree = vertices
        .iter()
        .copied()
        .max_by_key(|&vertex| {
            (
                subset_degree(graph, subset_mask, vertex),
                usize::MAX - vertex,
            )
        })
        .expect("vertices is nonempty");
    seeds.push(min_degree);
    seeds.push(max_degree);
    let (_, farthest_from_first) = bfs_distances(graph, subset_mask, vertices[0]);
    seeds.push(farthest_from_first);
    seeds.sort_unstable();
    seeds.dedup();
    seeds
}

fn subset_degree(graph: &CsrGraph, subset_mask: &[bool], vertex: usize) -> usize {
    graph
        .neighbors(vertex)
        .iter()
        .filter(|&&neighbor| subset_mask[neighbor])
        .count()
}

fn best_separator_candidate(
    graph: &CsrGraph,
    vertices: &[usize],
    subset_mask: &[bool],
    dist_a: &[usize],
    dist_b: &[usize],
    options: &NestedDissectionOptions,
) -> Option<SeparatorSplit> {
    let diameter = vertices
        .iter()
        .copied()
        .filter_map(|vertex| {
            let distance = dist_a[vertex];
            (distance != usize::MAX).then_some(distance)
        })
        .max()
        .unwrap_or(0);
    if diameter == 0 {
        return None;
    }

    let mut candidates = Vec::new();
    for bias in [0_usize, 1] {
        let mut labels = vec![VertexPartition::Unassigned; graph.vertex_count()];
        for &vertex in vertices {
            let delta = dist_a[vertex].abs_diff(dist_b[vertex]);
            labels[vertex] = if delta <= bias {
                VertexPartition::Separator
            } else if dist_a[vertex] < dist_b[vertex] {
                VertexPartition::Left
            } else {
                VertexPartition::Right
            };
        }
        if let Some(split) = finalize_separator_candidate(
            graph,
            vertices,
            subset_mask,
            labels,
            dist_a,
            dist_b,
            options,
        ) {
            candidates.push(split);
        }
    }

    let middle = diameter / 2;
    let mut levels = vec![middle];
    if middle > 0 {
        levels.push(middle - 1);
    }
    if middle + 1 < diameter {
        levels.push(middle + 1);
    }
    levels.sort_unstable();
    levels.dedup();

    for level in levels {
        let mut labels = vec![VertexPartition::Unassigned; graph.vertex_count()];
        for &vertex in vertices {
            labels[vertex] = if dist_a[vertex] == level {
                VertexPartition::Separator
            } else if dist_a[vertex] < level {
                VertexPartition::Left
            } else {
                VertexPartition::Right
            };
        }
        if let Some(split) = finalize_separator_candidate(
            graph,
            vertices,
            subset_mask,
            labels,
            dist_a,
            dist_b,
            options,
        ) {
            candidates.push(split);
        }
    }

    candidates
        .into_iter()
        .min_by_key(|split| separator_score(split, vertices.len()))
}

fn finalize_separator_candidate(
    graph: &CsrGraph,
    vertices: &[usize],
    subset_mask: &[bool],
    mut labels: Vec<VertexPartition>,
    dist_a: &[usize],
    dist_b: &[usize],
    options: &NestedDissectionOptions,
) -> Option<SeparatorSplit> {
    repair_cross_edges(graph, subset_mask, &mut labels, dist_a, dist_b, options)?;
    shrink_separator(graph, subset_mask, &mut labels, dist_a, dist_b);
    repair_cross_edges(graph, subset_mask, &mut labels, dist_a, dist_b, options)?;
    labels_to_split(vertices, &labels, options)
}

fn labels_to_split(
    vertices: &[usize],
    labels: &[VertexPartition],
    options: &NestedDissectionOptions,
) -> Option<SeparatorSplit> {
    let mut left = Vec::new();
    let mut right = Vec::new();
    let mut separator = Vec::new();
    for &vertex in vertices {
        match labels[vertex] {
            VertexPartition::Left => left.push(vertex),
            VertexPartition::Right => right.push(vertex),
            VertexPartition::Separator => separator.push(vertex),
            VertexPartition::Unassigned => return None,
        }
    }
    if left.is_empty() || right.is_empty() || separator.is_empty() {
        return None;
    }
    if (separator.len() as f64) > options.max_separator_fraction * (vertices.len() as f64) {
        return None;
    }
    Some(SeparatorSplit {
        left,
        right,
        separator,
    })
}

fn separator_score(split: &SeparatorSplit, total_vertices: usize) -> (u128, usize, usize, usize) {
    let left = split.left.len() as u128;
    let right = split.right.len() as u128;
    let separator = split.separator.len() as u128;
    let work = separator * total_vertices as u128 * 4 + left * left + right * right;
    (
        work,
        split.separator.len(),
        split.left.len().abs_diff(split.right.len()),
        split.left.len().max(split.right.len()),
    )
}

fn repair_cross_edges(
    graph: &CsrGraph,
    subset_mask: &[bool],
    labels: &mut [VertexPartition],
    dist_a: &[usize],
    dist_b: &[usize],
    options: &NestedDissectionOptions,
) -> Option<()> {
    for _ in 0..=options.max_cross_edge_repairs {
        let mut moved_vertex = None;
        'outer: for (vertex, &label) in labels.iter().enumerate() {
            if !subset_mask[vertex] || label != VertexPartition::Left {
                continue;
            }
            for &neighbor in graph.neighbors(vertex) {
                if subset_mask[neighbor] && labels[neighbor] == VertexPartition::Right {
                    let vertex_score =
                        separator_move_priority(graph, subset_mask, labels, dist_a, dist_b, vertex);
                    let neighbor_score = separator_move_priority(
                        graph,
                        subset_mask,
                        labels,
                        dist_a,
                        dist_b,
                        neighbor,
                    );
                    moved_vertex = Some(if vertex_score <= neighbor_score {
                        vertex
                    } else {
                        neighbor
                    });
                    break 'outer;
                }
            }
        }
        match moved_vertex {
            Some(vertex) => labels[vertex] = VertexPartition::Separator,
            None => return Some(()),
        }
    }
    None
}

fn separator_move_priority(
    graph: &CsrGraph,
    subset_mask: &[bool],
    labels: &[VertexPartition],
    dist_a: &[usize],
    dist_b: &[usize],
    vertex: usize,
) -> (usize, usize, usize, usize) {
    let label = labels[vertex];
    let mut same_side_neighbors = 0;
    let mut opposite_side_neighbors = 0;
    for &neighbor in graph.neighbors(vertex) {
        if !subset_mask[neighbor] {
            continue;
        }
        match (label, labels[neighbor]) {
            (VertexPartition::Left, VertexPartition::Left)
            | (VertexPartition::Right, VertexPartition::Right) => same_side_neighbors += 1,
            (VertexPartition::Left, VertexPartition::Right)
            | (VertexPartition::Right, VertexPartition::Left) => opposite_side_neighbors += 1,
            _ => {}
        }
    }
    (
        same_side_neighbors,
        opposite_side_neighbors,
        dist_a[vertex].abs_diff(dist_b[vertex]),
        vertex,
    )
}

fn shrink_separator(
    graph: &CsrGraph,
    subset_mask: &[bool],
    labels: &mut [VertexPartition],
    dist_a: &[usize],
    dist_b: &[usize],
) {
    loop {
        let mut changed = false;
        for vertex in 0..labels.len() {
            if !subset_mask[vertex] || labels[vertex] != VertexPartition::Separator {
                continue;
            }
            let mut touches_left = false;
            let mut touches_right = false;
            for &neighbor in graph.neighbors(vertex) {
                if !subset_mask[neighbor] {
                    continue;
                }
                match labels[neighbor] {
                    VertexPartition::Left => touches_left = true,
                    VertexPartition::Right => touches_right = true,
                    _ => {}
                }
                if touches_left && touches_right {
                    break;
                }
            }

            let new_label = match (touches_left, touches_right) {
                (true, false) => Some(VertexPartition::Left),
                (false, true) => Some(VertexPartition::Right),
                (false, false) => Some(if dist_a[vertex] <= dist_b[vertex] {
                    VertexPartition::Left
                } else {
                    VertexPartition::Right
                }),
                (true, true) => None,
            };

            if let Some(new_label) = new_label {
                labels[vertex] = new_label;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn bfs_distances(graph: &CsrGraph, subset_mask: &[bool], start: usize) -> (Vec<usize>, usize) {
    let mut distances = vec![usize::MAX; graph.vertex_count()];
    let mut queue = VecDeque::from([start]);
    let mut farthest = start;
    distances[start] = 0;
    while let Some(vertex) = queue.pop_front() {
        let next_distance = distances[vertex] + 1;
        if distances[vertex] > distances[farthest]
            || (distances[vertex] == distances[farthest] && vertex < farthest)
        {
            farthest = vertex;
        }
        for &neighbor in graph.neighbors(vertex) {
            if subset_mask[neighbor] && distances[neighbor] == usize::MAX {
                distances[neighbor] = next_distance;
                queue.push_back(neighbor);
            }
        }
    }
    (distances, farthest)
}
