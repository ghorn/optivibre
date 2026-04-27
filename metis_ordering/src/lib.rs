mod mmd;

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

    #[doc(hidden)]
    pub fn from_trusted_sorted_adjacency(offsets: Vec<usize>, neighbors: Vec<usize>) -> Self {
        #[cfg(debug_assertions)]
        {
            Self::new(offsets.clone(), neighbors.clone())
                .expect("trusted CSR graph must already be valid and sorted");
        }
        Self { offsets, neighbors }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetisNodeNdOptions {
    pub compress: bool,
    pub ccorder: bool,
    pub pfactor: usize,
}

impl MetisNodeNdOptions {
    pub const fn spral_default() -> Self {
        Self {
            compress: true,
            ccorder: false,
            pfactor: 0,
        }
    }
}

impl Default for MetisNodeNdOptions {
    fn default() -> Self {
        Self::spral_default()
    }
}

/// Rust production boundary for the METIS `NodeND` ordering used by SPRAL's
/// matching-order path.
///
/// This function is intentionally native-free, but full METIS 5.2.1
/// multilevel-separator parity is still tracked in
/// `docs/spral-matching-scaling-port.md`; callers that need native-oracle
/// parity must keep using the fail-closed parity tests until that audit closes.
pub fn metis_node_nd_order(graph: &CsrGraph) -> Result<OrderingSummary, OrderingError> {
    metis_node_nd_order_with_options(graph, MetisNodeNdOptions::spral_default())
}

pub fn metis_node_nd_order_with_options(
    graph: &CsrGraph,
    options: MetisNodeNdOptions,
) -> Result<OrderingSummary, OrderingError> {
    metis_node_nd_order_raw_with_options(&MetisGraph::from_csr_graph(graph), options)
}

fn metis_node_nd_order_raw(graph: &MetisGraph) -> Result<OrderingSummary, OrderingError> {
    metis_node_nd_order_raw_with_options(graph, MetisNodeNdOptions::spral_default())
}

fn metis_node_nd_order_raw_with_options(
    graph: &MetisGraph,
    options: MetisNodeNdOptions,
) -> Result<OrderingSummary, OrderingError> {
    if graph.vertex_count() == 0 {
        return Ok(OrderingSummary {
            permutation: Permutation::identity(0),
            stats: OrderingStats {
                connected_components: 0,
                separator_calls: 0,
                leaf_calls: 0,
                separator_vertices: 0,
                max_separator_size: 0,
            },
        });
    }
    if graph.vertex_count() == 1 {
        return Ok(OrderingSummary {
            permutation: Permutation::identity(1),
            stats: OrderingStats {
                connected_components: 0,
                separator_calls: 0,
                leaf_calls: 1,
                separator_vertices: 0,
                max_separator_size: 0,
            },
        });
    }

    let prepared = prepare_metis_node_nd_graph(graph, options)?;
    let config = MetisNodeNdConfig {
        compression_active: prepared.compression_active,
        ccorder: options.ccorder,
        nseps: prepared.nseps,
    };
    let base_summary = if options.ccorder {
        metis_recursive_node_nd_order_cc(&prepared.graph, config)?
    } else if prepared.graph.vertex_count() <= 53 {
        metis_one_level_node_nd_order(&prepared.graph, config)?
    } else {
        metis_recursive_node_nd_order(&prepared.graph, config)?
    };

    let permutation =
        expand_metis_node_nd_permutation(base_summary.permutation, prepared.expansion)?;
    Ok(OrderingSummary {
        permutation,
        stats: OrderingStats {
            connected_components: base_summary.stats.connected_components,
            separator_calls: base_summary.stats.separator_calls,
            leaf_calls: base_summary.stats.leaf_calls,
            separator_vertices: base_summary.stats.separator_vertices,
            max_separator_size: base_summary.stats.max_separator_size,
        },
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MetisNodeNdConfig {
    compression_active: bool,
    ccorder: bool,
    nseps: usize,
}

struct PreparedMetisNodeNdGraph {
    graph: MetisGraph,
    compression_active: bool,
    nseps: usize,
    expansion: MetisNodeNdExpansion,
}

enum MetisNodeNdExpansion {
    Identity,
    Compressed {
        original_vertices: Vec<Vec<usize>>,
    },
    Pruned {
        original_vertex_count: usize,
        kept_vertex_count: usize,
        piperm: Vec<usize>,
    },
}

fn prepare_metis_node_nd_graph(
    graph: &MetisGraph,
    options: MetisNodeNdOptions,
) -> Result<PreparedMetisNodeNdGraph, OrderingError> {
    if options.pfactor > 0
        && let Some(pruned) = prune_metis_node_nd_graph(graph, options.pfactor)?
    {
        return Ok(PreparedMetisNodeNdGraph {
            graph: pruned.graph,
            compression_active: false,
            nseps: 1,
            expansion: MetisNodeNdExpansion::Pruned {
                original_vertex_count: graph.vertex_count(),
                kept_vertex_count: pruned.kept_vertex_count,
                piperm: pruned.piperm,
            },
        });
    }

    if options.compress {
        let compressed = compress_metis_node_nd_graph(graph)?;
        if compressed.compression_active {
            let nseps =
                if (graph.vertex_count() as f64 / compressed.graph.vertex_count() as f64) > 1.5 {
                    2
                } else {
                    1
                };
            return Ok(PreparedMetisNodeNdGraph {
                graph: compressed.graph,
                compression_active: true,
                nseps,
                expansion: MetisNodeNdExpansion::Compressed {
                    original_vertices: compressed.original_vertices,
                },
            });
        }
    }

    Ok(PreparedMetisNodeNdGraph {
        graph: graph.clone(),
        compression_active: false,
        nseps: 1,
        expansion: MetisNodeNdExpansion::Identity,
    })
}

fn expand_metis_node_nd_permutation(
    base: Permutation,
    expansion: MetisNodeNdExpansion,
) -> Result<Permutation, OrderingError> {
    match expansion {
        MetisNodeNdExpansion::Identity => Ok(base),
        MetisNodeNdExpansion::Compressed { original_vertices } => {
            let original_vertex_count = original_vertices.iter().map(Vec::len).sum();
            let mut expanded_order = Vec::with_capacity(original_vertex_count);
            for &compressed_vertex in base.perm() {
                expanded_order.extend(original_vertices[compressed_vertex].iter().copied());
            }
            Permutation::new(expanded_order)
        }
        MetisNodeNdExpansion::Pruned {
            original_vertex_count,
            kept_vertex_count,
            piperm,
        } => {
            let mut old_to_new = vec![usize::MAX; original_vertex_count];
            for (kept, &original) in piperm.iter().take(kept_vertex_count).enumerate() {
                old_to_new[original] = base.inverse()[kept];
            }
            for (position, &original) in piperm.iter().enumerate().skip(kept_vertex_count) {
                old_to_new[original] = position;
            }
            let mut new_to_old = vec![usize::MAX; original_vertex_count];
            for (old, &new) in old_to_new.iter().enumerate() {
                if new == usize::MAX || new >= original_vertex_count {
                    return Err(OrderingError::Algorithm(
                        "METIS NodeND pruning expansion produced invalid position".into(),
                    ));
                }
                new_to_old[new] = old;
            }
            Permutation::new(new_to_old)
        }
    }
}

fn metis_one_level_node_nd_order(
    graph: &MetisGraph,
    config: MetisNodeNdConfig,
) -> Result<OrderingSummary, OrderingError> {
    debug_assert!(!config.ccorder);
    let vertex_count = graph.vertex_count();
    if vertex_count <= 1 {
        return Ok(OrderingSummary {
            permutation: Permutation::identity(vertex_count),
            stats: OrderingStats {
                connected_components: 0,
                separator_calls: 0,
                leaf_calls: vertex_count,
                separator_vertices: 0,
                max_separator_size: 0,
            },
        });
    }

    let mut rng = MetisRng::with_metis_seed(-1);
    let separator = metis_node_bisection_multiple_trace_with_rng(graph, config, &mut rng)?;
    let mut old_to_new = vec![usize::MAX; vertex_count];
    let mut last_vertex = vertex_count;
    for &separator_vertex in &separator.boundary {
        last_vertex -= 1;
        old_to_new[separator_vertex] = last_vertex;
    }

    let left = (0..vertex_count)
        .filter(|&vertex| separator.where_part[vertex] == 0)
        .collect::<Vec<_>>();
    let right = (0..vertex_count)
        .filter(|&vertex| separator.where_part[vertex] == 1)
        .collect::<Vec<_>>();

    assign_mmd_leaf_positions(
        graph,
        &left,
        last_vertex.saturating_sub(right.len()),
        &mut old_to_new,
    )?;
    assign_mmd_leaf_positions(graph, &right, last_vertex, &mut old_to_new)?;

    if let Some((vertex, _)) = old_to_new
        .iter()
        .enumerate()
        .find(|&(_, &position)| position == usize::MAX)
    {
        return Err(OrderingError::Algorithm(format!(
            "METIS one-level NodeND left vertex {vertex} unordered"
        )));
    }

    let mut new_to_old = vec![0usize; vertex_count];
    for (old, &new) in old_to_new.iter().enumerate() {
        new_to_old[new] = old;
    }
    Ok(OrderingSummary {
        permutation: Permutation::new(new_to_old)?,
        stats: OrderingStats {
            connected_components: 0,
            separator_calls: 1,
            leaf_calls: usize::from(!left.is_empty()) + usize::from(!right.is_empty()),
            separator_vertices: separator.boundary.len(),
            max_separator_size: separator.boundary.len(),
        },
    })
}

fn metis_recursive_node_nd_order(
    graph: &MetisGraph,
    config: MetisNodeNdConfig,
) -> Result<OrderingSummary, OrderingError> {
    debug_assert!(!config.ccorder);
    let vertex_count = graph.vertex_count();
    let mut old_to_new = vec![usize::MAX; vertex_count];
    let labels = (0..vertex_count).collect::<Vec<_>>();
    let mut rng = MetisRng::with_metis_seed(-1);
    let mut stats = OrderingStats {
        connected_components: 0,
        separator_calls: 0,
        leaf_calls: 0,
        separator_vertices: 0,
        max_separator_size: 0,
    };
    assign_nested_dissection_positions(
        graph,
        &labels,
        vertex_count,
        &mut old_to_new,
        &mut rng,
        config,
        &mut stats,
    )?;

    if let Some((vertex, _)) = old_to_new
        .iter()
        .enumerate()
        .find(|&(_, &position)| position == usize::MAX)
    {
        return Err(OrderingError::Algorithm(format!(
            "METIS recursive NodeND vertex {vertex} unordered"
        )));
    }

    let mut new_to_old = vec![0usize; vertex_count];
    for (old, &new) in old_to_new.iter().enumerate() {
        new_to_old[new] = old;
    }
    Ok(OrderingSummary {
        permutation: Permutation::new(new_to_old)?,
        stats,
    })
}

fn metis_recursive_node_nd_order_cc(
    graph: &MetisGraph,
    config: MetisNodeNdConfig,
) -> Result<OrderingSummary, OrderingError> {
    debug_assert!(config.ccorder);
    let vertex_count = graph.vertex_count();
    let mut old_to_new = vec![usize::MAX; vertex_count];
    let labels = (0..vertex_count).collect::<Vec<_>>();
    let mut rng = MetisRng::with_metis_seed(-1);
    let mut stats = OrderingStats {
        connected_components: 0,
        separator_calls: 0,
        leaf_calls: 0,
        separator_vertices: 0,
        max_separator_size: 0,
    };
    assign_nested_dissection_positions_cc(
        graph,
        &labels,
        vertex_count,
        &mut old_to_new,
        &mut rng,
        config,
        &mut stats,
    )?;

    if let Some((vertex, _)) = old_to_new
        .iter()
        .enumerate()
        .find(|&(_, &position)| position == usize::MAX)
    {
        return Err(OrderingError::Algorithm(format!(
            "METIS recursive CC NodeND vertex {vertex} unordered"
        )));
    }

    let mut new_to_old = vec![0usize; vertex_count];
    for (old, &new) in old_to_new.iter().enumerate() {
        new_to_old[new] = old;
    }
    Ok(OrderingSummary {
        permutation: Permutation::new(new_to_old)?,
        stats,
    })
}

fn assign_nested_dissection_positions(
    graph: &MetisGraph,
    labels: &[usize],
    last_vertex: usize,
    old_to_new: &mut [usize],
    rng: &mut MetisRng,
    config: MetisNodeNdConfig,
    stats: &mut OrderingStats,
) -> Result<(), OrderingError> {
    debug_assert_eq!(graph.vertex_count(), labels.len());
    if graph.vertex_count() == 0 {
        return Ok(());
    }

    let separator = metis_node_bisection_multiple_trace_with_rng(graph, config, rng)?;
    stats.separator_calls += 1;
    stats.separator_vertices += separator.boundary.len();
    stats.max_separator_size = stats.max_separator_size.max(separator.boundary.len());

    let mut next_last_vertex = last_vertex;
    for &separator_vertex in &separator.boundary {
        next_last_vertex -= 1;
        old_to_new[labels[separator_vertex]] = next_last_vertex;
    }

    let (left_graph, left_labels, right_graph, right_labels) =
        split_graph_order(graph, labels, &separator);

    let left_last_vertex = next_last_vertex - right_graph.vertex_count();
    if left_graph.vertex_count() > 120 && left_graph.directed_edge_count() > 0 {
        assign_nested_dissection_positions(
            &left_graph,
            &left_labels,
            left_last_vertex,
            old_to_new,
            rng,
            config,
            stats,
        )?;
    } else {
        assign_mmd_graph_positions(&left_graph, &left_labels, left_last_vertex, old_to_new)?;
        stats.leaf_calls += usize::from(left_graph.vertex_count() > 0);
    }

    if right_graph.vertex_count() > 120 && right_graph.directed_edge_count() > 0 {
        assign_nested_dissection_positions(
            &right_graph,
            &right_labels,
            next_last_vertex,
            old_to_new,
            rng,
            config,
            stats,
        )?;
    } else {
        assign_mmd_graph_positions(&right_graph, &right_labels, next_last_vertex, old_to_new)?;
        stats.leaf_calls += usize::from(right_graph.vertex_count() > 0);
    }
    Ok(())
}

fn assign_nested_dissection_positions_cc(
    graph: &MetisGraph,
    labels: &[usize],
    last_vertex: usize,
    old_to_new: &mut [usize],
    rng: &mut MetisRng,
    config: MetisNodeNdConfig,
    stats: &mut OrderingStats,
) -> Result<(), OrderingError> {
    debug_assert_eq!(graph.vertex_count(), labels.len());
    if graph.vertex_count() == 0 {
        return Ok(());
    }

    let separator = metis_node_bisection_multiple_trace_with_rng(graph, config, rng)?;
    stats.separator_calls += 1;
    stats.separator_vertices += separator.boundary.len();
    stats.max_separator_size = stats.max_separator_size.max(separator.boundary.len());

    let mut next_last_vertex = last_vertex;
    for &separator_vertex in &separator.boundary {
        next_last_vertex -= 1;
        old_to_new[labels[separator_vertex]] = next_last_vertex;
    }

    let mut components = find_separator_induced_components(graph, &separator);
    stats.connected_components += components.len().saturating_sub(1);
    let subgraphs = split_graph_order_cc(graph, labels, &separator, &mut components, rng)?;

    let mut removed_vertices = 0usize;
    for (subgraph, sublabels) in subgraphs {
        let subgraph_vertex_count = subgraph.vertex_count();
        let subgraph_last_vertex = next_last_vertex - removed_vertices;
        if subgraph.vertex_count() > 120 && subgraph.directed_edge_count() > 0 {
            assign_nested_dissection_positions_cc(
                &subgraph,
                &sublabels,
                subgraph_last_vertex,
                old_to_new,
                rng,
                config,
                stats,
            )?;
        } else {
            assign_mmd_graph_positions(&subgraph, &sublabels, subgraph_last_vertex, old_to_new)?;
            stats.leaf_calls += usize::from(subgraph.vertex_count() > 0);
        }
        removed_vertices += subgraph_vertex_count;
    }

    Ok(())
}

fn assign_mmd_graph_positions(
    graph: &MetisGraph,
    labels: &[usize],
    last_vertex: usize,
    old_to_new: &mut [usize],
) -> Result<(), OrderingError> {
    debug_assert_eq!(graph.vertex_count(), labels.len());
    if graph.vertex_count() == 0 {
        return Ok(());
    }
    let summary = mmd::mmd_order(graph)?;
    let first_vertex = last_vertex - graph.vertex_count();
    for (local, &global) in labels.iter().enumerate() {
        old_to_new[global] = first_vertex + summary.permutation.inverse()[local];
    }
    Ok(())
}

fn split_graph_order(
    graph: &MetisGraph,
    labels: &[usize],
    separator: &MetisNodeSeparatorTrace,
) -> (MetisGraph, Vec<usize>, MetisGraph, Vec<usize>) {
    let left_vertices = (0..graph.vertex_count())
        .filter(|&vertex| separator.where_part[vertex] == 0)
        .collect::<Vec<_>>();
    let right_vertices = (0..graph.vertex_count())
        .filter(|&vertex| separator.where_part[vertex] == 1)
        .collect::<Vec<_>>();
    let left_labels = left_vertices
        .iter()
        .map(|&vertex| labels[vertex])
        .collect::<Vec<_>>();
    let right_labels = right_vertices
        .iter()
        .map(|&vertex| labels[vertex])
        .collect::<Vec<_>>();
    let left = split_graph_order_part(graph, &separator.where_part, 0, &left_vertices);
    let right = split_graph_order_part(graph, &separator.where_part, 1, &right_vertices);
    (left, left_labels, right, right_labels)
}

fn split_graph_order_part(
    graph: &MetisGraph,
    where_part: &[usize],
    part: usize,
    vertices: &[usize],
) -> MetisGraph {
    let mut rename = vec![usize::MAX; graph.vertex_count()];
    for (local, &global) in vertices.iter().enumerate() {
        rename[global] = local;
    }

    let mut offsets = Vec::with_capacity(vertices.len() + 1);
    let mut neighbors = Vec::new();
    let mut vertex_weights = Vec::with_capacity(vertices.len());
    offsets.push(0);
    for &global in vertices {
        vertex_weights.push(graph.vertex_weight(global));
        for edge in graph.edge_range(global) {
            let neighbor = graph.neighbors[edge];
            if where_part[neighbor] == part {
                neighbors.push(rename[neighbor]);
            }
        }
        offsets.push(neighbors.len());
    }
    let edge_weights = vec![1; neighbors.len()];
    MetisGraph {
        offsets,
        neighbors,
        vertex_weights,
        edge_weights,
    }
}

fn find_separator_induced_components(
    graph: &MetisGraph,
    separator: &MetisNodeSeparatorTrace,
) -> Vec<Vec<usize>> {
    let vertex_count = graph.vertex_count();
    let mut touched = vec![false; vertex_count];
    for &boundary in &separator.boundary {
        touched[boundary] = true;
    }

    let non_separator_count = separator
        .where_part
        .iter()
        .filter(|&&part| part != 2)
        .count();
    if non_separator_count == 0 {
        return Vec::new();
    }

    let mut queue = Vec::with_capacity(non_separator_count);
    let mut first_start = 0usize;
    while first_start < vertex_count && separator.where_part[first_start] == 2 {
        first_start += 1;
    }
    if first_start == vertex_count {
        return Vec::new();
    }
    touched[first_start] = true;
    queue.push(first_start);

    let mut first = 0usize;
    let mut last = 1usize;
    let mut component_starts = vec![0usize];
    while first != non_separator_count {
        if first == last {
            component_starts.push(first);
            let mut next_start = 0usize;
            while next_start < vertex_count && touched[next_start] {
                next_start += 1;
            }
            if next_start == vertex_count {
                break;
            }
            queue.push(next_start);
            touched[next_start] = true;
            last += 1;
        }

        let vertex = queue[first];
        first += 1;
        for &neighbor in graph.neighbors(vertex) {
            if !touched[neighbor] {
                queue.push(neighbor);
                touched[neighbor] = true;
                last += 1;
            }
        }
    }
    component_starts.push(first);

    component_starts
        .windows(2)
        .map(|window| queue[window[0]..window[1]].to_vec())
        .collect()
}

fn split_graph_order_cc(
    graph: &MetisGraph,
    labels: &[usize],
    separator: &MetisNodeSeparatorTrace,
    components: &mut [Vec<usize>],
    rng: &mut MetisRng,
) -> Result<Vec<(MetisGraph, Vec<usize>)>, OrderingError> {
    let mut boundary_neighbor = vec![false; graph.vertex_count()];
    for &boundary in &separator.boundary {
        for &neighbor in graph.neighbors(boundary) {
            boundary_neighbor[neighbor] = true;
        }
    }

    let mut subgraphs = Vec::with_capacity(components.len());
    for component in components {
        let component_len = component.len();
        rng.rand_array_permute(component, component_len, false);
        let mut rename = vec![usize::MAX; graph.vertex_count()];
        for (local, &global) in component.iter().enumerate() {
            rename[global] = local;
        }

        let mut offsets = Vec::with_capacity(component.len() + 1);
        let mut neighbors = Vec::new();
        let mut vertex_weights = Vec::with_capacity(component.len());
        let mut sublabels = Vec::with_capacity(component.len());
        offsets.push(0);
        for &global in component.iter() {
            vertex_weights.push(graph.vertex_weight(global));
            sublabels.push(labels[global]);
            if boundary_neighbor[global] {
                for &neighbor in graph.neighbors(global) {
                    if separator.where_part[neighbor] != 2 {
                        let local_neighbor = rename[neighbor];
                        if local_neighbor == usize::MAX {
                            return Err(OrderingError::Algorithm(
                                "METIS CC split found edge across separator-induced components"
                                    .into(),
                            ));
                        }
                        neighbors.push(local_neighbor);
                    }
                }
            } else {
                for &neighbor in graph.neighbors(global) {
                    let local_neighbor = rename[neighbor];
                    if local_neighbor == usize::MAX {
                        return Err(OrderingError::Algorithm(
                            "METIS CC split interior edge left its component".into(),
                        ));
                    }
                    neighbors.push(local_neighbor);
                }
            }
            offsets.push(neighbors.len());
        }
        let edge_weights = vec![1; neighbors.len()];
        subgraphs.push((
            MetisGraph {
                offsets,
                neighbors,
                vertex_weights,
                edge_weights,
            },
            sublabels,
        ));
    }

    Ok(subgraphs)
}

fn assign_mmd_leaf_positions(
    graph: &MetisGraph,
    labels: &[usize],
    last_vertex: usize,
    old_to_new: &mut [usize],
) -> Result<(), OrderingError> {
    if labels.is_empty() {
        return Ok(());
    }
    let subgraph = induced_metis_subgraph(graph, labels);
    let summary = mmd::mmd_order(&subgraph)?;
    let first_vertex = last_vertex - labels.len();
    for (local, &global) in labels.iter().enumerate() {
        old_to_new[global] = first_vertex + summary.permutation.inverse()[local];
    }
    Ok(())
}

fn induced_metis_subgraph(graph: &MetisGraph, labels: &[usize]) -> MetisGraph {
    let mut local_index = vec![usize::MAX; graph.vertex_count()];
    for (local, &global) in labels.iter().enumerate() {
        local_index[global] = local;
    }

    let mut offsets = Vec::with_capacity(labels.len() + 1);
    let mut neighbors = Vec::new();
    offsets.push(0);
    for &global in labels {
        for &neighbor in graph.neighbors(global) {
            let local_neighbor = local_index[neighbor];
            if local_neighbor != usize::MAX {
                neighbors.push(local_neighbor);
            }
        }
        offsets.push(neighbors.len());
    }
    MetisGraph {
        offsets,
        neighbors: neighbors.clone(),
        vertex_weights: vec![1; labels.len()],
        edge_weights: vec![1; neighbors.len()],
    }
}

#[doc(hidden)]
pub fn metis_debug_irand_sequence(seed: i32, count: usize) -> Vec<usize> {
    let mut rng = MetisRng::with_metis_seed(seed);
    (0..count).map(|_| rng.rand()).collect()
}

#[doc(hidden)]
pub fn metis_debug_irand_array_permute(
    seed: i32,
    n: usize,
    nshuffles: usize,
    flag: bool,
) -> Vec<usize> {
    let mut rng = MetisRng::with_metis_seed(seed);
    let mut values = if flag {
        vec![0; n]
    } else {
        (0..n).rev().collect()
    };
    rng.rand_array_permute(&mut values, nshuffles, flag);
    values
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetisEdgePartitionTrace {
    pub mincut: isize,
    pub part_weights: [isize; 2],
    pub where_part: Vec<usize>,
    pub boundary: Vec<usize>,
    pub internal_degree: Vec<isize>,
    pub external_degree: Vec<isize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetisNodeSeparatorTrace {
    pub mincut: isize,
    pub part_weights: [isize; 3],
    pub where_part: Vec<usize>,
    pub boundary: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetisCoarsenTrace {
    pub vertex_count: usize,
    pub directed_edge_count: usize,
    pub offsets: Vec<usize>,
    pub neighbors: Vec<usize>,
    pub vertex_weights: Vec<isize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetisNodeNdTopSeparatorTrace {
    pub compression_active: bool,
    pub nseps: usize,
    pub compressed_vertex_count: usize,
    pub compressed_directed_edge_count: usize,
    pub compressed_original_vertices: Vec<Vec<usize>>,
    pub separator: MetisNodeSeparatorTrace,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetisCcComponentsTrace {
    pub separator: MetisNodeSeparatorTrace,
    pub cptr: Vec<usize>,
    pub cind: Vec<usize>,
    pub subgraph_labels: Vec<Vec<usize>>,
    pub subgraph_offsets: Vec<Vec<usize>>,
    pub subgraph_neighbors: Vec<Vec<usize>>,
}

#[doc(hidden)]
pub fn metis_debug_l1_edge_bisection_from_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
) -> Result<MetisEdgePartitionTrace, OrderingError> {
    let graph = spral_half_to_full_drop_diag(dimension, col_ptrs, row_indices)?;
    metis_l1_edge_bisection_trace(&graph)
}

#[doc(hidden)]
pub fn metis_debug_l1_coarsen_from_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
) -> Result<MetisCoarsenTrace, OrderingError> {
    let graph = spral_half_to_full_drop_diag(dimension, col_ptrs, row_indices)?;
    let mut rng = MetisRng::with_metis_seed(-1);
    let coarse = metis_l1_coarsen_graph(
        &graph,
        metis_ometis_l1_coarsen_to(graph.vertex_count()),
        &mut rng,
    )?;
    Ok(MetisCoarsenTrace {
        vertex_count: coarse.vertex_count(),
        directed_edge_count: coarse.directed_edge_count(),
        offsets: coarse.offsets,
        neighbors: coarse.neighbors,
        vertex_weights: coarse.vertex_weights,
    })
}

#[doc(hidden)]
pub fn metis_debug_l1_construct_separator_from_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
) -> Result<MetisNodeSeparatorTrace, OrderingError> {
    let graph = spral_half_to_full_drop_diag(dimension, col_ptrs, row_indices)?;
    metis_l1_construct_separator_trace(&graph)
}

#[doc(hidden)]
pub fn metis_debug_l1_projected_separator_from_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
) -> Result<MetisNodeSeparatorTrace, OrderingError> {
    let graph = spral_half_to_full_drop_diag(dimension, col_ptrs, row_indices)?;
    metis_l1_projected_separator_trace(&graph)
}

#[doc(hidden)]
pub fn metis_debug_node_nd_top_separator_from_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
) -> Result<MetisNodeNdTopSeparatorTrace, OrderingError> {
    metis_debug_node_nd_top_separator_from_lower_csc_with_options(
        dimension,
        col_ptrs,
        row_indices,
        MetisNodeNdOptions::spral_default(),
    )
}

#[doc(hidden)]
pub fn metis_debug_node_nd_top_separator_from_lower_csc_with_options(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    options: MetisNodeNdOptions,
) -> Result<MetisNodeNdTopSeparatorTrace, OrderingError> {
    let graph = spral_half_to_full_drop_diag(dimension, col_ptrs, row_indices)?;
    let prepared = prepare_metis_node_nd_graph(&graph, options)?;
    let config = MetisNodeNdConfig {
        compression_active: prepared.compression_active,
        ccorder: options.ccorder,
        nseps: prepared.nseps,
    };
    let mut rng = MetisRng::with_metis_seed(-1);
    let separator =
        metis_node_bisection_multiple_trace_with_rng(&prepared.graph, config, &mut rng)?;
    let compressed_original_vertices = match prepared.expansion {
        MetisNodeNdExpansion::Compressed { original_vertices } => original_vertices,
        MetisNodeNdExpansion::Identity => (0..prepared.graph.vertex_count())
            .map(|vertex| vec![vertex])
            .collect(),
        MetisNodeNdExpansion::Pruned { .. } => (0..prepared.graph.vertex_count())
            .map(|vertex| vec![vertex])
            .collect(),
    };
    Ok(MetisNodeNdTopSeparatorTrace {
        compression_active: prepared.compression_active,
        nseps: prepared.nseps,
        compressed_vertex_count: prepared.graph.vertex_count(),
        compressed_directed_edge_count: prepared.graph.directed_edge_count(),
        compressed_original_vertices,
        separator,
    })
}

#[doc(hidden)]
pub fn metis_debug_cc_components_from_lower_csc_with_options(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    options: MetisNodeNdOptions,
) -> Result<MetisCcComponentsTrace, OrderingError> {
    let graph = spral_half_to_full_drop_diag(dimension, col_ptrs, row_indices)?;
    let prepared = prepare_metis_node_nd_graph(&graph, options)?;
    let config = MetisNodeNdConfig {
        compression_active: prepared.compression_active,
        ccorder: options.ccorder,
        nseps: prepared.nseps,
    };
    let mut rng = MetisRng::with_metis_seed(-1);
    let separator =
        metis_node_bisection_multiple_trace_with_rng(&prepared.graph, config, &mut rng)?;
    let labels = (0..prepared.graph.vertex_count()).collect::<Vec<_>>();
    let mut components = find_separator_induced_components(&prepared.graph, &separator);
    let subgraphs = split_graph_order_cc(
        &prepared.graph,
        &labels,
        &separator,
        &mut components,
        &mut rng,
    )?;
    let mut cptr = Vec::with_capacity(components.len() + 1);
    let mut cind = Vec::new();
    cptr.push(0);
    for component in &components {
        cind.extend(component.iter().copied());
        cptr.push(cind.len());
    }
    Ok(MetisCcComponentsTrace {
        separator,
        cptr,
        cind,
        subgraph_labels: subgraphs.iter().map(|(_, labels)| labels.clone()).collect(),
        subgraph_offsets: subgraphs
            .iter()
            .map(|(graph, _)| graph.offsets.clone())
            .collect(),
        subgraph_neighbors: subgraphs
            .iter()
            .map(|(graph, _)| graph.neighbors.clone())
            .collect(),
    })
}

pub fn metis_node_nd_order_from_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
) -> Result<OrderingSummary, OrderingError> {
    let graph = spral_half_to_full_drop_diag(dimension, col_ptrs, row_indices)?;
    metis_node_nd_order_raw(&graph)
}

pub fn metis_node_nd_order_from_lower_csc_with_options(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    options: MetisNodeNdOptions,
) -> Result<OrderingSummary, OrderingError> {
    let graph = spral_half_to_full_drop_diag(dimension, col_ptrs, row_indices)?;
    metis_node_nd_order_raw_with_options(&graph, options)
}

#[doc(hidden)]
pub fn metis_mmd_order_from_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
) -> Result<OrderingSummary, OrderingError> {
    let graph = spral_half_to_full_drop_diag(dimension, col_ptrs, row_indices)?;
    mmd::mmd_order(&graph)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MetisGraph {
    offsets: Vec<usize>,
    neighbors: Vec<usize>,
    vertex_weights: Vec<isize>,
    edge_weights: Vec<isize>,
}

struct MetisCoarsenedLevel {
    graph: MetisGraph,
    cmap: Vec<usize>,
}

struct MetisCoarsening {
    graph: MetisGraph,
    maps: Vec<Vec<usize>>,
    graphs: Vec<MetisGraph>,
}

impl MetisGraph {
    fn from_csr_graph(graph: &CsrGraph) -> Self {
        Self {
            offsets: graph.offsets.clone(),
            neighbors: graph.neighbors.clone(),
            vertex_weights: vec![1; graph.vertex_count()],
            edge_weights: vec![1; graph.neighbors.len()],
        }
    }

    fn vertex_count(&self) -> usize {
        self.offsets.len() - 1
    }

    fn directed_edge_count(&self) -> usize {
        self.neighbors.len()
    }

    fn total_vertex_weight(&self) -> isize {
        self.vertex_weights.iter().sum()
    }

    fn vertex_weight(&self, vertex: usize) -> isize {
        self.vertex_weights[vertex]
    }

    fn degree(&self, vertex: usize) -> usize {
        self.offsets[vertex + 1] - self.offsets[vertex]
    }

    fn neighbors(&self, vertex: usize) -> &[usize] {
        &self.neighbors[self.offsets[vertex]..self.offsets[vertex + 1]]
    }

    fn edge_weight(&self, edge: usize) -> isize {
        self.edge_weights[edge]
    }

    fn edge_range(&self, vertex: usize) -> std::ops::Range<usize> {
        self.offsets[vertex]..self.offsets[vertex + 1]
    }
}

fn spral_half_to_full_drop_diag(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
) -> Result<MetisGraph, OrderingError> {
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

    let mut counts = vec![0usize; dimension];
    for col in 0..dimension {
        for &row in &row_indices[col_ptrs[col]..col_ptrs[col + 1]] {
            if row >= dimension {
                return Err(OrderingError::InvalidGraph(format!(
                    "row index {row} out of bounds for {dimension}x{dimension} matrix"
                )));
            }
            if row != col {
                counts[row] += 1;
                counts[col] += 1;
            }
        }
    }

    let mut ends = vec![0usize; dimension];
    let mut total = 0usize;
    for (vertex, count) in counts.into_iter().enumerate() {
        total += count;
        ends[vertex] = total;
    }
    let mut next = ends;
    let mut neighbors = vec![0usize; total];
    let edge_weights = vec![1isize; total];
    for col in 0..dimension {
        for &row in &row_indices[col_ptrs[col]..col_ptrs[col + 1]] {
            if row == col {
                continue;
            }
            next[row] -= 1;
            neighbors[next[row]] = col;
            next[col] -= 1;
            neighbors[next[col]] = row;
        }
    }
    let mut offsets = Vec::with_capacity(dimension + 1);
    offsets.extend(next);
    offsets.push(total);
    Ok(MetisGraph {
        offsets,
        neighbors,
        vertex_weights: vec![1; dimension],
        edge_weights,
    })
}

struct CompressedNodeNdGraph {
    graph: MetisGraph,
    original_vertices: Vec<Vec<usize>>,
    compression_active: bool,
}

struct PrunedNodeNdGraph {
    graph: MetisGraph,
    kept_vertex_count: usize,
    piperm: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetisPruneTrace {
    pub pruning_active: bool,
    pub kept_vertex_count: usize,
    pub piperm: Vec<usize>,
    pub offsets: Vec<usize>,
    pub neighbors: Vec<usize>,
    pub vertex_weights: Vec<isize>,
}

fn compress_metis_node_nd_graph(
    graph: &MetisGraph,
) -> Result<CompressedNodeNdGraph, OrderingError> {
    let vertex_count = graph.vertex_count();
    let mut mark = vec![usize::MAX; vertex_count];
    let mut map = vec![usize::MAX; vertex_count];
    let mut keys = (0..vertex_count)
        .map(|vertex| KeyVal {
            key: graph
                .neighbors(vertex)
                .iter()
                .copied()
                .fold(vertex, usize::wrapping_add),
            val: vertex,
        })
        .collect::<Vec<_>>();
    gk_ikvsorti_by_key(&mut keys);

    let mut cptr = Vec::with_capacity(vertex_count + 1);
    let mut cind = Vec::with_capacity(vertex_count);
    cptr.push(0);
    let mut compressed_vertex_count = 0usize;
    for index in 0..vertex_count {
        let vertex = keys[index].val;
        if map[vertex] != usize::MAX {
            continue;
        }
        mark[vertex] = index;
        for &neighbor in graph.neighbors(vertex) {
            mark[neighbor] = index;
        }
        map[vertex] = compressed_vertex_count;
        cind.push(vertex);

        for candidate_key in keys.iter().skip(index + 1) {
            let candidate = candidate_key.val;
            if keys[index].key != candidate_key.key
                || graph.degree(vertex) != graph.degree(candidate)
            {
                break;
            }
            if map[candidate] != usize::MAX {
                continue;
            }
            let identical = graph
                .neighbors(candidate)
                .iter()
                .all(|&neighbor| mark[neighbor] == index);
            if identical {
                map[candidate] = compressed_vertex_count;
                cind.push(candidate);
            }
        }
        compressed_vertex_count += 1;
        cptr.push(cind.len());
    }

    let mut original_vertices = cptr
        .windows(2)
        .map(|window| cind[window[0]..window[1]].to_vec())
        .collect::<Vec<_>>();
    if (compressed_vertex_count as f64) >= 0.85 * (vertex_count as f64) {
        original_vertices = (0..vertex_count).map(|vertex| vec![vertex]).collect();
        return Ok(CompressedNodeNdGraph {
            graph: graph.clone(),
            original_vertices,
            compression_active: false,
        });
    }

    let mut offsets = Vec::with_capacity(compressed_vertex_count + 1);
    let mut neighbors = Vec::new();
    mark.fill(usize::MAX);
    offsets.push(0);
    for compressed in 0..compressed_vertex_count {
        mark[compressed] = compressed;
        for &vertex in &cind[cptr[compressed]..cptr[compressed + 1]] {
            for &neighbor in graph.neighbors(vertex) {
                let compressed_neighbor = map[neighbor];
                if mark[compressed_neighbor] != compressed {
                    mark[compressed_neighbor] = compressed;
                    neighbors.push(compressed_neighbor);
                }
            }
        }
        offsets.push(neighbors.len());
    }
    Ok(CompressedNodeNdGraph {
        graph: MetisGraph {
            offsets,
            neighbors: neighbors.clone(),
            vertex_weights: cptr
                .windows(2)
                .map(|window| (window[1] - window[0]) as isize)
                .collect(),
            edge_weights: vec![1; neighbors.len()],
        },
        original_vertices,
        compression_active: true,
    })
}

fn prune_metis_node_nd_graph(
    graph: &MetisGraph,
    pfactor: usize,
) -> Result<Option<PrunedNodeNdGraph>, OrderingError> {
    let vertex_count = graph.vertex_count();
    if vertex_count == 0 {
        return Ok(None);
    }

    let threshold = 0.1 * pfactor as f64 * graph.directed_edge_count() as f64 / vertex_count as f64;
    let mut perm = vec![usize::MAX; vertex_count];
    let mut piperm = vec![usize::MAX; vertex_count];
    let mut kept_vertex_count = 0usize;
    let mut pruned_count = 0usize;
    let mut kept_directed_edges = 0usize;

    for (vertex, mapped_vertex) in perm.iter_mut().enumerate().take(vertex_count) {
        if (graph.degree(vertex) as f64) < threshold {
            *mapped_vertex = kept_vertex_count;
            piperm[kept_vertex_count] = vertex;
            kept_vertex_count += 1;
            kept_directed_edges += graph.degree(vertex);
        } else {
            pruned_count += 1;
            let position = vertex_count - pruned_count;
            *mapped_vertex = position;
            piperm[position] = vertex;
        }
    }

    if pruned_count == 0 || pruned_count == vertex_count {
        return Ok(None);
    }

    let mut offsets = Vec::with_capacity(kept_vertex_count + 1);
    let mut neighbors = Vec::with_capacity(kept_directed_edges);
    let mut vertex_weights = Vec::with_capacity(kept_vertex_count);
    offsets.push(0);
    for vertex in 0..vertex_count {
        if (graph.degree(vertex) as f64) < threshold {
            vertex_weights.push(graph.vertex_weight(vertex));
            for &neighbor in graph.neighbors(vertex) {
                let mapped = perm[neighbor];
                if mapped < kept_vertex_count {
                    neighbors.push(mapped);
                }
            }
            offsets.push(neighbors.len());
        }
    }
    let edge_weights = vec![1; neighbors.len()];
    Ok(Some(PrunedNodeNdGraph {
        graph: MetisGraph {
            offsets,
            neighbors,
            vertex_weights,
            edge_weights,
        },
        kept_vertex_count,
        piperm,
    }))
}

#[doc(hidden)]
pub fn metis_debug_prune_from_lower_csc(
    dimension: usize,
    col_ptrs: &[usize],
    row_indices: &[usize],
    pfactor: usize,
) -> Result<MetisPruneTrace, OrderingError> {
    let graph = spral_half_to_full_drop_diag(dimension, col_ptrs, row_indices)?;
    let pruned = prune_metis_node_nd_graph(&graph, pfactor)?;
    if let Some(pruned) = pruned {
        Ok(MetisPruneTrace {
            pruning_active: true,
            kept_vertex_count: pruned.kept_vertex_count,
            piperm: pruned.piperm,
            offsets: pruned.graph.offsets,
            neighbors: pruned.graph.neighbors,
            vertex_weights: pruned.graph.vertex_weights,
        })
    } else {
        Ok(MetisPruneTrace {
            pruning_active: false,
            kept_vertex_count: dimension,
            piperm: (0..dimension).collect(),
            offsets: graph.offsets,
            neighbors: graph.neighbors,
            vertex_weights: graph.vertex_weights,
        })
    }
}

#[derive(Clone, Copy)]
struct KeyVal {
    key: usize,
    val: usize,
}

fn gk_ikvsorti_by_key(base: &mut [KeyVal]) {
    const MAX_THRESH: usize = 8;

    if base.is_empty() {
        return;
    }
    if base.len() > MAX_THRESH {
        let mut lo = 0usize;
        let mut hi = base.len() - 1;
        let mut stack = Vec::<(usize, usize)>::with_capacity(8 * std::mem::size_of::<usize>());
        loop {
            let mut mid = lo + ((hi - lo) >> 1);
            if ikv_key_lt(base, mid, lo) {
                base.swap(mid, lo);
            }
            if ikv_key_lt(base, hi, mid) {
                base.swap(mid, hi);
                if ikv_key_lt(base, mid, lo) {
                    base.swap(mid, lo);
                }
            }

            let mut left = lo + 1;
            let mut right = hi - 1;
            loop {
                while ikv_key_lt(base, left, mid) {
                    left += 1;
                }
                while ikv_key_lt(base, mid, right) {
                    right -= 1;
                }
                if left < right {
                    base.swap(left, right);
                    if mid == left {
                        mid = right;
                    } else if mid == right {
                        mid = left;
                    }
                    left += 1;
                    right -= 1;
                } else if left == right {
                    left += 1;
                    right -= 1;
                    break;
                }
                if left > right {
                    break;
                }
            }

            if right.saturating_sub(lo) <= MAX_THRESH {
                if hi.saturating_sub(left) <= MAX_THRESH {
                    let Some((stack_lo, stack_hi)) = stack.pop() else {
                        break;
                    };
                    lo = stack_lo;
                    hi = stack_hi;
                } else {
                    lo = left;
                }
            } else if hi.saturating_sub(left) <= MAX_THRESH {
                hi = right;
            } else if right - lo > hi - left {
                stack.push((lo, right));
                lo = left;
            } else {
                stack.push((left, hi));
                hi = right;
            }
        }
    }

    let end = base.len() - 1;
    let mut smallest = 0usize;
    let threshold = MAX_THRESH.min(end);
    for run in 1..=threshold {
        if ikv_key_lt(base, run, smallest) {
            smallest = run;
        }
    }
    if smallest != 0 {
        base.swap(smallest, 0);
    }

    for run in 2..=end {
        let mut target = run - 1;
        while ikv_key_lt(base, run, target) {
            target -= 1;
        }
        target += 1;
        if target != run {
            let hold = base[run];
            for slot in (target + 1..=run).rev() {
                base[slot] = base[slot - 1];
            }
            base[target] = hold;
        }
    }
}

fn ikv_key_lt(base: &[KeyVal], lhs: usize, rhs: usize) -> bool {
    base[lhs].key < base[rhs].key
}

#[derive(Clone, Debug)]
struct MetisRng {
    state: u32,
}

impl MetisRng {
    fn with_metis_seed(seed: i32) -> Self {
        let seed = if seed == -1 { 4321 } else { seed };
        Self { state: seed as u32 }
    }

    fn rand(&mut self) -> usize {
        self.state = macos_libc_rand_next(self.state);
        self.state as usize
    }

    fn rand_in_range(&mut self, max: usize) -> usize {
        self.rand() % max
    }

    fn rand_array_permute(&mut self, values: &mut [usize], nshuffles: usize, flag: bool) {
        let n = values.len();
        if flag {
            for (index, value) in values.iter_mut().enumerate() {
                *value = index;
            }
        }

        if n == 0 {
            return;
        }
        if n < 10 {
            for _ in 0..n {
                let v = self.rand_in_range(n);
                let u = self.rand_in_range(n);
                values.swap(v, u);
            }
        } else {
            for _ in 0..nshuffles {
                let v = self.rand_in_range(n - 3);
                let u = self.rand_in_range(n - 3);
                values.swap(v, u + 2);
                values.swap(v + 1, u + 3);
                values.swap(v + 2, u);
                values.swap(v + 3, u + 1);
            }
        }
    }
}

fn macos_libc_rand_next(state: u32) -> u32 {
    ((u64::from(state) * 16_807) % 2_147_483_647) as u32
}

#[derive(Clone, Debug)]
struct EdgePartitionState {
    where_part: Vec<usize>,
    part_weights: [isize; 2],
    boundary_ptr: Vec<usize>,
    boundary: Vec<usize>,
    internal_degree: Vec<isize>,
    external_degree: Vec<isize>,
    mincut: isize,
}

impl EdgePartitionState {
    fn new(vertex_count: usize, total_weight: isize) -> Self {
        Self {
            where_part: vec![1; vertex_count],
            part_weights: [0, total_weight],
            boundary_ptr: vec![usize::MAX; vertex_count],
            boundary: Vec::with_capacity(vertex_count),
            internal_degree: vec![0; vertex_count],
            external_degree: vec![0; vertex_count],
            mincut: 0,
        }
    }

    fn insert_boundary(&mut self, vertex: usize) {
        debug_assert_eq!(self.boundary_ptr[vertex], usize::MAX);
        self.boundary_ptr[vertex] = self.boundary.len();
        self.boundary.push(vertex);
    }

    fn delete_boundary(&mut self, vertex: usize) {
        let position = self.boundary_ptr[vertex];
        debug_assert_ne!(position, usize::MAX);
        let moved = self
            .boundary
            .pop()
            .expect("boundary delete requires nonempty boundary");
        if position < self.boundary.len() {
            self.boundary[position] = moved;
            self.boundary_ptr[moved] = position;
        }
        self.boundary_ptr[vertex] = usize::MAX;
    }

    fn trace(&self) -> MetisEdgePartitionTrace {
        MetisEdgePartitionTrace {
            mincut: self.mincut,
            part_weights: self.part_weights,
            where_part: self.where_part.clone(),
            boundary: self.boundary.clone(),
            internal_degree: self.internal_degree.clone(),
            external_degree: self.external_degree.clone(),
        }
    }
}

#[derive(Clone, Debug)]
struct NodePartitionState {
    where_part: Vec<usize>,
    part_weights: [isize; 3],
    boundary_ptr: Vec<usize>,
    boundary: Vec<usize>,
    edegrees: Vec<[isize; 2]>,
    mincut: isize,
}

impl NodePartitionState {
    fn from_edge_boundary(graph: &MetisGraph, edge: &EdgePartitionState) -> Self {
        let mut where_part = edge.where_part.clone();
        for &vertex in &edge.boundary {
            if graph.degree(vertex) > 0 {
                where_part[vertex] = 2;
            }
        }
        let vertex_count = graph.vertex_count();
        Self {
            where_part,
            part_weights: [0; 3],
            boundary_ptr: vec![usize::MAX; vertex_count],
            boundary: Vec::with_capacity(vertex_count),
            edegrees: vec![[0; 2]; vertex_count],
            mincut: 0,
        }
    }

    fn from_where(where_part: Vec<usize>) -> Self {
        let vertex_count = where_part.len();
        Self {
            where_part,
            part_weights: [0; 3],
            boundary_ptr: vec![usize::MAX; vertex_count],
            boundary: Vec::with_capacity(vertex_count),
            edegrees: vec![[0; 2]; vertex_count],
            mincut: 0,
        }
    }

    fn insert_boundary(&mut self, vertex: usize) {
        debug_assert_eq!(self.boundary_ptr[vertex], usize::MAX);
        self.boundary_ptr[vertex] = self.boundary.len();
        self.boundary.push(vertex);
    }

    fn delete_boundary(&mut self, vertex: usize) {
        let position = self.boundary_ptr[vertex];
        debug_assert_ne!(position, usize::MAX);
        let moved = self
            .boundary
            .pop()
            .expect("boundary delete requires nonempty boundary");
        if position < self.boundary.len() {
            self.boundary[position] = moved;
            self.boundary_ptr[moved] = position;
        }
        self.boundary_ptr[vertex] = usize::MAX;
    }

    fn trace(&self) -> MetisNodeSeparatorTrace {
        MetisNodeSeparatorTrace {
            mincut: self.mincut,
            part_weights: self.part_weights,
            where_part: self.where_part.clone(),
            boundary: self.boundary.clone(),
        }
    }
}

fn metis_l1_edge_bisection_trace(
    graph: &MetisGraph,
) -> Result<MetisEdgePartitionTrace, OrderingError> {
    let (_, state, _) = metis_l1_edge_bisection_state(graph)?;
    Ok(state.trace())
}

fn metis_l1_construct_separator_trace(
    graph: &MetisGraph,
) -> Result<MetisNodeSeparatorTrace, OrderingError> {
    let (coarse_graph, edge_state, mut rng) = metis_l1_edge_bisection_state(graph)?;
    let mut state = NodePartitionState::from_edge_boundary(&coarse_graph, &edge_state);
    compute_2way_node_partition_params(&coarse_graph, &mut state);
    fm_2way_node_refine_2sided(&coarse_graph, &mut state, 1, false, &mut rng);
    fm_2way_node_refine_1sided(&coarse_graph, &mut state, 4, false, &mut rng);
    Ok(state.trace())
}

fn metis_l1_projected_separator_trace(
    graph: &MetisGraph,
) -> Result<MetisNodeSeparatorTrace, OrderingError> {
    let mut rng = MetisRng::with_metis_seed(-1);
    metis_l1_projected_separator_trace_with_rng(graph, &mut rng)
}

fn metis_l1_projected_separator_trace_with_rng(
    graph: &MetisGraph,
    rng: &mut MetisRng,
) -> Result<MetisNodeSeparatorTrace, OrderingError> {
    metis_l1_projected_separator_trace_with_config(
        graph,
        MetisNodeNdConfig {
            compression_active: false,
            ccorder: false,
            nseps: 1,
        },
        7,
        rng,
    )
}

fn metis_l1_projected_separator_trace_with_config(
    graph: &MetisGraph,
    config: MetisNodeNdConfig,
    initial_niparts: usize,
    rng: &mut MetisRng,
) -> Result<MetisNodeSeparatorTrace, OrderingError> {
    let vertex_count = graph.vertex_count();
    if vertex_count == 0 {
        return Ok(MetisNodeSeparatorTrace {
            mincut: 0,
            part_weights: [0; 3],
            where_part: Vec::new(),
            boundary: Vec::new(),
        });
    }

    let coarsen_to = metis_ometis_l1_coarsen_to(vertex_count);
    let coarsening = metis_l1_coarsen_graph_with_maps(graph, coarsen_to, rng)?;

    let niparts = if coarsening.graph.vertex_count() <= coarsen_to {
        initial_niparts / 2
    } else {
        initial_niparts
    }
    .max(1);
    let edge_state = if coarsening.graph.directed_edge_count() == 0 {
        random_bisection_edge_state(&coarsening.graph, niparts, rng)?
    } else {
        grow_bisection_edge_state(&coarsening.graph, niparts, rng)?
    };
    let mut coarse_state = NodePartitionState::from_edge_boundary(&coarsening.graph, &edge_state);
    compute_2way_node_partition_params(&coarsening.graph, &mut coarse_state);
    fm_2way_node_refine_2sided(
        &coarsening.graph,
        &mut coarse_state,
        1,
        config.compression_active,
        rng,
    );
    fm_2way_node_refine_1sided(
        &coarsening.graph,
        &mut coarse_state,
        4,
        config.compression_active,
        rng,
    );

    project_node_separator_state_through_coarsening(graph, &coarsening, coarse_state, config, rng)
}

fn project_node_separator_state_through_coarsening(
    graph: &MetisGraph,
    coarsening: &MetisCoarsening,
    coarse_state: NodePartitionState,
    config: MetisNodeNdConfig,
    rng: &mut MetisRng,
) -> Result<MetisNodeSeparatorTrace, OrderingError> {
    let mut state = coarse_state;
    for level_index in (0..coarsening.maps.len()).rev() {
        let finer_graph = if level_index == 0 {
            graph
        } else {
            &coarsening.graphs[level_index - 1]
        };
        let where_part = coarsening.maps[level_index]
            .iter()
            .map(|&coarse_vertex| state.where_part[coarse_vertex])
            .collect::<Vec<_>>();
        state = NodePartitionState::from_where(where_part);
        compute_2way_node_partition_params(finer_graph, &mut state);
        fm_2way_node_balance(finer_graph, &mut state, rng);
        fm_2way_node_refine_1sided(finer_graph, &mut state, 10, config.compression_active, rng);
    }
    Ok(state.trace())
}

fn metis_node_bisection_multiple_trace_with_rng(
    graph: &MetisGraph,
    config: MetisNodeNdConfig,
    rng: &mut MetisRng,
) -> Result<MetisNodeSeparatorTrace, OrderingError> {
    let multiple_threshold = if config.compression_active {
        1000
    } else {
        2000
    };
    if config.nseps <= 1 || graph.vertex_count() < multiple_threshold {
        return metis_l2_node_bisection_trace_with_rng(graph, config, rng);
    }

    let mut mincut = isize::MAX;
    let mut best: Option<MetisNodeSeparatorTrace> = None;
    let mut current: Option<MetisNodeSeparatorTrace> = None;
    for trial in 0..config.nseps {
        let separator = metis_l2_node_bisection_trace_with_rng(graph, config, rng)?;
        if trial == 0 || separator.mincut < mincut {
            mincut = separator.mincut;
            if trial + 1 < config.nseps {
                best = Some(separator.clone());
            }
        }
        let reached_zero = mincut == 0;
        current = Some(separator);
        if reached_zero || trial + 1 == config.nseps {
            break;
        }
    }
    let current = current.ok_or_else(|| {
        OrderingError::Algorithm("METIS NodeND separator retry had no trial".into())
    })?;
    if current.mincut == mincut {
        Ok(current)
    } else {
        best.ok_or_else(|| {
            OrderingError::Algorithm("METIS NodeND separator retry lost best separator".into())
        })
    }
}

fn metis_l2_node_bisection_trace_with_rng(
    graph: &MetisGraph,
    config: MetisNodeNdConfig,
    rng: &mut MetisRng,
) -> Result<MetisNodeSeparatorTrace, OrderingError> {
    if graph.vertex_count() < 5000 {
        metis_l1_projected_separator_trace_with_config(graph, config, 7, rng)
    } else {
        let coarsen_to = (graph.vertex_count() / 30).max(100);
        let coarsening = metis_coarsen_graph_nlevels_with_maps(graph, coarsen_to, 4, rng)?;
        let mut mincut = graph.total_vertex_weight();
        let mut best: Option<MetisNodeSeparatorTrace> = None;
        let mut current: Option<MetisNodeSeparatorTrace> = None;
        for run in 0..5 {
            let separator =
                metis_l1_projected_separator_trace_with_config(&coarsening.graph, config, 4, rng)?;
            if run == 0 || separator.mincut < mincut {
                mincut = separator.mincut;
                if run < 4 {
                    best = Some(separator.clone());
                }
            }
            let reached_zero = mincut == 0;
            current = Some(separator);
            if reached_zero {
                break;
            }
        }
        let current = current.ok_or_else(|| {
            OrderingError::Algorithm("METIS L2 separator retry had no trial".into())
        })?;
        let separator = if current.mincut == mincut {
            current
        } else {
            best.ok_or_else(|| {
                OrderingError::Algorithm("METIS L2 separator retry lost best separator".into())
            })?
        };
        let coarse_state = NodePartitionState::from_where(separator.where_part);
        project_node_separator_state_through_coarsening(
            graph,
            &coarsening,
            coarse_state,
            config,
            rng,
        )
    }
}

fn metis_l1_edge_bisection_state(
    graph: &MetisGraph,
) -> Result<(MetisGraph, EdgePartitionState, MetisRng), OrderingError> {
    let vertex_count = graph.vertex_count();
    if vertex_count == 0 {
        return Ok((
            graph.clone(),
            EdgePartitionState::new(0, 0),
            MetisRng::with_metis_seed(-1),
        ));
    }
    let coarsen_to = metis_ometis_l1_coarsen_to(vertex_count);

    let mut rng = MetisRng::with_metis_seed(-1);
    let coarse_graph = metis_l1_coarsen_graph(graph, coarsen_to, &mut rng)?;
    let niparts = if coarse_graph.vertex_count() <= coarsen_to {
        3
    } else {
        7
    };
    let state = if coarse_graph.directed_edge_count() == 0 {
        random_bisection_edge_state(&coarse_graph, niparts, &mut rng)?
    } else {
        grow_bisection_edge_state(&coarse_graph, niparts, &mut rng)?
    };
    Ok((coarse_graph, state, rng))
}

fn metis_ometis_l1_coarsen_to(vertex_count: usize) -> usize {
    let coarsen_to = vertex_count / 8;
    coarsen_to.clamp(40, 100)
}

fn metis_real_to_idx(value: f64) -> isize {
    value as isize
}

fn metis_l1_coarsen_graph(
    graph: &MetisGraph,
    coarsen_to: usize,
    rng: &mut MetisRng,
) -> Result<MetisGraph, OrderingError> {
    Ok(metis_l1_coarsen_graph_with_maps(graph, coarsen_to, rng)?.graph)
}

fn metis_l1_coarsen_graph_with_maps(
    graph: &MetisGraph,
    coarsen_to: usize,
    rng: &mut MetisRng,
) -> Result<MetisCoarsening, OrderingError> {
    metis_coarsen_graph_with_maps_until(graph, coarsen_to, usize::MAX, rng)
}

fn metis_coarsen_graph_nlevels_with_maps(
    graph: &MetisGraph,
    coarsen_to: usize,
    max_levels: usize,
    rng: &mut MetisRng,
) -> Result<MetisCoarsening, OrderingError> {
    metis_coarsen_graph_with_maps_until(graph, coarsen_to, max_levels, rng)
}

fn metis_coarsen_graph_with_maps_until(
    graph: &MetisGraph,
    coarsen_to: usize,
    max_levels: usize,
    rng: &mut MetisRng,
) -> Result<MetisCoarsening, OrderingError> {
    let max_vertex_weight =
        metis_real_to_idx(1.5 * graph.total_vertex_weight() as f64 / coarsen_to as f64);
    let mut current = graph.clone();
    let mut maps = Vec::new();
    let mut graphs = Vec::new();
    let mut eq_edge_weights = current
        .edge_weights
        .windows(2)
        .all(|window| window[0] == window[1]);

    for _ in 0..max_levels {
        let previous_vertex_count = current.vertex_count();
        let level = if eq_edge_weights || current.directed_edge_count() == 0 {
            metis_match_rm_coarsen(&current, max_vertex_weight, rng)?
        } else {
            metis_match_shem_coarsen(&current, max_vertex_weight, rng)?
        };
        current = level.graph;
        maps.push(level.cmap);
        graphs.push(current.clone());
        let keep_coarsening = current.vertex_count() > coarsen_to
            && (current.vertex_count() as f64) < 0.85 * previous_vertex_count as f64
            && current.directed_edge_count() > current.vertex_count() / 2;
        if !keep_coarsening {
            break;
        }
        eq_edge_weights = false;
    }

    Ok(MetisCoarsening {
        graph: current,
        maps,
        graphs,
    })
}

fn metis_match_rm_coarsen(
    graph: &MetisGraph,
    max_vertex_weight: isize,
    rng: &mut MetisRng,
) -> Result<MetisCoarsenedLevel, OrderingError> {
    const UNMATCHED_FOR_2HOP: f64 = 0.10;

    let vertex_count = graph.vertex_count();
    let mut match_to = vec![usize::MAX; vertex_count];
    let mut cmap = vec![usize::MAX; vertex_count];
    let mut tperm = vec![0usize; vertex_count];
    rng.rand_array_permute(&mut tperm, vertex_count / 8, true);

    let average_degree = if vertex_count == 0 {
        0
    } else {
        4 * (graph.directed_edge_count() / vertex_count)
    };
    let degrees = (0..vertex_count)
        .map(|vertex| {
            let bucket = ((1 + graph.degree(vertex)) as f64).sqrt() as usize;
            bucket.min(average_degree)
        })
        .collect::<Vec<_>>();
    let perm = bucket_sort_keys_inc(average_degree, &degrees, &tperm);

    let mut unmatched_for_2hop = 0usize;
    let mut last_unmatched = 0usize;
    let mut coarse_vertex_count = 0usize;
    for (permutation_index, &vertex) in perm.iter().enumerate() {
        if match_to[vertex] != usize::MAX {
            continue;
        }

        let mut match_vertex = vertex;
        if graph.vertex_weight(vertex) < max_vertex_weight {
            if graph.degree(vertex) == 0 {
                last_unmatched = permutation_index.max(last_unmatched) + 1;
                while last_unmatched < vertex_count {
                    let candidate = perm[last_unmatched];
                    if match_to[candidate] == usize::MAX {
                        match_vertex = candidate;
                        break;
                    }
                    last_unmatched += 1;
                }
            } else {
                for edge in graph.edge_range(vertex) {
                    let candidate = graph.neighbors[edge];
                    if match_to[candidate] == usize::MAX
                        && graph.vertex_weight(vertex) + graph.vertex_weight(candidate)
                            <= max_vertex_weight
                    {
                        match_vertex = candidate;
                        break;
                    }
                }

                if match_vertex == vertex && 2 * graph.vertex_weight(vertex) < max_vertex_weight {
                    unmatched_for_2hop += 1;
                    match_vertex = usize::MAX;
                }
            }
        }

        if match_vertex != usize::MAX {
            cmap[vertex] = coarse_vertex_count;
            cmap[match_vertex] = coarse_vertex_count;
            match_to[vertex] = match_vertex;
            match_to[match_vertex] = vertex;
            coarse_vertex_count += 1;
        }
    }

    if unmatched_for_2hop as f64 > UNMATCHED_FOR_2HOP * vertex_count as f64 {
        let _ = metis_match_2hop(
            graph,
            &perm,
            &mut match_to,
            &mut cmap,
            coarse_vertex_count,
            unmatched_for_2hop,
        );
    }

    coarse_vertex_count = 0;
    for vertex in 0..vertex_count {
        if match_to[vertex] == usize::MAX {
            match_to[vertex] = vertex;
            cmap[vertex] = coarse_vertex_count;
            coarse_vertex_count += 1;
        } else if vertex <= match_to[vertex] {
            cmap[vertex] = coarse_vertex_count;
            cmap[match_to[vertex]] = coarse_vertex_count;
            coarse_vertex_count += 1;
        }
    }

    let graph = create_coarse_graph_from_match(graph, coarse_vertex_count, &match_to, &cmap)?;
    Ok(MetisCoarsenedLevel { graph, cmap })
}

fn metis_match_shem_coarsen(
    graph: &MetisGraph,
    max_vertex_weight: isize,
    rng: &mut MetisRng,
) -> Result<MetisCoarsenedLevel, OrderingError> {
    const UNMATCHED_FOR_2HOP: f64 = 0.10;

    let vertex_count = graph.vertex_count();
    let mut match_to = vec![usize::MAX; vertex_count];
    let mut cmap = vec![usize::MAX; vertex_count];
    let mut tperm = vec![0usize; vertex_count];
    rng.rand_array_permute(&mut tperm, vertex_count / 8, true);

    let average_degree = if vertex_count == 0 {
        0
    } else {
        4 * (graph.directed_edge_count() / vertex_count)
    };
    let degrees = (0..vertex_count)
        .map(|vertex| {
            let bucket = ((1 + graph.degree(vertex)) as f64).sqrt() as usize;
            bucket.min(average_degree)
        })
        .collect::<Vec<_>>();
    let perm = bucket_sort_keys_inc(average_degree, &degrees, &tperm);

    let mut unmatched_for_2hop = 0usize;
    let mut last_unmatched = 0usize;
    let mut coarse_vertex_count = 0usize;
    for (permutation_index, &vertex) in perm.iter().enumerate() {
        if match_to[vertex] != usize::MAX {
            continue;
        }

        let mut match_vertex = vertex;
        let mut max_edge_weight = -1isize;
        if graph.vertex_weight(vertex) < max_vertex_weight {
            if graph.degree(vertex) == 0 {
                last_unmatched = permutation_index.max(last_unmatched) + 1;
                while last_unmatched < vertex_count {
                    let candidate = perm[last_unmatched];
                    if match_to[candidate] == usize::MAX {
                        match_vertex = candidate;
                        break;
                    }
                    last_unmatched += 1;
                }
            } else {
                for edge in graph.edge_range(vertex) {
                    let candidate = graph.neighbors[edge];
                    let edge_weight = graph.edge_weight(edge);
                    if match_to[candidate] == usize::MAX
                        && max_edge_weight < edge_weight
                        && graph.vertex_weight(vertex) + graph.vertex_weight(candidate)
                            <= max_vertex_weight
                    {
                        match_vertex = candidate;
                        max_edge_weight = edge_weight;
                    }
                }

                if match_vertex == vertex && 2 * graph.vertex_weight(vertex) < max_vertex_weight {
                    unmatched_for_2hop += 1;
                    match_vertex = usize::MAX;
                }
            }
        }

        if match_vertex != usize::MAX {
            cmap[vertex] = coarse_vertex_count;
            cmap[match_vertex] = coarse_vertex_count;
            match_to[vertex] = match_vertex;
            match_to[match_vertex] = vertex;
            coarse_vertex_count += 1;
        }
    }

    if unmatched_for_2hop as f64 > UNMATCHED_FOR_2HOP * vertex_count as f64 {
        let _ = metis_match_2hop(
            graph,
            &perm,
            &mut match_to,
            &mut cmap,
            coarse_vertex_count,
            unmatched_for_2hop,
        );
    }

    coarse_vertex_count = 0;
    for vertex in 0..vertex_count {
        if match_to[vertex] == usize::MAX {
            match_to[vertex] = vertex;
            cmap[vertex] = coarse_vertex_count;
            coarse_vertex_count += 1;
        } else if vertex <= match_to[vertex] {
            cmap[vertex] = coarse_vertex_count;
            cmap[match_to[vertex]] = coarse_vertex_count;
            coarse_vertex_count += 1;
        }
    }

    let graph = create_coarse_graph_from_match(graph, coarse_vertex_count, &match_to, &cmap)?;
    Ok(MetisCoarsenedLevel { graph, cmap })
}

fn metis_match_2hop(
    graph: &MetisGraph,
    perm: &[usize],
    match_to: &mut [usize],
    cmap: &mut [usize],
    mut coarse_vertex_count: usize,
    mut unmatched_count: usize,
) -> (usize, usize) {
    const UNMATCHED_FOR_2HOP: f64 = 0.10;

    let vertex_count = graph.vertex_count();
    (coarse_vertex_count, unmatched_count) = metis_match_2hop_any(
        graph,
        perm,
        match_to,
        cmap,
        coarse_vertex_count,
        unmatched_count,
        2,
    );
    (coarse_vertex_count, unmatched_count) = metis_match_2hop_all(
        graph,
        perm,
        match_to,
        cmap,
        coarse_vertex_count,
        unmatched_count,
        64,
    );
    if unmatched_count as f64 > 1.5 * UNMATCHED_FOR_2HOP * vertex_count as f64 {
        (coarse_vertex_count, unmatched_count) = metis_match_2hop_any(
            graph,
            perm,
            match_to,
            cmap,
            coarse_vertex_count,
            unmatched_count,
            3,
        );
    }
    if unmatched_count as f64 > 2.0 * UNMATCHED_FOR_2HOP * vertex_count as f64 {
        (coarse_vertex_count, unmatched_count) = metis_match_2hop_any(
            graph,
            perm,
            match_to,
            cmap,
            coarse_vertex_count,
            unmatched_count,
            vertex_count,
        );
    }
    (coarse_vertex_count, unmatched_count)
}

fn metis_match_2hop_any(
    graph: &MetisGraph,
    perm: &[usize],
    match_to: &mut [usize],
    cmap: &mut [usize],
    mut coarse_vertex_count: usize,
    mut unmatched_count: usize,
    max_degree: usize,
) -> (usize, usize) {
    let vertex_count = graph.vertex_count();
    let mut colptr = vec![0usize; vertex_count + 1];
    for (vertex, &matched) in match_to.iter().enumerate().take(vertex_count) {
        if matched == usize::MAX && graph.degree(vertex) < max_degree {
            for &neighbor in graph.neighbors(vertex) {
                colptr[neighbor] += 1;
            }
        }
    }
    make_csr(&mut colptr, vertex_count);

    let mut rowind = vec![0usize; colptr[vertex_count]];
    let mut next = colptr.clone();
    for &vertex in perm {
        if match_to[vertex] == usize::MAX && graph.degree(vertex) < max_degree {
            for &neighbor in graph.neighbors(vertex) {
                rowind[next[neighbor]] = vertex;
                next[neighbor] += 1;
            }
        }
    }

    for &vertex in perm {
        if colptr[vertex + 1] - colptr[vertex] < 2 {
            continue;
        }
        let mut end = colptr[vertex + 1];
        let mut cursor = colptr[vertex];
        while cursor < end {
            let lhs = rowind[cursor];
            if match_to[lhs] == usize::MAX {
                while end > cursor + 1 {
                    end -= 1;
                    let rhs = rowind[end];
                    if match_to[rhs] == usize::MAX {
                        cmap[lhs] = coarse_vertex_count;
                        cmap[rhs] = coarse_vertex_count;
                        coarse_vertex_count += 1;
                        match_to[lhs] = rhs;
                        match_to[rhs] = lhs;
                        unmatched_count -= 2;
                        break;
                    }
                }
            }
            cursor += 1;
        }
    }

    (coarse_vertex_count, unmatched_count)
}

fn metis_match_2hop_all(
    graph: &MetisGraph,
    perm: &[usize],
    match_to: &mut [usize],
    cmap: &mut [usize],
    mut coarse_vertex_count: usize,
    mut unmatched_count: usize,
    max_degree: usize,
) -> (usize, usize) {
    let vertex_count = graph.vertex_count();
    let mask = (isize::MAX as usize) / max_degree;
    let mut keys = Vec::with_capacity(unmatched_count);
    for &vertex in perm {
        let degree = graph.degree(vertex);
        if match_to[vertex] == usize::MAX && degree > 1 && degree < max_degree {
            let key_sum = graph
                .neighbors(vertex)
                .iter()
                .copied()
                .map(|neighbor| neighbor % mask)
                .sum::<usize>();
            keys.push(KeyVal {
                val: vertex,
                key: (key_sum % mask) * max_degree + degree,
            });
        }
    }
    gk_ikvsorti_by_key(&mut keys);

    let mut mark = vec![0usize; vertex_count];
    for key_index in 0..keys.len() {
        let vertex = keys[key_index].val;
        if match_to[vertex] != usize::MAX {
            continue;
        }

        for &neighbor in graph.neighbors(vertex) {
            mark[neighbor] = vertex;
        }

        for candidate_index in key_index + 1..keys.len() {
            let candidate = keys[candidate_index].val;
            if match_to[candidate] != usize::MAX {
                continue;
            }
            if keys[key_index].key != keys[candidate_index].key
                || graph.degree(vertex) != graph.degree(candidate)
            {
                break;
            }
            if graph
                .neighbors(candidate)
                .iter()
                .all(|&neighbor| mark[neighbor] == vertex)
            {
                cmap[vertex] = coarse_vertex_count;
                cmap[candidate] = coarse_vertex_count;
                coarse_vertex_count += 1;
                match_to[vertex] = candidate;
                match_to[candidate] = vertex;
                unmatched_count -= 2;
                break;
            }
        }
    }

    (coarse_vertex_count, unmatched_count)
}

fn make_csr(ptr: &mut [usize], n: usize) {
    for index in 1..n {
        ptr[index] += ptr[index - 1];
    }
    for index in (1..=n).rev() {
        ptr[index] = ptr[index - 1];
    }
    ptr[0] = 0;
}

fn bucket_sort_keys_inc(max_key: usize, keys: &[usize], tperm: &[usize]) -> Vec<usize> {
    let mut counts = vec![0usize; max_key + 2];
    for &key in keys {
        counts[key] += 1;
    }
    let mut sum = 0usize;
    for count in counts.iter_mut().take(max_key + 1) {
        let current = *count;
        *count = sum;
        sum += current;
    }

    let mut perm = vec![0usize; tperm.len()];
    for &vertex in tperm {
        let key = keys[vertex];
        perm[counts[key]] = vertex;
        counts[key] += 1;
    }
    perm
}

fn create_coarse_graph_from_match(
    graph: &MetisGraph,
    coarse_vertex_count: usize,
    match_to: &[usize],
    cmap: &[usize],
) -> Result<MetisGraph, OrderingError> {
    let vertex_count = graph.vertex_count();
    let mut offsets = Vec::with_capacity(coarse_vertex_count + 1);
    let mut neighbors = Vec::new();
    let mut edge_weights = Vec::new();
    let mut vertex_weights = Vec::with_capacity(coarse_vertex_count);
    offsets.push(0);

    for vertex in 0..vertex_count {
        let mate = match_to[vertex];
        if mate < vertex {
            continue;
        }
        let coarse_vertex = vertex_weights.len();
        if cmap[vertex] != coarse_vertex || cmap[mate] != coarse_vertex {
            return Err(OrderingError::Algorithm(
                "METIS Match_RM cmap order diverged from coarse contraction".into(),
            ));
        }

        vertex_weights.push(if vertex == mate {
            graph.vertex_weight(vertex)
        } else {
            graph.vertex_weight(vertex) + graph.vertex_weight(mate)
        });

        let degree_sum = graph.degree(vertex)
            + if vertex == mate {
                0
            } else {
                graph.degree(mate)
            };
        let (mut local_neighbors, mut local_weights) = if degree_sum < (HT_LENGTH >> 2) {
            coarse_edges_hash(graph, coarse_vertex, vertex, mate, cmap)
        } else {
            coarse_edges_direct(
                graph,
                coarse_vertex_count,
                coarse_vertex,
                vertex,
                mate,
                cmap,
            )
        };
        neighbors.append(&mut local_neighbors);
        edge_weights.append(&mut local_weights);
        offsets.push(neighbors.len());
    }

    Ok(MetisGraph {
        offsets,
        neighbors,
        vertex_weights,
        edge_weights,
    })
}

const HT_LENGTH: usize = (1 << 13) - 1;

fn coarse_edges_hash(
    graph: &MetisGraph,
    coarse_vertex: usize,
    vertex: usize,
    mate: usize,
    cmap: &[usize],
) -> (Vec<usize>, Vec<isize>) {
    let mut htable = vec![usize::MAX; HT_LENGTH + 1];
    let mut local_neighbors = vec![coarse_vertex];
    let mut local_weights = vec![0isize];
    htable[coarse_vertex & HT_LENGTH] = 0;

    coarse_edges_hash_accumulate(
        graph,
        vertex,
        cmap,
        &mut htable,
        &mut local_neighbors,
        &mut local_weights,
    );
    if vertex != mate {
        coarse_edges_hash_accumulate(
            graph,
            mate,
            cmap,
            &mut htable,
            &mut local_neighbors,
            &mut local_weights,
        );
    }

    let last = local_neighbors.len() - 1;
    local_neighbors[0] = local_neighbors[last];
    local_weights[0] = local_weights[last];
    local_neighbors.pop();
    local_weights.pop();
    (local_neighbors, local_weights)
}

fn coarse_edges_hash_accumulate(
    graph: &MetisGraph,
    vertex: usize,
    cmap: &[usize],
    htable: &mut [usize],
    local_neighbors: &mut Vec<usize>,
    local_weights: &mut Vec<isize>,
) {
    for edge in graph.edge_range(vertex) {
        let coarse_neighbor = cmap[graph.neighbors[edge]];
        let mut hash = coarse_neighbor & HT_LENGTH;
        while htable[hash] != usize::MAX && local_neighbors[htable[hash]] != coarse_neighbor {
            hash = (hash + 1) & HT_LENGTH;
        }
        let position = htable[hash];
        if position == usize::MAX {
            htable[hash] = local_neighbors.len();
            local_neighbors.push(coarse_neighbor);
            local_weights.push(graph.edge_weight(edge));
        } else {
            local_weights[position] += graph.edge_weight(edge);
        }
    }
}

fn coarse_edges_direct(
    graph: &MetisGraph,
    coarse_vertex_count: usize,
    coarse_vertex: usize,
    vertex: usize,
    mate: usize,
    cmap: &[usize],
) -> (Vec<usize>, Vec<isize>) {
    let mut dtable = vec![usize::MAX; coarse_vertex_count];
    let mut local_neighbors = Vec::new();
    let mut local_weights = Vec::new();

    coarse_edges_direct_accumulate(
        graph,
        vertex,
        cmap,
        &mut dtable,
        &mut local_neighbors,
        &mut local_weights,
    );
    if vertex != mate {
        coarse_edges_direct_accumulate(
            graph,
            mate,
            cmap,
            &mut dtable,
            &mut local_neighbors,
            &mut local_weights,
        );

        if let Some(position) = dtable.get(coarse_vertex).copied()
            && position != usize::MAX
        {
            let last = local_neighbors.len() - 1;
            local_neighbors[position] = local_neighbors[last];
            local_weights[position] = local_weights[last];
            local_neighbors.pop();
            local_weights.pop();
        }
    }

    (local_neighbors, local_weights)
}

fn coarse_edges_direct_accumulate(
    graph: &MetisGraph,
    vertex: usize,
    cmap: &[usize],
    dtable: &mut [usize],
    local_neighbors: &mut Vec<usize>,
    local_weights: &mut Vec<isize>,
) {
    for edge in graph.edge_range(vertex) {
        let coarse_neighbor = cmap[graph.neighbors[edge]];
        let position = dtable[coarse_neighbor];
        if position == usize::MAX {
            dtable[coarse_neighbor] = local_neighbors.len();
            local_neighbors.push(coarse_neighbor);
            local_weights.push(graph.edge_weight(edge));
        } else {
            local_weights[position] += graph.edge_weight(edge);
        }
    }
}

fn random_bisection_edge_state(
    graph: &MetisGraph,
    niparts: usize,
    rng: &mut MetisRng,
) -> Result<EdgePartitionState, OrderingError> {
    debug_assert_eq!(graph.directed_edge_count(), 0);
    let vertex_count = graph.vertex_count();
    let total_weight = graph.total_vertex_weight();
    let ubfactor = 1.0 + 0.001 * 200.0 + 0.000_049_9;
    let zero_max_weight = metis_real_to_idx(ubfactor * total_weight as f64 * 0.5);
    let mut best_cut = None;
    let mut best_where = vec![1; vertex_count];

    for run in 0..niparts {
        let mut state = EdgePartitionState::new(vertex_count, total_weight);

        if run > 0 {
            let mut perm = vec![0usize; vertex_count];
            rng.rand_array_permute(&mut perm, vertex_count / 2, true);
            let mut part_weights = [0isize, total_weight];
            for &vertex in &perm {
                let vertex_weight = graph.vertex_weight(vertex);
                if part_weights[0] + vertex_weight < zero_max_weight {
                    state.where_part[vertex] = 0;
                    part_weights[0] += vertex_weight;
                    part_weights[1] -= vertex_weight;
                    if part_weights[0] > zero_max_weight {
                        break;
                    }
                }
            }
        }

        compute_2way_partition_params(graph, &mut state);
        balance_2way_no_edge(graph, &mut state, rng);
        fm_2way_cut_refine(graph, &mut state, 4, rng);

        if run == 0 || best_cut.is_some_and(|cut| cut > state.mincut) {
            best_cut = Some(state.mincut);
            best_where.clone_from(&state.where_part);
            if state.mincut == 0 {
                break;
            }
        }
    }

    let mut state = EdgePartitionState::new(vertex_count, total_weight);
    state.where_part = best_where;
    compute_2way_partition_params(graph, &mut state);
    Ok(state)
}

fn balance_2way_no_edge(graph: &MetisGraph, state: &mut EdgePartitionState, rng: &mut MetisRng) {
    debug_assert_eq!(graph.directed_edge_count(), 0);
    let vertex_count = graph.vertex_count();
    if vertex_count == 0 {
        return;
    }

    let target_weights = [
        metis_real_to_idx(0.5 * graph.total_vertex_weight() as f64),
        graph.total_vertex_weight() - metis_real_to_idx(0.5 * graph.total_vertex_weight() as f64),
    ];
    if (target_weights[0] - state.part_weights[0]).abs()
        < 3 * graph.total_vertex_weight() / vertex_count as isize
    {
        return;
    }

    let mut perm = vec![0usize; vertex_count];
    rng.rand_array_permute(&mut perm, vertex_count / 5, true);
    let min_difference = (target_weights[0] - state.part_weights[0]).abs();
    let from = if state.part_weights[0] < target_weights[0] {
        1
    } else {
        0
    };
    let to = (from + 1) % 2;
    let mut queue = SourcePriorityQueue::new(vertex_count);
    for &vertex in &perm {
        if state.where_part[vertex] == from && graph.vertex_weight(vertex) <= min_difference {
            queue.insert(vertex, 0);
        }
    }

    while let Some(high_gain) = queue.get_top() {
        let high_gain_weight = graph.vertex_weight(high_gain);
        if state.part_weights[to] + high_gain_weight > target_weights[to] {
            break;
        }

        state.part_weights[to] += high_gain_weight;
        state.part_weights[from] -= high_gain_weight;
        state.where_part[high_gain] = to;
    }
}

fn grow_bisection_edge_state(
    graph: &MetisGraph,
    niparts: usize,
    rng: &mut MetisRng,
) -> Result<EdgePartitionState, OrderingError> {
    let vertex_count = graph.vertex_count();
    let total_weight = graph.total_vertex_weight();
    let ubfactor = 1.0 + 0.001 * 200.0 + 0.000_049_9;
    let one_max_weight = metis_real_to_idx(ubfactor * total_weight as f64 * 0.5);
    let one_min_weight = metis_real_to_idx((1.0 / ubfactor) * total_weight as f64 * 0.5);
    let mut best_cut = None;
    let mut best_where = vec![1; vertex_count];

    for run in 0..niparts {
        let mut state = EdgePartitionState::new(vertex_count, total_weight);
        let mut touched = vec![false; vertex_count];
        let mut queue = vec![0usize; vertex_count.max(1)];
        let mut bfs_weights = [0isize, total_weight];

        queue[0] = rng.rand_in_range(vertex_count);
        touched[queue[0]] = true;
        let mut first = 0usize;
        let mut last = 1usize;
        let mut nleft = vertex_count - 1;
        let mut drain = false;

        loop {
            if first == last {
                if nleft == 0 || drain {
                    break;
                }
                let mut k = rng.rand_in_range(nleft);
                let mut next_start = None;
                for (vertex, &was_touched) in touched.iter().enumerate() {
                    if !was_touched {
                        if k == 0 {
                            next_start = Some(vertex);
                            break;
                        }
                        k -= 1;
                    }
                }
                let vertex = next_start.ok_or_else(|| {
                    OrderingError::Algorithm(
                        "METIS GrowBisection disconnected start was not found".into(),
                    )
                })?;
                queue[0] = vertex;
                touched[vertex] = true;
                first = 0;
                last = 1;
                nleft -= 1;
            }

            let vertex = queue[first];
            let vertex_weight = graph.vertex_weight(vertex);
            first += 1;
            if bfs_weights[0] > 0 && bfs_weights[1] - vertex_weight < one_min_weight {
                drain = true;
                continue;
            }

            state.where_part[vertex] = 0;
            bfs_weights[0] += vertex_weight;
            bfs_weights[1] -= vertex_weight;
            if bfs_weights[1] <= one_max_weight {
                break;
            }

            drain = false;
            for &neighbor in graph.neighbors(vertex) {
                if !touched[neighbor] {
                    queue[last] = neighbor;
                    last += 1;
                    touched[neighbor] = true;
                    nleft -= 1;
                }
            }
        }

        if bfs_weights[1] == 0 {
            state.where_part[rng.rand_in_range(vertex_count)] = 1;
        }
        if bfs_weights[0] == 0 {
            state.where_part[rng.rand_in_range(vertex_count)] = 0;
        }

        compute_2way_partition_params(graph, &mut state);
        ensure_2way_balance_is_source_noop(graph, &state)?;
        fm_2way_cut_refine(graph, &mut state, 10, rng);

        if run == 0 || best_cut.is_some_and(|cut| cut > state.mincut) {
            best_cut = Some(state.mincut);
            best_where.clone_from(&state.where_part);
            if state.mincut == 0 {
                break;
            }
        }
    }

    let mut state = EdgePartitionState::new(vertex_count, total_weight);
    state.where_part = best_where;
    compute_2way_partition_params(graph, &mut state);
    Ok(state)
}

fn compute_2way_partition_params(graph: &MetisGraph, state: &mut EdgePartitionState) {
    let vertex_count = graph.vertex_count();
    state.part_weights = [0, 0];
    state.boundary.clear();
    state.boundary_ptr.fill(usize::MAX);
    state.mincut = 0;

    for (vertex, &part) in state.where_part.iter().enumerate() {
        state.part_weights[part] += graph.vertex_weight(vertex);
    }

    for vertex in 0..vertex_count {
        let part = state.where_part[vertex];
        let mut internal = 0isize;
        let mut external = 0isize;
        for edge in graph.edge_range(vertex) {
            let neighbor = graph.neighbors[edge];
            let edge_weight = graph.edge_weight(edge);
            if part == state.where_part[neighbor] {
                internal += edge_weight;
            } else {
                external += edge_weight;
            }
        }
        state.internal_degree[vertex] = internal;
        state.external_degree[vertex] = external;
        if external > 0 || graph.degree(vertex) == 0 {
            state.insert_boundary(vertex);
            state.mincut += external;
        }
    }
    state.mincut /= 2;
}

fn ensure_2way_balance_is_source_noop(
    graph: &MetisGraph,
    state: &EdgePartitionState,
) -> Result<(), OrderingError> {
    let total_weight = graph.total_vertex_weight();
    let vertex_count = total_weight as f64;
    let ubfactor = 1.0 + 0.001 * 200.0 + 0.000_049_9;
    let imbalance = state
        .part_weights
        .iter()
        .map(|&weight| weight as f64 * 2.0 / vertex_count)
        .fold(0.0, f64::max)
        - ubfactor;
    if imbalance <= 0.0 {
        return Ok(());
    }

    let target_left = metis_real_to_idx(0.5 * total_weight as f64);
    let balance_slop = 3 * total_weight / graph.vertex_count() as isize;
    if (target_left - state.part_weights[0]).abs() < balance_slop {
        return Ok(());
    }

    if state.boundary.is_empty() {
        Err(OrderingError::Algorithm(
            "METIS General2WayBalance branch is not yet ported".into(),
        ))
    } else {
        Err(OrderingError::Algorithm(
            "METIS Bnd2WayBalance branch is not yet ported".into(),
        ))
    }
}

fn fm_2way_cut_refine(
    graph: &MetisGraph,
    state: &mut EdgePartitionState,
    niter: usize,
    rng: &mut MetisRng,
) {
    let vertex_count = graph.vertex_count();
    let total_weight = graph.total_vertex_weight();
    let target_weights = [
        metis_real_to_idx(total_weight as f64 * 0.5),
        total_weight - metis_real_to_idx(total_weight as f64 * 0.5),
    ];
    let limit = metis_real_to_idx(0.01 * vertex_count as f64).clamp(15, 100);
    let average_vertex_weight = ((state.part_weights[0] + state.part_weights[1]) / 20)
        .min(2 * (state.part_weights[0] + state.part_weights[1]) / vertex_count as isize);
    let original_difference = (target_weights[0] - state.part_weights[0]).abs();
    let mut moved = vec![-1isize; vertex_count];
    let mut swaps = Vec::with_capacity(vertex_count);

    for _pass in 0..niter {
        let mut queues = [
            SourcePriorityQueue::new(vertex_count),
            SourcePriorityQueue::new(vertex_count),
        ];
        let mut mincut_order = -1isize;
        let mut new_cut = state.mincut;
        let mut min_cut = state.mincut;
        let initial_cut = state.mincut;
        let mut min_difference = (target_weights[0] - state.part_weights[0]).abs();
        let mut boundary_count = state.boundary.len();
        swaps.clear();

        let mut permutation = vec![0; boundary_count];
        rng.rand_array_permute(&mut permutation, boundary_count, true);
        for &boundary_position in &permutation {
            let vertex = state.boundary[boundary_position];
            let part = state.where_part[vertex];
            queues[part].insert(
                vertex,
                state.external_degree[vertex] - state.internal_degree[vertex],
            );
        }

        for swap_index in 0..vertex_count {
            let from = if target_weights[0] - state.part_weights[0]
                < target_weights[1] - state.part_weights[1]
            {
                0
            } else {
                1
            };
            let to = (from + 1) % 2;
            let Some(high_gain) = queues[from].get_top() else {
                break;
            };

            let gain = state.external_degree[high_gain] - state.internal_degree[high_gain];
            let high_gain_weight = graph.vertex_weight(high_gain);
            new_cut -= gain;
            state.part_weights[to] += high_gain_weight;
            state.part_weights[from] -= high_gain_weight;

            let current_difference = (target_weights[0] - state.part_weights[0]).abs();
            if (new_cut < min_cut
                && current_difference <= original_difference + average_vertex_weight)
                || (new_cut == min_cut && current_difference < min_difference)
            {
                min_cut = new_cut;
                min_difference = current_difference;
                mincut_order = swap_index as isize;
            } else if swap_index as isize - mincut_order > limit {
                state.part_weights[from] += high_gain_weight;
                state.part_weights[to] -= high_gain_weight;
                break;
            }

            state.where_part[high_gain] = to;
            moved[high_gain] = swap_index as isize;
            swaps.push(high_gain);

            std::mem::swap(
                &mut state.internal_degree[high_gain],
                &mut state.external_degree[high_gain],
            );
            if state.external_degree[high_gain] == 0 && graph.degree(high_gain) > 0 {
                state.delete_boundary(high_gain);
                boundary_count -= 1;
            }

            for edge in graph.edge_range(high_gain) {
                let neighbor = graph.neighbors[edge];
                let weight = graph.edge_weight(edge);
                let edge_delta = if to == state.where_part[neighbor] {
                    weight
                } else {
                    -weight
                };
                state.internal_degree[neighbor] += edge_delta;
                state.external_degree[neighbor] -= edge_delta;

                if state.boundary_ptr[neighbor] != usize::MAX {
                    if state.external_degree[neighbor] == 0 {
                        state.delete_boundary(neighbor);
                        boundary_count -= 1;
                        if moved[neighbor] == -1 {
                            queues[state.where_part[neighbor]].delete(neighbor);
                        }
                    } else if moved[neighbor] == -1 {
                        queues[state.where_part[neighbor]].update(
                            neighbor,
                            state.external_degree[neighbor] - state.internal_degree[neighbor],
                        );
                    }
                } else if state.external_degree[neighbor] > 0 {
                    state.insert_boundary(neighbor);
                    boundary_count += 1;
                    if moved[neighbor] == -1 {
                        queues[state.where_part[neighbor]].insert(
                            neighbor,
                            state.external_degree[neighbor] - state.internal_degree[neighbor],
                        );
                    }
                }
            }
        }

        for &vertex in &swaps {
            moved[vertex] = -1;
        }
        let mut rollback_index = swaps.len() as isize - 1;
        while rollback_index > mincut_order {
            let high_gain = swaps[rollback_index as usize];
            let to = (state.where_part[high_gain] + 1) % 2;
            state.where_part[high_gain] = to;
            std::mem::swap(
                &mut state.internal_degree[high_gain],
                &mut state.external_degree[high_gain],
            );
            if state.external_degree[high_gain] == 0
                && state.boundary_ptr[high_gain] != usize::MAX
                && graph.degree(high_gain) > 0
            {
                state.delete_boundary(high_gain);
                boundary_count -= 1;
            } else if state.external_degree[high_gain] > 0
                && state.boundary_ptr[high_gain] == usize::MAX
            {
                state.insert_boundary(high_gain);
                boundary_count += 1;
            }

            let high_gain_weight = graph.vertex_weight(high_gain);
            state.part_weights[to] += high_gain_weight;
            state.part_weights[(to + 1) % 2] -= high_gain_weight;
            for edge in graph.edge_range(high_gain) {
                let neighbor = graph.neighbors[edge];
                let weight = graph.edge_weight(edge);
                let edge_delta = if to == state.where_part[neighbor] {
                    weight
                } else {
                    -weight
                };
                state.internal_degree[neighbor] += edge_delta;
                state.external_degree[neighbor] -= edge_delta;

                if state.boundary_ptr[neighbor] != usize::MAX
                    && state.external_degree[neighbor] == 0
                {
                    state.delete_boundary(neighbor);
                    boundary_count -= 1;
                }
                if state.boundary_ptr[neighbor] == usize::MAX && state.external_degree[neighbor] > 0
                {
                    state.insert_boundary(neighbor);
                    boundary_count += 1;
                }
            }
            rollback_index -= 1;
        }

        state.mincut = min_cut;
        debug_assert_eq!(boundary_count, state.boundary.len());

        if mincut_order <= 0 || min_cut == initial_cut {
            break;
        }
    }
}

fn compute_2way_node_partition_params(graph: &MetisGraph, state: &mut NodePartitionState) {
    let vertex_count = graph.vertex_count();
    state.part_weights = [0; 3];
    state.boundary.clear();
    state.boundary_ptr.fill(usize::MAX);
    state.edegrees.fill([0; 2]);

    for vertex in 0..vertex_count {
        let part = state.where_part[vertex];
        debug_assert!(part <= 2);
        state.part_weights[part] += graph.vertex_weight(vertex);

        if part == 2 {
            state.insert_boundary(vertex);
            for &neighbor in graph.neighbors(vertex) {
                let other = state.where_part[neighbor];
                if other != 2 {
                    state.edegrees[vertex][other] += graph.vertex_weight(neighbor);
                }
            }
        }
    }
    state.mincut = state.part_weights[2];
}

fn fm_2way_node_balance(graph: &MetisGraph, state: &mut NodePartitionState, rng: &mut MetisRng) {
    let vertex_count = graph.vertex_count();
    if vertex_count == 0 {
        return;
    }
    let ubfactor = 1.0 + 0.001 * 200.0 + 0.000_049_9;
    let mut bad_max_weight =
        metis_real_to_idx(0.5 * ubfactor * (state.part_weights[0] + state.part_weights[1]) as f64);
    if state.part_weights[0].max(state.part_weights[1]) < bad_max_weight {
        return;
    }
    if (state.part_weights[0] - state.part_weights[1]).abs()
        < 3 * graph.total_vertex_weight() / vertex_count as isize
    {
        return;
    }

    let to = if state.part_weights[0] < state.part_weights[1] {
        0
    } else {
        1
    };
    let other = (to + 1) % 2;
    let mut queue = SourcePriorityQueue::new(vertex_count);
    let mut moved = vec![-1isize; vertex_count];

    let boundary_count = state.boundary.len();
    let mut permutation = vec![0usize; boundary_count];
    rng.rand_array_permute(&mut permutation, boundary_count, true);
    for &boundary_position in &permutation {
        let vertex = state.boundary[boundary_position];
        debug_assert_eq!(state.where_part[vertex], 2);
        queue.insert(
            vertex,
            graph.vertex_weight(vertex) - state.edegrees[vertex][other],
        );
    }

    for _ in 0..vertex_count {
        let Some(high_gain) = queue.get_top() else {
            break;
        };
        moved[high_gain] = 1;

        let high_gain_weight = graph.vertex_weight(high_gain);
        let gain = high_gain_weight - state.edegrees[high_gain][other];
        bad_max_weight = metis_real_to_idx(
            0.5 * ubfactor * (state.part_weights[0] + state.part_weights[1]) as f64,
        );

        if state.part_weights[to] > state.part_weights[other] {
            break;
        }
        if gain < 0 && state.part_weights[other] < bad_max_weight {
            break;
        }
        if state.part_weights[to] + high_gain_weight > bad_max_weight {
            continue;
        }

        debug_assert_ne!(state.boundary_ptr[high_gain], usize::MAX);
        state.part_weights[2] -= gain;
        state.delete_boundary(high_gain);
        state.part_weights[to] += high_gain_weight;
        state.where_part[high_gain] = to;

        for &neighbor in graph.neighbors(high_gain) {
            if state.where_part[neighbor] == 2 {
                state.edegrees[neighbor][to] += high_gain_weight;
            } else if state.where_part[neighbor] == other {
                let neighbor_weight = graph.vertex_weight(neighbor);
                debug_assert_eq!(state.boundary_ptr[neighbor], usize::MAX);
                state.insert_boundary(neighbor);
                state.where_part[neighbor] = 2;
                state.part_weights[other] -= neighbor_weight;

                state.edegrees[neighbor] = [0; 2];
                for &neighbor_neighbor in graph.neighbors(neighbor) {
                    if state.where_part[neighbor_neighbor] != 2 {
                        state.edegrees[neighbor][state.where_part[neighbor_neighbor]] +=
                            graph.vertex_weight(neighbor_neighbor);
                    } else {
                        debug_assert_ne!(state.boundary_ptr[neighbor_neighbor], usize::MAX);
                        let old_gain = graph.vertex_weight(neighbor_neighbor)
                            - state.edegrees[neighbor_neighbor][other];
                        state.edegrees[neighbor_neighbor][other] -= neighbor_weight;

                        if moved[neighbor_neighbor] == -1 {
                            queue.update(neighbor_neighbor, old_gain + neighbor_weight);
                        }
                    }
                }

                queue.insert(neighbor, neighbor_weight - state.edegrees[neighbor][other]);
            }
        }
    }

    state.mincut = state.part_weights[2];
}

fn fm_2way_node_refine_2sided(
    graph: &MetisGraph,
    state: &mut NodePartitionState,
    niter: usize,
    compression_active: bool,
    rng: &mut MetisRng,
) {
    let vertex_count = graph.vertex_count();
    if vertex_count == 0 {
        return;
    }
    let ubfactor = 1.0 + 0.001 * 200.0 + 0.000_049_9;
    let bad_max_weight = metis_real_to_idx(
        0.5 * ubfactor
            * (state.part_weights[0] + state.part_weights[1] + state.part_weights[2]) as f64,
    );
    let mut moved = vec![-1isize; vertex_count];
    let mut swaps = vec![0usize; vertex_count];
    let mut mptr = vec![0usize; vertex_count + 1];
    let mut mind = vec![0usize; 2 * vertex_count];

    for pass in 0..niter {
        moved.fill(-1);
        let mut queues = [
            SourcePriorityQueue::new(vertex_count),
            SourcePriorityQueue::new(vertex_count),
        ];

        let mut mincut_order = -1isize;
        let init_cut = state.mincut;
        let mut mincut = state.mincut;
        let initial_boundary_count = state.boundary.len();
        let mut boundary_count = initial_boundary_count;

        let mut permutation = vec![0usize; initial_boundary_count];
        rng.rand_array_permute(&mut permutation, initial_boundary_count, true);
        for &boundary_position in &permutation {
            let vertex = state.boundary[boundary_position];
            debug_assert_eq!(state.where_part[vertex], 2);
            let vertex_weight = graph.vertex_weight(vertex);
            queues[0].insert(vertex, vertex_weight - state.edegrees[vertex][1]);
            queues[1].insert(vertex, vertex_weight - state.edegrees[vertex][0]);
        }

        mptr[0] = 0;
        let mut nmind = 0usize;
        let mut mindiff = (state.part_weights[0] - state.part_weights[1]).abs();
        let limit = if compression_active {
            (5 * initial_boundary_count).min(400)
        } else {
            (2 * initial_boundary_count).min(300)
        } as isize;
        let mut performed_swaps = 0usize;

        for swap_index in 0..vertex_count {
            let mut to: usize;
            let top = [queues[0].see_top_val(), queues[1].see_top_val()];
            if let [Some(left_top), Some(right_top)] = top {
                let gains = [
                    graph.vertex_weight(left_top) - state.edegrees[left_top][1],
                    graph.vertex_weight(right_top) - state.edegrees[right_top][0],
                ];
                to = if gains[0] > gains[1] {
                    0
                } else if gains[0] < gains[1] {
                    1
                } else {
                    pass % 2
                };
                if state.part_weights[to] + graph.vertex_weight(top[to].expect("top exists"))
                    > bad_max_weight
                {
                    to = (to + 1) % 2;
                }
            } else if top[0].is_none() && top[1].is_none() {
                break;
            } else if top[0].is_some_and(|vertex| {
                state.part_weights[0] + graph.vertex_weight(vertex) <= bad_max_weight
            }) {
                to = 0;
            } else if top[1].is_some_and(|vertex| {
                state.part_weights[1] + graph.vertex_weight(vertex) <= bad_max_weight
            }) {
                to = 1;
            } else {
                break;
            }

            let other = (to + 1) % 2;
            let Some(high_gain) = queues[to].get_top() else {
                break;
            };
            if moved[high_gain] == -1 {
                queues[other].delete(high_gain);
            }
            debug_assert_ne!(state.boundary_ptr[high_gain], usize::MAX);

            if nmind + graph.degree(high_gain) >= 2 * vertex_count - 1 {
                break;
            }

            let high_gain_weight = graph.vertex_weight(high_gain);
            let sep_delta = high_gain_weight - state.edegrees[high_gain][other];
            state.part_weights[2] -= sep_delta;

            let newdiff = (state.part_weights[to] + high_gain_weight
                - (state.part_weights[other] - state.edegrees[high_gain][other]))
                .abs();
            if state.part_weights[2] < mincut
                || (state.part_weights[2] == mincut && newdiff < mindiff)
            {
                mincut = state.part_weights[2];
                mincut_order = swap_index as isize;
                mindiff = newdiff;
            } else if swap_index as isize - mincut_order > 2 * limit
                || (swap_index as isize - mincut_order > limit
                    && (state.part_weights[2] as f64) > 1.10 * (mincut as f64))
            {
                state.part_weights[2] += sep_delta;
                break;
            }

            state.delete_boundary(high_gain);
            boundary_count -= 1;
            state.part_weights[to] += high_gain_weight;
            state.where_part[high_gain] = to;
            moved[high_gain] = swap_index as isize;
            swaps[swap_index] = high_gain;
            performed_swaps = swap_index + 1;

            for &neighbor in graph.neighbors(high_gain) {
                if state.where_part[neighbor] == 2 {
                    let old_gain = graph.vertex_weight(neighbor) - state.edegrees[neighbor][to];
                    state.edegrees[neighbor][to] += high_gain_weight;
                    if moved[neighbor] == -1 || moved[neighbor] == -((2 + other) as isize) {
                        queues[other].update(neighbor, old_gain - high_gain_weight);
                    }
                } else if state.where_part[neighbor] == other {
                    let neighbor_weight = graph.vertex_weight(neighbor);
                    debug_assert_eq!(state.boundary_ptr[neighbor], usize::MAX);
                    state.insert_boundary(neighbor);
                    boundary_count += 1;
                    mind[nmind] = neighbor;
                    nmind += 1;
                    state.where_part[neighbor] = 2;
                    state.part_weights[other] -= neighbor_weight;

                    state.edegrees[neighbor] = [0; 2];
                    for &neighbor_neighbor in graph.neighbors(neighbor) {
                        if state.where_part[neighbor_neighbor] != 2 {
                            state.edegrees[neighbor][state.where_part[neighbor_neighbor]] +=
                                graph.vertex_weight(neighbor_neighbor);
                        } else {
                            let old_gain = graph.vertex_weight(neighbor_neighbor)
                                - state.edegrees[neighbor_neighbor][other];
                            state.edegrees[neighbor_neighbor][other] -= neighbor_weight;
                            if moved[neighbor_neighbor] == -1
                                || moved[neighbor_neighbor] == -((2 + to) as isize)
                            {
                                queues[to].update(neighbor_neighbor, old_gain + neighbor_weight);
                            }
                        }
                    }

                    if moved[neighbor] == -1 {
                        queues[to]
                            .insert(neighbor, neighbor_weight - state.edegrees[neighbor][other]);
                        moved[neighbor] = -((2 + to) as isize);
                    }
                }
            }
            mptr[swap_index + 1] = nmind;
        }

        for &vertex in &swaps[..performed_swaps] {
            moved[vertex] = -1;
        }

        for swap_index in ((mincut_order + 1).max(0) as usize..performed_swaps).rev() {
            let high_gain = swaps[swap_index];
            let to = state.where_part[high_gain];
            let other = (to + 1) % 2;
            let high_gain_weight = graph.vertex_weight(high_gain);
            state.part_weights[2] += high_gain_weight;
            state.part_weights[to] -= high_gain_weight;
            state.where_part[high_gain] = 2;
            state.insert_boundary(high_gain);
            boundary_count += 1;

            state.edegrees[high_gain] = [0; 2];
            for &neighbor in graph.neighbors(high_gain) {
                if state.where_part[neighbor] == 2 {
                    state.edegrees[neighbor][to] -= high_gain_weight;
                } else {
                    state.edegrees[high_gain][state.where_part[neighbor]] +=
                        graph.vertex_weight(neighbor);
                }
            }

            for &vertex in mind
                .iter()
                .take(mptr[swap_index + 1])
                .skip(mptr[swap_index])
            {
                let vertex_weight = graph.vertex_weight(vertex);
                debug_assert_eq!(state.where_part[vertex], 2);
                state.where_part[vertex] = other;
                state.part_weights[other] += vertex_weight;
                state.part_weights[2] -= vertex_weight;
                state.delete_boundary(vertex);
                boundary_count -= 1;
                for &neighbor in graph.neighbors(vertex) {
                    if state.where_part[neighbor] == 2 {
                        state.edegrees[neighbor][other] += vertex_weight;
                    }
                }
            }
        }

        state.mincut = mincut;
        debug_assert_eq!(boundary_count, state.boundary.len());

        if mincut_order == -1 || mincut >= init_cut {
            break;
        }
    }
}

fn fm_2way_node_refine_1sided(
    graph: &MetisGraph,
    state: &mut NodePartitionState,
    niter: usize,
    compression_active: bool,
    rng: &mut MetisRng,
) {
    let vertex_count = graph.vertex_count();
    if vertex_count == 0 {
        return;
    }
    let ubfactor = 1.0 + 0.001 * 200.0 + 0.000_049_9;
    let bad_max_weight = metis_real_to_idx(
        0.5 * ubfactor
            * (state.part_weights[0] + state.part_weights[1] + state.part_weights[2]) as f64,
    );
    let mut swaps = vec![0usize; vertex_count];
    let mut mptr = vec![0usize; vertex_count + 1];
    let mut mind = vec![0usize; 2 * vertex_count];

    let mut to = if state.part_weights[0] < state.part_weights[1] {
        1
    } else {
        0
    };
    for pass in 0..2 * niter {
        let other = to;
        to = (to + 1) % 2;

        let mut queue = SourcePriorityQueue::new(vertex_count);
        let mut mincut_order = -1isize;
        let init_cut = state.mincut;
        let mut mincut = state.mincut;
        let initial_boundary_count = state.boundary.len();
        let mut boundary_count = initial_boundary_count;

        let mut permutation = vec![0usize; initial_boundary_count];
        rng.rand_array_permute(&mut permutation, initial_boundary_count, true);
        for &boundary_position in &permutation {
            let vertex = state.boundary[boundary_position];
            debug_assert_eq!(state.where_part[vertex], 2);
            queue.insert(
                vertex,
                graph.vertex_weight(vertex) - state.edegrees[vertex][other],
            );
        }

        let limit = if compression_active {
            (5 * initial_boundary_count).min(500)
        } else {
            (3 * initial_boundary_count).min(300)
        } as isize;
        mptr[0] = 0;
        let mut nmind = 0usize;
        let mut mindiff = (state.part_weights[0] - state.part_weights[1]).abs();
        let mut performed_swaps = 0usize;

        for swap_index in 0..vertex_count {
            let Some(high_gain) = queue.get_top() else {
                break;
            };
            debug_assert_ne!(state.boundary_ptr[high_gain], usize::MAX);

            if nmind + graph.degree(high_gain) >= 2 * vertex_count - 1 {
                break;
            }
            let high_gain_weight = graph.vertex_weight(high_gain);
            if state.part_weights[to] + high_gain_weight > bad_max_weight {
                break;
            }

            let sep_delta = high_gain_weight - state.edegrees[high_gain][other];
            state.part_weights[2] -= sep_delta;

            let newdiff = (state.part_weights[to] + high_gain_weight
                - (state.part_weights[other] - state.edegrees[high_gain][other]))
                .abs();
            if state.part_weights[2] < mincut
                || (state.part_weights[2] == mincut && newdiff < mindiff)
            {
                mincut = state.part_weights[2];
                mincut_order = swap_index as isize;
                mindiff = newdiff;
            } else if swap_index as isize - mincut_order > 3 * limit
                || (swap_index as isize - mincut_order > limit
                    && (state.part_weights[2] as f64) > 1.10 * (mincut as f64))
            {
                state.part_weights[2] += sep_delta;
                break;
            }

            state.delete_boundary(high_gain);
            boundary_count -= 1;
            state.part_weights[to] += high_gain_weight;
            state.where_part[high_gain] = to;
            swaps[swap_index] = high_gain;
            performed_swaps = swap_index + 1;

            for &neighbor in graph.neighbors(high_gain) {
                if state.where_part[neighbor] == 2 {
                    state.edegrees[neighbor][to] += high_gain_weight;
                } else if state.where_part[neighbor] == other {
                    let neighbor_weight = graph.vertex_weight(neighbor);
                    debug_assert_eq!(state.boundary_ptr[neighbor], usize::MAX);
                    state.insert_boundary(neighbor);
                    boundary_count += 1;
                    mind[nmind] = neighbor;
                    nmind += 1;
                    state.where_part[neighbor] = 2;
                    state.part_weights[other] -= neighbor_weight;

                    state.edegrees[neighbor] = [0; 2];
                    for &neighbor_neighbor in graph.neighbors(neighbor) {
                        if state.where_part[neighbor_neighbor] != 2 {
                            state.edegrees[neighbor][state.where_part[neighbor_neighbor]] +=
                                graph.vertex_weight(neighbor_neighbor);
                        } else {
                            state.edegrees[neighbor_neighbor][other] -= neighbor_weight;
                            queue.update(
                                neighbor_neighbor,
                                graph.vertex_weight(neighbor_neighbor)
                                    - state.edegrees[neighbor_neighbor][other],
                            );
                        }
                    }

                    queue.insert(neighbor, neighbor_weight - state.edegrees[neighbor][other]);
                }
            }
            mptr[swap_index + 1] = nmind;
        }

        for swap_index in ((mincut_order + 1).max(0) as usize..performed_swaps).rev() {
            let high_gain = swaps[swap_index];
            debug_assert_eq!(state.where_part[high_gain], to);
            let high_gain_weight = graph.vertex_weight(high_gain);
            state.part_weights[2] += high_gain_weight;
            state.part_weights[to] -= high_gain_weight;
            state.where_part[high_gain] = 2;
            state.insert_boundary(high_gain);
            boundary_count += 1;

            state.edegrees[high_gain] = [0; 2];
            for &neighbor in graph.neighbors(high_gain) {
                if state.where_part[neighbor] == 2 {
                    state.edegrees[neighbor][to] -= high_gain_weight;
                } else {
                    state.edegrees[high_gain][state.where_part[neighbor]] +=
                        graph.vertex_weight(neighbor);
                }
            }

            for &vertex in mind
                .iter()
                .take(mptr[swap_index + 1])
                .skip(mptr[swap_index])
            {
                let vertex_weight = graph.vertex_weight(vertex);
                debug_assert_eq!(state.where_part[vertex], 2);
                state.where_part[vertex] = other;
                state.part_weights[other] += vertex_weight;
                state.part_weights[2] -= vertex_weight;
                state.delete_boundary(vertex);
                boundary_count -= 1;
                for &neighbor in graph.neighbors(vertex) {
                    if state.where_part[neighbor] == 2 {
                        state.edegrees[neighbor][other] += vertex_weight;
                    }
                }
            }
        }

        state.mincut = mincut;
        debug_assert_eq!(boundary_count, state.boundary.len());

        if pass % 2 == 1 && (mincut_order == -1 || mincut >= init_cut) {
            break;
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PriorityEntry {
    key: isize,
    value: usize,
}

#[derive(Clone, Debug)]
struct SourcePriorityQueue {
    heap: Vec<PriorityEntry>,
    locator: Vec<usize>,
}

impl SourcePriorityQueue {
    fn new(max_nodes: usize) -> Self {
        Self {
            heap: Vec::new(),
            locator: vec![usize::MAX; max_nodes],
        }
    }

    fn insert(&mut self, node: usize, key: isize) {
        debug_assert_eq!(self.locator[node], usize::MAX);
        let mut position = self.heap.len();
        self.heap.push(PriorityEntry { key, value: node });
        while position > 0 {
            let parent = (position - 1) >> 1;
            if key > self.heap[parent].key {
                self.heap[position] = self.heap[parent];
                self.locator[self.heap[position].value] = position;
                position = parent;
            } else {
                break;
            }
        }
        self.heap[position] = PriorityEntry { key, value: node };
        self.locator[node] = position;
    }

    fn delete(&mut self, node: usize) {
        let position = self.locator[node];
        debug_assert_ne!(position, usize::MAX);
        self.locator[node] = usize::MAX;
        let Some(last) = self.heap.pop() else {
            return;
        };
        if position < self.heap.len() && last.value != node {
            self.filter_replacement(position, last);
        }
    }

    fn update(&mut self, node: usize, new_key: isize) {
        let position = self.locator[node];
        debug_assert_ne!(position, usize::MAX);
        let old_key = self.heap[position].key;
        if new_key == old_key {
            return;
        }
        self.filter_replacement(
            position,
            PriorityEntry {
                key: new_key,
                value: node,
            },
        );
    }

    fn get_top(&mut self) -> Option<usize> {
        if self.heap.is_empty() {
            return None;
        }
        let top = self.heap[0].value;
        self.locator[top] = usize::MAX;
        let last = self.heap.pop().expect("nonempty heap");
        if !self.heap.is_empty() {
            self.filter_replacement(0, last);
        }
        Some(top)
    }

    fn see_top_val(&self) -> Option<usize> {
        self.heap.first().map(|entry| entry.value)
    }

    fn filter_replacement(&mut self, mut position: usize, entry: PriorityEntry) {
        if position > 0 && entry.key > self.heap[position].key {
            while position > 0 {
                let parent = (position - 1) >> 1;
                if entry.key > self.heap[parent].key {
                    self.heap[position] = self.heap[parent];
                    self.locator[self.heap[position].value] = position;
                    position = parent;
                } else {
                    break;
                }
            }
        } else {
            let len = self.heap.len();
            while 2 * position + 1 < len {
                let mut child = 2 * position + 1;
                if self.heap[child].key > entry.key {
                    if child + 1 < len && self.heap[child + 1].key > self.heap[child].key {
                        child += 1;
                    }
                    self.heap[position] = self.heap[child];
                    self.locator[self.heap[position].value] = position;
                    position = child;
                } else if child + 1 < len && self.heap[child + 1].key > entry.key {
                    child += 1;
                    self.heap[position] = self.heap[child];
                    self.locator[self.heap[position].value] = position;
                    position = child;
                } else {
                    break;
                }
            }
        }
        self.heap[position] = entry;
        self.locator[entry.value] = position;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_permutation(summary: &OrderingSummary, n: usize) {
        assert_eq!(summary.permutation.len(), n);
        let mut seen = summary.permutation.perm().to_vec();
        seen.sort_unstable();
        assert_eq!(seen, (0..n).collect::<Vec<_>>());
    }

    fn lower_csc_pattern_from_edges(
        dimension: usize,
        edges: &[(usize, usize)],
    ) -> (Vec<usize>, Vec<usize>) {
        let mut cols = vec![Vec::new(); dimension];
        for (col, col_rows) in cols.iter_mut().enumerate() {
            col_rows.push(col);
        }
        for &(lhs, rhs) in edges {
            let (col, row) = if lhs <= rhs { (lhs, rhs) } else { (rhs, lhs) };
            cols[col].push(row);
        }
        let mut col_ptrs = Vec::with_capacity(dimension + 1);
        let mut row_indices = Vec::new();
        col_ptrs.push(0);
        for col_rows in &mut cols {
            col_rows.sort_unstable();
            col_rows.dedup();
            row_indices.extend(col_rows.iter().copied());
            col_ptrs.push(row_indices.len());
        }
        (col_ptrs, row_indices)
    }

    fn twin_path_edges(groups: usize) -> Vec<(usize, usize)> {
        let mut edges = Vec::with_capacity(groups + 4 * groups.saturating_sub(1));
        for group in 0..groups {
            let base = 2 * group;
            edges.push((base, base + 1));
            if group + 1 < groups {
                let next = base + 2;
                edges.push((base, next));
                edges.push((base, next + 1));
                edges.push((base + 1, next));
                edges.push((base + 1, next + 1));
            }
        }
        edges
    }

    #[test]
    fn metis_node_nd_handles_single_vertex() {
        let graph = CsrGraph::from_edges(1, &[]).unwrap();
        let summary = metis_node_nd_order(&graph).unwrap();
        assert_eq!(summary.permutation.perm(), &[0]);
    }

    #[test]
    fn metis_node_nd_orders_path_without_changing_existing_apis() {
        let graph = CsrGraph::from_edges(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]).unwrap();
        let summary = metis_node_nd_order(&graph).unwrap();
        assert_permutation(&summary, 5);
        let explicit =
            metis_node_nd_order_with_options(&graph, MetisNodeNdOptions::spral_default()).unwrap();
        assert_eq!(summary, explicit);
        let amd = approximate_minimum_degree_order(&graph).unwrap();
        assert_permutation(&amd, 5);
    }

    #[test]
    fn metis_node_nd_options_can_force_no_compression() {
        let dimension = 69;
        let mut col_ptrs = Vec::with_capacity(dimension + 1);
        let mut row_indices = Vec::new();
        col_ptrs.push(0);
        for col in 0..dimension {
            row_indices.extend(col..dimension);
            col_ptrs.push(row_indices.len());
        }
        let default_summary =
            metis_node_nd_order_from_lower_csc(dimension, &col_ptrs, &row_indices).unwrap();
        let no_compress = metis_node_nd_order_from_lower_csc_with_options(
            dimension,
            &col_ptrs,
            &row_indices,
            MetisNodeNdOptions {
                compress: false,
                ..MetisNodeNdOptions::spral_default()
            },
        )
        .unwrap();
        assert_permutation(&default_summary, dimension);
        assert_permutation(&no_compress, dimension);
        assert_ne!(default_summary.permutation, no_compress.permutation);
    }

    #[test]
    fn metis_node_nd_prune_trace_covers_no_partial_and_all_pruned() {
        let star_edges = (1..10).map(|leaf| (0, leaf)).collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(10, &star_edges);

        let no_prune = metis_debug_prune_from_lower_csc(10, &col_ptrs, &row_indices, 100).unwrap();
        assert!(!no_prune.pruning_active);
        assert_eq!(no_prune.kept_vertex_count, 10);
        assert_eq!(
            no_prune.offsets,
            &[0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        );

        let partial = metis_debug_prune_from_lower_csc(10, &col_ptrs, &row_indices, 20).unwrap();
        assert!(partial.pruning_active);
        assert_eq!(partial.kept_vertex_count, 9);
        assert_eq!(partial.piperm, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]);
        assert_eq!(partial.offsets, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert!(partial.neighbors.is_empty());
        assert_eq!(partial.vertex_weights, &[1; 9]);

        let all_pruned = metis_debug_prune_from_lower_csc(10, &col_ptrs, &row_indices, 1).unwrap();
        assert!(!all_pruned.pruning_active);
        assert_eq!(all_pruned.kept_vertex_count, 10);
    }

    #[test]
    fn metis_node_nd_orders_empty_three_fixture() {
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(3, &[]);
        let summary = metis_node_nd_order_from_lower_csc(3, &col_ptrs, &row_indices).unwrap();
        assert_eq!(summary.permutation.inverse(), &[2, 0, 1]);
        assert_eq!(summary.permutation.perm(), &[1, 2, 0]);
    }

    #[test]
    fn metis_node_nd_compression_expands_duplicate_isolates() {
        let graph = CsrGraph::from_edges(3, &[(0, 1)]).unwrap();
        let summary = metis_node_nd_order(&graph).unwrap();
        assert_permutation(&summary, 3);
        assert!(summary.permutation.perm().contains(&2));
    }

    #[test]
    fn spral_half_to_full_drop_diag_preserves_source_row_order() {
        let col_ptrs = [0, 3, 5, 6];
        let row_indices = [0, 1, 2, 1, 2, 2];
        let graph = spral_half_to_full_drop_diag(3, &col_ptrs, &row_indices).unwrap();
        assert_eq!(graph.offsets, vec![0, 2, 4, 6]);
        assert_eq!(graph.neighbors(0), &[2, 1]);
        assert_eq!(graph.neighbors(1), &[2, 0]);
        assert_eq!(graph.neighbors(2), &[1, 0]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metis_rng_matches_source_built_default_seed_sequence() {
        assert_eq!(
            metis_debug_irand_sequence(-1, 12),
            &[
                72_623_047,
                804_839_433,
                2_084_341_625,
                1_776_441_511,
                187_331_136,
                263_376_250,
                600_837_283,
                804_107_187,
                514_901_338,
                1_735_174_003,
                241_542_161,
                855_007_097,
            ]
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metis_irand_array_permute_matches_gklib_template_branches() {
        assert_eq!(
            metis_debug_irand_array_permute(-1, 6, 6, true),
            &[4, 0, 2, 5, 1, 3]
        );
        assert_eq!(
            metis_debug_irand_array_permute(-1, 12, 4, true),
            &[10, 5, 4, 0, 11, 9, 8, 3, 6, 1, 7, 2]
        );
        assert_eq!(
            metis_debug_irand_array_permute(123, 10, 5, false),
            &[7, 9, 2, 1, 5, 4, 3, 8, 6, 0]
        );
    }

    #[test]
    fn metis_l1_edge_bisection_matches_native_path_stage_fixture() {
        let edges = (0..5)
            .map(|vertex| (vertex, vertex + 1))
            .collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(6, &edges);
        let trace =
            metis_debug_l1_edge_bisection_from_lower_csc(6, &col_ptrs, &row_indices).unwrap();
        assert_eq!(trace.mincut, 1);
        assert_eq!(trace.part_weights, [3, 3]);
        assert_eq!(trace.where_part, &[1, 1, 1, 0, 0, 0]);
        assert_eq!(trace.boundary, &[2, 3]);
        assert_eq!(trace.internal_degree, &[1, 2, 1, 1, 2, 1]);
        assert_eq!(trace.external_degree, &[0, 0, 1, 1, 0, 0]);
    }

    #[test]
    fn metis_l1_edge_bisection_matches_native_star_stage_fixture() {
        let edges = (1..7).map(|leaf| (0, leaf)).collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(7, &edges);
        let trace =
            metis_debug_l1_edge_bisection_from_lower_csc(7, &col_ptrs, &row_indices).unwrap();
        assert_eq!(trace.mincut, 3);
        assert_eq!(trace.part_weights, [3, 4]);
        assert_eq!(trace.where_part, &[1, 0, 1, 0, 1, 1, 0]);
        assert_eq!(trace.boundary, &[0, 1, 3, 6]);
        assert_eq!(trace.internal_degree, &[3, 0, 1, 0, 1, 1, 0]);
        assert_eq!(trace.external_degree, &[3, 1, 0, 1, 0, 0, 1]);
    }

    #[test]
    fn metis_l1_construct_separator_matches_native_path_stage_fixture() {
        let edges = (0..5)
            .map(|vertex| (vertex, vertex + 1))
            .collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(6, &edges);
        let trace =
            metis_debug_l1_construct_separator_from_lower_csc(6, &col_ptrs, &row_indices).unwrap();
        assert_eq!(trace.mincut, 1);
        assert_eq!(trace.part_weights, [3, 2, 1]);
        assert_eq!(trace.where_part, &[1, 1, 2, 0, 0, 0]);
        assert_eq!(trace.boundary, &[2]);
    }

    #[test]
    fn metis_l1_construct_separator_matches_native_star_stage_fixture() {
        let edges = (1..7).map(|leaf| (0, leaf)).collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(7, &edges);
        let trace =
            metis_debug_l1_construct_separator_from_lower_csc(7, &col_ptrs, &row_indices).unwrap();
        assert_eq!(trace.mincut, 1);
        assert_eq!(trace.part_weights, [3, 3, 1]);
        assert_eq!(trace.where_part, &[2, 0, 1, 0, 1, 1, 0]);
        assert_eq!(trace.boundary, &[0]);
    }

    #[test]
    fn metis_l1_match_rm_coarsening_matches_native_path_54_fixture() {
        let edges = (0..53)
            .map(|vertex| (vertex, vertex + 1))
            .collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(54, &edges);
        let trace = metis_debug_l1_coarsen_from_lower_csc(54, &col_ptrs, &row_indices).unwrap();
        assert_eq!(trace.vertex_count, 30);
        assert_eq!(trace.directed_edge_count, 58);
        assert_eq!(
            trace.offsets,
            &[
                0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
                43, 45, 47, 49, 51, 53, 55, 57, 58,
            ]
        );
        assert_eq!(
            trace.neighbors,
            &[
                1, 2, 0, 3, 1, 4, 2, 5, 3, 4, 6, 7, 5, 6, 8, 9, 7, 10, 8, 11, 9, 12, 10, 13, 11,
                14, 12, 13, 15, 16, 14, 17, 15, 18, 16, 17, 19, 20, 18, 21, 19, 22, 20, 21, 23, 24,
                22, 25, 23, 26, 24, 27, 25, 28, 26, 29, 27, 28,
            ]
        );
        assert_eq!(
            trace.vertex_weights,
            &[
                2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2,
                2, 1
            ]
        );
    }

    #[test]
    fn metis_l1_edge_bisection_matches_native_path_54_match_rm_fixture() {
        let edges = (0..53)
            .map(|vertex| (vertex, vertex + 1))
            .collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(54, &edges);
        let trace =
            metis_debug_l1_edge_bisection_from_lower_csc(54, &col_ptrs, &row_indices).unwrap();
        assert_eq!(trace.mincut, 1);
        assert_eq!(trace.part_weights, [27, 27]);
        assert_eq!(
            trace.where_part,
            &[
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0,
            ]
        );
        assert_eq!(trace.boundary, &[14, 15]);
        assert_eq!(
            trace.internal_degree,
            &[
                1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 1
            ]
        );
        assert_eq!(
            trace.external_degree,
            &[
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0
            ]
        );
    }

    #[test]
    fn metis_l1_construct_separator_matches_native_path_54_match_rm_fixture() {
        let edges = (0..53)
            .map(|vertex| (vertex, vertex + 1))
            .collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(54, &edges);
        let trace =
            metis_debug_l1_construct_separator_from_lower_csc(54, &col_ptrs, &row_indices).unwrap();
        assert_eq!(trace.mincut, 1);
        assert_eq!(trace.part_weights, [27, 26, 1]);
        assert_eq!(
            trace.where_part,
            &[
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0,
            ]
        );
        assert_eq!(trace.boundary, &[14]);
    }

    #[test]
    fn metis_l1_projected_separator_matches_native_path_54_fixture() {
        let edges = (0..53)
            .map(|vertex| (vertex, vertex + 1))
            .collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(54, &edges);
        let trace =
            metis_debug_l1_projected_separator_from_lower_csc(54, &col_ptrs, &row_indices).unwrap();
        assert_eq!(trace.mincut, 1);
        assert_eq!(trace.part_weights, [27, 26, 1]);
        assert_eq!(
            trace.where_part,
            &[
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]
        );
        assert_eq!(trace.boundary, &[26]);
    }

    #[test]
    fn metis_node_nd_matches_native_path_one_level_fixture() {
        let edges = (0..5)
            .map(|vertex| (vertex, vertex + 1))
            .collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(6, &edges);
        let summary = metis_node_nd_order_from_lower_csc(6, &col_ptrs, &row_indices).unwrap();
        assert_eq!(summary.permutation.inverse(), &[4, 3, 5, 1, 2, 0]);
        assert_eq!(summary.permutation.perm(), &[5, 3, 4, 1, 0, 2]);
    }

    #[test]
    fn metis_node_nd_matches_native_star_one_level_fixture() {
        let edges = (1..7).map(|leaf| (0, leaf)).collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(7, &edges);
        let summary = metis_node_nd_order_from_lower_csc(7, &col_ptrs, &row_indices).unwrap();
        assert_eq!(summary.permutation.inverse(), &[6, 2, 5, 1, 4, 3, 0]);
        assert_eq!(summary.permutation.perm(), &[6, 3, 1, 5, 4, 2, 0]);
    }

    #[test]
    fn metis_node_nd_matches_native_path_54_projected_fixture() {
        let edges = (0..53)
            .map(|vertex| (vertex, vertex + 1))
            .collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(54, &edges);
        let summary = metis_node_nd_order_from_lower_csc(54, &col_ptrs, &row_indices).unwrap();
        assert_eq!(
            summary.permutation.inverse(),
            &[
                28, 40, 48, 39, 46, 38, 50, 37, 45, 36, 52, 35, 44, 34, 49, 33, 43, 32, 51, 31, 42,
                30, 47, 29, 41, 27, 53, 1, 15, 13, 22, 12, 20, 11, 24, 10, 19, 9, 26, 8, 18, 7, 23,
                6, 17, 5, 25, 4, 16, 3, 21, 2, 14, 0,
            ]
        );
        assert_eq!(
            summary.permutation.perm(),
            &[
                53, 27, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 52, 28, 48, 44, 40, 36, 32,
                50, 30, 42, 34, 46, 38, 25, 0, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 24, 20,
                16, 12, 8, 4, 22, 2, 14, 6, 18, 10, 26,
            ]
        );
    }

    #[test]
    fn metis_node_nd_matches_native_path_121_projected_fixture() {
        let edges = (0..120)
            .map(|vertex| (vertex, vertex + 1))
            .collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(121, &edges);
        let summary = metis_node_nd_order_from_lower_csc(121, &col_ptrs, &row_indices).unwrap();
        assert_eq!(
            summary.permutation.inverse(),
            &[
                61, 90, 105, 89, 114, 88, 104, 87, 112, 86, 103, 85, 118, 84, 102, 83, 111, 82,
                101, 81, 116, 80, 100, 79, 110, 78, 99, 77, 119, 76, 98, 75, 109, 74, 97, 73, 115,
                72, 96, 71, 108, 70, 95, 69, 117, 68, 94, 67, 107, 66, 93, 65, 113, 64, 92, 63,
                106, 62, 91, 60, 120, 1, 30, 45, 29, 54, 28, 44, 27, 52, 26, 43, 25, 58, 24, 42,
                23, 51, 22, 41, 21, 56, 20, 40, 19, 50, 18, 39, 17, 59, 16, 38, 15, 49, 14, 37, 13,
                55, 12, 36, 11, 48, 10, 35, 9, 57, 8, 34, 7, 47, 6, 33, 5, 53, 4, 32, 3, 46, 2, 31,
                0,
            ]
        );
        assert_eq!(
            summary.permutation.perm(),
            &[
                120, 61, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90, 88,
                86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 119, 115, 111, 107, 103, 99,
                95, 91, 87, 83, 79, 75, 71, 67, 63, 117, 109, 101, 93, 85, 77, 69, 113, 65, 97, 81,
                105, 73, 89, 59, 0, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27,
                25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 58, 54, 50, 46, 42, 38, 34, 30, 26,
                22, 18, 14, 10, 6, 2, 56, 48, 40, 32, 24, 16, 8, 52, 4, 36, 20, 44, 12, 28, 60,
            ]
        );
    }

    #[test]
    fn metis_node_nd_orders_path_300_recursive_fixture() {
        let edges = (0..299)
            .map(|vertex| (vertex, vertex + 1))
            .collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(300, &edges);
        let summary = metis_node_nd_order_from_lower_csc(300, &col_ptrs, &row_indices).unwrap();
        assert_permutation(&summary, 300);
        assert!(summary.stats.separator_calls >= 2);
    }

    #[test]
    fn metis_node_nd_orders_path_1000_multilevel_recursive_fixture() {
        let edges = (0..999)
            .map(|vertex| (vertex, vertex + 1))
            .collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(1000, &edges);
        let summary = metis_node_nd_order_from_lower_csc(1000, &col_ptrs, &row_indices).unwrap();
        assert_permutation(&summary, 1000);
        assert!(summary.stats.separator_calls >= 4);
    }

    #[test]
    fn metis_node_nd_orders_twin_path_2400_compression_retry_fixture() {
        let edges = twin_path_edges(1200);
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(2400, &edges);
        let summary = metis_node_nd_order_from_lower_csc(2400, &col_ptrs, &row_indices).unwrap();
        assert_permutation(&summary, 2400);
        assert!(summary.stats.separator_calls >= 1);
    }

    #[test]
    fn metis_node_nd_orders_path_5000_l2_fixture() {
        let edges = (0..4999)
            .map(|vertex| (vertex, vertex + 1))
            .collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(5000, &edges);
        let summary = metis_node_nd_order_from_lower_csc(5000, &col_ptrs, &row_indices).unwrap();
        assert_permutation(&summary, 5000);
        assert!(summary.stats.separator_calls >= 1);
    }

    #[test]
    fn metis_node_nd_compresses_complete_graph_with_gklib_tie_order() {
        let dimension = 69;
        let mut col_ptrs = Vec::with_capacity(dimension + 1);
        let mut row_indices = Vec::new();
        col_ptrs.push(0);
        for col in 0..dimension {
            row_indices.extend(col..dimension);
            col_ptrs.push(row_indices.len());
        }
        let summary =
            metis_node_nd_order_from_lower_csc(dimension, &col_ptrs, &row_indices).unwrap();
        assert_eq!(
            summary.permutation.perm(),
            &[
                0, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 51, 52, 67, 66, 65,
                64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 35, 34, 33, 15, 14, 13, 12, 11, 10,
                9, 8, 7, 6, 5, 4, 3, 2, 1, 16, 17, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21,
                20, 19, 18, 68
            ]
        );
    }

    #[test]
    fn metis_node_nd_from_lower_csc_drops_diagonal() {
        let col_ptrs = [0, 2, 4, 5];
        let row_indices = [0, 1, 1, 2, 2];
        let summary = metis_node_nd_order_from_lower_csc(3, &col_ptrs, &row_indices).unwrap();
        assert_permutation(&summary, 3);
    }

    #[test]
    fn metis_mmd_order_matches_native_genmmd_path_fixture() {
        let edges = (0..5)
            .map(|vertex| (vertex, vertex + 1))
            .collect::<Vec<_>>();
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(6, &edges);
        let summary = metis_mmd_order_from_lower_csc(6, &col_ptrs, &row_indices).unwrap();
        assert_eq!(summary.permutation.perm(), &[5, 0, 3, 1, 4, 2]);
        assert_eq!(summary.permutation.inverse(), &[1, 3, 5, 2, 4, 0]);
    }

    #[test]
    fn metis_mmd_order_matches_native_genmmd_empty_fixture() {
        let (col_ptrs, row_indices) = lower_csc_pattern_from_edges(5, &[]);
        let summary = metis_mmd_order_from_lower_csc(5, &col_ptrs, &row_indices).unwrap();
        assert_eq!(summary.permutation.perm(), &[4, 3, 2, 1, 0]);
        assert_eq!(summary.permutation.inverse(), &[4, 3, 2, 1, 0]);
    }
}
