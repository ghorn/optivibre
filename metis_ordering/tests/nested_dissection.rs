use std::collections::BTreeSet;

use metis_ordering::{
    CsrGraph, NestedDissectionOptions, Permutation, approximate_minimum_degree_order,
    nested_dissection_order,
};

fn path_graph(size: usize) -> CsrGraph {
    let edges = (0..size.saturating_sub(1))
        .map(|idx| (idx, idx + 1))
        .collect::<Vec<_>>();
    CsrGraph::from_edges(size, &edges).expect("path graph")
}

fn grid_graph(rows: usize, cols: usize) -> CsrGraph {
    let mut edges = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let index = row * cols + col;
            if row + 1 < rows {
                edges.push((index, index + cols));
            }
            if col + 1 < cols {
                edges.push((index, index + 1));
            }
        }
    }
    CsrGraph::from_edges(rows * cols, &edges).expect("grid graph")
}

fn fill_proxy(graph: &CsrGraph, permutation: &Permutation) -> usize {
    let mut adjacency = vec![BTreeSet::new(); graph.vertex_count()];
    for vertex in 0..graph.vertex_count() {
        let permuted_vertex = permutation.inverse()[vertex];
        for &neighbor in graph.neighbors(vertex) {
            let permuted_neighbor = permutation.inverse()[neighbor];
            adjacency[permuted_vertex].insert(permuted_neighbor);
        }
    }

    let mut fill_nnz = 0;
    for column in 0..graph.vertex_count() {
        let active = adjacency[column]
            .iter()
            .copied()
            .filter(|&neighbor| neighbor > column)
            .collect::<Vec<_>>();
        fill_nnz += active.len() + 1;
        for index in 0..active.len() {
            for jdx in (index + 1)..active.len() {
                let lhs = active[index];
                let rhs = active[jdx];
                adjacency[lhs].insert(rhs);
                adjacency[rhs].insert(lhs);
            }
        }
    }
    fill_nnz
}

#[test]
fn symmetric_csc_graph_builder_deduplicates_structure() {
    let graph = CsrGraph::from_symmetric_csc(4, &[0, 2, 4, 6, 7], &[0, 1, 0, 2, 1, 3, 2])
        .expect("valid symmetric graph");
    assert_eq!(graph.vertex_count(), 4);
    assert_eq!(graph.edge_count(), 3);
    assert_eq!(graph.neighbors(0), &[1]);
    assert_eq!(graph.neighbors(1), &[0, 2]);
    assert_eq!(graph.neighbors(2), &[1, 3]);
    assert_eq!(graph.neighbors(3), &[2]);
}

#[test]
fn nested_dissection_returns_valid_permutation_on_path_graph() {
    let summary = nested_dissection_order(
        &path_graph(17),
        &NestedDissectionOptions {
            leaf_size: 4,
            ..NestedDissectionOptions::default()
        },
    )
    .expect("ordering succeeds");
    let permutation = Permutation::new(summary.permutation.perm().to_vec()).expect("valid");
    assert_eq!(permutation.len(), 17);
    assert!(summary.stats.separator_calls > 0);
    assert!(summary.stats.leaf_calls > 0);
}

#[test]
fn disconnected_components_are_ordered_without_failure() {
    let graph = CsrGraph::from_edges(6, &[(0, 1), (1, 2), (3, 4)]).expect("valid graph");
    let summary = nested_dissection_order(
        &graph,
        &NestedDissectionOptions {
            leaf_size: 2,
            ..NestedDissectionOptions::default()
        },
    )
    .expect("ordering succeeds");
    assert_eq!(summary.permutation.len(), 6);
    assert!(summary.stats.connected_components > 0);
}

#[test]
fn approximate_minimum_degree_returns_valid_fill_reducing_order() {
    let graph = grid_graph(4, 4);
    let summary = approximate_minimum_degree_order(&graph).expect("ordering succeeds");
    assert_eq!(summary.permutation.len(), 16);
    let natural = Permutation::identity(16);
    assert!(fill_proxy(&graph, &summary.permutation) <= fill_proxy(&graph, &natural));
}
