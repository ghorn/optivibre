use std::collections::BTreeSet;
use std::mem::size_of;

use anyhow::{Result, bail};
use metis_ordering::{CsrGraph, Permutation};

use crate::corpus::SymmetricPatternMatrix;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactSymbolicMetrics {
    pub fill_nnz: usize,
    pub column_counts: Vec<usize>,
    pub elimination_tree: Vec<Option<usize>>,
    pub column_pattern: Vec<Vec<usize>>,
    pub etree_height: usize,
    pub memory_bytes: usize,
}

pub fn permutation_is_valid(permutation: &Permutation, expected_len: usize) -> bool {
    permutation.len() == expected_len
        && permutation
            .perm()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            == expected_len
}

pub fn connected_component_count(graph: &CsrGraph) -> usize {
    let mut visited = vec![false; graph.vertex_count()];
    let mut components = 0;
    for start in 0..graph.vertex_count() {
        if visited[start] {
            continue;
        }
        components += 1;
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(vertex) = stack.pop() {
            for &neighbor in graph.neighbors(vertex) {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
    }
    components
}

pub fn exact_symbolic_metrics(
    matrix: &SymmetricPatternMatrix,
    permutation: &Permutation,
) -> Result<ExactSymbolicMetrics> {
    if matrix.dimension() != permutation.len() {
        bail!(
            "matrix dimension {} does not match permutation length {}",
            matrix.dimension(),
            permutation.len()
        );
    }
    let graph = matrix.to_graph()?;
    let mut adjacency = vec![BTreeSet::new(); graph.vertex_count()];
    for original_vertex in 0..graph.vertex_count() {
        let permuted_vertex = permutation.inverse()[original_vertex];
        for &original_neighbor in graph.neighbors(original_vertex) {
            let permuted_neighbor = permutation.inverse()[original_neighbor];
            adjacency[permuted_vertex].insert(permuted_neighbor);
        }
    }

    let mut column_counts = vec![1; graph.vertex_count()];
    let mut elimination_tree = vec![None; graph.vertex_count()];
    let mut column_pattern = vec![Vec::new(); graph.vertex_count()];
    for column in 0..graph.vertex_count() {
        let active_neighbors = adjacency[column]
            .iter()
            .copied()
            .filter(|&neighbor| neighbor > column)
            .collect::<Vec<_>>();
        elimination_tree[column] = active_neighbors.first().copied();
        column_counts[column] = active_neighbors.len() + 1;
        column_pattern[column].push(column);
        column_pattern[column].extend(active_neighbors.iter().copied());

        for index in 0..active_neighbors.len() {
            for jdx in (index + 1)..active_neighbors.len() {
                let lhs = active_neighbors[index];
                let rhs = active_neighbors[jdx];
                adjacency[lhs].insert(rhs);
                adjacency[rhs].insert(lhs);
            }
        }
    }

    let fill_nnz = column_counts.iter().sum();
    let etree_height = elimination_tree_height(&elimination_tree);
    let memory_bytes = symbolic_memory_bytes(graph.vertex_count(), fill_nnz);
    Ok(ExactSymbolicMetrics {
        fill_nnz,
        column_counts,
        elimination_tree,
        column_pattern,
        etree_height,
        memory_bytes,
    })
}

pub fn elimination_tree_height(tree: &[Option<usize>]) -> usize {
    let mut max_height = 0;
    for start in 0..tree.len() {
        let mut current = Some(start);
        let mut height = 0;
        while let Some(node) = current {
            height += 1;
            current = tree[node];
        }
        max_height = max_height.max(height);
    }
    max_height
}

pub fn symbolic_memory_bytes(dimension: usize, fill_nnz: usize) -> usize {
    (dimension + 1 + fill_nnz) * size_of::<usize>() + fill_nnz * size_of::<f64>()
}

pub fn tree_parent_validity(tree: &[Option<usize>]) -> bool {
    tree.iter()
        .enumerate()
        .all(|(index, parent)| parent.is_none_or(|parent| parent > index && parent < tree.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::SymmetricPatternMatrix;
    use metis_ordering::Permutation;

    #[test]
    fn exact_symbolic_metrics_match_path_structure() {
        let matrix = SymmetricPatternMatrix::from_undirected_edges(4, &[(0, 1), (1, 2), (2, 3)])
            .expect("path matrix");
        let metrics = exact_symbolic_metrics(&matrix, &Permutation::identity(4)).expect("metrics");
        assert_eq!(metrics.fill_nnz, 7);
        assert_eq!(metrics.column_counts, vec![2, 2, 2, 1]);
        assert!(tree_parent_validity(&metrics.elimination_tree));
    }
}
