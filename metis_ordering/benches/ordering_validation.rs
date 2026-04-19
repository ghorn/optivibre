use std::collections::BTreeSet;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use metis_ordering::{CsrGraph, NestedDissectionOptions, Permutation, nested_dissection_order};

fn path_edges(size: usize) -> Vec<(usize, usize)> {
    (0..size.saturating_sub(1))
        .map(|index| (index, index + 1))
        .collect()
}

fn grid_edges(rows: usize, cols: usize) -> Vec<(usize, usize)> {
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
    edges
}

fn ladder_edges(rungs: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for rung in 0..rungs {
        let top = rung;
        let bottom = rung + rungs;
        edges.push((top, bottom));
        if rung + 1 < rungs {
            edges.push((top, top + 1));
            edges.push((bottom, bottom + 1));
        }
    }
    edges
}

fn exact_fill_proxy(graph: &CsrGraph, permutation: &Permutation) -> usize {
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

fn ordering_graphs() -> Vec<(&'static str, CsrGraph)> {
    vec![
        (
            "path_256",
            CsrGraph::from_edges(256, &path_edges(256)).expect("path graph"),
        ),
        (
            "grid_20x20",
            CsrGraph::from_edges(400, &grid_edges(20, 20)).expect("grid graph"),
        ),
        (
            "ladder_160",
            CsrGraph::from_edges(160, &ladder_edges(80)).expect("ladder graph"),
        ),
    ]
}

fn graph_ingestion_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("metis_ordering/graph_ingestion");
    group.bench_function("grid_24x24_from_edges", |bench| {
        let edges = grid_edges(24, 24);
        bench.iter_batched(
            || edges.clone(),
            |edges| CsrGraph::from_edges(24 * 24, &edges).expect("graph"),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn nested_dissection_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("metis_ordering/nested_dissection");
    let options = NestedDissectionOptions {
        leaf_size: 16,
        ..NestedDissectionOptions::default()
    };
    for (name, graph) in ordering_graphs() {
        group.bench_with_input(BenchmarkId::from_parameter(name), &graph, |bench, graph| {
            bench.iter(|| nested_dissection_order(graph, &options).expect("ordering"));
        });
    }
    group.finish();
}

fn fill_proxy_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("metis_ordering/fill_proxy");
    let options = NestedDissectionOptions {
        leaf_size: 12,
        ..NestedDissectionOptions::default()
    };
    for (name, graph) in ordering_graphs() {
        let summary = nested_dissection_order(&graph, &options).expect("ordering");
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &summary.permutation,
            |bench, permutation| {
                bench.iter(|| exact_fill_proxy(&graph, permutation));
            },
        );
    }
    group.finish();
}

criterion_group!(
    ordering_validation,
    graph_ingestion_bench,
    nested_dissection_bench,
    fill_proxy_bench
);
criterion_main!(ordering_validation);
