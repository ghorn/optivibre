#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let vertex_count = usize::from((data[0] % 31) + 1);
    let leaf_size = usize::from((data[1] % 15) + 1);
    let edges = data[2..]
        .chunks_exact(2)
        .filter_map(|pair| {
            let lhs = usize::from(pair[0]) % vertex_count;
            let rhs = usize::from(pair[1]) % vertex_count;
            (lhs != rhs).then_some((lhs, rhs))
        })
        .collect::<Vec<_>>();
    if let Ok(graph) = metis_ordering::CsrGraph::from_edges(vertex_count, &edges) {
        let _ = metis_ordering::nested_dissection_order(
            &graph,
            &metis_ordering::NestedDissectionOptions {
                leaf_size,
                ..metis_ordering::NestedDissectionOptions::default()
            },
        );
    }
});
