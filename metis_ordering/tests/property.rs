use std::collections::BTreeSet;

use metis_ordering::{CsrGraph, NestedDissectionOptions, nested_dissection_order};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRng, TestRunner};

fn deterministic_runner(cases: u32) -> TestRunner {
    TestRunner::new_with_rng(
        Config {
            cases,
            failure_persistence: None,
            rng_seed: RngSeed::Fixed(0x5eed_1234),
            ..Config::default()
        },
        TestRng::deterministic_rng(RngAlgorithm::ChaCha),
    )
}

#[test]
fn property_nested_dissection_returns_valid_permutations() {
    let strategy = (
        1_usize..24,
        vec((0_usize..48, 0_usize..48), 0..96),
        1_usize..8,
    );
    let mut runner = deterministic_runner(96);
    runner
        .run(&strategy, |(dimension, raw_edges, leaf_size)| {
            let edges = raw_edges
                .into_iter()
                .filter_map(|(lhs, rhs)| {
                    let lhs = lhs % dimension;
                    let rhs = rhs % dimension;
                    (lhs != rhs).then_some((lhs, rhs))
                })
                .collect::<Vec<_>>();
            let graph = CsrGraph::from_edges(dimension, &edges).expect("valid generated graph");
            let summary = nested_dissection_order(
                &graph,
                &NestedDissectionOptions {
                    leaf_size: leaf_size.min(dimension.max(1)),
                    ..NestedDissectionOptions::default()
                },
            )
            .expect("ordering should succeed");
            let unique = summary
                .permutation
                .perm()
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();

            prop_assert_eq!(summary.permutation.len(), dimension);
            prop_assert_eq!(unique.len(), dimension);
            prop_assert!(summary.stats.max_separator_size <= dimension);
            prop_assert!(summary.stats.separator_vertices <= dimension * dimension);
            prop_assert!(summary.stats.separator_calls + summary.stats.leaf_calls > 0);
            Ok(())
        })
        .expect("property should hold");
}

#[test]
fn property_malformed_csr_inputs_are_rejected() {
    let strategy = prop_oneof![
        Just((Vec::<usize>::new(), Vec::<usize>::new())),
        Just((vec![1, 1], Vec::<usize>::new())),
        Just((vec![0, 2, 1], vec![0, 1])),
        Just((vec![0, 1], vec![0])),
        Just((vec![0, 1], vec![2])),
        (1_usize..8).prop_map(|len| {
            let mut offsets = vec![0, 2];
            offsets.resize(len + 1, 2);
            (offsets, vec![1, 0])
        }),
    ];

    let mut runner = deterministic_runner(48);
    runner
        .run(&strategy, |(offsets, neighbors)| {
            prop_assert!(CsrGraph::new(offsets, neighbors).is_err());
            Ok(())
        })
        .expect("malformed CSR should be rejected");
}
