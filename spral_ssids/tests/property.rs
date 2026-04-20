use std::collections::BTreeSet;

use metis_ordering::NestedDissectionOptions;
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::test_runner::{Config, RngAlgorithm, RngSeed, TestRng, TestRunner};
use spral_ssids::{
    NumericFactorOptions, OrderingStrategy, SsidsOptions, SymmetricCscMatrix, analyse, factorize,
};

fn deterministic_runner(cases: u32) -> TestRunner {
    TestRunner::new_with_rng(
        Config {
            cases,
            failure_persistence: None,
            rng_seed: RngSeed::Fixed(0x5eed_5678),
            ..Config::default()
        },
        TestRng::deterministic_rng(RngAlgorithm::ChaCha),
    )
}

fn symmetric_lower_csc(dimension: usize, entries: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
    let mut columns = vec![BTreeSet::new(); dimension];
    for (index, rows) in columns.iter_mut().enumerate() {
        rows.insert(index);
    }
    for &(lhs, rhs) in entries {
        let row = lhs.max(rhs);
        let col = lhs.min(rhs);
        columns[col].insert(row);
    }

    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for rows in columns {
        row_indices.extend(rows);
        col_ptrs.push(row_indices.len());
    }
    (col_ptrs, row_indices)
}

fn weighted_lower_csc(
    dimension: usize,
    entries: &[(usize, usize, f64)],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut columns = vec![BTreeSet::new(); dimension];
    let mut weights = vec![std::collections::BTreeMap::new(); dimension];
    for (index, rows) in columns.iter_mut().enumerate() {
        rows.insert(index);
        weights[index].insert(index, 1.0);
    }
    for &(lhs, rhs, value) in entries {
        let row = lhs.max(rhs);
        let col = lhs.min(rhs);
        columns[col].insert(row);
        weights[col].insert(row, value);
    }

    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for column in 0..dimension {
        for row in &columns[column] {
            row_indices.push(*row);
            values.push(*weights[column].get(row).expect("weight"));
        }
        col_ptrs.push(row_indices.len());
    }
    (col_ptrs, row_indices, values)
}

fn tree_parent_valid(tree: &[Option<usize>]) -> bool {
    tree.iter()
        .enumerate()
        .all(|(index, parent)| parent.is_none_or(|parent| parent > index && parent < tree.len()))
}

#[test]
fn property_symbolic_analysis_matches_structural_invariants() {
    let strategy = (1_usize..20, vec((0_usize..40, 0_usize..40), 0..96));
    let mut runner = deterministic_runner(96);
    runner
        .run(&strategy, |(dimension, raw_entries)| {
            let entries = raw_entries
                .into_iter()
                .filter_map(|(lhs, rhs)| {
                    let lhs = lhs % dimension;
                    let rhs = rhs % dimension;
                    (lhs != rhs).then_some((lhs, rhs))
                })
                .collect::<Vec<_>>();
            let (col_ptrs, row_indices) = symmetric_lower_csc(dimension, &entries);
            let matrix =
                SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, None).expect("matrix");

            for ordering in [
                OrderingStrategy::Natural,
                OrderingStrategy::ApproximateMinimumDegree,
                OrderingStrategy::NestedDissection(NestedDissectionOptions::default()),
            ] {
                let (symbolic, info) =
                    analyse(matrix, &SsidsOptions { ordering }).expect("analysis succeeds");
                prop_assert_eq!(symbolic.permutation.len(), dimension);
                prop_assert_eq!(symbolic.elimination_tree.len(), dimension);
                prop_assert_eq!(symbolic.column_counts.len(), dimension);
                prop_assert_eq!(symbolic.column_pattern.len(), dimension);
                prop_assert!(tree_parent_valid(&symbolic.elimination_tree));
                prop_assert_eq!(
                    symbolic.column_counts.iter().sum::<usize>(),
                    info.estimated_fill_nnz
                );
                for (column, pattern) in symbolic.column_pattern.iter().enumerate() {
                    prop_assert_eq!(pattern.first().copied(), Some(column));
                    prop_assert_eq!(pattern.len(), symbolic.column_counts[column]);
                    prop_assert!(pattern.windows(2).all(|window| window[0] < window[1]));
                }
            }
            Ok(())
        })
        .expect("symbolic invariants should hold");
}

#[test]
fn property_numeric_factorization_solves_diagonally_dominant_systems() {
    let strategy = (
        1_usize..12,
        vec((0_usize..40, 0_usize..40, -50_i32..50), 0..72),
    );
    let mut runner = deterministic_runner(48);
    runner
        .run(&strategy, |(dimension, raw_entries)| {
            let filtered = raw_entries
                .into_iter()
                .filter_map(|(lhs, rhs, raw_weight)| {
                    let lhs = lhs % dimension;
                    let rhs = rhs % dimension;
                    let weight = f64::from(raw_weight) / 200.0;
                    (lhs != rhs && weight.abs() > 1e-6).then_some((lhs, rhs, weight))
                })
                .collect::<Vec<_>>();

            let mut row_abs_sums = vec![0.0; dimension];
            for &(lhs, rhs, weight) in &filtered {
                row_abs_sums[lhs] += weight.abs();
                row_abs_sums[rhs] += weight.abs();
            }
            let mut weighted_entries = filtered;
            for (index, row_sum) in row_abs_sums.iter().copied().enumerate() {
                let diagonal = row_sum + 1.0 + (index as f64 * 0.05);
                weighted_entries.push((index, index, diagonal));
            }

            let (col_ptrs, row_indices, values) = weighted_lower_csc(dimension, &weighted_entries);
            let matrix = SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, Some(&values))
                .expect("matrix");
            let rhs = (0..dimension)
                .map(|index| 1.0 + index as f64 * 0.25)
                .collect::<Vec<_>>();

            for ordering in [
                OrderingStrategy::Natural,
                OrderingStrategy::NestedDissection(NestedDissectionOptions::default()),
            ] {
                let (symbolic, _) =
                    analyse(matrix, &SsidsOptions { ordering }).expect("analysis succeeds");
                let (mut factor, info) =
                    factorize(matrix, &symbolic, &NumericFactorOptions::default())
                        .expect("factor succeeds");
                prop_assert!(info.factorization_residual_max_abs <= 1e-8);
                let solution = factor.solve(&rhs).expect("solve succeeds");
                prop_assert!(solution.iter().all(|value| value.is_finite()));
                let mut residual = vec![0.0; dimension];
                for col in 0..dimension {
                    for index in col_ptrs[col]..col_ptrs[col + 1] {
                        let row = row_indices[index];
                        let value = values[index];
                        residual[row] += value * solution[col];
                        if row != col {
                            residual[col] += value * solution[row];
                        }
                    }
                }
                let residual_inf = residual
                    .iter()
                    .zip(rhs.iter())
                    .map(|(lhs, rhs)| (lhs - rhs).abs())
                    .fold(0.0, f64::max);
                prop_assert!(residual_inf <= 1e-7);
            }
            Ok(())
        })
        .expect("numeric factorization invariants should hold");
}

#[test]
fn property_malformed_csc_inputs_are_rejected() {
    let strategy = prop_oneof![
        Just((2_usize, vec![0, 2, 1], vec![0, 1], None::<Vec<f64>>)),
        Just((2_usize, vec![0, 1, 2], vec![0, 2], None::<Vec<f64>>)),
        Just((2_usize, vec![0, 1, 2], vec![0, 1], Some(vec![1.0]))),
        Just((
            3_usize,
            vec![1, 1, 1, 1],
            Vec::<usize>::new(),
            None::<Vec<f64>>
        )),
        (2_usize..8).prop_map(|dimension| { (dimension, vec![0, 1], vec![0], None::<Vec<f64>>,) }),
    ];

    let mut runner = deterministic_runner(48);
    runner
        .run(&strategy, |(dimension, col_ptrs, row_indices, values)| {
            prop_assert!(
                SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, values.as_deref())
                    .is_err()
            );
            Ok(())
        })
        .expect("malformed CSC should be rejected");
}
