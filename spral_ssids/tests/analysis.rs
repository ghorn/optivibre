use metis_ordering::NestedDissectionOptions;
use spral_ssids::{AnalyseInfo, OrderingStrategy, SsidsOptions, SymmetricCscMatrix, analyse};

fn tridiagonal_pattern(dimension: usize) -> (Vec<usize>, Vec<usize>) {
    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for col in 0..dimension {
        row_indices.push(col);
        if col + 1 < dimension {
            row_indices.push(col + 1);
        }
        col_ptrs.push(row_indices.len());
    }
    (col_ptrs, row_indices)
}

fn grid_pattern(rows: usize, cols: usize) -> (Vec<usize>, Vec<usize>) {
    let dimension = rows * cols;
    let mut columns = vec![Vec::new(); dimension];
    for (index, column) in columns.iter_mut().enumerate() {
        column.push(index);
    }
    for row in 0..rows {
        for col in 0..cols {
            let index = row * cols + col;
            if row + 1 < rows {
                columns[index].push(index + cols);
            }
            if col + 1 < cols {
                columns[index].push(index + 1);
            }
        }
    }
    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for mut rows in columns {
        rows.sort_unstable();
        rows.dedup();
        row_indices.extend(rows);
        col_ptrs.push(row_indices.len());
    }
    (col_ptrs, row_indices)
}

#[test]
fn symbolic_analysis_returns_valid_factor_metadata() {
    let (col_ptrs, row_indices) = tridiagonal_pattern(8);
    let matrix = SymmetricCscMatrix::new(8, &col_ptrs, &row_indices, None).expect("valid matrix");
    let (symbolic, info): (_, AnalyseInfo) =
        analyse(matrix, &SsidsOptions::default()).expect("analysis succeeds");
    assert_eq!(symbolic.permutation.len(), 8);
    assert_eq!(symbolic.elimination_tree.len(), 8);
    assert_eq!(symbolic.column_counts.len(), 8);
    assert_eq!(symbolic.column_pattern.len(), 8);
    assert!(info.estimated_fill_nnz >= 8);
    assert!(info.supernode_count > 0);
    assert!(
        matches!(
            info.ordering_kind,
            "auto_natural" | "auto_approximate_minimum_degree" | "auto_nested_dissection"
        ),
        "unexpected default ordering kind: {}",
        info.ordering_kind
    );
}

#[test]
fn auto_ordering_never_exceeds_explicit_candidate_fill_on_grid_pattern() {
    let (col_ptrs, row_indices) = grid_pattern(4, 4);
    let matrix = SymmetricCscMatrix::new(16, &col_ptrs, &row_indices, None).expect("valid matrix");
    let (_, natural_info) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::Natural,
        },
    )
    .expect("natural analysis succeeds");
    let (_, amd_info) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::ApproximateMinimumDegree,
        },
    )
    .expect("amd analysis succeeds");
    let (_, nd_info) = analyse(
        matrix,
        &SsidsOptions {
            ordering: OrderingStrategy::NestedDissection(NestedDissectionOptions::default()),
        },
    )
    .expect("nd analysis succeeds");
    let (_, auto_info) = analyse(matrix, &SsidsOptions::default()).expect("auto analysis succeeds");

    let best_explicit = natural_info
        .estimated_fill_nnz
        .min(amd_info.estimated_fill_nnz)
        .min(nd_info.estimated_fill_nnz);
    assert!(auto_info.estimated_fill_nnz <= best_explicit);
}
