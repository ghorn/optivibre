use optimization::{CCS, CompiledNlpProblem};
use sparse_validation::{SymmetricPatternMatrix, exact_symbolic_metrics};
use spral_ssids::{OrderingStrategy, SsidsOptions, SymmetricCscMatrix, analyse};

#[path = "support/generated_problem.rs"]
#[allow(dead_code)]
mod generated_problem;

use generated_problem::{
    CallbackBackend, hanging_chain_problem, hs021_problem, hs035_problem, hs071_problem,
};

fn symmetric_pattern_from_ccs(ccs: &CCS) -> SymmetricPatternMatrix {
    let entries = (0..ccs.ncol)
        .flat_map(|col| {
            ccs.row_indices[ccs.col_ptrs[col]..ccs.col_ptrs[col + 1]]
                .iter()
                .map(move |&row| (row, col))
        })
        .collect::<Vec<_>>();
    SymmetricPatternMatrix::from_coordinate_entries(ccs.nrow, &entries, true)
        .expect("symmetric CCS should convert")
}

fn normal_pattern_from_jacobian(ccs: &CCS) -> SymmetricPatternMatrix {
    let mut row_to_columns = vec![Vec::new(); ccs.nrow];
    for col in 0..ccs.ncol {
        for &row in &ccs.row_indices[ccs.col_ptrs[col]..ccs.col_ptrs[col + 1]] {
            row_to_columns[row].push(col);
        }
    }
    let mut entries = Vec::new();
    for columns in row_to_columns {
        for index in 0..columns.len() {
            for jdx in index..columns.len() {
                entries.push((columns[index], columns[jdx]));
            }
        }
    }
    SymmetricPatternMatrix::from_coordinate_entries(ccs.ncol, &entries, true)
        .expect("normal-equation pattern should convert")
}

fn assert_symbolic_matches_exact(label: &str, matrix: &SymmetricPatternMatrix) {
    let csc = SymmetricCscMatrix::new(
        matrix.dimension(),
        matrix.col_ptrs(),
        matrix.row_indices(),
        None,
    )
    .expect("matrix should be valid");

    for ordering in [
        OrderingStrategy::Natural,
        OrderingStrategy::NestedDissection(Default::default()),
    ] {
        let (symbolic, info) = analyse(csc, &SsidsOptions { ordering }).expect(label);
        let exact = exact_symbolic_metrics(matrix, &symbolic.permutation).expect(label);
        assert_eq!(symbolic.elimination_tree, exact.elimination_tree, "{label}");
        assert_eq!(symbolic.column_counts, exact.column_counts, "{label}");
        assert_eq!(symbolic.column_pattern, exact.column_pattern, "{label}");
        assert_eq!(info.estimated_fill_nnz, exact.fill_nnz, "{label}");
    }
}

fn assert_problem_patterns(problem_name: &str, problem: &impl CompiledNlpProblem) {
    assert_symbolic_matches_exact(
        &format!("{problem_name}/hessian"),
        &symmetric_pattern_from_ccs(problem.lagrangian_hessian_ccs()),
    );

    if problem.equality_jacobian_ccs().nnz() > 0 {
        assert_symbolic_matches_exact(
            &format!("{problem_name}/equality_normal"),
            &normal_pattern_from_jacobian(problem.equality_jacobian_ccs()),
        );
    }
    if problem.inequality_jacobian_ccs().nnz() > 0 {
        assert_symbolic_matches_exact(
            &format!("{problem_name}/inequality_normal"),
            &normal_pattern_from_jacobian(problem.inequality_jacobian_ccs()),
        );
    }
}

#[test]
fn spral_symbolic_analysis_accepts_generated_hanging_chain_hessian_pattern() {
    let problem = hanging_chain_problem(CallbackBackend::Aot).expect("generated problem");
    let hessian = problem.lagrangian_hessian_ccs();
    let matrix =
        SymmetricCscMatrix::new(hessian.nrow, &hessian.col_ptrs, &hessian.row_indices, None)
            .expect("valid Hessian structure");
    let (symbolic, info) = analyse(matrix, &SsidsOptions::default()).expect("analysis succeeds");
    assert_eq!(symbolic.permutation.len(), hessian.nrow);
    assert_eq!(symbolic.elimination_tree.len(), hessian.nrow);
    assert_eq!(symbolic.column_counts.len(), hessian.nrow);
    assert!(info.estimated_fill_nnz >= hessian.nrow);
}

#[test]
fn spral_symbolic_analysis_matches_exact_workspace_patterns() {
    assert_problem_patterns(
        "hanging_chain",
        &hanging_chain_problem(CallbackBackend::Aot).expect("generated problem"),
    );
    assert_problem_patterns(
        "hs021",
        &hs021_problem(CallbackBackend::Aot).expect("generated problem"),
    );
    assert_problem_patterns(
        "hs035",
        &hs035_problem(CallbackBackend::Aot).expect("generated problem"),
    );
    assert_problem_patterns(
        "hs071",
        &hs071_problem(CallbackBackend::Aot).expect("generated problem"),
    );
}
