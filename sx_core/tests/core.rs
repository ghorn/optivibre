use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use approx::assert_abs_diff_eq;
use proptest::prelude::*;
use rstest::rstest;
use serde::Deserialize;
use sx_core::{BinaryOp, CCS, HessianStrategy, NamedMatrix, NodeView, SX, SXFunction, SXMatrix};

fn eval(expr: SX, vars: &HashMap<u32, f64>) -> f64 {
    match expr.inspect() {
        NodeView::Constant(v) => v,
        NodeView::Symbol { .. } => vars[&expr.id()],
        NodeView::Binary { op, lhs, rhs } => match op {
            BinaryOp::Add => eval(lhs, vars) + eval(rhs, vars),
            BinaryOp::Sub => eval(lhs, vars) - eval(rhs, vars),
            BinaryOp::Mul => eval(lhs, vars) * eval(rhs, vars),
            BinaryOp::Div => eval(lhs, vars) / eval(rhs, vars),
        },
    }
}

fn assert_matrix_values_match(lhs: &SXMatrix, rhs: &SXMatrix, vars: &HashMap<u32, f64>) {
    assert_eq!(lhs.shape(), rhs.shape());
    assert_eq!(lhs.ccs(), rhs.ccs());
    let (nrow, ncol) = lhs.shape();
    for col in 0..ncol {
        for row in 0..nrow {
            assert_abs_diff_eq!(
                eval(lhs.get(row, col), vars),
                eval(rhs.get(row, col), vars),
                epsilon = 1e-9
            );
        }
    }
}

#[rstest]
fn sx_cse_and_constant_folding() {
    let x = SX::sym("x");
    let y = SX::sym("y");
    assert_eq!(x + SX::zero(), x);
    assert_eq!(x * SX::one(), x);
    assert_eq!(x + y, y + x);
    assert_eq!((x + y).id(), (y + x).id());
}

#[rstest]
fn ccs_transpose_roundtrip() {
    let ccs = CCS::from_positions(3, 2, &[(0, 0), (2, 0), (1, 1)]).unwrap();
    assert_eq!(ccs.transpose().transpose(), ccs);
}

#[rstest]
fn gradient_matches_manual_rosenbrock() {
    let x0 = SX::sym("x0");
    let x1 = SX::sym("x1");
    let x = SXMatrix::dense_column(vec![x0, x1]).unwrap();
    let f = SXMatrix::scalar((1.0 - x0).sqr() + 100.0 * (x1 - x0.sqr()).sqr());
    let grad = f.gradient(&x).unwrap();

    let values = HashMap::from([(x0.id(), 1.2), (x1.id(), 1.5)]);
    let g0 = eval(grad.nz(0), &values);
    let g1 = eval(grad.nz(1), &values);
    assert_abs_diff_eq!(g0, -28.4, epsilon = 1e-9);
    assert_abs_diff_eq!(g1, 12.0, epsilon = 1e-9);
}

#[rstest]
fn jacobian_and_hessian_shapes() {
    let x0 = SX::sym("x0");
    let x1 = SX::sym("x1");
    let x = SXMatrix::dense_column(vec![x0, x1]).unwrap();
    let f = SXMatrix::dense_column(vec![x0 + x1, x0 * x1]).unwrap();

    let jac = f.jacobian(&x).unwrap();
    assert_eq!(jac.ccs().nrow(), 2);
    assert_eq!(jac.ccs().ncol(), 2);

    let rosen = SXMatrix::scalar((1.0 - x0).sqr() + 100.0 * (x1 - x0.sqr()).sqr());
    let hess = rosen.hessian(&x).unwrap();
    assert_eq!(hess.ccs().nrow(), 2);
    assert_eq!(hess.ccs().ncol(), 2);
    assert!(hess.ccs().positions().iter().all(|(row, col)| row >= col));
}

#[rstest]
fn hessian_strategies_match_on_rosenbrock() {
    let x0 = SX::sym("hx0");
    let x1 = SX::sym("hx1");
    let x = SXMatrix::dense_column(vec![x0, x1]).unwrap();
    let rosen = SXMatrix::scalar((1.0 - x0).sqr() + 100.0 * (x1 - x0.sqr()).sqr());
    let values = HashMap::from([(x0.id(), 1.2), (x1.id(), 1.5)]);

    let reference = rosen
        .hessian_with_strategy(&x, HessianStrategy::LowerTriangleByColumn)
        .unwrap();
    for strategy in HessianStrategy::ALL {
        let candidate = rosen.hessian_with_strategy(&x, strategy).unwrap();
        assert_matrix_values_match(&reference, &candidate, &values);
    }

    let default_hessian = rosen.hessian(&x).unwrap();
    let selected_outputs_hessian = rosen
        .hessian_with_strategy(&x, HessianStrategy::LowerTriangleSelectedOutputs)
        .unwrap();
    assert_eq!(default_hessian, selected_outputs_hessian);
}

#[rstest]
fn sx_function_validates_free_symbols() {
    let x = SXMatrix::sym_dense("x", 2, 1).unwrap();
    let y = SX::sym("y");
    let output = SXMatrix::scalar(y);
    let err = SXFunction::new(
        "bad",
        vec![NamedMatrix::new("x", x).unwrap()],
        vec![NamedMatrix::new("out", output).unwrap()],
    )
    .unwrap_err();
    assert!(err.to_string().contains("free symbol"));
}

#[derive(Debug, Deserialize)]
struct ParityManifest {
    casadi_commit: String,
    entries: Vec<ParityEntry>,
}

#[derive(Debug, Deserialize)]
struct ParityEntry {
    casadi_case: String,
    rust_test: Option<String>,
    status: String,
}

#[rstest]
fn parity_manifest_is_pinned_and_supported_cases_have_tests() {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../parity/casadi_v1.toml");
    let manifest: ParityManifest =
        toml::from_str(&fs::read_to_string(manifest_path).unwrap()).unwrap();
    assert_eq!(
        manifest.casadi_commit,
        "838d585b15f3de6f2a1e8d058c91e445a01d43e3"
    );
    assert!(
        manifest
            .entries
            .iter()
            .inspect(|entry| assert!(!entry.casadi_case.trim().is_empty()))
            .filter(|entry| entry.status == "supported")
            .all(|entry| entry.rust_test.is_some())
    );
}

prop_compose! {
    fn expr_coeffs()(a in -3.0f64..3.0, b in -3.0f64..3.0, c in -3.0f64..3.0, d in 0.5f64..3.0) -> (f64, f64, f64, f64) {
        (a, b, c, d)
    }
}

prop_compose! {
    fn hessian_expr_coeffs()(
        a in -2.0f64..2.0,
        b in -2.0f64..2.0,
        c in -2.0f64..2.0,
        d in 0.5f64..3.0,
        e in -2.0f64..2.0,
        f in -2.0f64..2.0,
    ) -> (f64, f64, f64, f64, f64, f64) {
        (a, b, c, d, e, f)
    }
}

proptest! {
    #[test]
    fn forward_reverse_directionals_match_finite_difference((a, b, c, d) in expr_coeffs(), x0v in -1.5f64..1.5, x1v in -1.5f64..1.5, s0 in -1.0f64..1.0, s1 in -1.0f64..1.0) {
        let x0 = SX::sym("px0");
        let x1 = SX::sym("px1");
        let x = SXMatrix::dense_column(vec![x0, x1]).unwrap();
        let expr = SXMatrix::scalar(((a * x0 + b * x1) * (c * x0 - x1)) / d);
        let seed = SXMatrix::dense_column(vec![SX::from(s0), SX::from(s1)]).unwrap();

        let forward = expr.forward(&x, &seed).unwrap().scalar_expr().unwrap();
        let reverse = expr.reverse(&x, &SXMatrix::scalar(1.0)).unwrap();

        let vars = HashMap::from([(x0.id(), x0v), (x1.id(), x1v)]);
        let eps = 1e-6;
        let base = eval(expr.scalar_expr().unwrap(), &vars);
        let bumped = {
            let vars_eps = HashMap::from([(x0.id(), x0v + eps * s0), (x1.id(), x1v + eps * s1)]);
            eval(expr.scalar_expr().unwrap(), &vars_eps)
        };
        let fd = (bumped - base) / eps;
        prop_assert!((eval(forward, &vars) - fd).abs() < 1e-3);
        prop_assert_eq!(reverse.nnz(), 2);
    }

    #[test]
    fn hessian_strategies_match_on_random_nonlinear_scalar(
        (a, b, c, d, e, f) in hessian_expr_coeffs(),
        x0v in -1.5f64..1.5,
        x1v in -1.5f64..1.5,
        x2v in -1.5f64..1.5,
    ) {
        let x0 = SX::sym("phx0");
        let x1 = SX::sym("phx1");
        let x2 = SX::sym("phx2");
        let x = SXMatrix::dense_column(vec![x0, x1, x2]).unwrap();
        let expr = SXMatrix::scalar(
            ((a * x0 + b * x1) * (c * x1 - x2)) / d
                + e * x0 * x2
                + f * x1.sqr()
                + x0 * x1 * x2
                + x0.sqr() * x2,
        );
        let vars = HashMap::from([(x0.id(), x0v), (x1.id(), x1v), (x2.id(), x2v)]);

        let reference = expr
            .hessian_with_strategy(&x, HessianStrategy::LowerTriangleByColumn)
            .unwrap();
        for strategy in HessianStrategy::ALL {
            let candidate = expr.hessian_with_strategy(&x, strategy).unwrap();
            prop_assert_eq!(candidate.ccs(), reference.ccs());
            let (nrow, ncol) = candidate.shape();
            for col in 0..ncol {
                for row in 0..nrow {
                    prop_assert!(
                        (eval(candidate.get(row, col), &vars) - eval(reference.get(row, col), &vars)).abs() < 1e-9
                    );
                }
            }
        }
    }
}
