use anyhow::Result;
use sx_core::{HessianOptions, HessianStrategy, Index, NamedMatrix, SX, SXFunction, SXMatrix};

#[derive(Clone, Debug)]
pub struct ExampleArtifact {
    pub module_name: String,
    pub function: SXFunction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdCostScenario {
    ReverseGradient,
    ForwardSweep,
    Jacobian,
    Hessian,
}

#[derive(Clone, Debug)]
pub struct AdCostCase {
    pub key: &'static str,
    pub scenario: AdCostScenario,
    pub size: Index,
    pub sweep_count: Index,
    pub original: SXFunction,
    pub augmented: SXFunction,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdCostExpectations {
    pub description: &'static str,
    pub exact_original_ops: usize,
    pub exact_augmented_ops: usize,
    pub directional_ratio_limit: Option<f64>,
    pub normalized_ratio_limit: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct HessianStrategyCase {
    pub strategy: HessianStrategy,
    pub function: SXFunction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HessianStrategyExpectation {
    pub exact_ops: usize,
}

pub const AD_REVERSE_SIZE: Index = 96;
pub const AD_FORWARD_SIZE: Index = 160;
pub const AD_JACOBIAN_SIZE: Index = 32;
pub const AD_HESSIAN_SIZE: Index = 24;

impl AdCostScenario {
    pub fn label(self) -> &'static str {
        match self {
            Self::ReverseGradient => "reverse-gradient",
            Self::ForwardSweep => "forward-sweep",
            Self::Jacobian => "jacobian",
            Self::Hessian => "hessian",
        }
    }
}

impl AdCostCase {
    pub fn expectations(&self) -> AdCostExpectations {
        ad_cost_expectations(self.scenario)
    }
}

pub fn ad_cost_expectations(scenario: AdCostScenario) -> AdCostExpectations {
    match scenario {
        AdCostScenario::ReverseGradient => AdCostExpectations {
            description: "Scalar primal; reverse AD returns value plus gradient.",
            exact_original_ops: 952,
            exact_augmented_ops: 2376,
            directional_ratio_limit: Some(5.0),
            normalized_ratio_limit: None,
        },
        AdCostScenario::ForwardSweep => AdCostExpectations {
            description: "Vector primal; one forward sweep returns primal plus directional.",
            exact_original_ops: 1281,
            exact_augmented_ops: 2880,
            directional_ratio_limit: Some(5.0),
            normalized_ratio_limit: None,
        },
        AdCostScenario::Jacobian => AdCostExpectations {
            description: "Vector primal; full Jacobian cost normalized by sweep count.",
            exact_original_ops: 322,
            exact_augmented_ops: 4081,
            directional_ratio_limit: None,
            normalized_ratio_limit: Some(8.0),
        },
        AdCostScenario::Hessian => AdCostExpectations {
            description: "Scalar primal; Hessian cost normalized by sweep count.",
            exact_original_ops: 232,
            exact_augmented_ops: 11159,
            directional_ratio_limit: None,
            normalized_ratio_limit: Some(8.0),
        },
    }
}

fn named_function(
    name: &str,
    inputs: Vec<(&str, SXMatrix)>,
    outputs: Vec<(&str, SXMatrix)>,
) -> Result<SXFunction> {
    SXFunction::new(
        name,
        inputs
            .into_iter()
            .map(|(slot_name, matrix)| NamedMatrix::new(slot_name, matrix))
            .collect::<std::result::Result<Vec<_>, _>>()?,
        outputs
            .into_iter()
            .map(|(slot_name, matrix)| NamedMatrix::new(slot_name, matrix))
            .collect::<std::result::Result<Vec<_>, _>>()?,
    )
    .map_err(Into::into)
}

fn single_io_function(
    name: &str,
    input_name: &str,
    input: SXMatrix,
    output_name: &str,
    output: SXMatrix,
) -> Result<SXFunction> {
    named_function(name, vec![(input_name, input)], vec![(output_name, output)])
}

fn rosenbrock_bundle() -> Result<Vec<ExampleArtifact>> {
    let x = SXMatrix::sym_dense("x", 2, 1)?;
    let x0 = x.nz(0);
    let x1 = x.nz(1);
    let objective = SXMatrix::scalar((1.0 - x0).sqr() + 100.0 * (x1 - x0.sqr()).sqr());
    let gradient = objective.gradient(&x)?;
    let hessian = objective.hessian(&x)?;

    Ok(vec![
        ExampleArtifact {
            module_name: "rosenbrock_objective".into(),
            function: single_io_function(
                "rosenbrock_objective",
                "x",
                x.clone(),
                "objective",
                objective,
            )?,
        },
        ExampleArtifact {
            module_name: "rosenbrock_gradient".into(),
            function: single_io_function(
                "rosenbrock_gradient",
                "x",
                x.clone(),
                "gradient",
                gradient,
            )?,
        },
        ExampleArtifact {
            module_name: "rosenbrock_hessian".into(),
            function: single_io_function("rosenbrock_hessian", "x", x, "hessian", hessian)?,
        },
    ])
}

fn casadi_rosenbrock_nlp_bundle() -> Result<Vec<ExampleArtifact>> {
    let vars = SXMatrix::sym_dense("x", 3, 1)?;
    let x = vars.nz(0);
    let y = vars.nz(1);
    let z = vars.nz(2);
    let objective = SXMatrix::scalar(x.sqr() + 100.0 * z.sqr());
    let gradient = objective.gradient(&vars)?;
    let constraints = SXMatrix::dense_column(vec![z + (1.0 - x).sqr() - y])?;
    let jacobian = constraints.jacobian(&vars)?;
    let lambda_eq = SXMatrix::sym_dense("lambda_eq", 1, 1)?;
    let lagrangian = SXMatrix::scalar(objective.nz(0) + lambda_eq.nz(0) * constraints.nz(0));
    let lagrangian_hessian = lagrangian.hessian(&vars)?;

    Ok(vec![
        ExampleArtifact {
            module_name: "casadi_rosenbrock_nlp_objective".into(),
            function: single_io_function(
                "casadi_rosenbrock_nlp_objective",
                "x",
                vars.clone(),
                "objective",
                objective,
            )?,
        },
        ExampleArtifact {
            module_name: "casadi_rosenbrock_nlp_gradient".into(),
            function: single_io_function(
                "casadi_rosenbrock_nlp_gradient",
                "x",
                vars.clone(),
                "gradient",
                gradient,
            )?,
        },
        ExampleArtifact {
            module_name: "casadi_rosenbrock_nlp_constraints".into(),
            function: single_io_function(
                "casadi_rosenbrock_nlp_constraints",
                "x",
                vars.clone(),
                "constraints",
                constraints,
            )?,
        },
        ExampleArtifact {
            module_name: "casadi_rosenbrock_nlp_jacobian".into(),
            function: single_io_function(
                "casadi_rosenbrock_nlp_jacobian",
                "x",
                vars.clone(),
                "jacobian",
                jacobian,
            )?,
        },
        ExampleArtifact {
            module_name: "casadi_rosenbrock_nlp_lagrangian_hessian".into(),
            function: named_function(
                "casadi_rosenbrock_nlp_lagrangian_hessian",
                vec![("x", vars), ("lambda_eq", lambda_eq)],
                vec![("hessian", lagrangian_hessian)],
            )?,
        },
    ])
}

fn simple_nlp_bundle() -> Result<Vec<ExampleArtifact>> {
    let x = SXMatrix::sym_dense("x", 2, 1)?;
    let objective = SXMatrix::scalar(x.nz(0).sqr() + x.nz(1).sqr());
    let gradient = objective.gradient(&x)?;
    let constraints = SXMatrix::dense_column(vec![x.nz(0) + x.nz(1) - 10.0])?;
    let jacobian = constraints.jacobian(&x)?;
    let lambda_eq = SXMatrix::sym_dense("lambda_eq", 1, 1)?;
    let lagrangian = SXMatrix::scalar(objective.nz(0) + lambda_eq.nz(0) * constraints.nz(0));
    let lagrangian_hessian = lagrangian.hessian(&x)?;

    Ok(vec![
        ExampleArtifact {
            module_name: "simple_nlp_objective".into(),
            function: single_io_function(
                "simple_nlp_objective",
                "x",
                x.clone(),
                "objective",
                objective,
            )?,
        },
        ExampleArtifact {
            module_name: "simple_nlp_gradient".into(),
            function: single_io_function(
                "simple_nlp_gradient",
                "x",
                x.clone(),
                "gradient",
                gradient,
            )?,
        },
        ExampleArtifact {
            module_name: "simple_nlp_constraints".into(),
            function: single_io_function(
                "simple_nlp_constraints",
                "x",
                x.clone(),
                "constraints",
                constraints,
            )?,
        },
        ExampleArtifact {
            module_name: "simple_nlp_jacobian".into(),
            function: single_io_function(
                "simple_nlp_jacobian",
                "x",
                x.clone(),
                "jacobian",
                jacobian,
            )?,
        },
        ExampleArtifact {
            module_name: "simple_nlp_lagrangian_hessian".into(),
            function: named_function(
                "simple_nlp_lagrangian_hessian",
                vec![("x", x), ("lambda_eq", lambda_eq)],
                vec![("hessian", lagrangian_hessian)],
            )?,
        },
    ])
}

fn constrained_rosenbrock_bundle() -> Result<Vec<ExampleArtifact>> {
    let x = SXMatrix::sym_dense("x", 2, 1)?;
    let x0 = x.nz(0);
    let x1 = x.nz(1);
    let objective = SXMatrix::scalar((1.0 - x0).sqr() + 100.0 * (x1 - x0.sqr()).sqr());
    let gradient = objective.gradient(&x)?;
    let constraints = SXMatrix::dense_column(vec![x0 + x1 - 1.0])?;
    let jacobian = constraints.jacobian(&x)?;
    let hessian = objective.hessian(&x)?;
    let lambda_eq = SXMatrix::sym_dense("lambda_eq", 1, 1)?;
    let lagrangian = SXMatrix::scalar(objective.nz(0) + lambda_eq.nz(0) * constraints.nz(0));
    let lagrangian_hessian = lagrangian.hessian(&x)?;

    Ok(vec![
        ExampleArtifact {
            module_name: "constrained_rosenbrock_objective".into(),
            function: single_io_function(
                "constrained_rosenbrock_objective",
                "x",
                x.clone(),
                "objective",
                objective,
            )?,
        },
        ExampleArtifact {
            module_name: "constrained_rosenbrock_constraints".into(),
            function: single_io_function(
                "constrained_rosenbrock_constraints",
                "x",
                x.clone(),
                "constraints",
                constraints,
            )?,
        },
        ExampleArtifact {
            module_name: "constrained_rosenbrock_gradient".into(),
            function: single_io_function(
                "constrained_rosenbrock_gradient",
                "x",
                x.clone(),
                "gradient",
                gradient,
            )?,
        },
        ExampleArtifact {
            module_name: "constrained_rosenbrock_jacobian".into(),
            function: single_io_function(
                "constrained_rosenbrock_jacobian",
                "x",
                x.clone(),
                "jacobian",
                jacobian,
            )?,
        },
        ExampleArtifact {
            module_name: "constrained_rosenbrock_hessian".into(),
            function: single_io_function(
                "constrained_rosenbrock_hessian",
                "x",
                x.clone(),
                "hessian",
                hessian,
            )?,
        },
        ExampleArtifact {
            module_name: "constrained_rosenbrock_lagrangian_hessian".into(),
            function: named_function(
                "constrained_rosenbrock_lagrangian_hessian",
                vec![("x", x), ("lambda_eq", lambda_eq)],
                vec![("hessian", lagrangian_hessian)],
            )?,
        },
    ])
}

fn hanging_chain_bundle() -> Result<Vec<ExampleArtifact>> {
    let points = 4;
    let vars = SXMatrix::sym_dense("q", 2 * points, 1)?;
    let mut objective_terms = Vec::new();
    for idx in 0..points {
        objective_terms.push(vars.nz(2 * idx + 1));
    }
    let objective = SXMatrix::scalar(
        objective_terms
            .into_iter()
            .fold(SX::zero(), |acc, term| acc + term),
    );

    let anchor_left = (0.0, 0.0);
    let anchor_right = (3.0, 0.0);
    let link_length_sq = 0.75f64 * 0.75f64;
    let mut constraints = Vec::new();
    let mut prev_x = SX::from(anchor_left.0);
    let mut prev_y = SX::from(anchor_left.1);
    for idx in 0..points {
        let xi = vars.nz(2 * idx);
        let yi = vars.nz(2 * idx + 1);
        constraints.push((xi - prev_x).sqr() + (yi - prev_y).sqr() - link_length_sq);
        prev_x = xi;
        prev_y = yi;
    }
    constraints
        .push((prev_x - anchor_right.0).sqr() + (prev_y - anchor_right.1).sqr() - link_length_sq);
    let constraints = SXMatrix::dense_column(constraints)?;
    let gradient = objective.gradient(&vars)?;
    let jacobian = constraints.jacobian(&vars)?;
    let hessian = objective.hessian(&vars)?;
    let lambda_eq = SXMatrix::sym_dense("lambda_eq", points + 1, 1)?;
    let mut lagrangian_expr = objective.nz(0);
    for idx in 0..=points {
        lagrangian_expr += lambda_eq.nz(idx) * constraints.nz(idx);
    }
    let lagrangian_hessian = SXMatrix::scalar(lagrangian_expr).hessian(&vars)?;

    Ok(vec![
        ExampleArtifact {
            module_name: "hanging_chain_objective".into(),
            function: single_io_function(
                "hanging_chain_objective",
                "q",
                vars.clone(),
                "objective",
                objective,
            )?,
        },
        ExampleArtifact {
            module_name: "hanging_chain_constraints".into(),
            function: single_io_function(
                "hanging_chain_constraints",
                "q",
                vars.clone(),
                "constraints",
                constraints,
            )?,
        },
        ExampleArtifact {
            module_name: "hanging_chain_gradient".into(),
            function: single_io_function(
                "hanging_chain_gradient",
                "q",
                vars.clone(),
                "gradient",
                gradient,
            )?,
        },
        ExampleArtifact {
            module_name: "hanging_chain_jacobian".into(),
            function: single_io_function(
                "hanging_chain_jacobian",
                "q",
                vars.clone(),
                "jacobian",
                jacobian,
            )?,
        },
        ExampleArtifact {
            module_name: "hanging_chain_hessian".into(),
            function: single_io_function(
                "hanging_chain_hessian",
                "q",
                vars.clone(),
                "hessian",
                hessian,
            )?,
        },
        ExampleArtifact {
            module_name: "hanging_chain_lagrangian_hessian".into(),
            function: named_function(
                "hanging_chain_lagrangian_hessian",
                vec![("q", vars), ("lambda_eq", lambda_eq)],
                vec![("hessian", lagrangian_hessian)],
            )?,
        },
    ])
}

fn hs021_bundle() -> Result<Vec<ExampleArtifact>> {
    let x = SXMatrix::sym_dense("x", 2, 1)?;
    let x0 = x.nz(0);
    let x1 = x.nz(1);
    let objective = SXMatrix::scalar(0.01 * x0.sqr() + x1.sqr() - 100.0);
    let gradient = objective.gradient(&x)?;
    let inequalities = SXMatrix::dense_column(vec![
        -10.0 * x0 + x1 + 10.0,
        2.0 - x0,
        x0 - 50.0,
        -50.0 - x1,
        x1 - 50.0,
    ])?;
    let inequality_jacobian = inequalities.jacobian(&x)?;
    let mu = SXMatrix::sym_dense("mu", 5, 1)?;
    let mut lagrangian_expr = objective.nz(0);
    for idx in 0..5 {
        lagrangian_expr += mu.nz(idx) * inequalities.nz(idx);
    }
    let lagrangian_hessian = SXMatrix::scalar(lagrangian_expr).hessian(&x)?;

    Ok(vec![
        ExampleArtifact {
            module_name: "hs021_objective".into(),
            function: single_io_function(
                "hs021_objective",
                "x",
                x.clone(),
                "objective",
                objective,
            )?,
        },
        ExampleArtifact {
            module_name: "hs021_gradient".into(),
            function: single_io_function("hs021_gradient", "x", x.clone(), "gradient", gradient)?,
        },
        ExampleArtifact {
            module_name: "hs021_inequalities".into(),
            function: single_io_function(
                "hs021_inequalities",
                "x",
                x.clone(),
                "inequalities",
                inequalities,
            )?,
        },
        ExampleArtifact {
            module_name: "hs021_inequality_jacobian".into(),
            function: single_io_function(
                "hs021_inequality_jacobian",
                "x",
                x.clone(),
                "jacobian",
                inequality_jacobian,
            )?,
        },
        ExampleArtifact {
            module_name: "hs021_lagrangian_hessian".into(),
            function: named_function(
                "hs021_lagrangian_hessian",
                vec![("x", x), ("mu", mu)],
                vec![("hessian", lagrangian_hessian)],
            )?,
        },
    ])
}

fn hs035_bundle() -> Result<Vec<ExampleArtifact>> {
    let x = SXMatrix::sym_dense("x", 3, 1)?;
    let x0 = x.nz(0);
    let x1 = x.nz(1);
    let x2 = x.nz(2);
    let objective = SXMatrix::scalar(
        9.0 - 8.0 * x0 - 6.0 * x1 - 4.0 * x2
            + 2.0 * x0.sqr()
            + 2.0 * x1.sqr()
            + x2.sqr()
            + 2.0 * x0 * x1
            + 2.0 * x0 * x2,
    );
    let gradient = objective.gradient(&x)?;
    let inequalities = SXMatrix::dense_column(vec![x0 + x1 + 2.0 * x2 - 3.0, -x0, -x1, -x2])?;
    let inequality_jacobian = inequalities.jacobian(&x)?;
    let mu = SXMatrix::sym_dense("mu", 4, 1)?;
    let mut lagrangian_expr = objective.nz(0);
    for idx in 0..4 {
        lagrangian_expr += mu.nz(idx) * inequalities.nz(idx);
    }
    let lagrangian_hessian = SXMatrix::scalar(lagrangian_expr).hessian(&x)?;

    Ok(vec![
        ExampleArtifact {
            module_name: "hs035_objective".into(),
            function: single_io_function(
                "hs035_objective",
                "x",
                x.clone(),
                "objective",
                objective,
            )?,
        },
        ExampleArtifact {
            module_name: "hs035_gradient".into(),
            function: single_io_function("hs035_gradient", "x", x.clone(), "gradient", gradient)?,
        },
        ExampleArtifact {
            module_name: "hs035_inequalities".into(),
            function: single_io_function(
                "hs035_inequalities",
                "x",
                x.clone(),
                "inequalities",
                inequalities,
            )?,
        },
        ExampleArtifact {
            module_name: "hs035_inequality_jacobian".into(),
            function: single_io_function(
                "hs035_inequality_jacobian",
                "x",
                x.clone(),
                "jacobian",
                inequality_jacobian,
            )?,
        },
        ExampleArtifact {
            module_name: "hs035_lagrangian_hessian".into(),
            function: named_function(
                "hs035_lagrangian_hessian",
                vec![("x", x), ("mu", mu)],
                vec![("hessian", lagrangian_hessian)],
            )?,
        },
    ])
}

fn hs071_bundle() -> Result<Vec<ExampleArtifact>> {
    let x = SXMatrix::sym_dense("x", 4, 1)?;
    let x0 = x.nz(0);
    let x1 = x.nz(1);
    let x2 = x.nz(2);
    let x3 = x.nz(3);
    let objective = SXMatrix::scalar(x0 * x3 * (x0 + x1 + x2) + x2);
    let gradient = objective.gradient(&x)?;
    let equalities =
        SXMatrix::dense_column(vec![x0.sqr() + x1.sqr() + x2.sqr() + x3.sqr() - 40.0])?;
    let equality_jacobian = equalities.jacobian(&x)?;
    let inequalities = SXMatrix::dense_column(vec![
        25.0 - x0 * x1 * x2 * x3,
        1.0 - x0,
        1.0 - x1,
        1.0 - x2,
        1.0 - x3,
        x0 - 5.0,
        x1 - 5.0,
        x2 - 5.0,
        x3 - 5.0,
    ])?;
    let inequality_jacobian = inequalities.jacobian(&x)?;
    let lambda_eq = SXMatrix::sym_dense("lambda_eq", 1, 1)?;
    let mu = SXMatrix::sym_dense("mu", 9, 1)?;
    let mut lagrangian_expr = objective.nz(0) + lambda_eq.nz(0) * equalities.nz(0);
    for idx in 0..9 {
        lagrangian_expr += mu.nz(idx) * inequalities.nz(idx);
    }
    let lagrangian_hessian = SXMatrix::scalar(lagrangian_expr).hessian(&x)?;

    Ok(vec![
        ExampleArtifact {
            module_name: "hs071_objective".into(),
            function: single_io_function(
                "hs071_objective",
                "x",
                x.clone(),
                "objective",
                objective,
            )?,
        },
        ExampleArtifact {
            module_name: "hs071_gradient".into(),
            function: single_io_function("hs071_gradient", "x", x.clone(), "gradient", gradient)?,
        },
        ExampleArtifact {
            module_name: "hs071_equalities".into(),
            function: single_io_function(
                "hs071_equalities",
                "x",
                x.clone(),
                "equalities",
                equalities,
            )?,
        },
        ExampleArtifact {
            module_name: "hs071_equality_jacobian".into(),
            function: single_io_function(
                "hs071_equality_jacobian",
                "x",
                x.clone(),
                "jacobian",
                equality_jacobian,
            )?,
        },
        ExampleArtifact {
            module_name: "hs071_inequalities".into(),
            function: single_io_function(
                "hs071_inequalities",
                "x",
                x.clone(),
                "inequalities",
                inequalities,
            )?,
        },
        ExampleArtifact {
            module_name: "hs071_inequality_jacobian".into(),
            function: single_io_function(
                "hs071_inequality_jacobian",
                "x",
                x.clone(),
                "jacobian",
                inequality_jacobian,
            )?,
        },
        ExampleArtifact {
            module_name: "hs071_lagrangian_hessian".into(),
            function: named_function(
                "hs071_lagrangian_hessian",
                vec![("x", x), ("lambda_eq", lambda_eq), ("mu", mu)],
                vec![("hessian", lagrangian_hessian)],
            )?,
        },
    ])
}

fn parameterized_quadratic_bundle() -> Result<Vec<ExampleArtifact>> {
    let x = SXMatrix::sym_dense("x", 2, 1)?;
    let p = SXMatrix::sym_dense("p", 2, 1)?;
    let x0 = x.nz(0);
    let x1 = x.nz(1);
    let p0 = p.nz(0);
    let p1 = p.nz(1);
    let objective = SXMatrix::scalar((x0 - p0).sqr() + (x1 - p1).sqr());
    let gradient = objective.gradient(&x)?;
    let equalities = SXMatrix::dense_column(vec![x0 + x1 - 1.0])?;
    let equality_jacobian = equalities.jacobian(&x)?;
    let lambda_eq = SXMatrix::sym_dense("lambda_eq", 1, 1)?;
    let lagrangian = SXMatrix::scalar(objective.nz(0) + lambda_eq.nz(0) * equalities.nz(0));
    let lagrangian_hessian = lagrangian.hessian(&x)?;

    Ok(vec![
        ExampleArtifact {
            module_name: "parameterized_quadratic_objective".into(),
            function: named_function(
                "parameterized_quadratic_objective",
                vec![("x", x.clone()), ("p", p.clone())],
                vec![("objective", objective)],
            )?,
        },
        ExampleArtifact {
            module_name: "parameterized_quadratic_gradient".into(),
            function: named_function(
                "parameterized_quadratic_gradient",
                vec![("x", x.clone()), ("p", p.clone())],
                vec![("gradient", gradient)],
            )?,
        },
        ExampleArtifact {
            module_name: "parameterized_quadratic_equalities".into(),
            function: named_function(
                "parameterized_quadratic_equalities",
                vec![("x", x.clone()), ("p", p.clone())],
                vec![("equalities", equalities)],
            )?,
        },
        ExampleArtifact {
            module_name: "parameterized_quadratic_equality_jacobian".into(),
            function: named_function(
                "parameterized_quadratic_equality_jacobian",
                vec![("x", x.clone()), ("p", p.clone())],
                vec![("jacobian", equality_jacobian)],
            )?,
        },
        ExampleArtifact {
            module_name: "parameterized_quadratic_lagrangian_hessian".into(),
            function: named_function(
                "parameterized_quadratic_lagrangian_hessian",
                vec![("x", x), ("p", p), ("lambda_eq", lambda_eq)],
                vec![("hessian", lagrangian_hessian)],
            )?,
        },
    ])
}

fn scalar_chain_from_vector(input: &SXMatrix) -> SX {
    let mut state = input.nz(0) / 2.0 + 1.0;
    for idx in 1..input.nnz() {
        let xi = input.nz(idx);
        let shift = 1.0 + 0.01 * idx as f64;
        let numerator = (state + shift * xi) * (xi + shift);
        let denominator = 1.0 + (xi - state).sqr();
        state = numerator / denominator + 0.1 * state;
    }
    state
}

fn vector_chain_from_scalar(length: Index) -> Result<(SXMatrix, SXMatrix)> {
    let input = SXMatrix::sym_dense("t", 1, 1)?;
    let t = input.nz(0);
    let mut state = t / 3.0 + 1.0;
    let mut outputs = Vec::new();
    for idx in 0..length {
        let shift = 0.25 + 0.01 * idx as f64;
        let numerator = state * (t + shift);
        let denominator = 1.0 + (state - shift).sqr();
        state = numerator / denominator + shift * t;
        outputs.push(state);
    }
    Ok((input, SXMatrix::dense_column(outputs)?))
}

fn vector_chain_from_vector(input: &SXMatrix) -> Result<SXMatrix> {
    let mut state = input.nz(0) / 2.0 + 1.0;
    let mut outputs = Vec::new();
    for idx in 0..input.nnz() {
        let xi = input.nz(idx);
        let shift = 0.5 + 0.01 * idx as f64;
        let numerator = (state + shift * xi) * (xi + 1.0);
        let denominator = 1.0 + (state - xi).sqr();
        state = numerator / denominator;
        outputs.push(state + xi / (1.0 + shift));
    }
    Ok(SXMatrix::dense_column(outputs)?)
}

fn reverse_gradient_case(size: Index) -> Result<AdCostCase> {
    Ok(AdCostCase {
        key: "reverse_gradient",
        scenario: AdCostScenario::ReverseGradient,
        size,
        sweep_count: 1,
        original: build_reverse_gradient_original_function(size)?,
        augmented: build_reverse_gradient_augmented_function(size)?,
    })
}

pub fn build_reverse_gradient_original_function(size: Index) -> Result<SXFunction> {
    let x = SXMatrix::sym_dense("x", size, 1)?;
    let value = SXMatrix::scalar(scalar_chain_from_vector(&x));
    single_io_function("ad_reverse_gradient_original", "x", x, "value", value)
}

pub fn build_reverse_gradient_augmented_function(size: Index) -> Result<SXFunction> {
    let x = SXMatrix::sym_dense("x", size, 1)?;
    let value = SXMatrix::scalar(scalar_chain_from_vector(&x));
    let gradient = value.gradient(&x)?;
    named_function(
        "ad_reverse_gradient_with_gradient",
        vec![("x", x)],
        vec![("value", value), ("gradient", gradient)],
    )
}

pub fn build_reverse_gradient_case(size: Index) -> Result<AdCostCase> {
    reverse_gradient_case(size)
}

fn forward_sweep_case(size: Index) -> Result<AdCostCase> {
    Ok(AdCostCase {
        key: "forward_sweep",
        scenario: AdCostScenario::ForwardSweep,
        size,
        sweep_count: 1,
        original: build_forward_sweep_original_function(size)?,
        augmented: build_forward_sweep_augmented_function(size)?,
    })
}

pub fn build_forward_sweep_original_function(size: Index) -> Result<SXFunction> {
    let (t, outputs) = vector_chain_from_scalar(size)?;
    single_io_function("ad_forward_sweep_original", "t", t, "outputs", outputs)
}

pub fn build_forward_sweep_augmented_function(size: Index) -> Result<SXFunction> {
    let (t, outputs) = vector_chain_from_scalar(size)?;
    let seed = SXMatrix::scalar(1.0);
    let directional = outputs.forward(&t, &seed)?;
    named_function(
        "ad_forward_sweep_with_directional",
        vec![("t", t)],
        vec![("outputs", outputs), ("directional", directional)],
    )
}

pub fn build_forward_sweep_case(size: Index) -> Result<AdCostCase> {
    forward_sweep_case(size)
}

fn jacobian_case(size: Index) -> Result<AdCostCase> {
    Ok(AdCostCase {
        key: "jacobian",
        scenario: AdCostScenario::Jacobian,
        size,
        sweep_count: size,
        original: build_jacobian_original_function(size)?,
        augmented: build_jacobian_augmented_function(size)?,
    })
}

pub fn build_jacobian_original_function(size: Index) -> Result<SXFunction> {
    let x = SXMatrix::sym_dense("x", size, 1)?;
    let outputs = vector_chain_from_vector(&x)?;
    single_io_function("ad_jacobian_original", "x", x, "outputs", outputs)
}

pub fn build_jacobian_augmented_function(size: Index) -> Result<SXFunction> {
    let x = SXMatrix::sym_dense("x", size, 1)?;
    let outputs = vector_chain_from_vector(&x)?;
    let jacobian = outputs.jacobian(&x)?;
    named_function(
        "ad_jacobian_with_jacobian",
        vec![("x", x)],
        vec![("outputs", outputs), ("jacobian", jacobian)],
    )
}

pub fn build_jacobian_case(size: Index) -> Result<AdCostCase> {
    jacobian_case(size)
}

fn hessian_case(size: Index) -> Result<AdCostCase> {
    Ok(AdCostCase {
        key: "hessian",
        scenario: AdCostScenario::Hessian,
        size,
        sweep_count: size,
        original: build_hessian_original_function(size)?,
        augmented: build_hessian_augmented_function(size)?,
    })
}

pub fn build_hessian_original_function(size: Index) -> Result<SXFunction> {
    let x = SXMatrix::sym_dense("x", size, 1)?;
    let value = SXMatrix::scalar(scalar_chain_from_vector(&x));
    single_io_function("ad_hessian_original", "x", x, "value", value)
}

pub fn build_hessian_augmented_function(size: Index) -> Result<SXFunction> {
    let x = SXMatrix::sym_dense("x", size, 1)?;
    let value = SXMatrix::scalar(scalar_chain_from_vector(&x));
    let hessian = value.hessian_with_options(&x, HessianOptions::default())?;
    named_function(
        "ad_hessian_with_hessian",
        vec![("x", x)],
        vec![("value", value), ("hessian", hessian)],
    )
}

pub fn build_hessian_augmented_function_with_strategy(
    size: Index,
    strategy: HessianStrategy,
) -> Result<SXFunction> {
    let x = SXMatrix::sym_dense("x", size, 1)?;
    let value = SXMatrix::scalar(scalar_chain_from_vector(&x));
    let hessian = value.hessian_with_options(&x, HessianOptions::with_strategy(strategy))?;
    named_function(
        &format!("ad_hessian_with_hessian_{}", strategy.key()),
        vec![("x", x)],
        vec![("value", value), ("hessian", hessian)],
    )
}

pub fn build_hessian_case(size: Index) -> Result<AdCostCase> {
    hessian_case(size)
}

pub fn hessian_strategy_cases(size: Index) -> Result<Vec<HessianStrategyCase>> {
    HessianStrategy::ALL
        .into_iter()
        .map(|strategy| {
            Ok(HessianStrategyCase {
                strategy,
                function: build_hessian_augmented_function_with_strategy(size, strategy)?,
            })
        })
        .collect()
}

pub fn hessian_strategy_expectation(strategy: HessianStrategy) -> HessianStrategyExpectation {
    match strategy {
        HessianStrategy::LowerTriangleByColumn
        | HessianStrategy::LowerTriangleSelectedOutputs
        | HessianStrategy::LowerTriangleColored => HessianStrategyExpectation { exact_ops: 11_159 },
    }
}

pub fn ad_cost_cases() -> Result<Vec<AdCostCase>> {
    Ok(vec![
        reverse_gradient_case(AD_REVERSE_SIZE)?,
        forward_sweep_case(AD_FORWARD_SIZE)?,
        jacobian_case(AD_JACOBIAN_SIZE)?,
        hessian_case(AD_HESSIAN_SIZE)?,
    ])
}

pub fn ad_cost_artifacts() -> Result<Vec<ExampleArtifact>> {
    let mut artifacts = Vec::new();
    for case in ad_cost_cases()? {
        artifacts.push(ExampleArtifact {
            module_name: case.original.name().to_string(),
            function: case.original,
        });
        artifacts.push(ExampleArtifact {
            module_name: case.augmented.name().to_string(),
            function: case.augmented,
        });
    }
    for strategy_case in hessian_strategy_cases(AD_HESSIAN_SIZE)? {
        artifacts.push(ExampleArtifact {
            module_name: strategy_case.function.name().to_string(),
            function: strategy_case.function,
        });
    }
    Ok(artifacts)
}

pub fn all_examples() -> Result<Vec<ExampleArtifact>> {
    let mut artifacts = Vec::new();
    artifacts.extend(rosenbrock_bundle()?);
    artifacts.extend(casadi_rosenbrock_nlp_bundle()?);
    artifacts.extend(simple_nlp_bundle()?);
    artifacts.extend(constrained_rosenbrock_bundle()?);
    artifacts.extend(hs021_bundle()?);
    artifacts.extend(hs035_bundle()?);
    artifacts.extend(hs071_bundle()?);
    artifacts.extend(parameterized_quadratic_bundle()?);
    artifacts.extend(hanging_chain_bundle()?);
    Ok(artifacts)
}

pub fn all_generated_artifacts() -> Result<Vec<ExampleArtifact>> {
    let mut artifacts = all_examples()?;
    artifacts.extend(ad_cost_artifacts()?);
    Ok(artifacts)
}
