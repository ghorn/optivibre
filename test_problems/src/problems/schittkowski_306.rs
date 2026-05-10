use optimization::{SymbolicNlpOutputs, TypedRuntimeNlpBounds};
use sx_core::SX;

use super::{
    CaseMetadata, Pair, ProblemCase, TypedProblemData, VecN, exact_solution_validation,
    make_typed_case, objective_validation, symbolic_compile,
};
use crate::manifest::ProblemTestSet;

const SOURCE: &str = "schittkowski_306";
const X_TOL: f64 = 1e-5;
const OBJECTIVE_TOL: f64 = 1e-8;
const PRIMAL_TOL: f64 = 1e-7;
const DUAL_TOL: f64 = 1e-5;
const COMPLEMENTARITY_TOL: f64 = 1e-6;

pub(crate) fn cases() -> Vec<ProblemCase> {
    vec![
        tp001(),
        tp002(),
        tp003(),
        tp004(),
        tp005(),
        tp006(),
        tp007(),
        tp008(),
        tp009(),
        tp010(),
    ]
}

fn metadata(id: &'static str, variant: &'static str, description: &'static str) -> CaseMetadata {
    CaseMetadata::new(id, variant, variant, SOURCE, description, false)
        .with_test_set(ProblemTestSet::Schittkowski306)
}

fn tp001() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), (), _, _>(
        metadata(
            "schittkowski_tp001",
            "tp001",
            "Schittkowski TP001 Rosenbrock problem with one lower bound",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), (), _>(
                "schittkowski_tp001",
                |x, ()| SymbolicNlpOutputs {
                    objective: rosenbrock_objective(x.x, x.y),
                    equalities: (),
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: -2.0, y: 1.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: None,
                        y: Some(-1.5),
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[1.0, 1.0],
            X_TOL,
            0.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}

fn tp002() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), (), _, _>(
        metadata(
            "schittkowski_tp002",
            "tp002",
            "Schittkowski TP002 Rosenbrock problem with active lower bound",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), (), _>(
                "schittkowski_tp002",
                |x, ()| SymbolicNlpOutputs {
                    objective: rosenbrock_objective(x.x, x.y),
                    equalities: (),
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: -2.0, y: 1.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: None,
                        y: Some(1.5),
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &tp002_solution(),
            5e-5,
            tp002_objective(),
            1e-7,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}

fn tp003() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), (), _, _>(
        metadata(
            "schittkowski_tp003",
            "tp003",
            "Schittkowski TP003 nearly linear objective with lower bound",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), (), _>(
                "schittkowski_tp003",
                |x, ()| SymbolicNlpOutputs {
                    objective: x.y + 1e-5 * (x.y - x.x).sqr(),
                    equalities: (),
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 10.0, y: 1.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: None,
                        y: Some(0.0),
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[0.0, 0.0],
            X_TOL,
            0.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}

fn tp004() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), (), _, _>(
        metadata(
            "schittkowski_tp004",
            "tp004",
            "Schittkowski TP004 cubic objective with lower bounds",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), (), _>(
                "schittkowski_tp004",
                |x, ()| SymbolicNlpOutputs {
                    objective: (x.x + 1.0).powf(3.0) / 3.0 + x.y,
                    equalities: (),
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 1.125, y: 0.125 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: Some(1.0),
                        y: Some(0.0),
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[1.0, 0.0],
            X_TOL,
            8.0 / 3.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}

fn tp005() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), (), _, _>(
        metadata(
            "schittkowski_tp005",
            "tp005",
            "Schittkowski TP005 bounded sinusoidal quadratic objective",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), (), _>(
                "schittkowski_tp005",
                |x, ()| SymbolicNlpOutputs {
                    objective: (x.x + x.y).sin() + (x.x - x.y).sqr() - 1.5 * x.x + 2.5 * x.y + 1.0,
                    equalities: (),
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 0.0, y: 0.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: Some(-1.5),
                        y: Some(-3.0),
                    }),
                    variable_upper: Some(Pair {
                        x: Some(4.0),
                        y: Some(3.0),
                    }),
                    inequality_lower: None,
                    inequality_upper: None,
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &tp005_solution(),
            X_TOL,
            -3.0_f64.sqrt() / 2.0 - std::f64::consts::PI / 3.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}

fn tp006() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), SX, (), _, _>(
        metadata(
            "schittkowski_tp006",
            "tp006",
            "Schittkowski TP006 Rosenbrock equality-constrained problem",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), SX, (), _>(
                "schittkowski_tp006",
                |x, ()| SymbolicNlpOutputs {
                    objective: (1.0 - x.x).sqr(),
                    equalities: 10.0 * (x.y - x.x.sqr()),
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: -1.2, y: 1.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(
            &[1.0, 1.0],
            X_TOL,
            0.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}

fn tp007() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), SX, (), _, _>(
        metadata(
            "schittkowski_tp007",
            "tp007",
            "Schittkowski TP007 logarithmic objective with circular equality",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), SX, (), _>(
                "schittkowski_tp007",
                |x, ()| SymbolicNlpOutputs {
                    objective: (1.0 + x.x.sqr()).log() - x.y,
                    equalities: (1.0 + x.x.sqr()).sqr() + x.y.sqr() - 4.0,
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 2.0, y: 2.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(
            &[0.0, 3.0_f64.sqrt()],
            X_TOL,
            -3.0_f64.sqrt(),
            1e-7,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}

fn tp008() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), VecN<SX, 2>, (), _, _>(
        metadata(
            "schittkowski_tp008",
            "tp008",
            "Schittkowski TP008 constant objective with two nonlinear equalities",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), VecN<SX, 2>, (), _>(
                "schittkowski_tp008",
                |x, ()| SymbolicNlpOutputs {
                    objective: -SX::one(),
                    equalities: VecN {
                        values: [x.x.sqr() + x.y.sqr() - 25.0, x.x * x.y - 9.0],
                    },
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 2.0, y: 1.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        objective_validation(-1.0, OBJECTIVE_TOL, PRIMAL_TOL, DUAL_TOL, None),
    )
}

fn tp009() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), SX, (), _, _>(
        metadata(
            "schittkowski_tp009",
            "tp009",
            "Schittkowski TP009 sinusoidal objective with linear equality",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), SX, (), _>(
                "schittkowski_tp009",
                |x, ()| SymbolicNlpOutputs {
                    objective: (std::f64::consts::PI * x.x / 12.0).sin()
                        * (std::f64::consts::PI * x.y / 16.0).cos(),
                    equalities: 4.0 * x.x - 3.0 * x.y,
                    inequalities: (),
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 0.0, y: 0.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(
            &[-3.0, -4.0],
            X_TOL,
            -0.5,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}

fn tp010() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), SX, _, _>(
        metadata(
            "schittkowski_tp010",
            "tp010",
            "Schittkowski TP010 linear objective with one nonlinear inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), SX, _>(
                "schittkowski_tp010",
                |x, ()| SymbolicNlpOutputs {
                    objective: x.x - x.y,
                    equalities: (),
                    inequalities: 3.0 * x.x.sqr() - 2.0 * x.x * x.y + x.y.sqr() - 1.0,
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: -10.0, y: 10.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: None,
                    variable_upper: None,
                    inequality_lower: Some(None),
                    inequality_upper: Some(Some(0.0)),
                    scaling: None,
                },
            })
        },
        exact_solution_validation(
            &[0.0, 1.0],
            X_TOL,
            -1.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}

fn rosenbrock_objective(x: SX, y: SX) -> SX {
    100.0 * (y - x.sqr()).sqr() + (1.0 - x).sqr()
}

fn tp002_solution() -> [f64; 2] {
    let w1 = (598.0_f64 / 1200.0).sqrt();
    let angle = (0.0025 / w1.powi(3)).acos() / 3.0;
    let x = 2.0 * w1 * angle.cos();
    [x, 1.5]
}

fn tp002_objective() -> f64 {
    let [x, y] = tp002_solution();
    100.0 * (y - x.powi(2)).powi(2) + (1.0 - x).powi(2)
}

fn tp005_solution() -> [f64; 2] {
    let x = 0.5 - std::f64::consts::PI / 3.0;
    [x, x - 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_objective_values_match_validated_values() {
        assert!((rosenbrock_value(1.0, 1.0) - 0.0).abs() <= 1e-14);
        let [x2, y2] = tp002_solution();
        assert!((tp002_objective() - rosenbrock_value(x2, y2)).abs() <= 1e-14);
        assert!((tp003_value(0.0, 0.0) - 0.0).abs() <= 1e-14);
        assert!((tp004_value(1.0, 0.0) - 8.0 / 3.0).abs() <= 1e-14);
        let [x5, y5] = tp005_solution();
        let f5 = tp005_value(x5, y5);
        assert!((f5 - (-3.0_f64.sqrt() / 2.0 - std::f64::consts::PI / 3.0)).abs() <= 1e-12);
        assert!((tp006_value(1.0, 1.0) - 0.0).abs() <= 1e-14);
        assert!((tp006_eq(1.0, 1.0) - 0.0).abs() <= 1e-14);
        assert!((tp007_value(0.0, 3.0_f64.sqrt()) + 3.0_f64.sqrt()).abs() <= 1e-14);
        assert!((tp007_eq(0.0, 3.0_f64.sqrt()) - 0.0).abs() <= 1e-14);
        let [x8, y8] = tp008_solution();
        assert!((tp008_eq1(x8, y8) - 0.0).abs() <= 1e-14);
        assert!((tp008_eq2(x8, y8) - 0.0).abs() <= 1e-14);
        assert!((tp009_value(-3.0, -4.0) + 0.5).abs() <= 1e-14);
        assert!((tp009_eq(-3.0, -4.0) - 0.0).abs() <= 1e-14);
        assert!((tp010_value(0.0, 1.0) + 1.0).abs() <= 1e-14);
        assert!((tp010_ineq(0.0, 1.0) - 0.0).abs() <= 1e-14);
    }

    #[test]
    fn documented_starting_points_evaluate_finitely() {
        let values = [
            rosenbrock_value(-2.0, 1.0),
            tp003_value(10.0, 1.0),
            tp004_value(1.125, 0.125),
            tp005_value(0.0, 0.0),
            tp006_value(-1.2, 1.0),
            tp006_eq(-1.2, 1.0),
            tp007_value(2.0, 2.0),
            tp007_eq(2.0, 2.0),
            tp008_eq1(2.0, 1.0),
            tp008_eq2(2.0, 1.0),
            tp009_value(0.0, 0.0),
            tp009_eq(0.0, 0.0),
            tp010_value(-10.0, 10.0),
            tp010_ineq(-10.0, 10.0),
        ];
        assert!(values.into_iter().all(f64::is_finite));
    }

    fn rosenbrock_value(x: f64, y: f64) -> f64 {
        100.0 * (y - x.powi(2)).powi(2) + (1.0 - x).powi(2)
    }

    fn tp003_value(x: f64, y: f64) -> f64 {
        y + 1e-5 * (y - x).powi(2)
    }

    fn tp004_value(x: f64, y: f64) -> f64 {
        (x + 1.0).powi(3) / 3.0 + y
    }

    fn tp005_value(x: f64, y: f64) -> f64 {
        (x + y).sin() + (x - y).powi(2) - 1.5 * x + 2.5 * y + 1.0
    }

    fn tp006_value(x: f64, _y: f64) -> f64 {
        (1.0 - x).powi(2)
    }

    fn tp006_eq(x: f64, y: f64) -> f64 {
        10.0 * (y - x.powi(2))
    }

    fn tp007_value(x: f64, y: f64) -> f64 {
        (1.0 + x.powi(2)).ln() - y
    }

    fn tp007_eq(x: f64, y: f64) -> f64 {
        (1.0 + x.powi(2)).powi(2) + y.powi(2) - 4.0
    }

    fn tp008_solution() -> [f64; 2] {
        let x = ((25.0 + 301.0_f64.sqrt()) / 2.0).sqrt();
        [x, 9.0 / x]
    }

    fn tp008_eq1(x: f64, y: f64) -> f64 {
        x.powi(2) + y.powi(2) - 25.0
    }

    fn tp008_eq2(x: f64, y: f64) -> f64 {
        x * y - 9.0
    }

    fn tp009_value(x: f64, y: f64) -> f64 {
        (std::f64::consts::PI * x / 12.0).sin() * (std::f64::consts::PI * y / 16.0).cos()
    }

    fn tp009_eq(x: f64, y: f64) -> f64 {
        4.0 * x - 3.0 * y
    }

    fn tp010_value(x: f64, y: f64) -> f64 {
        x - y
    }

    fn tp010_ineq(x: f64, y: f64) -> f64 {
        3.0 * x.powi(2) - 2.0 * x * y + y.powi(2) - 1.0
    }
}
