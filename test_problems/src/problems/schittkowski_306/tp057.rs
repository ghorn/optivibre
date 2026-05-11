use super::helpers::*;
use super::*;

pub(super) fn tp057() -> ProblemCase {
    make_typed_case::<Pair<SX>, (), (), SX, _, _>(
        metadata(
            "schittkowski_tp057",
            "tp057",
            "Schittkowski TP057 two-parameter exponential data fit with one nonlinear inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<Pair<SX>, (), (), SX, _>(
                "schittkowski_tp057",
                |x, ()| {
                    let (a, b) = tp057_data();
                    let mut objective = SX::zero();
                    for i in 0..44 {
                        let residual = b[i] - x.x - (0.49 - x.x) * (-(x.y * (a[i] - 8.0))).exp();
                        objective += residual.sqr();
                    }
                    SymbolicNlpOutputs {
                        objective,
                        equalities: (),
                        inequalities: x.x * x.y - 0.49 * x.y + 0.09,
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: Pair { x: 0.42, y: 5.0 },
                parameters: (),
                bounds: TypedRuntimeNlpBounds {
                    variable_lower: Some(Pair {
                        x: Some(0.4),
                        y: Some(-4.0),
                    }),
                    variable_upper: None,
                    inequality_lower: None,
                    inequality_upper: Some(Some(0.0)),
                    scaling: None,
                },
            })
        },
        objective_validation(
            0.0284596697213,
            1e-8,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
