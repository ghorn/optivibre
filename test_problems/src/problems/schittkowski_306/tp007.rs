use super::helpers::*;
use super::*;

pub(super) fn tp007() -> ProblemCase {
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
