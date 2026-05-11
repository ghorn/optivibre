use super::helpers::*;
use super::*;

pub(super) fn tp008() -> ProblemCase {
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
