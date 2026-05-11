use super::helpers::*;
use super::*;

pub(super) fn tp029() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), (), SX, _, _>(
        metadata(
            "schittkowski_tp029",
            "tp029",
            "Schittkowski TP029 cubic product objective with ellipsoidal inequality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), (), SX, _>(
                "schittkowski_tp029",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: -x0 * x1 * x2,
                        equalities: (),
                        inequalities: x0.sqr() + 2.0 * x1.sqr() + 4.0 * x2.sqr() - 48.0,
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [1.0, 1.0, 1.0],
                },
                parameters: (),
                bounds: scalar_inequality_upper_bound_for_vec3(),
            })
        },
        objective_validation(
            -16.0 * 2.0_f64.sqrt(),
            1e-7,
            PRIMAL_TOL,
            DUAL_TOL,
            Some(COMPLEMENTARITY_TOL),
        ),
    )
}
