use super::helpers::*;
use super::*;

pub(super) fn tp042() -> ProblemCase {
    make_typed_case::<VecN<SX, 4>, (), VecN<SX, 2>, (), _, _>(
        metadata(
            "schittkowski_tp042",
            "tp042",
            "Schittkowski TP042 distance objective with one linear and one nonlinear equality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 4>, (), VecN<SX, 2>, (), _>(
                "schittkowski_tp042",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    let x3 = x.values[3];
                    SymbolicNlpOutputs {
                        objective: (x0 - 1.0).sqr()
                            + (x1 - 2.0).sqr()
                            + (x2 - 3.0).sqr()
                            + (x3 - 4.0).sqr(),
                        equalities: VecN {
                            values: [x0 - 2.0, x2.sqr() + x3.sqr() - 2.0],
                        },
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN { values: [1.0; 4] },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(
            &[2.0, 2.0, 0.72_f64.sqrt(), 1.28_f64.sqrt()],
            X_TOL,
            28.0 - 10.0 * 2.0_f64.sqrt(),
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
