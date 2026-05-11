use super::helpers::*;
use super::*;

pub(super) fn tp027() -> ProblemCase {
    make_typed_case::<VecN<SX, 3>, (), SX, (), _, _>(
        metadata(
            "schittkowski_tp027",
            "tp027",
            "Schittkowski TP027 Rosenbrock objective with one nonlinear equality",
        ),
        |options| {
            let compiled = symbolic_compile::<VecN<SX, 3>, (), SX, (), _>(
                "schittkowski_tp027",
                |x, ()| {
                    let x0 = x.values[0];
                    let x1 = x.values[1];
                    let x2 = x.values[2];
                    SymbolicNlpOutputs {
                        objective: rosenbrock_objective(x0, x1),
                        equalities: x0 + x2.sqr() + 1.0,
                        inequalities: (),
                    }
                },
                options,
            )?;
            Ok(TypedProblemData {
                compiled,
                x0: VecN {
                    values: [2.0, 2.0, 2.0],
                },
                parameters: (),
                bounds: TypedRuntimeNlpBounds::default(),
            })
        },
        exact_solution_validation(
            &[-1.0, 1.0, 0.0],
            X_TOL,
            4.0,
            OBJECTIVE_TOL,
            PRIMAL_TOL,
            DUAL_TOL,
            None,
        ),
    )
}
