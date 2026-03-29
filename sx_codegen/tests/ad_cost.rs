use examples_source::{
    AD_HESSIAN_SIZE, AdCostScenario, ad_cost_cases, hessian_strategy_cases,
    hessian_strategy_expectation,
};
use sx_codegen::lower_function;
use sx_core::HessianStrategy;

fn lowered_op_count(function: &sx_core::SXFunction) -> usize {
    lower_function(function).unwrap().instructions.len()
}

#[test]
fn ad_cost_baselines_match_expected_lowered_op_counts() {
    for case in ad_cost_cases().unwrap() {
        let expectations = case.expectations();
        let original_ops = lowered_op_count(&case.original);
        let augmented_ops = lowered_op_count(&case.augmented);
        assert_eq!(
            original_ops, expectations.exact_original_ops,
            "{} original op count changed",
            case.key
        );
        assert_eq!(
            augmented_ops, expectations.exact_augmented_ops,
            "{} augmented op count changed",
            case.key
        );
    }
}

#[test]
fn directional_ad_costs_stay_within_constant_factor_of_primal_eval() {
    for case in ad_cost_cases().unwrap() {
        if !matches!(
            case.scenario,
            AdCostScenario::ReverseGradient | AdCostScenario::ForwardSweep
        ) {
            continue;
        }

        let original_ops = lowered_op_count(&case.original);
        let augmented_ops = lowered_op_count(&case.augmented);
        let ratio = augmented_ops as f64 / original_ops as f64;

        assert!(
            ratio
                <= case
                    .expectations()
                    .directional_ratio_limit
                    .expect("directional ratio limit must exist"),
            "{} exceeded the expected AD constant factor: original_ops={}, augmented_ops={}, ratio={ratio:.3}",
            case.key,
            original_ops,
            augmented_ops,
        );
    }
}

#[test]
fn higher_order_costs_scale_with_the_expected_number_of_sweeps() {
    for case in ad_cost_cases().unwrap() {
        if !matches!(
            case.scenario,
            AdCostScenario::Jacobian | AdCostScenario::Hessian
        ) {
            continue;
        }

        let original_ops = lowered_op_count(&case.original);
        let augmented_ops = lowered_op_count(&case.augmented);
        let normalized_ratio =
            augmented_ops as f64 / (original_ops as f64 * case.sweep_count as f64);

        assert!(
            normalized_ratio
                <= case
                    .expectations()
                    .normalized_ratio_limit
                    .expect("normalized ratio limit must exist"),
            "{} scaled worse than expected: original_ops={}, augmented_ops={}, sweep_count={}, normalized_ratio={normalized_ratio:.3}",
            case.key,
            original_ops,
            augmented_ops,
            case.sweep_count,
        );
    }
}

#[test]
fn hessian_strategy_op_counts_are_pinned_and_identical() {
    let strategy_cases = hessian_strategy_cases(AD_HESSIAN_SIZE).unwrap();
    let mut reference = None;
    for case in strategy_cases {
        let op_count = lowered_op_count(&case.function);
        assert_eq!(
            op_count,
            hessian_strategy_expectation(case.strategy).exact_ops,
            "{} op count changed",
            case.strategy.key()
        );
        if let Some(reference_count) = reference {
            assert_eq!(
                op_count,
                reference_count,
                "{} no longer lowers to the same op count as strategy 1",
                case.strategy.key()
            );
        } else {
            assert_eq!(case.strategy, HessianStrategy::LowerTriangleByColumn);
            reference = Some(op_count);
        }
    }
}
