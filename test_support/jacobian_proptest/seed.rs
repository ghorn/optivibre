use proptest::prelude::*;

use crate::jacobian_proptest::generate::{ProfileMode, PropertyScenario};

#[derive(Clone, Debug)]
pub enum ExprSeed {
    Const(i16),
    Input(u8),
    Unary {
        op: u8,
        arg: Box<ExprSeed>,
    },
    Binary {
        op: u8,
        lhs: Box<ExprSeed>,
        rhs: Box<ExprSeed>,
    },
    Call {
        helper: u8,
        output: u8,
        args: Vec<ExprSeed>,
    },
}

#[derive(Clone, Debug)]
pub struct FunctionSeed {
    pub input_count: u8,
    pub outputs: Vec<ExprSeed>,
}

#[derive(Clone, Debug)]
pub struct CaseSeed {
    pub root_input_count: u8,
    pub helper_count_hint: u8,
    pub helpers: Vec<FunctionSeed>,
    pub root_outputs: Vec<ExprSeed>,
    pub input_box_kinds: Vec<u8>,
    pub sample_codes: Vec<i16>,
    pub preferred_helper: u8,
    pub profile_hint: u8,
}

fn expr_seed_strategy(max_nodes: u32, call_weight: u32) -> BoxedStrategy<ExprSeed> {
    let leaf = prop_oneof![
        any::<i16>().prop_map(ExprSeed::Const),
        any::<u8>().prop_map(ExprSeed::Input),
    ];
    leaf.prop_recursive(max_nodes, max_nodes.saturating_mul(8), 2, move |inner| {
        prop_oneof![
            2 => (any::<u8>(), inner.clone()).prop_map(|(op, arg)| ExprSeed::Unary {
                op,
                arg: Box::new(arg),
            }),
            3 => (any::<u8>(), inner.clone(), inner.clone()).prop_map(|(op, lhs, rhs)| ExprSeed::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            }),
            call_weight => (
                any::<u8>(),
                any::<u8>(),
                prop::collection::vec(inner, 0..=4),
            )
                .prop_map(|(helper, output, args)| ExprSeed::Call {
                    helper,
                    output,
                    args,
                }),
        ]
    })
    .boxed()
}

pub fn case_seed_strategy(scenario: &'static PropertyScenario) -> BoxedStrategy<CaseSeed> {
    let config = scenario.generator;
    let call_weight = match scenario.profile_mode {
        ProfileMode::ForceCallHeavy => 6,
        ProfileMode::Mixed => 2,
    };
    let exprs = expr_seed_strategy(config.max_nodes_per_expr as u32, call_weight);
    let min_helpers = match scenario.profile_mode {
        ProfileMode::ForceCallHeavy => 2,
        ProfileMode::Mixed => 0,
    };
    let min_outputs = scenario.requirements.min_root_outputs.max(1);
    (
        any::<u8>(),
        any::<u8>(),
        prop::collection::vec(
            (
                any::<u8>(),
                prop::collection::vec(exprs.clone(), 1..=config.max_helper_outputs),
            )
                .prop_map(|(input_count, outputs)| FunctionSeed {
                    input_count,
                    outputs,
                }),
            min_helpers..=config.max_helpers,
        ),
        prop::collection::vec(exprs, min_outputs..=config.max_outputs),
        prop::collection::vec(any::<u8>(), 1..=config.max_inputs),
        prop::collection::vec(any::<i16>(), 1..=config.max_inputs),
        any::<u8>(),
        any::<u8>(),
    )
        .prop_map(
            |(
                root_input_count,
                helper_count_hint,
                helpers,
                root_outputs,
                input_box_kinds,
                sample_codes,
                preferred_helper,
                profile_hint,
            )| CaseSeed {
                root_input_count,
                helper_count_hint,
                helpers,
                root_outputs,
                input_box_kinds,
                sample_codes,
                preferred_helper,
                profile_hint,
            },
        )
        .boxed()
}
