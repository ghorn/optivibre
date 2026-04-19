use std::cmp::max;
use std::collections::BTreeMap;

use crate::jacobian_proptest::ast::{
    BinaryOpAst, CaseProfile, ExprAst, FunctionAst, GeneratedCase, OperatorTier, UnaryOpAst,
    compute_case_features,
};
use crate::jacobian_proptest::domain::{
    InputBox, InputBoxFamily, RangeCert, add as add_range, cos as cos_range, div as div_range,
    exp as exp_range, family_interval, log as log_range, mul as mul_range, negate as negate_range,
    sin as sin_range, sqrt as sqrt_range, square as square_range, sub as sub_range,
};
use crate::jacobian_proptest::seed::{CaseSeed, ExprSeed, FunctionSeed};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileMode {
    Mixed,
    ForceCallHeavy,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GeneratedCaseRequirements {
    pub min_root_outputs: usize,
    pub require_calls: bool,
    pub require_multi_output_helper: bool,
    pub require_repeated_helper_calls: bool,
    pub require_nested_helper_calls: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeneratorConfig {
    pub max_inputs: usize,
    pub max_outputs: usize,
    pub max_helpers: usize,
    pub max_helper_outputs: usize,
    pub max_nodes_per_expr: usize,
    pub max_call_depth: usize,
    pub exp_input_cap: f64,
    pub positive_margin: f64,
    pub nonzero_margin: f64,
    pub output_abs_cap: f64,
    pub fd_step: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PropertyScenario {
    pub name: &'static str,
    pub accepted_cases: u32,
    pub max_global_rejects: u32,
    pub operator_tier: OperatorTier,
    pub generator: GeneratorConfig,
    pub profile_mode: ProfileMode,
    pub requirements: GeneratedCaseRequirements,
}

#[derive(Clone, Debug, Default)]
pub struct CoverageCounters {
    pub accepted_valid_cases: u32,
    pub static_rejects: u32,
    pub runtime_nonfinite_rejects: u32,
    pub fd_nonfinite_rejects: u32,
    pub accepted_by_profile: BTreeMap<&'static str, u32>,
    pub accepted_call_cases: u32,
    pub accepted_repeated_helper_cases: u32,
    pub accepted_multi_output_helper_cases: u32,
    pub accepted_tier2_cases: u32,
}

#[derive(Clone, Debug, Default)]
pub struct CoverageSnapshot {
    pub accepted_valid_cases: u32,
    pub static_rejects: u32,
    pub runtime_nonfinite_rejects: u32,
    pub fd_nonfinite_rejects: u32,
}

impl CoverageCounters {
    pub fn snapshot(&self) -> CoverageSnapshot {
        CoverageSnapshot {
            accepted_valid_cases: self.accepted_valid_cases,
            static_rejects: self.static_rejects,
            runtime_nonfinite_rejects: self.runtime_nonfinite_rejects,
            fd_nonfinite_rejects: self.fd_nonfinite_rejects,
        }
    }

    pub fn record_accept(&mut self, case: &GeneratedCase) {
        self.accepted_valid_cases += 1;
        *self
            .accepted_by_profile
            .entry(case.profile.label())
            .or_default() += 1;
        if case.features.call_count > 0 {
            self.accepted_call_cases += 1;
        }
        if case.features.repeated_helper_calls > 0 {
            self.accepted_repeated_helper_cases += 1;
        }
        if case.features.helper_multi_output_count > 0 {
            self.accepted_multi_output_helper_cases += 1;
        }
        if case.features.tier2_op_count > 0 {
            self.accepted_tier2_cases += 1;
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RejectReason {
    Structural(String),
    Domain(String),
    Scenario(String),
}

impl CaseProfile {
    pub fn label(self) -> &'static str {
        match self {
            Self::Mixed => "mixed",
            Self::CallHeavy => "call_heavy",
            Self::BinaryHeavy => "binary_heavy",
            Self::UnaryChain => "unary_chain",
            Self::RepeatedCalls => "repeated_calls",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    AnyFinite,
    PositiveFinite,
    NonZeroFinite,
    UnitBounded,
}

#[derive(Clone, Debug)]
struct CertifiedExpr {
    expr: ExprAst,
    range: RangeCert,
}

pub fn generate_case_from_seed(
    scenario: &PropertyScenario,
    seed: &CaseSeed,
) -> Result<GeneratedCase, RejectReason> {
    let config = scenario.generator;
    let profile = match scenario.profile_mode {
        ProfileMode::ForceCallHeavy => CaseProfile::CallHeavy,
        ProfileMode::Mixed => choose_profile(seed.profile_hint),
    };
    let root_input_count = clamp_count(seed.root_input_count, 1, config.max_inputs);
    let input_box = build_input_box(root_input_count, seed);
    let sample_input = build_sample_input(&input_box, seed, config.fd_step);
    let helper_count = max(
        required_helper_count(profile),
        clamp_count(seed.helper_count_hint, 0, config.max_helpers),
    );

    let mut helpers = Vec::with_capacity(helper_count);
    for helper_index in 0..helper_count {
        let helper_seed = seed
            .helpers
            .get(helper_index)
            .cloned()
            .unwrap_or_else(|| FunctionSeed {
                input_count: root_input_count as u8,
                outputs: vec![ExprSeed::Input(0)],
            });
        let helper = build_function_ast(
            &helper_seed,
            &helpers,
            profile,
            scenario.operator_tier,
            &config,
            helper_index,
        );
        helpers.push(helper);
    }

    if matches!(profile, CaseProfile::CallHeavy | CaseProfile::RepeatedCalls)
        && !helpers.is_empty()
        && !helpers.iter().any(|helper| helper.outputs.len() > 1)
    {
        helpers[0].outputs.push(ExprAst::Unary {
            op: UnaryOpAst::Sin,
            arg: Box::new(ExprAst::Input(0)),
        });
    }

    let root_seed = FunctionSeed {
        input_count: root_input_count as u8,
        outputs: seed.root_outputs.clone(),
    };
    let mut root = build_function_ast(
        &root_seed,
        &helpers,
        profile,
        scenario.operator_tier,
        &config,
        helpers.len(),
    );
    root.input_count = root_input_count;
    root.outputs.truncate(config.max_outputs);
    while root.outputs.len() < scenario.requirements.min_root_outputs.max(1) {
        root.outputs
            .push(ExprAst::Input(root.outputs.len() % root_input_count));
    }

    let certified_output_ranges = root
        .outputs
        .iter()
        .map(|expr| certify_expr(expr, &input_box.intervals, &helpers, &config))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| RejectReason::Domain("root output certification failed".into()))?;
    if certified_output_ranges
        .iter()
        .any(|range| !range.within_abs_cap(config.output_abs_cap))
    {
        return Err(RejectReason::Domain(
            "root output range exceeds configured absolute cap".into(),
        ));
    }

    let generated = compute_case_features(
        &root,
        &helpers,
        scenario.operator_tier,
        &input_box,
        certified_output_ranges,
        profile,
        sample_input,
    );
    validate_generated_case(scenario, &generated)?;
    Ok(generated)
}

fn choose_profile(hint: u8) -> CaseProfile {
    match hint % 20 {
        0..=7 => CaseProfile::Mixed,
        8..=11 => CaseProfile::CallHeavy,
        12..=15 => CaseProfile::RepeatedCalls,
        16..=18 => CaseProfile::BinaryHeavy,
        _ => CaseProfile::UnaryChain,
    }
}

fn required_helper_count(profile: CaseProfile) -> usize {
    match profile {
        CaseProfile::CallHeavy | CaseProfile::RepeatedCalls => 2,
        _ => 0,
    }
}

fn clamp_count(value: u8, min_value: usize, max_value: usize) -> usize {
    min_value.max((value as usize % (max_value + 1)).min(max_value))
}

fn build_input_box(count: usize, seed: &CaseSeed) -> InputBox {
    let mut intervals = Vec::with_capacity(count);
    for index in 0..count {
        let family = match seed.input_box_kinds.get(index).copied().unwrap_or(0) % 3 {
            0 => InputBoxFamily::SymmetricFinite,
            1 => InputBoxFamily::PositiveFinite,
            _ => InputBoxFamily::ShiftedFinite,
        };
        intervals.push(family_interval(family));
    }
    InputBox::new(intervals)
}

fn build_sample_input(input_box: &InputBox, seed: &CaseSeed, fd_step: f64) -> Vec<f64> {
    input_box
        .intervals
        .iter()
        .enumerate()
        .map(|(index, interval)| {
            let raw = f64::from(seed.sample_codes.get(index).copied().unwrap_or_default());
            let alpha = ((raw + 32768.0) / 65535.0).clamp(0.15, 0.85);
            let lower = interval.lower + 2.0 * fd_step;
            let upper = interval.upper - 2.0 * fd_step;
            if lower >= upper {
                0.5 * (interval.lower + interval.upper)
            } else {
                lower + alpha * (upper - lower)
            }
        })
        .collect()
}

fn build_function_ast(
    seed: &FunctionSeed,
    helpers_so_far: &[FunctionAst],
    profile: CaseProfile,
    operator_tier: OperatorTier,
    config: &GeneratorConfig,
    helper_index: usize,
) -> FunctionAst {
    let input_count = clamp_count(seed.input_count, 1, config.max_inputs);
    let local_input_box = InputBox::new(
        (0..input_count)
            .map(|slot| match (helper_index + slot) % 3 {
                0 => family_interval(InputBoxFamily::SymmetricFinite),
                1 => family_interval(InputBoxFamily::PositiveFinite),
                _ => family_interval(InputBoxFamily::ShiftedFinite),
            })
            .collect(),
    );
    let output_count = seed
        .outputs
        .len()
        .max(1)
        .min(config.max_helper_outputs.max(config.max_outputs));
    let mut outputs = Vec::with_capacity(output_count);
    for output_index in 0..output_count {
        let expr_seed = seed
            .outputs
            .get(output_index)
            .unwrap_or(&ExprSeed::Input(0));
        let expr = build_family_expr(
            expr_seed,
            Family::AnyFinite,
            config.max_nodes_per_expr,
            input_count,
            &local_input_box.intervals,
            helpers_so_far,
            operator_tier,
            profile,
            config,
            0,
        );
        outputs.push(expr.expr);
    }
    FunctionAst {
        input_count,
        outputs,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "generator state is kept explicit so property-case assembly stays transparent"
)]
fn build_family_expr(
    seed: &ExprSeed,
    family: Family,
    budget: usize,
    local_input_count: usize,
    local_input_ranges: &[RangeCert],
    helpers_so_far: &[FunctionAst],
    operator_tier: OperatorTier,
    profile: CaseProfile,
    config: &GeneratorConfig,
    call_depth: usize,
) -> CertifiedExpr {
    let fallback = fallback_leaf(seed, family, local_input_count, local_input_ranges);
    if budget <= 1 {
        return fallback;
    }
    let choice = seed_choice(seed);
    let candidate_count = match family {
        Family::AnyFinite => {
            if matches!(operator_tier, OperatorTier::Tier2Domain) {
                10
            } else {
                7
            }
        }
        Family::PositiveFinite => 5,
        Family::NonZeroFinite => 4,
        Family::UnitBounded => 3,
    };
    for offset in 0..candidate_count {
        let idx = (choice + offset) % candidate_count;
        if let Some(expr) = try_build_candidate(
            idx,
            seed,
            family,
            budget,
            local_input_count,
            local_input_ranges,
            helpers_so_far,
            operator_tier,
            profile,
            config,
            call_depth,
        ) {
            return expr;
        }
    }
    fallback
}

#[allow(clippy::too_many_arguments)]
fn try_build_candidate(
    idx: usize,
    seed: &ExprSeed,
    family: Family,
    budget: usize,
    local_input_count: usize,
    local_input_ranges: &[RangeCert],
    helpers_so_far: &[FunctionAst],
    operator_tier: OperatorTier,
    profile: CaseProfile,
    config: &GeneratorConfig,
    call_depth: usize,
) -> Option<CertifiedExpr> {
    let child_budget = budget.saturating_sub(1).max(1);
    match family {
        Family::AnyFinite => match idx {
            0 => call_leaf_for_family(
                seed,
                family,
                local_input_count,
                local_input_ranges,
                helpers_so_far,
                operator_tier,
                profile,
                config,
                call_depth,
                child_budget,
            )
            .or_else(|| {
                Some(fallback_leaf(
                    seed,
                    family,
                    local_input_count,
                    local_input_ranges,
                ))
            }),
            1 => unary_expr(
                UnaryOpAst::Neg,
                build_family_expr(
                    seed_child(seed, 0),
                    Family::AnyFinite,
                    child_budget,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                ),
                negate_range,
            ),
            2 => unary_simple(
                UnaryOpAst::Sin,
                build_family_expr(
                    seed_child(seed, 0),
                    Family::AnyFinite,
                    child_budget,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                ),
                sin_range,
            ),
            3 => unary_simple(
                UnaryOpAst::Cos,
                build_family_expr(
                    seed_child(seed, 0),
                    Family::AnyFinite,
                    child_budget,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                ),
                cos_range,
            ),
            4 => {
                let child = build_family_expr(
                    seed_child(seed, 0),
                    Family::UnitBounded,
                    child_budget,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                );
                unary_exp(child, config.exp_input_cap)
            }
            5 => binary_expr(
                BinaryOpAst::Add,
                build_family_expr(
                    seed_child(seed, 0),
                    Family::AnyFinite,
                    child_budget / 2 + 1,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                ),
                build_family_expr(
                    seed_child(seed, 1),
                    Family::AnyFinite,
                    child_budget / 2 + 1,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                ),
                add_range,
            ),
            6 => binary_expr(
                if matches!(profile, CaseProfile::BinaryHeavy) {
                    BinaryOpAst::Mul
                } else {
                    BinaryOpAst::Sub
                },
                build_family_expr(
                    seed_child(seed, 0),
                    Family::AnyFinite,
                    child_budget / 2 + 1,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                ),
                build_family_expr(
                    seed_child(seed, 1),
                    Family::AnyFinite,
                    child_budget / 2 + 1,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                ),
                if matches!(profile, CaseProfile::BinaryHeavy) {
                    mul_range
                } else {
                    sub_range
                },
            ),
            7 if matches!(operator_tier, OperatorTier::Tier2Domain) => {
                let child = build_family_expr(
                    seed_child(seed, 0),
                    Family::PositiveFinite,
                    child_budget,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                );
                unary_sqrt(child, config.positive_margin)
            }
            8 if matches!(operator_tier, OperatorTier::Tier2Domain) => {
                let child = build_family_expr(
                    seed_child(seed, 0),
                    Family::PositiveFinite,
                    child_budget,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                );
                unary_log(child, config.positive_margin)
            }
            9 if matches!(operator_tier, OperatorTier::Tier2Domain) => {
                let lhs = build_family_expr(
                    seed_child(seed, 0),
                    Family::AnyFinite,
                    child_budget / 2 + 1,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                );
                let rhs = build_family_expr(
                    seed_child(seed, 1),
                    Family::NonZeroFinite,
                    child_budget / 2 + 1,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                );
                binary_div(lhs, rhs, config.nonzero_margin)
            }
            _ => None,
        },
        Family::PositiveFinite => match idx {
            0 => call_leaf_for_family(
                seed,
                family,
                local_input_count,
                local_input_ranges,
                helpers_so_far,
                operator_tier,
                profile,
                config,
                call_depth,
                child_budget,
            ),
            1 => {
                let child = build_family_expr(
                    seed_child(seed, 0),
                    Family::AnyFinite,
                    child_budget,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                );
                let sq = unary_expr(UnaryOpAst::Square, child, square_range)?;
                let margin = RangeCert::exact(config.positive_margin + const_value(seed, 0).abs());
                binary_expr(
                    BinaryOpAst::Add,
                    CertifiedExpr {
                        expr: ExprAst::Const(margin.lower),
                        range: margin,
                    },
                    sq,
                    add_range,
                )
            }
            2 => {
                let child = build_family_expr(
                    seed_child(seed, 0),
                    Family::UnitBounded,
                    child_budget,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                );
                unary_exp(child, config.exp_input_cap)
            }
            3 => {
                let lhs = build_family_expr(
                    seed_child(seed, 0),
                    Family::PositiveFinite,
                    child_budget / 2 + 1,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                );
                let rhs = build_family_expr(
                    seed_child(seed, 1),
                    Family::PositiveFinite,
                    child_budget / 2 + 1,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                );
                binary_expr(BinaryOpAst::Add, lhs, rhs, add_range)
            }
            4 => {
                let lhs = build_family_expr(
                    seed_child(seed, 0),
                    Family::PositiveFinite,
                    child_budget / 2 + 1,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                );
                let rhs = build_family_expr(
                    seed_child(seed, 1),
                    Family::PositiveFinite,
                    child_budget / 2 + 1,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                );
                binary_expr(BinaryOpAst::Mul, lhs, rhs, mul_range)
            }
            _ => None,
        },
        Family::NonZeroFinite => match idx {
            0 => call_leaf_for_family(
                seed,
                family,
                local_input_count,
                local_input_ranges,
                helpers_so_far,
                operator_tier,
                profile,
                config,
                call_depth,
                child_budget,
            ),
            1 => Some(build_family_expr(
                seed_child(seed, 0),
                Family::PositiveFinite,
                child_budget,
                local_input_count,
                local_input_ranges,
                helpers_so_far,
                operator_tier,
                profile,
                config,
                call_depth,
            )),
            2 => {
                let positive = build_family_expr(
                    seed_child(seed, 0),
                    Family::PositiveFinite,
                    child_budget,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                );
                let sign = if const_value(seed, 1).is_sign_negative() {
                    -1.0
                } else {
                    1.0
                };
                binary_expr(
                    BinaryOpAst::Mul,
                    CertifiedExpr {
                        expr: ExprAst::Const(sign),
                        range: RangeCert::exact(sign),
                    },
                    positive,
                    mul_range,
                )
            }
            3 => Some(fallback_leaf(
                seed,
                Family::PositiveFinite,
                local_input_count,
                local_input_ranges,
            )),
            _ => None,
        },
        Family::UnitBounded => match idx {
            0 => call_leaf_for_family(
                seed,
                family,
                local_input_count,
                local_input_ranges,
                helpers_so_far,
                operator_tier,
                profile,
                config,
                call_depth,
                child_budget,
            ),
            1 => {
                let child = build_family_expr(
                    seed_child(seed, 0),
                    Family::AnyFinite,
                    child_budget,
                    local_input_count,
                    local_input_ranges,
                    helpers_so_far,
                    operator_tier,
                    profile,
                    config,
                    call_depth,
                );
                let sine = unary_simple(UnaryOpAst::Sin, child, sin_range)?;
                let scale = 0.2 + (const_value(seed, 1).abs() % 0.6);
                binary_expr(
                    BinaryOpAst::Mul,
                    CertifiedExpr {
                        expr: ExprAst::Const(scale),
                        range: RangeCert::exact(scale),
                    },
                    sine,
                    mul_range,
                )
            }
            2 => Some(fallback_leaf(
                seed,
                Family::UnitBounded,
                local_input_count,
                local_input_ranges,
            )),
            _ => None,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn call_leaf_for_family(
    seed: &ExprSeed,
    family: Family,
    local_input_count: usize,
    local_input_ranges: &[RangeCert],
    helpers_so_far: &[FunctionAst],
    operator_tier: OperatorTier,
    profile: CaseProfile,
    config: &GeneratorConfig,
    call_depth: usize,
    budget: usize,
) -> Option<CertifiedExpr> {
    if helpers_so_far.is_empty() || call_depth >= config.max_call_depth {
        return None;
    }
    let (helper_hint, output_hint, arg_seeds) = match seed {
        ExprSeed::Call {
            helper,
            output,
            args,
        } => (usize::from(*helper), usize::from(*output), args.as_slice()),
        _ => (seed_choice(seed), seed_choice(seed), &[][..]),
    };
    for helper_index in helper_visit_order(helpers_so_far.len(), helper_hint, profile) {
        let helper = &helpers_so_far[helper_index];
        let mut arg_exprs = Vec::with_capacity(helper.input_count);
        for arg_index in 0..helper.input_count {
            let arg_seed = arg_seeds
                .get(arg_index)
                .unwrap_or_else(|| seed_child(seed, arg_index));
            arg_exprs.push(build_family_expr(
                arg_seed,
                Family::AnyFinite,
                budget.saturating_sub(1).max(1),
                local_input_count,
                local_input_ranges,
                helpers_so_far,
                operator_tier,
                profile,
                config,
                call_depth + 1,
            ));
        }
        let arg_ranges = arg_exprs.iter().map(|arg| arg.range).collect::<Vec<_>>();
        for output_index in output_visit_order(helper.outputs.len(), output_hint) {
            let range = certify_expr(
                &helper.outputs[output_index],
                &arg_ranges,
                helpers_so_far,
                config,
            )?;
            if range_matches_family(range, family, config) {
                return Some(CertifiedExpr {
                    expr: ExprAst::Call {
                        helper: helper_index,
                        output: output_index,
                        args: arg_exprs.iter().map(|arg| arg.expr.clone()).collect(),
                    },
                    range,
                });
            }
        }
    }
    None
}

fn helper_visit_order(count: usize, preferred: usize, profile: CaseProfile) -> Vec<usize> {
    if count == 0 {
        return Vec::new();
    }
    let anchor = if matches!(profile, CaseProfile::CallHeavy | CaseProfile::RepeatedCalls) {
        preferred % count
    } else {
        count - 1 - (preferred % count)
    };
    let mut order = vec![anchor];
    for index in 0..count {
        if index != anchor {
            order.push(index);
        }
    }
    order
}

fn output_visit_order(count: usize, preferred: usize) -> Vec<usize> {
    if count == 0 {
        return Vec::new();
    }
    let anchor = preferred % count;
    let mut order = vec![anchor];
    for index in 0..count {
        if index != anchor {
            order.push(index);
        }
    }
    order
}

fn range_matches_family(range: RangeCert, family: Family, config: &GeneratorConfig) -> bool {
    match family {
        Family::AnyFinite => range.finite_guaranteed,
        Family::PositiveFinite => range.is_positive_with_margin(config.positive_margin),
        Family::NonZeroFinite => range.is_nonzero_with_margin(config.nonzero_margin),
        Family::UnitBounded => range.finite_guaranteed && range.lower >= -0.8 && range.upper <= 0.8,
    }
}

fn fallback_leaf(
    seed: &ExprSeed,
    family: Family,
    local_input_count: usize,
    local_input_ranges: &[RangeCert],
) -> CertifiedExpr {
    match family {
        Family::AnyFinite => match seed {
            ExprSeed::Input(index) => {
                let index = (*index as usize) % local_input_count;
                CertifiedExpr {
                    expr: ExprAst::Input(index),
                    range: local_input_ranges[index],
                }
            }
            _ => {
                let value = const_value(seed, 0);
                CertifiedExpr {
                    expr: ExprAst::Const(value),
                    range: RangeCert::exact(value),
                }
            }
        },
        Family::PositiveFinite => {
            let value = 0.5 + const_value(seed, 0).abs();
            CertifiedExpr {
                expr: ExprAst::Const(value),
                range: RangeCert::exact(value),
            }
        }
        Family::NonZeroFinite => {
            let value = 0.5 + const_value(seed, 0).abs();
            let signed = if const_value(seed, 1).is_sign_negative() {
                -value
            } else {
                value
            };
            CertifiedExpr {
                expr: ExprAst::Const(signed),
                range: RangeCert::exact(signed),
            }
        }
        Family::UnitBounded => {
            let value = (const_value(seed, 0) / 3.0).clamp(-0.8, 0.8);
            CertifiedExpr {
                expr: ExprAst::Const(value),
                range: RangeCert::exact(value),
            }
        }
    }
}

fn seed_choice(seed: &ExprSeed) -> usize {
    match seed {
        ExprSeed::Const(value) => value.unsigned_abs() as usize,
        ExprSeed::Input(index) => *index as usize,
        ExprSeed::Unary { op, .. } => *op as usize,
        ExprSeed::Binary { op, .. } => *op as usize,
        ExprSeed::Call { helper, output, .. } => usize::from(*helper) + usize::from(*output),
    }
}

fn const_value(seed: &ExprSeed, salt: usize) -> f64 {
    let base = match seed {
        ExprSeed::Const(value) => f64::from(*value) / 512.0,
        ExprSeed::Input(index) => f64::from(*index) / 32.0,
        ExprSeed::Unary { op, .. } => f64::from(*op) / 64.0,
        ExprSeed::Binary { op, .. } => f64::from(*op) / 64.0,
        ExprSeed::Call { helper, output, .. } => (f64::from(*helper) + f64::from(*output)) / 32.0,
    };
    (base + salt as f64 / 16.0).clamp(-1.5, 1.5)
}

fn seed_child(seed: &ExprSeed, child_index: usize) -> &ExprSeed {
    match seed {
        ExprSeed::Unary { arg, .. } => arg,
        ExprSeed::Binary { lhs, rhs, .. } => {
            if child_index == 0 {
                lhs
            } else {
                rhs
            }
        }
        ExprSeed::Call { args, .. } => args.get(child_index).unwrap_or(seed),
        ExprSeed::Const(_) | ExprSeed::Input(_) => seed,
    }
}

fn unary_expr(
    op: UnaryOpAst,
    child: CertifiedExpr,
    range_fn: impl Fn(RangeCert) -> Option<RangeCert>,
) -> Option<CertifiedExpr> {
    let range = range_fn(child.range)?;
    Some(CertifiedExpr {
        expr: ExprAst::Unary {
            op,
            arg: Box::new(child.expr),
        },
        range,
    })
}

fn unary_simple(
    op: UnaryOpAst,
    child: CertifiedExpr,
    range_fn: impl Fn(RangeCert) -> RangeCert,
) -> Option<CertifiedExpr> {
    let range = range_fn(child.range);
    Some(CertifiedExpr {
        expr: ExprAst::Unary {
            op,
            arg: Box::new(child.expr),
        },
        range,
    })
}

fn unary_exp(child: CertifiedExpr, cap: f64) -> Option<CertifiedExpr> {
    let range = exp_range(child.range, cap)?;
    Some(CertifiedExpr {
        expr: ExprAst::Unary {
            op: UnaryOpAst::Exp,
            arg: Box::new(child.expr),
        },
        range,
    })
}

fn unary_sqrt(child: CertifiedExpr, margin: f64) -> Option<CertifiedExpr> {
    let range = sqrt_range(child.range, margin)?;
    Some(CertifiedExpr {
        expr: ExprAst::Unary {
            op: UnaryOpAst::Sqrt,
            arg: Box::new(child.expr),
        },
        range,
    })
}

fn unary_log(child: CertifiedExpr, margin: f64) -> Option<CertifiedExpr> {
    let range = log_range(child.range, margin)?;
    Some(CertifiedExpr {
        expr: ExprAst::Unary {
            op: UnaryOpAst::Log,
            arg: Box::new(child.expr),
        },
        range,
    })
}

fn binary_expr(
    op: BinaryOpAst,
    lhs: CertifiedExpr,
    rhs: CertifiedExpr,
    range_fn: impl Fn(RangeCert, RangeCert) -> Option<RangeCert>,
) -> Option<CertifiedExpr> {
    let range = range_fn(lhs.range, rhs.range)?;
    Some(CertifiedExpr {
        expr: ExprAst::Binary {
            op,
            lhs: Box::new(lhs.expr),
            rhs: Box::new(rhs.expr),
        },
        range,
    })
}

fn binary_div(lhs: CertifiedExpr, rhs: CertifiedExpr, margin: f64) -> Option<CertifiedExpr> {
    let range = div_range(lhs.range, rhs.range, margin)?;
    Some(CertifiedExpr {
        expr: ExprAst::Binary {
            op: BinaryOpAst::Div,
            lhs: Box::new(lhs.expr),
            rhs: Box::new(rhs.expr),
        },
        range,
    })
}

fn certify_expr(
    expr: &ExprAst,
    inputs: &[RangeCert],
    helpers: &[FunctionAst],
    config: &GeneratorConfig,
) -> Option<RangeCert> {
    match expr {
        ExprAst::Const(value) => Some(RangeCert::exact(*value)),
        ExprAst::Input(index) => inputs.get(*index).copied(),
        ExprAst::Unary { op, arg } => {
            let arg = certify_expr(arg, inputs, helpers, config)?;
            match op {
                UnaryOpAst::Neg => negate_range(arg),
                UnaryOpAst::Sin => Some(sin_range(arg)),
                UnaryOpAst::Cos => Some(cos_range(arg)),
                UnaryOpAst::Exp => exp_range(arg, config.exp_input_cap),
                UnaryOpAst::Sqrt => sqrt_range(arg, config.positive_margin),
                UnaryOpAst::Log => log_range(arg, config.positive_margin),
                UnaryOpAst::Square => square_range(arg),
            }
        }
        ExprAst::Binary { op, lhs, rhs } => {
            let lhs = certify_expr(lhs, inputs, helpers, config)?;
            let rhs = certify_expr(rhs, inputs, helpers, config)?;
            match op {
                BinaryOpAst::Add => add_range(lhs, rhs),
                BinaryOpAst::Sub => sub_range(lhs, rhs),
                BinaryOpAst::Mul => mul_range(lhs, rhs),
                BinaryOpAst::Div => div_range(lhs, rhs, config.nonzero_margin),
            }
        }
        ExprAst::Call {
            helper,
            output,
            args,
        } => {
            let function = helpers.get(*helper)?;
            let arg_ranges = (0..function.input_count)
                .map(|index| {
                    if let Some(arg) = args.get(index) {
                        certify_expr(arg, inputs, helpers, config)
                    } else {
                        Some(RangeCert::exact(0.0))
                    }
                })
                .collect::<Option<Vec<_>>>()?;
            certify_expr(function.outputs.get(*output)?, &arg_ranges, helpers, config)
        }
    }
}

fn validate_generated_case(
    scenario: &PropertyScenario,
    case: &GeneratedCase,
) -> Result<(), RejectReason> {
    if scenario.requirements.require_calls && case.features.call_count == 0 {
        return Err(RejectReason::Scenario(
            "case does not contain helper calls".into(),
        ));
    }
    if scenario.requirements.require_multi_output_helper
        && case.features.helper_multi_output_count == 0
    {
        return Err(RejectReason::Scenario(
            "case does not contain a multi-output helper".into(),
        ));
    }
    if scenario.requirements.require_repeated_helper_calls
        && case.features.repeated_helper_calls == 0
    {
        return Err(RejectReason::Scenario(
            "case does not repeat helper calls".into(),
        ));
    }
    if scenario.requirements.require_nested_helper_calls && case.features.nested_helper_calls == 0 {
        return Err(RejectReason::Scenario(
            "case does not contain nested helper calls".into(),
        ));
    }
    Ok(())
}
