#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterAcceptanceMode {
    ObjectiveArmijo,
    ViolationReduction,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct FilterEntry {
    pub objective: f64,
    pub violation: f64,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct FilterInfo {
    pub current: FilterEntry,
    pub entries: Vec<FilterEntry>,
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub accepted_mode: Option<FilterAcceptanceMode>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FilterParameters {
    pub gamma_objective: f64,
    pub gamma_violation: f64,
    pub armijo_c1: f64,
    pub violation_tol: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FilterTrialAssessment {
    pub acceptance_mode: Option<FilterAcceptanceMode>,
    pub objective_armijo_satisfied: bool,
    pub objective_armijo_tolerance_adjusted: bool,
    pub filter_acceptable: bool,
    pub filter_dominated: bool,
    pub filter_sufficient_objective_reduction: bool,
    pub filter_sufficient_violation_reduction: bool,
}

fn absolute_tolerance(value: f64) -> f64 {
    value.abs().max(1.0) * 1e-12
}

pub(crate) fn entry(objective: f64, violation: f64) -> FilterEntry {
    FilterEntry {
        objective,
        violation,
    }
}

fn objective_armijo_assessment(
    objective_value: f64,
    trial_objective: f64,
    alpha: f64,
    objective_directional_derivative: f64,
    armijo_c1: f64,
) -> (bool, bool) {
    let armijo_rhs = objective_value + armijo_c1 * alpha * objective_directional_derivative;
    let armijo_abs_tol = absolute_tolerance(objective_value);
    let strict = trial_objective <= armijo_rhs;
    let tolerance_adjusted = !strict && trial_objective <= armijo_rhs + armijo_abs_tol;
    (strict || tolerance_adjusted, tolerance_adjusted)
}

pub(crate) fn reduction_assessment(
    objective_value: f64,
    trial_objective: f64,
    target_objective: f64,
) -> (bool, bool) {
    let objective_abs_tol = absolute_tolerance(objective_value);
    let strict = trial_objective <= target_objective;
    let tolerance_adjusted = !strict && trial_objective <= target_objective + objective_abs_tol;
    (strict || tolerance_adjusted, tolerance_adjusted)
}

fn sufficient_violation_reduction(
    current_violation: f64,
    trial_violation: f64,
    gamma_violation: f64,
) -> bool {
    let target = if current_violation > 0.0 {
        (1.0 - gamma_violation).max(0.0) * current_violation
    } else {
        current_violation
    };
    trial_violation <= target + absolute_tolerance(current_violation)
}

fn dominates_trial(
    entry: &FilterEntry,
    trial: &FilterEntry,
    gamma_objective: f64,
    gamma_violation: f64,
) -> bool {
    let violation_barrier = (1.0 - gamma_violation).max(0.0) * entry.violation;
    let objective_barrier = entry.objective - gamma_objective * trial.violation.max(1e-12);
    trial.violation >= violation_barrier - absolute_tolerance(entry.violation)
        && trial.objective >= objective_barrier - absolute_tolerance(entry.objective)
}

fn is_acceptable(
    entries: &[FilterEntry],
    trial: &FilterEntry,
    gamma_objective: f64,
    gamma_violation: f64,
) -> bool {
    !entries
        .iter()
        .any(|entry| dominates_trial(entry, trial, gamma_objective, gamma_violation))
}

fn entries_match(lhs: &FilterEntry, rhs: &FilterEntry) -> bool {
    let violation_scale = lhs.violation.abs().max(rhs.violation.abs()).max(1.0);
    let objective_scale = lhs.objective.abs().max(rhs.objective.abs()).max(1.0);
    (lhs.violation - rhs.violation).abs() <= 1e-12 * violation_scale
        && (lhs.objective - rhs.objective).abs() <= 1e-12 * objective_scale
}

pub(crate) fn update_frontier(entries: &mut Vec<FilterEntry>, accepted: FilterEntry) {
    entries.retain(|entry| {
        entries_match(entry, &accepted)
            || accepted.violation > entry.violation + absolute_tolerance(entry.violation)
            || accepted.objective > entry.objective + absolute_tolerance(entry.objective)
    });
    if !entries.iter().any(|entry| entries_match(entry, &accepted)) {
        entries.push(accepted);
    }
}

pub(crate) fn assess_trial_with_objective_status(
    entries: &[FilterEntry],
    current: &FilterEntry,
    trial: &FilterEntry,
    objective_satisfied: bool,
    objective_tolerance_adjusted: bool,
    parameters: FilterParameters,
) -> FilterTrialAssessment {
    let near_feasible_nonworsening = current.violation <= parameters.violation_tol
        && trial.violation <= current.violation + absolute_tolerance(current.violation)
        && trial.objective <= current.objective + absolute_tolerance(current.objective);
    let filter_acceptable = near_feasible_nonworsening
        || is_acceptable(
            entries,
            trial,
            parameters.gamma_objective,
            parameters.gamma_violation,
        );
    let filter_sufficient_violation_reduction = sufficient_violation_reduction(
        current.violation,
        trial.violation,
        parameters.gamma_violation,
    );
    let filter_sufficient_objective_reduction = objective_satisfied || near_feasible_nonworsening;
    let acceptance_mode = if !filter_acceptable {
        None
    } else if filter_sufficient_objective_reduction {
        Some(FilterAcceptanceMode::ObjectiveArmijo)
    } else if filter_sufficient_violation_reduction {
        Some(FilterAcceptanceMode::ViolationReduction)
    } else {
        None
    };
    FilterTrialAssessment {
        acceptance_mode,
        objective_armijo_satisfied: objective_satisfied,
        objective_armijo_tolerance_adjusted: objective_tolerance_adjusted,
        filter_acceptable,
        filter_dominated: !filter_acceptable,
        filter_sufficient_objective_reduction,
        filter_sufficient_violation_reduction,
    }
}

pub(crate) fn assess_trial(
    entries: &[FilterEntry],
    current: &FilterEntry,
    trial: &FilterEntry,
    alpha: f64,
    objective_directional_derivative: f64,
    parameters: FilterParameters,
) -> FilterTrialAssessment {
    let (objective_armijo_satisfied, objective_armijo_tolerance_adjusted) =
        objective_armijo_assessment(
            current.objective,
            trial.objective,
            alpha,
            objective_directional_derivative,
            parameters.armijo_c1,
        );
    assess_trial_with_objective_status(
        entries,
        current,
        trial,
        objective_armijo_satisfied,
        objective_armijo_tolerance_adjusted,
        parameters,
    )
}
