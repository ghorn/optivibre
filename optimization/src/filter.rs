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
    pub gamma_phi: f64,
    pub gamma_theta: f64,
    pub eta_phi: f64,
    pub theta_max: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FilterTrialAssessment {
    pub acceptance_mode: Option<FilterAcceptanceMode>,
    pub objective_armijo_satisfied: bool,
    pub objective_armijo_tolerance_adjusted: bool,
    pub current_iterate_acceptable: bool,
    pub filter_acceptable: bool,
    pub filter_dominated: bool,
    pub filter_theta_acceptable: bool,
    pub filter_sufficient_objective_reduction: bool,
    pub filter_sufficient_violation_reduction: bool,
    pub switching_condition_satisfied: bool,
}

fn absolute_tolerance(value: f64) -> f64 {
    100.0 * f64::EPSILON * value.abs().max(1.0)
}

pub(crate) fn entry(objective: f64, violation: f64) -> FilterEntry {
    FilterEntry {
        objective,
        violation,
    }
}

pub(crate) fn augment_entry(
    objective_value: f64,
    violation: f64,
    gamma_phi: f64,
    gamma_theta: f64,
) -> FilterEntry {
    FilterEntry {
        objective: objective_value - gamma_phi * violation,
        violation: (1.0 - gamma_theta).max(0.0) * violation,
    }
}

fn objective_armijo_assessment(
    objective_value: f64,
    trial_objective: f64,
    alpha: f64,
    objective_directional_derivative: f64,
    eta_phi: f64,
) -> (bool, bool) {
    let armijo_rhs = objective_value + eta_phi * alpha * objective_directional_derivative;
    let armijo_abs_tol = absolute_tolerance(objective_value);
    let strict = trial_objective <= armijo_rhs;
    let tolerance_adjusted = !strict && trial_objective <= armijo_rhs + armijo_abs_tol;
    (strict || tolerance_adjusted, tolerance_adjusted)
}

fn sufficient_violation_reduction(
    current_violation: f64,
    trial_violation: f64,
    gamma_theta: f64,
) -> bool {
    let target = if current_violation > 0.0 {
        (1.0 - gamma_theta).max(0.0) * current_violation
    } else {
        current_violation
    };
    trial_violation <= target + absolute_tolerance(current_violation)
}

fn theta_acceptable(trial_violation: f64, theta_max: f64) -> bool {
    trial_violation <= theta_max + absolute_tolerance(theta_max)
}

fn dominates_trial(
    entry: &FilterEntry,
    trial: &FilterEntry,
    gamma_phi: f64,
    gamma_theta: f64,
) -> bool {
    let violation_barrier = (1.0 - gamma_theta).max(0.0) * entry.violation;
    let objective_barrier = entry.objective - gamma_phi * trial.violation.max(1e-12);
    trial.violation >= violation_barrier - absolute_tolerance(entry.violation)
        && trial.objective >= objective_barrier - absolute_tolerance(entry.objective)
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

pub(crate) fn assess_trial(
    entries: &[FilterEntry],
    current: &FilterEntry,
    trial: &FilterEntry,
    alpha: f64,
    barrier_directional_derivative: f64,
    switching_condition_satisfied: bool,
    armijo_required: bool,
    parameters: FilterParameters,
) -> FilterTrialAssessment {
    let (objective_armijo_satisfied, objective_armijo_tolerance_adjusted) =
        objective_armijo_assessment(
            current.objective,
            trial.objective,
            alpha,
            barrier_directional_derivative,
            parameters.eta_phi,
        );
    let filter_theta_acceptable = theta_acceptable(trial.violation, parameters.theta_max);
    let filter_dominated = entries
        .iter()
        .any(|entry| dominates_trial(entry, trial, parameters.gamma_phi, parameters.gamma_theta));
    let filter_acceptable = filter_theta_acceptable && !filter_dominated;
    let filter_sufficient_violation_reduction =
        sufficient_violation_reduction(current.violation, trial.violation, parameters.gamma_theta);
    let filter_sufficient_objective_reduction = objective_armijo_satisfied
        || trial.objective
            <= current.objective - parameters.gamma_phi * current.violation
                + absolute_tolerance(current.objective);
    let current_iterate_acceptable = if armijo_required {
        objective_armijo_satisfied
    } else {
        filter_sufficient_violation_reduction || filter_sufficient_objective_reduction
    };
    let acceptance_mode = if !current_iterate_acceptable || !filter_acceptable {
        None
    } else if switching_condition_satisfied && objective_armijo_satisfied {
        Some(FilterAcceptanceMode::ObjectiveArmijo)
    } else {
        Some(FilterAcceptanceMode::ViolationReduction)
    };
    FilterTrialAssessment {
        acceptance_mode,
        objective_armijo_satisfied,
        objective_armijo_tolerance_adjusted,
        current_iterate_acceptable,
        filter_acceptable,
        filter_dominated,
        filter_theta_acceptable,
        filter_sufficient_objective_reduction,
        filter_sufficient_violation_reduction,
        switching_condition_satisfied,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(theta_max: f64, _theta_min: f64) -> FilterParameters {
        FilterParameters {
            gamma_phi: 1.0e-4,
            gamma_theta: 0.1,
            eta_phi: 1.0e-8,
            theta_max,
        }
    }

    #[test]
    fn objective_improvement_can_accept_far_from_feasibility_without_armijo() {
        let current = entry(0.0, 10.0);
        let trial = entry(-1.0, 11.0);
        let assessment = assess_trial(
            &[],
            &current,
            &trial,
            1.0,
            -1.0,
            false,
            false,
            params(20.0, 1.0e-8),
        );
        assert!(assessment.current_iterate_acceptable);
        assert_eq!(
            assessment.acceptance_mode,
            Some(FilterAcceptanceMode::ViolationReduction)
        );
    }

    #[test]
    fn feasibility_reducing_h_step_is_accepted() {
        let current = entry(0.0, 10.0);
        let trial = entry(1.0, 5.0);
        let assessment = assess_trial(
            &[],
            &current,
            &trial,
            1.0,
            1.0,
            false,
            false,
            params(20.0, 1.0e-8),
        );
        assert!(assessment.current_iterate_acceptable);
        assert!(assessment.filter_acceptable);
        assert!(assessment.filter_sufficient_violation_reduction);
        assert_eq!(
            assessment.acceptance_mode,
            Some(FilterAcceptanceMode::ViolationReduction)
        );
    }

    #[test]
    fn objective_step_accepts_when_switching_condition_and_armijo_hold() {
        let current = entry(5.0, 1.0e-9);
        let trial = entry(4.0, 5.0e-10);
        let assessment = assess_trial(
            &[],
            &current,
            &trial,
            1.0,
            -1.0,
            true,
            true,
            params(1.0, 1.0e-8),
        );
        assert!(assessment.current_iterate_acceptable);
        assert!(assessment.filter_acceptable);
        assert_eq!(
            assessment.acceptance_mode,
            Some(FilterAcceptanceMode::ObjectiveArmijo)
        );
    }

    #[test]
    fn switching_condition_without_armijo_stays_h_type() {
        let current = entry(5.0, 1.0e-2);
        let trial = entry(5.1, 5.0e-3);
        let assessment = assess_trial(
            &[],
            &current,
            &trial,
            1.0,
            -1.0,
            true,
            false,
            params(1.0, 1.0e-8),
        );
        assert!(assessment.current_iterate_acceptable);
        assert_eq!(
            assessment.acceptance_mode,
            Some(FilterAcceptanceMode::ViolationReduction)
        );
    }

    #[test]
    fn near_feasible_h_step_is_still_accepted_without_switching_condition() {
        let current = entry(5.0, 1.0e-9);
        let trial = entry(6.0, 5.0e-10);
        let assessment = assess_trial(
            &[],
            &current,
            &trial,
            1.0,
            -1.0,
            false,
            false,
            params(1.0, 1.0e-8),
        );
        assert!(assessment.current_iterate_acceptable);
        assert_eq!(
            assessment.acceptance_mode,
            Some(FilterAcceptanceMode::ViolationReduction)
        );
    }

    #[test]
    fn non_switching_objective_reduction_step_is_treated_as_h_type() {
        let current = entry(5.0, 1.0e-9);
        let trial = entry(4.0, 1.1e-9);
        let assessment = assess_trial(
            &[],
            &current,
            &trial,
            1.0,
            -1.0,
            false,
            false,
            params(1.0, 1.0e-8),
        );
        assert!(assessment.current_iterate_acceptable);
        assert_eq!(
            assessment.acceptance_mode,
            Some(FilterAcceptanceMode::ViolationReduction)
        );
    }

    #[test]
    fn theta_max_rejects_trial() {
        let current = entry(0.0, 1.0);
        let trial = entry(-1.0, 10.0);
        let assessment = assess_trial(
            &[],
            &current,
            &trial,
            1.0,
            -1.0,
            false,
            false,
            params(1.0, 1.0e-8),
        );
        assert!(!assessment.filter_theta_acceptable);
        assert!(!assessment.filter_acceptable);
        assert_eq!(assessment.acceptance_mode, None);
    }

    #[test]
    fn augment_entry_applies_filter_margins() {
        let entry = augment_entry(10.0, 4.0, 0.25, 0.1);
        assert_eq!(entry.objective, 9.0);
        assert_eq!(entry.violation, 3.6);
    }
}
