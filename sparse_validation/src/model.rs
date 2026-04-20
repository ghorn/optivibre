use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationTier {
    Pr,
    Scheduled,
    Local,
}

impl ValidationTier {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Pr => "pr",
            Self::Scheduled => "scheduled",
            Self::Local => "local",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusTag {
    Path,
    Grid,
    Tree,
    Disconnected,
    BandedSpd,
    ArrowKkt,
    TwoByTwoPivot,
    DelayedPivot,
    Adversarial,
    TinyExact,
    Public,
    FiniteElement,
    Structural,
    Workspace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusSourceKind {
    Generated,
    Downloaded,
    Workspace,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusCaseMetadata {
    pub case_id: String,
    pub description: String,
    pub source: CorpusSourceKind,
    pub dimension: usize,
    pub nnz: usize,
    pub tags: Vec<CorpusTag>,
    pub exact_oracle: bool,
    pub protected_fill_regression: bool,
    pub location: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationOutcome {
    Passed,
    Failed,
    SkippedNotRequested,
    SkippedReferenceUnavailable,
    SkippedNotImplemented,
    SkippedCorpusUnavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderingMethod {
    Natural,
    Amd,
    RustNestedDissection,
    NativeMetis,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrderingQualityMetrics {
    pub permutation_valid: bool,
    pub fill_nnz: usize,
    pub fill_ratio_vs_natural: Option<f64>,
    pub etree_height: usize,
    pub memory_bytes: usize,
    pub max_separator_fraction: Option<f64>,
    pub component_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OrderingValidationResult {
    pub method: OrderingMethod,
    pub outcome: ValidationOutcome,
    pub elapsed_ms: f64,
    pub metrics: Option<OrderingQualityMetrics>,
    pub notes: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolicStrategy {
    Natural,
    RustNestedDissection,
    NativeSpral,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SymbolicAnalysisMetrics {
    pub fill_nnz: usize,
    pub exact_fill_pattern_match: Option<bool>,
    pub exact_column_counts_match: Option<bool>,
    pub tree_parent_valid: bool,
    pub etree_height: usize,
    pub supernode_count: usize,
    pub memory_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SymbolicValidationResult {
    pub strategy: SymbolicStrategy,
    pub outcome: ValidationOutcome,
    pub elapsed_ms: f64,
    pub metrics: Option<SymbolicAnalysisMetrics>,
    pub notes: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericStrategy {
    Natural,
    ApproximateMinimumDegree,
    Auto,
    RustNestedDissection,
    NativeSpral,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NumericFactorizationMetrics {
    pub factorization_residual_max_abs: f64,
    pub solve_residual_inf_norm: f64,
    pub solution_inf_error: f64,
    pub refactorization_residual_max_abs: Option<f64>,
    pub refactorization_solve_residual_inf_norm: Option<f64>,
    pub refactorization_solution_inf_error: Option<f64>,
    pub inertia_match: Option<bool>,
    pub positive_eigenvalues: usize,
    pub negative_eigenvalues: usize,
    pub zero_eigenvalues: usize,
    pub stored_nnz: usize,
    pub factor_storage_bytes: usize,
    pub supernode_count: usize,
    pub max_supernode_width: usize,
    pub two_by_two_pivots: usize,
    pub delayed_pivots: usize,
    pub refactor_speedup_vs_factor: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NumericValidationResult {
    pub strategy: NumericStrategy,
    pub outcome: ValidationOutcome,
    pub factor_elapsed_ms: f64,
    pub solve_elapsed_ms: f64,
    pub refactor_elapsed_ms: Option<f64>,
    pub metrics: Option<NumericFactorizationMetrics>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RobustnessCaseResult {
    pub scenario: String,
    pub target: String,
    pub outcome: ValidationOutcome,
    pub duration_ms: f64,
    pub error_kind: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedCase {
    pub case_id: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvironmentStamp {
    pub operating_system: String,
    pub architecture: String,
    pub rustc_version: String,
    pub git_sha: Option<String>,
    pub hostname: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BaselineSummary {
    pub previous_generated_at_utc: String,
    pub ordering_median_ratio: Option<f64>,
    pub symbolic_median_ratio: Option<f64>,
    pub numeric_median_ratio: Option<f64>,
    pub slower_ordering_entries: Vec<String>,
    pub slower_symbolic_entries: Vec<String>,
    pub slower_numeric_entries: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaseValidationReport {
    pub case: CorpusCaseMetadata,
    pub ordering: Vec<OrderingValidationResult>,
    pub symbolic: Vec<SymbolicValidationResult>,
    pub numeric: Vec<NumericValidationResult>,
    pub failures: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub total_cases: usize,
    pub executed_cases: usize,
    pub skipped_cases: usize,
    pub failed_ordering_results: usize,
    pub failed_symbolic_results: usize,
    pub failed_numeric_results: usize,
    pub failed_robustness_results: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValidationSuiteReport {
    pub tier: ValidationTier,
    pub generated_at_utc: String,
    pub environment: EnvironmentStamp,
    pub requested_native_metis: bool,
    pub requested_native_spral: bool,
    pub skipped_cases: Vec<SkippedCase>,
    pub cases: Vec<CaseValidationReport>,
    pub robustness: Vec<RobustnessCaseResult>,
    pub summary: ValidationSummary,
    pub baseline: Option<BaselineSummary>,
}
