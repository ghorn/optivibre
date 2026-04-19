mod corpus;
mod matrix_market;
mod metrics;
mod model;
mod references;
mod report;
mod runner;

pub use corpus::{
    DownloadedCorpusSpec, SymmetricPatternMatrix, corpus_download_target_path,
    downloaded_public_corpus_specs, extract_downloaded_corpus_archive,
};
pub use matrix_market::{MatrixMarketError, parse_matrix_market_file, parse_matrix_market_str};
pub use metrics::{ExactSymbolicMetrics, exact_symbolic_metrics};
pub use model::{
    BaselineSummary, CaseValidationReport, CorpusCaseMetadata, CorpusSourceKind, CorpusTag,
    EnvironmentStamp, NumericFactorizationMetrics, NumericStrategy, NumericValidationResult,
    OrderingMethod, OrderingQualityMetrics, OrderingValidationResult, RobustnessCaseResult,
    SkippedCase, SymbolicAnalysisMetrics, SymbolicStrategy, SymbolicValidationResult,
    ValidationOutcome, ValidationSuiteReport, ValidationSummary, ValidationTier,
};
pub use references::{NativeMetisReference, NativeSpralNumericResult, NativeSpralReference};
pub use report::{
    apply_baseline_summary, render_html_report, render_markdown_report, summarize_report,
};
pub use runner::{ValidationRunConfig, run_validation_suite};
