use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use bench_report::{MarkdownReportOptions, SuiteReport, render_markdown_report_with_options};
use clap::{Args, Parser, Subcommand, ValueEnum};
use sparse_validation::{
    ValidationRunConfig, ValidationSuiteReport, ValidationTier, apply_baseline_summary,
    corpus_download_target_path, downloaded_public_corpus_specs, extract_downloaded_corpus_archive,
    render_html_report, render_markdown_report, run_validation_suite,
};

fn workspace_root() -> Result<PathBuf> {
    std::env::current_dir().context("xtask must run from the workspace root")
}

#[derive(Debug, Parser)]
#[command(
    name = "xtask",
    about = "Internal report and audit task runner for ad_codegen_rs."
)]
struct XtaskCli {
    #[command(subcommand)]
    command: Option<XtaskCommand>,
}

#[derive(Debug, Clone, Subcommand)]
enum XtaskCommand {
    AdCostReport,
    CasadiParityReport,
    SparseValidation(SparseValidationArgs),
}

#[derive(Debug, Clone, Args)]
struct SparseValidationArgs {
    #[arg(long, value_enum, default_value_t = SparseValidationTierArg::Pr)]
    tier: SparseValidationTierArg,
    #[arg(long)]
    with_native_metis: bool,
    #[arg(long)]
    with_native_spral: bool,
    #[arg(long)]
    download_corpus: bool,
    #[arg(long)]
    output_dir: Option<PathBuf>,
    #[arg(long)]
    baseline_json: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SparseValidationTierArg {
    Pr,
    Scheduled,
    Local,
}

impl From<SparseValidationTierArg> for ValidationTier {
    fn from(value: SparseValidationTierArg) -> Self {
        match value {
            SparseValidationTierArg::Pr => ValidationTier::Pr,
            SparseValidationTierArg::Scheduled => ValidationTier::Scheduled,
            SparseValidationTierArg::Local => ValidationTier::Local,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReportPaths {
    dir: PathBuf,
    debug_json: PathBuf,
    release_json: PathBuf,
    markdown: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParityReportPaths {
    dir: PathBuf,
    markdown: PathBuf,
    json: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SparseValidationReportPaths {
    dir: PathBuf,
    json: PathBuf,
    markdown: PathBuf,
    html: PathBuf,
}

fn report_paths(root: &Path) -> ReportPaths {
    let dir = root.join("target/reports");
    ReportPaths {
        debug_json: dir.join("ad_cost_debug.json"),
        release_json: dir.join("ad_cost_release.json"),
        markdown: dir.join("ad_cost_report.md"),
        dir,
    }
}

fn parity_report_paths(root: &Path) -> ParityReportPaths {
    let dir = root.join("target/reports");
    ParityReportPaths {
        markdown: dir.join("casadi_parity_audit.md"),
        json: dir.join("casadi_parity_audit.json"),
        dir,
    }
}

fn sparse_validation_paths(
    root: &Path,
    output_dir: Option<&Path>,
    tier: ValidationTier,
) -> SparseValidationReportPaths {
    let dir = output_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| root.join("target/reports/sparse_validation"));
    let stem = format!("sparse_validation_{}", tier.label());
    SparseValidationReportPaths {
        json: dir.join(format!("{stem}.json")),
        markdown: dir.join(format!("{stem}.md")),
        html: dir.join(format!("{stem}.html")),
        dir,
    }
}

fn sparse_validation_corpus_root(root: &Path) -> PathBuf {
    root.join("target/validation_corpus")
}

fn ad_cost_suite_args(release: bool) -> Vec<&'static str> {
    let mut args = vec!["run", "--quiet"];
    if release {
        args.push("--release");
    }
    args.extend([
        "-p",
        "examples_run",
        "--features",
        "ad-bench-artifacts",
        "--bin",
        "ad_cost_suite",
        "--",
        "--json",
    ]);
    args
}

fn validate_suite_profiles(debug: &SuiteReport, release: &SuiteReport) -> Result<()> {
    if debug.build_profile != "debug" || release.build_profile != "release" {
        bail!(
            "unexpected suite profiles: debug={}, release={}",
            debug.build_profile,
            release.build_profile
        );
    }
    Ok(())
}

fn run_status(args: &[&str]) -> Result<()> {
    let status = Command::new("cargo")
        .args(args)
        .status()
        .with_context(|| format!("failed to run cargo {}", args.join(" ")))?;
    if !status.success() {
        bail!("cargo {} failed with status {status}", args.join(" "));
    }
    Ok(())
}

fn run_suite(args: &[&str]) -> Result<SuiteReport> {
    let output = Command::new("cargo")
        .args(args)
        .env("RUST_MIN_STACK", "67108864")
        .output()
        .with_context(|| format!("failed to run cargo {}", args.join(" ")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("cargo {} failed:\n{stderr}", args.join(" "));
    }
    serde_json::from_slice(&output.stdout).context("failed to parse suite JSON output")
}

fn generate_ad_cost_report() -> Result<()> {
    let root = workspace_root()?;
    let paths = report_paths(&root);
    fs::create_dir_all(&paths.dir)?;

    run_status(&[
        "test",
        "-p",
        "sx_core",
        "--test",
        "core",
        "hessian_strategies",
    ])?;
    run_status(&["test", "-p", "sx_codegen", "--test", "ad_cost"])?;
    let debug_args = ad_cost_suite_args(false);
    let release_args = ad_cost_suite_args(true);
    let debug = run_suite(debug_args.as_slice())?;
    let release = run_suite(release_args.as_slice())?;

    validate_suite_profiles(&debug, &release)?;

    fs::write(&paths.debug_json, serde_json::to_vec_pretty(&debug)?)?;
    fs::write(&paths.release_json, serde_json::to_vec_pretty(&release)?)?;

    let markdown = render_markdown_report_with_options(
        &MarkdownReportOptions {
            title: "AD Cost Report".into(),
            command: Some("cargo run -p xtask -- ad-cost-report".into()),
            include_lowered_op_explanation: true,
        },
        &debug,
        &release,
    );
    fs::write(&paths.markdown, markdown)?;
    println!("{}", paths.markdown.display());
    Ok(())
}

fn generate_casadi_parity_report() -> Result<()> {
    let root = workspace_root()?;
    let paths = parity_report_paths(&root);
    fs::create_dir_all(&paths.dir)?;
    let output = Command::new("python3")
        .arg("scripts/casadi_parity_audit.py")
        .output()
        .context("failed to run scripts/casadi_parity_audit.py")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "casadi parity report generation failed with status {}:\n{stderr}",
            output.status
        );
    }
    if !paths.markdown.exists() {
        bail!("expected parity markdown report was not generated");
    }
    if !paths.json.exists() {
        bail!("expected parity JSON report was not generated");
    }
    println!("{}", paths.markdown.display());
    Ok(())
}

fn generate_sparse_validation_report(args: &SparseValidationArgs) -> Result<()> {
    let root = workspace_root()?;
    let tier = ValidationTier::from(args.tier);
    let paths = sparse_validation_paths(&root, args.output_dir.as_deref(), tier);
    let corpus_root = sparse_validation_corpus_root(&root);
    fs::create_dir_all(&paths.dir)?;

    if args.download_corpus {
        download_sparse_validation_corpus(&corpus_root)?;
    }

    let config = ValidationRunConfig {
        tier,
        corpus_root,
        with_native_metis: args.with_native_metis,
        with_native_spral: args.with_native_spral,
    };
    let mut report = run_validation_suite(&config)?;
    if let Some(path) = &args.baseline_json {
        let bytes = fs::read(path).with_context(|| {
            format!(
                "failed to read sparse-validation baseline {}",
                path.display()
            )
        })?;
        let previous: ValidationSuiteReport =
            serde_json::from_slice(&bytes).with_context(|| {
                format!(
                    "failed to parse sparse-validation baseline {}",
                    path.display()
                )
            })?;
        apply_baseline_summary(&mut report, &previous);
    }

    fs::write(&paths.json, serde_json::to_vec_pretty(&report)?)?;
    fs::write(&paths.markdown, render_markdown_report(&report))?;
    fs::write(&paths.html, render_html_report(&report))?;
    ensure_sparse_validation_gates(&report)?;
    println!("{}", paths.markdown.display());
    Ok(())
}

fn download_sparse_validation_corpus(corpus_root: &Path) -> Result<()> {
    fs::create_dir_all(corpus_root)?;
    for spec in downloaded_public_corpus_specs() {
        let target = corpus_download_target_path(corpus_root, spec);
        if target.exists() {
            continue;
        }
        let archive = corpus_root.join(format!("{}.tar.gz", spec.id));
        let status = Command::new("curl")
            .args([
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                spec.url,
                "--output",
            ])
            .arg(&archive)
            .status()
            .with_context(|| format!("failed to invoke curl for {}", spec.url))?;
        if !status.success() {
            bail!(
                "curl failed while downloading {} with status {status}",
                spec.url
            );
        }
        extract_downloaded_corpus_archive(&archive, corpus_root, spec).with_context(|| {
            format!(
                "failed to extract sparse validation archive {}",
                archive.display()
            )
        })?;
        fs::remove_file(&archive)
            .with_context(|| format!("failed to remove temporary archive {}", archive.display()))?;
    }
    Ok(())
}

fn ensure_sparse_validation_gates(report: &ValidationSuiteReport) -> Result<()> {
    let mut failures = Vec::new();
    if report.summary.failed_ordering_results > 0 {
        failures.push(format!(
            "ordering failures: {}",
            report.summary.failed_ordering_results
        ));
    }
    if report.summary.failed_symbolic_results > 0 {
        failures.push(format!(
            "symbolic failures: {}",
            report.summary.failed_symbolic_results
        ));
    }
    if report.summary.failed_numeric_results > 0 {
        failures.push(format!(
            "numeric failures: {}",
            report.summary.failed_numeric_results
        ));
    }
    if report.summary.failed_robustness_results > 0 {
        failures.push(format!(
            "robustness failures: {}",
            report.summary.failed_robustness_results
        ));
    }
    for case in &report.cases {
        if !case.failures.is_empty() {
            failures.push(format!(
                "{}: {}",
                case.case.case_id,
                case.failures.join("; ")
            ));
        }
    }
    if failures.is_empty() {
        return Ok(());
    }
    bail!(
        "sparse validation reported failures:\n{}",
        failures.join("\n")
    )
}

fn main() -> Result<()> {
    match XtaskCli::parse()
        .command
        .unwrap_or(XtaskCommand::AdCostReport)
    {
        XtaskCommand::AdCostReport => generate_ad_cost_report(),
        XtaskCommand::CasadiParityReport => generate_casadi_parity_report(),
        XtaskCommand::SparseValidation(args) => generate_sparse_validation_report(&args),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_suite(profile: &str) -> SuiteReport {
        SuiteReport {
            build_profile: profile.into(),
            samples: 1,
            target_ms: 1,
            cases: Vec::new(),
            hessian_strategies: Vec::new(),
            properties: Vec::new(),
        }
    }

    #[test]
    fn report_paths_point_into_target_reports() {
        let root = Path::new("/tmp/workspace");
        let paths = report_paths(root);
        assert_eq!(paths.dir, PathBuf::from("/tmp/workspace/target/reports"));
        assert_eq!(
            paths.debug_json,
            PathBuf::from("/tmp/workspace/target/reports/ad_cost_debug.json")
        );
        assert_eq!(
            paths.release_json,
            PathBuf::from("/tmp/workspace/target/reports/ad_cost_release.json")
        );
        assert_eq!(
            paths.markdown,
            PathBuf::from("/tmp/workspace/target/reports/ad_cost_report.md")
        );
    }

    #[test]
    fn parity_report_paths_point_into_target_reports() {
        let root = Path::new("/tmp/workspace");
        let paths = parity_report_paths(root);
        assert_eq!(paths.dir, PathBuf::from("/tmp/workspace/target/reports"));
        assert_eq!(
            paths.markdown,
            PathBuf::from("/tmp/workspace/target/reports/casadi_parity_audit.md")
        );
        assert_eq!(
            paths.json,
            PathBuf::from("/tmp/workspace/target/reports/casadi_parity_audit.json")
        );
    }

    #[test]
    fn sparse_validation_paths_default_under_target_reports() {
        let root = Path::new("/tmp/workspace");
        let paths = sparse_validation_paths(root, None, ValidationTier::Scheduled);
        assert_eq!(
            paths.dir,
            PathBuf::from("/tmp/workspace/target/reports/sparse_validation")
        );
        assert_eq!(
            paths.json,
            PathBuf::from(
                "/tmp/workspace/target/reports/sparse_validation/sparse_validation_scheduled.json"
            )
        );
        assert_eq!(
            paths.markdown,
            PathBuf::from(
                "/tmp/workspace/target/reports/sparse_validation/sparse_validation_scheduled.md"
            )
        );
        assert_eq!(
            paths.html,
            PathBuf::from(
                "/tmp/workspace/target/reports/sparse_validation/sparse_validation_scheduled.html"
            )
        );
    }

    #[test]
    fn sparse_validation_corpus_root_lives_under_target() {
        let root = Path::new("/tmp/workspace");
        assert_eq!(
            sparse_validation_corpus_root(root),
            PathBuf::from("/tmp/workspace/target/validation_corpus")
        );
    }

    #[test]
    fn ad_cost_suite_args_match_expected_debug_and_release_invocations() {
        let debug = ad_cost_suite_args(false);
        let release = ad_cost_suite_args(true);

        assert!(debug.starts_with(&["run", "--quiet"]));
        assert!(!debug.contains(&"--release"));
        assert!(release.starts_with(&["run", "--quiet", "--release"]));
        assert_eq!(debug.last(), Some(&"--json"));
        assert_eq!(release.last(), Some(&"--json"));
    }

    #[test]
    fn validate_suite_profiles_accepts_expected_pair() {
        assert!(validate_suite_profiles(&empty_suite("debug"), &empty_suite("release")).is_ok());
    }

    #[test]
    fn validate_suite_profiles_rejects_unexpected_pair() {
        let result = validate_suite_profiles(&empty_suite("release"), &empty_suite("debug"));
        assert!(result.is_err());
        if let Err(err) = result {
            assert!(err.to_string().contains("unexpected suite profiles"));
        }
    }

    #[test]
    fn sparse_validation_tier_arg_maps_to_validation_tier() {
        assert_eq!(
            ValidationTier::from(SparseValidationTierArg::Pr),
            ValidationTier::Pr
        );
        assert_eq!(
            ValidationTier::from(SparseValidationTierArg::Scheduled),
            ValidationTier::Scheduled
        );
        assert_eq!(
            ValidationTier::from(SparseValidationTierArg::Local),
            ValidationTier::Local
        );
    }
}
