use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};
use bench_report::{MarkdownReportOptions, SuiteReport, render_markdown_report_with_options};

fn workspace_root() -> Result<PathBuf> {
    env::current_dir().context("xtask must run from the workspace root")
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReportPaths {
    dir: PathBuf,
    debug_json: PathBuf,
    release_json: PathBuf,
    markdown: PathBuf,
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

fn main() -> Result<()> {
    match env::args().nth(1).as_deref() {
        Some("ad-cost-report") | None => generate_ad_cost_report(),
        Some(other) => bail!("unknown xtask command: {other}"),
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
}
