use std::fs;
use std::path::PathBuf;

use anyhow::{Result, bail};
use test_problems::{
    CallPolicyMode, JitOptLevel, ProblemRunOptions, ProblemSpeed, RunRequest, SolverKind,
    render_markdown_report, render_terminal_report, run_cases, write_dashboard,
    write_html_report, write_json_report, write_transcript_artifacts,
};

fn main() -> Result<()> {
    let mut request = RunRequest {
        progress: true,
        ..RunRequest::default()
    };
    let mut output_dir = PathBuf::from("target/test-problems");
    let mut args = std::env::args().skip(1);
    let mut problems = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--problem" => {
                let Some(problem) = args.next() else {
                    bail!("--problem requires an id");
                };
                problems.push(problem);
            }
            "--solver" => {
                let Some(value) = args.next() else {
                    bail!("--solver requires sqp|nlip|ipopt|both|all");
                };
                request.solvers = match value.as_str() {
                    "sqp" => vec![SolverKind::Sqp],
                    "nlip" | "ip" => vec![SolverKind::Nlip],
                    #[cfg(feature = "ipopt")]
                    "ipopt" => vec![SolverKind::Ipopt],
                    "both" => vec![SolverKind::Sqp, SolverKind::Nlip],
                    #[cfg(not(feature = "ipopt"))]
                    "all" => vec![SolverKind::Sqp, SolverKind::Nlip],
                    #[cfg(feature = "ipopt")]
                    "all" => vec![SolverKind::Sqp, SolverKind::Nlip, SolverKind::Ipopt],
                    _ => bail!("unknown solver: {value}"),
                };
            }
            "--jobs" => {
                let Some(value) = args.next() else {
                    bail!("--jobs requires a value");
                };
                request.jobs = Some(value.parse()?);
            }
            "--problem-set" => {
                let Some(value) = args.next() else {
                    bail!("--problem-set requires fast|slow|all");
                };
                request.problem_set = match value.as_str() {
                    "fast" => Some(ProblemSpeed::Fast),
                    "slow" => Some(ProblemSpeed::Slow),
                    "all" => None,
                    _ => bail!("unknown problem set: {value}"),
                };
            }
            "--jit-opt" => {
                let Some(value) = args.next() else {
                    bail!("--jit-opt requires 0|2|3|s|all");
                };
                let jit_opt_levels = match value.as_str() {
                    "all" => vec![
                        JitOptLevel::O0,
                        JitOptLevel::O2,
                        JitOptLevel::O3,
                        JitOptLevel::Os,
                    ],
                    "0" | "o0" | "O0" => vec![JitOptLevel::O0],
                    "2" | "o2" | "O2" => vec![JitOptLevel::O2],
                    "3" | "o3" | "O3" => vec![JitOptLevel::O3],
                    "s" | "os" | "Os" | "OS" => vec![JitOptLevel::Os],
                    _ => bail!("unknown jit opt level: {value}"),
                };
                let call_policies = request
                    .run_options
                    .iter()
                    .map(|options| options.call_policy)
                    .collect::<Vec<_>>();
                request.run_options = jit_opt_levels
                    .into_iter()
                    .flat_map(|jit_opt_level| {
                        call_policies.iter().copied().map(move |call_policy| ProblemRunOptions {
                            jit_opt_level,
                            call_policy,
                        })
                    })
                    .collect();
            }
            "--call-policy" => {
                let Some(value) = args.next() else {
                    bail!(
                        "--call-policy requires inline_at_call|inline_at_lowering|inline_in_llvm|no_inline_llvm|all"
                    );
                };
                let call_policies = match value.as_str() {
                    "all" => vec![
                        CallPolicyMode::InlineAtCall,
                        CallPolicyMode::InlineAtLowering,
                        CallPolicyMode::InlineInLlvm,
                        CallPolicyMode::NoInlineLlvm,
                    ],
                    "inline_at_call" => vec![CallPolicyMode::InlineAtCall],
                    "inline_at_lowering" => vec![CallPolicyMode::InlineAtLowering],
                    "inline_in_llvm" => vec![CallPolicyMode::InlineInLlvm],
                    "no_inline_llvm" => vec![CallPolicyMode::NoInlineLlvm],
                    _ => bail!("unknown call policy: {value}"),
                };
                let jit_opt_levels = request
                    .run_options
                    .iter()
                    .map(|options| options.jit_opt_level)
                    .collect::<Vec<_>>();
                request.run_options = jit_opt_levels
                    .into_iter()
                    .flat_map(|jit_opt_level| {
                        call_policies.iter().copied().map(move |call_policy| ProblemRunOptions {
                            jit_opt_level,
                            call_policy,
                        })
                    })
                    .collect();
            }
            "--output-dir" => {
                let Some(value) = args.next() else {
                    bail!("--output-dir requires a path");
                };
                output_dir = PathBuf::from(value);
            }
            "--include-skipped" => {
                request.include_skipped = true;
            }
            "--no-progress" => {
                request.progress = false;
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            _ => bail!("unknown argument: {arg}"),
        }
    }

    if !problems.is_empty() {
        request.problem_ids = Some(problems);
    }

    fs::create_dir_all(&output_dir)?;
    let mut results = run_cases(&request)?;
    write_transcript_artifacts(&mut results, &output_dir)?;
    let markdown = render_markdown_report(&results);
    let terminal = render_terminal_report(&results);
    let markdown_path = output_dir.join("report.md");
    let html_path = output_dir.join("report.html");
    let json_path = output_dir.join("report.json");
    let dashboard_path = output_dir.join("dashboard.html");
    fs::write(&markdown_path, &markdown)?;
    write_html_report(&results, &html_path)?;
    write_json_report(&results, &json_path)?;
    write_dashboard(&results, &dashboard_path)?;

    println!("{terminal}");
    println!(
        "Wrote:\n- {}\n- {}\n- {}\n- {}",
        markdown_path.display(),
        html_path.display(),
        json_path.display(),
        dashboard_path.display()
    );
    Ok(())
}

fn print_help() {
    println!(
        "Usage: cargo run --release -p test_problems -- [--problem ID]... [--solver sqp|nlip|ipopt|both|all] [--problem-set fast|slow|all] [--jit-opt 0|2|3|s|all] [--call-policy inline_at_call|inline_at_lowering|inline_in_llvm|no_inline_llvm|all] [--jobs N] [--output-dir DIR] [--include-skipped] [--no-progress]"
    );
}
