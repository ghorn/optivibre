use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use test_problems::{
    CallPolicyMode, JitOptLevel, ProblemRunOptions, ProblemSpeed, RunRequest, SolverKind,
    render_markdown_report, render_terminal_report, run_cases, write_dashboard, write_html_report,
    write_json_report, write_transcript_artifacts,
};

#[derive(Debug, Parser)]
#[command(
    name = "test_problems",
    about = "Run the nonlinear test-problem suite and emit report artifacts."
)]
struct TestProblemsCli {
    #[arg(long = "problem")]
    problems: Vec<String>,
    #[arg(long, value_enum)]
    solver: Option<CliSolverSelection>,
    #[arg(long = "problem-set", value_enum)]
    problem_set: Option<CliProblemSetSelection>,
    #[arg(long = "jit-opt", value_enum)]
    jit_opt: Option<CliJitOptSelection>,
    #[arg(long = "call-policy", value_enum)]
    call_policy: Option<CliCallPolicySelection>,
    #[arg(long, value_parser = parse_positive_usize)]
    jobs: Option<usize>,
    #[arg(long = "output-dir", default_value = "target/test-problems")]
    output_dir: PathBuf,
    #[arg(long)]
    include_skipped: bool,
    #[arg(long)]
    no_progress: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliSolverSelection {
    Sqp,
    #[value(alias = "ip")]
    Nlip,
    #[cfg(feature = "ipopt")]
    Ipopt,
    Both,
    All,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliProblemSetSelection {
    Fast,
    Slow,
    All,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliJitOptSelection {
    #[value(name = "0", alias = "o0", alias = "O0")]
    O0,
    #[value(name = "2", alias = "o2", alias = "O2")]
    O2,
    #[value(name = "3", alias = "o3", alias = "O3")]
    O3,
    #[value(name = "s", alias = "os", alias = "Os", alias = "OS")]
    Os,
    All,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliCallPolicySelection {
    InlineAtCall,
    InlineAtLowering,
    InlineInLlvm,
    NoInlineLlvm,
    All,
}

fn main() -> Result<()> {
    let cli = TestProblemsCli::parse();
    let mut request = RunRequest {
        progress: !cli.no_progress,
        ..RunRequest::default()
    };

    if !cli.problems.is_empty() {
        request.problem_ids = Some(cli.problems);
    }
    if let Some(selection) = cli.solver {
        request.solvers = selection.solvers();
    }
    if let Some(jobs) = cli.jobs {
        request.jobs = Some(jobs);
    }
    if let Some(selection) = cli.problem_set {
        request.problem_set = selection.problem_speed();
    }
    if let Some(selection) = cli.jit_opt {
        let call_policies = request
            .run_options
            .iter()
            .map(|options| options.call_policy)
            .collect::<Vec<_>>();
        request.run_options = selection
            .jit_opt_levels()
            .into_iter()
            .flat_map(|jit_opt_level| {
                call_policies
                    .iter()
                    .copied()
                    .map(move |call_policy| ProblemRunOptions {
                        jit_opt_level,
                        call_policy,
                    })
            })
            .collect();
    }
    if let Some(selection) = cli.call_policy {
        let jit_opt_levels = request
            .run_options
            .iter()
            .map(|options| options.jit_opt_level)
            .collect::<Vec<_>>();
        request.run_options = jit_opt_levels
            .into_iter()
            .flat_map(|jit_opt_level| {
                selection
                    .call_policies()
                    .into_iter()
                    .map(move |call_policy| ProblemRunOptions {
                        jit_opt_level,
                        call_policy,
                    })
            })
            .collect();
    }
    request.include_skipped = cli.include_skipped;

    fs::create_dir_all(&cli.output_dir)?;
    let mut results = run_cases(&request)?;
    write_transcript_artifacts(&mut results, &cli.output_dir)?;
    let markdown = render_markdown_report(&results);
    let terminal = render_terminal_report(&results);
    let markdown_path = cli.output_dir.join("report.md");
    let html_path = cli.output_dir.join("report.html");
    let json_path = cli.output_dir.join("report.json");
    let dashboard_path = cli.output_dir.join("dashboard.html");
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

fn parse_positive_usize(value: &str) -> std::result::Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("invalid positive integer `{value}`"))?;
    if parsed == 0 {
        return Err("value must be greater than zero".to_string());
    }
    Ok(parsed)
}

impl CliSolverSelection {
    fn solvers(self) -> Vec<SolverKind> {
        match self {
            Self::Sqp => vec![SolverKind::Sqp],
            Self::Nlip => vec![SolverKind::Nlip],
            #[cfg(feature = "ipopt")]
            Self::Ipopt => vec![SolverKind::Ipopt],
            Self::Both => vec![SolverKind::Sqp, SolverKind::Nlip],
            Self::All => {
                #[cfg(feature = "ipopt")]
                {
                    vec![SolverKind::Sqp, SolverKind::Nlip, SolverKind::Ipopt]
                }
                #[cfg(not(feature = "ipopt"))]
                {
                    vec![SolverKind::Sqp, SolverKind::Nlip]
                }
            }
        }
    }
}

impl CliProblemSetSelection {
    fn problem_speed(self) -> Option<ProblemSpeed> {
        match self {
            Self::Fast => Some(ProblemSpeed::Fast),
            Self::Slow => Some(ProblemSpeed::Slow),
            Self::All => None,
        }
    }
}

impl CliJitOptSelection {
    fn jit_opt_levels(self) -> Vec<JitOptLevel> {
        match self {
            Self::O0 => vec![JitOptLevel::O0],
            Self::O2 => vec![JitOptLevel::O2],
            Self::O3 => vec![JitOptLevel::O3],
            Self::Os => vec![JitOptLevel::Os],
            Self::All => vec![
                JitOptLevel::O0,
                JitOptLevel::O2,
                JitOptLevel::O3,
                JitOptLevel::Os,
            ],
        }
    }
}

impl CliCallPolicySelection {
    fn call_policies(self) -> Vec<CallPolicyMode> {
        match self {
            Self::InlineAtCall => vec![CallPolicyMode::InlineAtCall],
            Self::InlineAtLowering => vec![CallPolicyMode::InlineAtLowering],
            Self::InlineInLlvm => vec![CallPolicyMode::InlineInLlvm],
            Self::NoInlineLlvm => vec![CallPolicyMode::NoInlineLlvm],
            Self::All => vec![
                CallPolicyMode::InlineAtCall,
                CallPolicyMode::InlineAtLowering,
                CallPolicyMode::InlineInLlvm,
                CallPolicyMode::NoInlineLlvm,
            ],
        }
    }
}
