use std::time::Duration;

use optimization::{CallPolicy, CallPolicyConfig, FunctionCompileOptions, LlvmOptimizationLevel};
use serde::{Deserialize, Serialize, ser::SerializeStruct};

use crate::manifest::{KnownStatus, ProblemTestSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverKind {
    Sqp,
    Nlip,
    #[cfg(feature = "ipopt")]
    Ipopt,
}

impl SolverKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Sqp => "sqp",
            Self::Nlip => "nlip",
            #[cfg(feature = "ipopt")]
            Self::Ipopt => "ipopt",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JitOptLevel {
    O0,
    O2,
    #[default]
    O3,
    Os,
}

impl JitOptLevel {
    pub const fn label(self) -> &'static str {
        match self {
            Self::O0 => "O0",
            Self::O2 => "O2",
            Self::O3 => "O3",
            Self::Os => "Os",
        }
    }

    pub const fn into_llvm(self) -> LlvmOptimizationLevel {
        match self {
            Self::O0 => LlvmOptimizationLevel::O0,
            Self::O2 => LlvmOptimizationLevel::O2,
            Self::O3 => LlvmOptimizationLevel::O3,
            Self::Os => LlvmOptimizationLevel::Os,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProblemRunOptions {
    pub jit_opt_level: JitOptLevel,
    pub call_policy: CallPolicyMode,
}

impl ProblemRunOptions {
    pub fn label(self) -> String {
        format!(
            "{} / {}",
            self.jit_opt_level.label(),
            self.call_policy.label()
        )
    }

    pub const fn compile_options(self) -> FunctionCompileOptions {
        FunctionCompileOptions {
            opt_level: self.jit_opt_level.into_llvm(),
            call_policy: CallPolicyConfig {
                default_policy: self.call_policy.into_sx(),
                respect_function_overrides: true,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallPolicyMode {
    InlineAtCall,
    #[default]
    InlineAtLowering,
    InlineInLlvm,
    NoInlineLlvm,
}

impl CallPolicyMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::InlineAtCall => "inline_at_call",
            Self::InlineAtLowering => "inline_at_lowering",
            Self::InlineInLlvm => "inline_in_llvm",
            Self::NoInlineLlvm => "no_inline_llvm",
        }
    }

    pub const fn into_sx(self) -> CallPolicy {
        match self {
            Self::InlineAtCall => CallPolicy::InlineAtCall,
            Self::InlineAtLowering => CallPolicy::InlineAtLowering,
            Self::InlineInLlvm => CallPolicy::InlineInLLVM,
            Self::NoInlineLlvm => CallPolicy::NoInlineLLVM,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Passed,
    ReducedAccuracy,
    FailedValidation,
    SolveError,
    Skipped,
}

impl RunStatus {
    pub const fn accepted(self) -> bool {
        matches!(self, Self::Passed | Self::ReducedAccuracy)
    }

    pub const fn failed(self) -> bool {
        matches!(self, Self::FailedValidation | Self::SolveError)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationTier {
    #[default]
    Failed,
    Passed,
    ReducedAccuracy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProblemDescriptor {
    pub id: String,
    pub test_set: ProblemTestSet,
    pub family: String,
    pub variant: String,
    pub source: String,
    pub description: String,
    pub parameterized: bool,
    pub num_vars: usize,
    pub num_eq: usize,
    pub num_ineq: usize,
    pub num_box: usize,
    pub dof: usize,
    pub constrained: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SolverMetrics {
    pub iterations: Option<usize>,
    pub objective: Option<f64>,
    pub equality_inf: Option<f64>,
    pub inequality_inf: Option<f64>,
    pub primal_inf: Option<f64>,
    pub dual_inf: Option<f64>,
    pub complementarity_inf: Option<f64>,
    pub elastic_recovery_activations: Option<usize>,
    pub elastic_recovery_qp_solves: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SolverTimingBreakdown {
    pub function_creation_time: Option<Duration>,
    pub derivative_generation_time: Option<Duration>,
    pub jit_time: Option<Duration>,
    pub compile_wall_time: Option<Duration>,
    pub solve_time: Option<Duration>,
    pub total_wall_time: Duration,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ValidationOutcome {
    pub tier: ValidationTier,
    pub tolerance: String,
    pub detail: String,
}

impl ValidationOutcome {
    pub const fn passed(&self) -> bool {
        matches!(
            self.tier,
            ValidationTier::Passed | ValidationTier::ReducedAccuracy
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FilterReplayPoint {
    pub violation: f64,
    pub objective: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FilterReplayFrame {
    pub iteration: usize,
    pub phase: String,
    pub current: FilterReplayPoint,
    #[serde(default)]
    pub frontier: Vec<FilterReplayPoint>,
    #[serde(default)]
    pub rejected_trials: Vec<FilterReplayPoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_mode: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FilterReplay {
    #[serde(default)]
    pub frames: Vec<FilterReplayFrame>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SetupProfileBreakdown {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbolic_construction_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objective_gradient_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub equality_jacobian_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inequality_jacobian_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lagrangian_assembly_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hessian_generation_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lowering_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llvm_jit_s: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CompileStatsSummary {
    pub symbolic_function_count: usize,
    pub call_site_count: usize,
    pub max_call_depth: usize,
    pub inline_at_call_policy_count: usize,
    pub inline_at_lowering_policy_count: usize,
    pub inline_in_llvm_policy_count: usize,
    pub no_inline_llvm_policy_count: usize,
    pub overrides_applied: usize,
    pub overrides_ignored: usize,
    pub inlines_at_call: usize,
    pub inlines_at_lowering: usize,
    pub llvm_subfunctions_emitted: usize,
    pub llvm_call_instructions_emitted: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CompileReportSummary {
    pub setup: SetupProfileBreakdown,
    pub stats: CompileStatsSummary,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResultCacheStatus {
    Miss,
    Hit,
    Bypassed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResultCacheInfo {
    pub status: ResultCacheStatus,
    pub key: String,
    pub written_at_unix_ms: Option<u128>,
    pub lookup_time_s: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProblemRunRecord {
    pub id: String,
    pub solver: SolverKind,
    pub options: ProblemRunOptions,
    pub expected: KnownStatus,
    pub max_iters_limit: usize,
    pub status: RunStatus,
    pub descriptor: ProblemDescriptor,
    pub solution: Option<Vec<f64>>,
    pub metrics: SolverMetrics,
    pub timing: SolverTimingBreakdown,
    pub validation: ValidationOutcome,
    pub solver_thresholds: Option<String>,
    pub solver_settings: Option<String>,
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compile_report: Option<CompileReportSummary>,
    #[serde(skip)]
    pub console_output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub console_output_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter_replay: Option<FilterReplay>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache: Option<ResultCacheInfo>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunArtifacts {
    pub markdown_report: String,
    pub json_report: String,
    pub dashboard_html: String,
}

pub struct ProblemCase {
    pub id: &'static str,
    pub test_set: ProblemTestSet,
    pub family: &'static str,
    pub variant: &'static str,
    pub source: &'static str,
    pub description: &'static str,
    pub parameterized: bool,
    pub(crate) run_fn: Box<
        dyn Fn(SolverKind, ProblemRunOptions, usize, KnownStatus) -> ProblemRunRecord + Send + Sync,
    >,
}

impl ProblemCase {
    pub fn run(
        &self,
        solver: SolverKind,
        options: ProblemRunOptions,
        max_iters_limit: usize,
        expected: KnownStatus,
    ) -> ProblemRunRecord {
        (self.run_fn)(solver, options, max_iters_limit, expected)
    }
}

impl std::fmt::Debug for ProblemCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProblemCase")
            .field("id", &self.id)
            .field("test_set", &self.test_set)
            .field("family", &self.family)
            .field("variant", &self.variant)
            .field("source", &self.source)
            .field("description", &self.description)
            .field("parameterized", &self.parameterized)
            .finish_non_exhaustive()
    }
}

impl Serialize for SolverTimingBreakdown {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("SolverTimingBreakdown", 6)?;
        state.serialize_field(
            "function_creation_time",
            &self
                .function_creation_time
                .map(|duration| duration.as_secs_f64()),
        )?;
        state.serialize_field(
            "derivative_generation_time",
            &self
                .derivative_generation_time
                .map(|duration| duration.as_secs_f64()),
        )?;
        state.serialize_field(
            "jit_time",
            &self.jit_time.map(|duration| duration.as_secs_f64()),
        )?;
        state.serialize_field(
            "compile_wall_time",
            &self
                .compile_wall_time
                .map(|duration| duration.as_secs_f64()),
        )?;
        state.serialize_field(
            "solve_time",
            &self.solve_time.map(|duration| duration.as_secs_f64()),
        )?;
        state.serialize_field("total_wall_time", &self.total_wall_time.as_secs_f64())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for SolverTimingBreakdown {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            function_creation_time: Option<f64>,
            derivative_generation_time: Option<f64>,
            jit_time: Option<f64>,
            compile_wall_time: Option<f64>,
            solve_time: Option<f64>,
            total_wall_time: f64,
        }

        let wire = Wire::deserialize(deserializer)?;
        Ok(Self {
            function_creation_time: wire.function_creation_time.map(Duration::from_secs_f64),
            derivative_generation_time: wire
                .derivative_generation_time
                .map(Duration::from_secs_f64),
            jit_time: wire.jit_time.map(Duration::from_secs_f64),
            compile_wall_time: wire.compile_wall_time.map(Duration::from_secs_f64),
            solve_time: wire.solve_time.map(Duration::from_secs_f64),
            total_wall_time: Duration::from_secs_f64(wire.total_wall_time),
        })
    }
}
