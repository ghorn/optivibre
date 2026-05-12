const PALETTE = ["#f7b267", "#5bd1b5", "#7cc6fe", "#f25f5c", "#d7aefb", "#b8f2e6"] as const;
const PLOT_LINE_WIDTH = Object.freeze({
  primary: 1.2,
  secondary: 0.95,
  bound: 0.7,
  tolerance: 0.6,
  scenePrimary: 1.4,
  sceneSecondary: 0.9,
  shape: 0.8,
});
const PLOT_MARKER_SIZE = Object.freeze({
  primary: 2.4,
  secondary: 2.0,
});
const EQ_INF_LABEL = "‖eq‖∞";
const INEQ_INF_LABEL = "‖ineq₊‖∞";
const DUAL_INF_LABEL = "‖∇L‖∞";
const TRUST_REGION_RADIUS_LABEL = "TR radius";
const STEP_INF_LABEL = "‖Δx‖∞";
const ITERATION_ARTIFACT_RENDER_STRIDE = 1;

type EnumValue<T extends Record<string, number>> = T[keyof T];
type NumericPoint = [number, number];
type NumericRange = [number, number];
type JsonPrimitive = string | number | boolean | null;
type JsonValue = JsonPrimitive | JsonObject | JsonArray;
type JsonArray = JsonValue[];

interface JsonObject {
  [key: string]: JsonValue | undefined;
}

const CONTROL_SECTION = Object.freeze({
  transcription: 0,
  solver: 1,
  problem: 2,
} as const);
const CONTROL_SECTION_FROM_WIRE = Object.freeze({
  transcription: CONTROL_SECTION.transcription,
  solver: CONTROL_SECTION.solver,
  problem: CONTROL_SECTION.problem,
} as const);
const CONTROL_PANEL = Object.freeze({
  sxFunctions: 0,
} as const);
const CONTROL_PANEL_FROM_WIRE = Object.freeze({
  sx_functions: CONTROL_PANEL.sxFunctions,
} as const);
const CONTROL_EDITOR = Object.freeze({
  slider: 0,
  select: 1,
  text: 2,
  checkbox: 3,
} as const);
const CONTROL_EDITOR_FROM_WIRE = Object.freeze({
  slider: CONTROL_EDITOR.slider,
  select: CONTROL_EDITOR.select,
  text: CONTROL_EDITOR.text,
  checkbox: CONTROL_EDITOR.checkbox,
} as const);
const CONTROL_SEMANTIC = Object.freeze({
  transcriptionMethod: 0,
  transcriptionIntervals: 1,
  collocationFamily: 2,
  collocationDegree: 3,
  solverMethod: 4,
  solverGlobalization: 5,
  solverMaxIterations: 6,
  solverHessianRegularization: 7,
  solverNlipLinearSolver: 8,
  solverDualTolerance: 9,
  solverConstraintTolerance: 10,
  solverComplementarityTolerance: 11,
  solverExactMeritPenalty: 12,
  solverPenaltyIncreaseFactor: 13,
  solverMaxPenaltyUpdates: 14,
  solverArmijoC1: 15,
  solverWolfeC2: 16,
  solverLineSearchBeta: 17,
  solverLineSearchMaxSteps: 18,
  solverMinStep: 19,
  solverFilterGammaObjective: 20,
  solverFilterGammaViolation: 21,
  solverFilterThetaMaxFactor: 22,
  solverFilterSwitchingReferenceMin: 23,
  solverFilterSwitchingViolationFactor: 24,
  solverFilterSwitchingLinearizedReductionFactor: 25,
  solverTrustRegionInitialRadius: 26,
  solverTrustRegionMaxRadius: 27,
  solverTrustRegionMinRadius: 28,
  solverTrustRegionShrinkFactor: 29,
  solverTrustRegionGrowFactor: 30,
  solverTrustRegionAcceptRatio: 31,
  solverTrustRegionExpandRatio: 32,
  solverTrustRegionBoundaryFraction: 33,
  solverTrustRegionMaxContractions: 34,
  solverTrustRegionFixedPenalty: 35,
  sxFunctionOption: 36,
  problemParameter: 37,
  solverNlipSpralPivotMethod: 38,
  solverNlipSpralZeroPivotAction: 39,
  solverNlipSpralSmallPivot: 40,
  solverNlipSpralPivotU: 41,
  timeGrid: 42,
  timeGridStrength: 43,
  timeGridFocusCenter: 44,
  timeGridFocusWidth: 45,
  timeGridBreakpoint: 46,
  timeGridFirstIntervalFraction: 47,
  solverProfile: 48,
  solverOverallTolerance: 49,
} as const);
const CONTROL_SEMANTIC_FROM_WIRE = Object.freeze({
  transcription_method: CONTROL_SEMANTIC.transcriptionMethod,
  transcription_intervals: CONTROL_SEMANTIC.transcriptionIntervals,
  collocation_family: CONTROL_SEMANTIC.collocationFamily,
  collocation_degree: CONTROL_SEMANTIC.collocationDegree,
  time_grid: CONTROL_SEMANTIC.timeGrid,
  time_grid_strength: CONTROL_SEMANTIC.timeGridStrength,
  time_grid_focus_center: CONTROL_SEMANTIC.timeGridFocusCenter,
  time_grid_focus_width: CONTROL_SEMANTIC.timeGridFocusWidth,
  time_grid_breakpoint: CONTROL_SEMANTIC.timeGridBreakpoint,
  time_grid_first_interval_fraction: CONTROL_SEMANTIC.timeGridFirstIntervalFraction,
  solver_method: CONTROL_SEMANTIC.solverMethod,
  solver_profile: CONTROL_SEMANTIC.solverProfile,
  solver_globalization: CONTROL_SEMANTIC.solverGlobalization,
  solver_max_iterations: CONTROL_SEMANTIC.solverMaxIterations,
  solver_overall_tolerance: CONTROL_SEMANTIC.solverOverallTolerance,
  solver_overall_tol: CONTROL_SEMANTIC.solverOverallTolerance,
  solver_hessian_regularization: CONTROL_SEMANTIC.solverHessianRegularization,
  solver_nlip_linear_solver: CONTROL_SEMANTIC.solverNlipLinearSolver,
  solver_nlip_spral_pivot_method: CONTROL_SEMANTIC.solverNlipSpralPivotMethod,
  solver_nlip_spral_zero_pivot_action: CONTROL_SEMANTIC.solverNlipSpralZeroPivotAction,
  solver_nlip_spral_small_pivot: CONTROL_SEMANTIC.solverNlipSpralSmallPivot,
  solver_nlip_spral_pivot_u: CONTROL_SEMANTIC.solverNlipSpralPivotU,
  solver_dual_tolerance: CONTROL_SEMANTIC.solverDualTolerance,
  solver_constraint_tolerance: CONTROL_SEMANTIC.solverConstraintTolerance,
  solver_complementarity_tolerance: CONTROL_SEMANTIC.solverComplementarityTolerance,
  solver_exact_merit_penalty: CONTROL_SEMANTIC.solverExactMeritPenalty,
  solver_penalty_increase_factor: CONTROL_SEMANTIC.solverPenaltyIncreaseFactor,
  solver_max_penalty_updates: CONTROL_SEMANTIC.solverMaxPenaltyUpdates,
  solver_armijo_c1: CONTROL_SEMANTIC.solverArmijoC1,
  solver_wolfe_c2: CONTROL_SEMANTIC.solverWolfeC2,
  solver_line_search_beta: CONTROL_SEMANTIC.solverLineSearchBeta,
  solver_line_search_max_steps: CONTROL_SEMANTIC.solverLineSearchMaxSteps,
  solver_min_step: CONTROL_SEMANTIC.solverMinStep,
  solver_filter_gamma_objective: CONTROL_SEMANTIC.solverFilterGammaObjective,
  solver_filter_gamma_violation: CONTROL_SEMANTIC.solverFilterGammaViolation,
  solver_filter_theta_max_factor: CONTROL_SEMANTIC.solverFilterThetaMaxFactor,
  solver_filter_switching_reference_min: CONTROL_SEMANTIC.solverFilterSwitchingReferenceMin,
  solver_filter_switching_violation_factor:
    CONTROL_SEMANTIC.solverFilterSwitchingViolationFactor,
  solver_filter_switching_linearized_reduction_factor:
    CONTROL_SEMANTIC.solverFilterSwitchingLinearizedReductionFactor,
  solver_trust_region_initial_radius: CONTROL_SEMANTIC.solverTrustRegionInitialRadius,
  solver_trust_region_max_radius: CONTROL_SEMANTIC.solverTrustRegionMaxRadius,
  solver_trust_region_min_radius: CONTROL_SEMANTIC.solverTrustRegionMinRadius,
  solver_trust_region_shrink_factor: CONTROL_SEMANTIC.solverTrustRegionShrinkFactor,
  solver_trust_region_grow_factor: CONTROL_SEMANTIC.solverTrustRegionGrowFactor,
  solver_trust_region_accept_ratio: CONTROL_SEMANTIC.solverTrustRegionAcceptRatio,
  solver_trust_region_expand_ratio: CONTROL_SEMANTIC.solverTrustRegionExpandRatio,
  solver_trust_region_boundary_fraction: CONTROL_SEMANTIC.solverTrustRegionBoundaryFraction,
  solver_trust_region_max_contractions: CONTROL_SEMANTIC.solverTrustRegionMaxContractions,
  solver_trust_region_fixed_penalty: CONTROL_SEMANTIC.solverTrustRegionFixedPenalty,
  sx_function_option: CONTROL_SEMANTIC.sxFunctionOption,
  problem_parameter: CONTROL_SEMANTIC.problemParameter,
} as const);
const CONTROL_VISIBILITY = Object.freeze({
  always: 0,
  directCollocationOnly: 1,
  multipleShootingOnly: 2,
} as const);
const CONTROL_VISIBILITY_FROM_WIRE = Object.freeze({
  always: CONTROL_VISIBILITY.always,
  direct_collocation_only: CONTROL_VISIBILITY.directCollocationOnly,
  multiple_shooting_only: CONTROL_VISIBILITY.multipleShootingOnly,
} as const);
const CONTROL_VALUE_DISPLAY = Object.freeze({
  scalar: 0,
  integer: 1,
  scientific: 2,
} as const);
const CONTROL_VALUE_DISPLAY_FROM_WIRE = Object.freeze({
  scalar: CONTROL_VALUE_DISPLAY.scalar,
  integer: CONTROL_VALUE_DISPLAY.integer,
  scientific: CONTROL_VALUE_DISPLAY.scientific,
} as const);
const LOG_LEVEL = Object.freeze({
  console: 0,
  info: 1,
  warning: 2,
  error: 3,
} as const);
const LOG_LEVEL_FROM_WIRE = Object.freeze({
  console: LOG_LEVEL.console,
  info: LOG_LEVEL.info,
  warning: LOG_LEVEL.warning,
  error: LOG_LEVEL.error,
} as const);
const TIME_SERIES_ROLE = Object.freeze({
  data: 0,
  lowerBound: 1,
  upperBound: 2,
} as const);
const TIME_SERIES_ROLE_FROM_WIRE = Object.freeze({
  data: TIME_SERIES_ROLE.data,
  lower_bound: TIME_SERIES_ROLE.lowerBound,
  upper_bound: TIME_SERIES_ROLE.upperBound,
} as const);
const SOLVER_STATUS_KIND = Object.freeze({
  success: 0,
  warning: 1,
  error: 2,
  info: 3,
} as const);
const SOLVER_STATUS_KIND_FROM_WIRE = Object.freeze({
  success: SOLVER_STATUS_KIND.success,
  warning: SOLVER_STATUS_KIND.warning,
  error: SOLVER_STATUS_KIND.error,
  info: SOLVER_STATUS_KIND.info,
} as const);
const CONSTRAINT_PANEL_SEVERITY = Object.freeze({
  fullAccuracy: 0,
  reducedAccuracy: 1,
  violated: 2,
} as const);
const CONSTRAINT_PANEL_SEVERITY_FROM_WIRE = Object.freeze({
  full_accuracy: CONSTRAINT_PANEL_SEVERITY.fullAccuracy,
  reduced_accuracy: CONSTRAINT_PANEL_SEVERITY.reducedAccuracy,
  violated: CONSTRAINT_PANEL_SEVERITY.violated,
} as const);
const CONSTRAINT_PANEL_CATEGORY = Object.freeze({
  boundaryEquality: 0,
  boundaryInequality: 1,
  path: 2,
  continuityState: 3,
  continuityControl: 4,
  collocationState: 5,
  collocationControl: 6,
  finalTime: 7,
} as const);
const CONSTRAINT_PANEL_CATEGORY_FROM_WIRE = Object.freeze({
  boundary_equality: CONSTRAINT_PANEL_CATEGORY.boundaryEquality,
  boundary_inequality: CONSTRAINT_PANEL_CATEGORY.boundaryInequality,
  path: CONSTRAINT_PANEL_CATEGORY.path,
  continuity_state: CONSTRAINT_PANEL_CATEGORY.continuityState,
  continuity_control: CONSTRAINT_PANEL_CATEGORY.continuityControl,
  collocation_state: CONSTRAINT_PANEL_CATEGORY.collocationState,
  collocation_control: CONSTRAINT_PANEL_CATEGORY.collocationControl,
  final_time: CONSTRAINT_PANEL_CATEGORY.finalTime,
} as const);
const CONSTRAINT_PANEL_BOUND_SIDE = Object.freeze({
  none: 0,
  lower: 1,
  upper: 2,
  both: 3,
} as const);
const CONSTRAINT_PANEL_BOUND_SIDE_FROM_WIRE = Object.freeze({
  none: CONSTRAINT_PANEL_BOUND_SIDE.none,
  lower: CONSTRAINT_PANEL_BOUND_SIDE.lower,
  upper: CONSTRAINT_PANEL_BOUND_SIDE.upper,
  both: CONSTRAINT_PANEL_BOUND_SIDE.both,
} as const);
const METRIC_KEY = Object.freeze({
  custom: 0,
  transcriptionMethod: 1,
  intervalCount: 2,
  collocationNodeCount: 3,
  termination: 4,
  distance: 5,
  finalTime: 6,
  bestGlideAlpha: 7,
  terminalLiftToDrag: 8,
  peakAltitude: 9,
  trimCost: 10,
  finalX: 11,
  finalY: 12,
  maxY: 13,
  minY: 14,
  peakJerk: 15,
  transferTime: 16,
  targetX: 17,
  maxSwing: 18,
  maxAccel: 19,
  maxJerk: 20,
  duration: 21,
  upwindTarget: 22,
  upwindDistance: 23,
  maxSpeed: 24,
  tackCount: 25,
  centerlineError: 26,
  maxCrossTrack: 27,
  timeGrid: 28,
} as const);
const METRIC_KEY_FROM_WIRE = Object.freeze({
  custom: METRIC_KEY.custom,
  transcription_method: METRIC_KEY.transcriptionMethod,
  interval_count: METRIC_KEY.intervalCount,
  collocation_node_count: METRIC_KEY.collocationNodeCount,
  time_grid: METRIC_KEY.timeGrid,
  termination: METRIC_KEY.termination,
  distance: METRIC_KEY.distance,
  final_time: METRIC_KEY.finalTime,
  best_glide_alpha: METRIC_KEY.bestGlideAlpha,
  terminal_lift_to_drag: METRIC_KEY.terminalLiftToDrag,
  peak_altitude: METRIC_KEY.peakAltitude,
  trim_cost: METRIC_KEY.trimCost,
  final_x: METRIC_KEY.finalX,
  final_y: METRIC_KEY.finalY,
  max_y: METRIC_KEY.maxY,
  min_y: METRIC_KEY.minY,
  peak_jerk: METRIC_KEY.peakJerk,
  transfer_time: METRIC_KEY.transferTime,
  target_x: METRIC_KEY.targetX,
  max_swing: METRIC_KEY.maxSwing,
  max_accel: METRIC_KEY.maxAccel,
  max_jerk: METRIC_KEY.maxJerk,
  duration: METRIC_KEY.duration,
  upwind_target: METRIC_KEY.upwindTarget,
  upwind_distance: METRIC_KEY.upwindDistance,
  max_speed: METRIC_KEY.maxSpeed,
  tack_count: METRIC_KEY.tackCount,
  centerline_error: METRIC_KEY.centerlineError,
  max_cross_track: METRIC_KEY.maxCrossTrack,
} as const);
const SOLVE_PHASE = Object.freeze({
  initial: 0,
  acceptedStep: 1,
  postConvergence: 2,
  converged: 3,
  regular: 4,
  restoration: 5,
} as const);
const FILTER_ACCEPTANCE_MODE = Object.freeze({
  objectiveArmijo: 0,
  violationReduction: 1,
} as const);
const FILTER_ACCEPTANCE_MODE_FROM_WIRE = Object.freeze({
  objective_armijo: FILTER_ACCEPTANCE_MODE.objectiveArmijo,
  violation_reduction: FILTER_ACCEPTANCE_MODE.violationReduction,
} as const);
const SOLVE_PHASE_FROM_WIRE = Object.freeze({
  initial: SOLVE_PHASE.initial,
  accepted_step: SOLVE_PHASE.acceptedStep,
  post_convergence: SOLVE_PHASE.postConvergence,
  converged: SOLVE_PHASE.converged,
  regular: SOLVE_PHASE.regular,
  restoration: SOLVE_PHASE.restoration,
} as const);
const SOLVER_METHOD = Object.freeze({
  sqp: 0,
  nlip: 1,
  ipopt: 2,
} as const);
const TIME_GRID = Object.freeze({
  uniform: 0,
  cosine: 1,
  tanh: 2,
  geometricStart: 3,
  geometricEnd: 4,
  focus: 5,
  piecewise: 6,
} as const);
const SOLVER_METHOD_FROM_WIRE = Object.freeze({
  sqp: SOLVER_METHOD.sqp,
  nlip: SOLVER_METHOD.nlip,
  ipopt: SOLVER_METHOD.ipopt,
} as const);
const NLIP_LINEAR_SOLVER = Object.freeze({
  ssidsRs: 0,
  spralSrc: 1,
  sparseQdldl: 2,
  auto: 3,
} as const);
const SQP_GLOBALIZATION = Object.freeze({
  lineSearchFilter: 0,
  lineSearchMerit: 1,
  trustRegionFilter: 2,
  trustRegionMerit: 3,
} as const);
const SOLVE_STAGE = Object.freeze({
  symbolicSetup: 0,
  jitCompilation: 1,
  solving: 2,
} as const);
const SOLVE_STAGE_FROM_WIRE = Object.freeze({
  symbolic_setup: SOLVE_STAGE.symbolicSetup,
  jit_compilation: SOLVE_STAGE.jitCompilation,
  solving: SOLVE_STAGE.solving,
} as const);
const STREAM_EVENT_KIND = Object.freeze({
  status: 0,
  log: 1,
  iteration: 2,
  final: 3,
  error: 4,
} as const);
const FILTER_TRACE = Object.freeze({
  history: 0,
  recent: 1,
  frontier: 2,
  current: 3,
} as const);
const FILTER_RECENT_POINT_LIMIT = 48;
const STREAM_EVENT_KIND_FROM_WIRE = Object.freeze({
  status: STREAM_EVENT_KIND.status,
  log: STREAM_EVENT_KIND.log,
  iteration: STREAM_EVENT_KIND.iteration,
  final: STREAM_EVENT_KIND.final,
  error: STREAM_EVENT_KIND.error,
} as const);
const PROBLEM_ID = Object.freeze({
  optimalDistanceGlider: 0,
  albatrossDynamicSoaring: 1,
  linearSManeuver: 2,
  sailboatUpwind: 3,
  craneTransfer: 4,
  hangingChainStatic: 5,
  rosenbrockVariants: 6,
} as const);
const PROBLEM_ID_FROM_WIRE = Object.freeze({
  optimal_distance_glider: PROBLEM_ID.optimalDistanceGlider,
  albatross_dynamic_soaring: PROBLEM_ID.albatrossDynamicSoaring,
  linear_s_maneuver: PROBLEM_ID.linearSManeuver,
  sailboat_upwind: PROBLEM_ID.sailboatUpwind,
  crane_transfer: PROBLEM_ID.craneTransfer,
  hanging_chain_static: PROBLEM_ID.hangingChainStatic,
  rosenbrock_variants: PROBLEM_ID.rosenbrockVariants,
} as const);
const ALBATROSS_DESIGN_PREFIXES = ["delta_l", "h0", "vx0", "tf"] as const;
type AlbatrossDesignPrefix = (typeof ALBATROSS_DESIGN_PREFIXES)[number];
const COMPILE_CACHE_STATE = Object.freeze({
  warming: 0,
  ready: 1,
} as const);
const COMPILE_CACHE_STATE_FROM_WIRE = Object.freeze({
  warming: COMPILE_CACHE_STATE.warming,
  ready: COMPILE_CACHE_STATE.ready,
} as const);

type ControlSectionCode = EnumValue<typeof CONTROL_SECTION>;
type ControlPanelCode = EnumValue<typeof CONTROL_PANEL>;
type ControlEditorCode = EnumValue<typeof CONTROL_EDITOR>;
type ControlSemanticCode = EnumValue<typeof CONTROL_SEMANTIC>;
type ControlVisibilityCode = EnumValue<typeof CONTROL_VISIBILITY>;
type ControlValueDisplayCode = EnumValue<typeof CONTROL_VALUE_DISPLAY>;
type LogLevelCode = EnumValue<typeof LOG_LEVEL>;
type TimeSeriesRoleCode = EnumValue<typeof TIME_SERIES_ROLE>;
type SolverStatusKindCode = EnumValue<typeof SOLVER_STATUS_KIND>;
type ConstraintPanelSeverityCode = EnumValue<typeof CONSTRAINT_PANEL_SEVERITY>;
type ConstraintPanelCategoryCode = EnumValue<typeof CONSTRAINT_PANEL_CATEGORY>;
type ConstraintPanelBoundSideCode = EnumValue<typeof CONSTRAINT_PANEL_BOUND_SIDE>;
type MetricKeyCode = EnumValue<typeof METRIC_KEY>;
type SolvePhaseCode = EnumValue<typeof SOLVE_PHASE>;
type SolverMethodCode = EnumValue<typeof SOLVER_METHOD>;
type SolveStageCode = EnumValue<typeof SOLVE_STAGE>;
type StreamEventKindCode = EnumValue<typeof STREAM_EVENT_KIND>;
type ProblemIdCode = EnumValue<typeof PROBLEM_ID>;
type CompileCacheStateCode = EnumValue<typeof COMPILE_CACHE_STATE>;

interface ControlChoice {
  value: number;
  label: string;
}

interface ControlProfileDefault {
  profile: number;
  value: number;
}

interface WireControlSpec {
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  unit: string;
  help: string;
  section?: string | number;
  panel?: string | number;
  editor?: string | number;
  visibility?: string | number;
  semantic?: string | number;
  value_display?: string | number;
  choices?: ControlChoice[];
  profile_defaults?: ControlProfileDefault[];
}

interface ControlSpec {
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
  default: number;
  unit: string;
  help: string;
  section: ControlSectionCode;
  panel: ControlPanelCode | null;
  editor: ControlEditorCode;
  visibility: ControlVisibilityCode;
  semantic: ControlSemanticCode;
  value_display: ControlValueDisplayCode;
  choices: ControlChoice[];
  profile_defaults: ControlProfileDefault[];
}

interface LatexSection {
  title: string;
  entries: string[];
}

interface WireProblemSpec {
  id: string | number;
  name: string;
  description: string;
  controls?: WireControlSpec[];
  math_sections?: LatexSection[];
  notes?: string[];
}

interface ProblemSpec {
  wire_id: string;
  id: ProblemIdCode;
  name: string;
  description: string;
  controls: ControlSpec[];
  math_sections: LatexSection[];
  notes: string[];
}

interface WireCompileCacheStatus {
  problem_id: string | number;
  problem_name: string;
  variant_id: string;
  variant_label: string;
  state: string | number;
  symbolic_build_s?: number | null;
  symbolic_derivatives_s?: number | null;
  symbolic_setup_s?: number | null;
  jit_s?: number | null;
  jit_disk_cache_hit?: boolean;
  phase_details?: SolverPhaseDetails;
  compile_report?: WireCompileReportSummary | null;
}

interface WireCompileCacheSnapshot {
  entries?: WireCompileCacheStatus[];
}

interface WireCompileKernelSummary {
  name: string;
  lowering_s?: number | null;
  llvm_cache_key_s?: number | null;
  llvm_s?: number | null;
  llvm_cache_hit?: boolean;
  llvm_module_build_s?: number | null;
  llvm_optimization_s?: number | null;
  llvm_object_emit_s?: number | null;
  llvm_ir_fingerprint_s?: number | null;
  context_s?: number | null;
  llvm_cache_check_s?: number | null;
  llvm_cache_read_s?: number | null;
  llvm_cache_write_s?: number | null;
  llvm_cache_load_s?: number | null;
  llvm_cache_materialize_s?: number | null;
  object_size_bytes?: number | null;
  llvm_root_instructions_emitted: number;
  llvm_total_instructions_emitted: number;
  llvm_subfunctions_emitted: number;
  llvm_call_instructions_emitted: number;
}

interface WireCompileReportSummary {
  symbolic_construction_s?: number | null;
  objective_gradient_s?: number | null;
  equality_jacobian_s?: number | null;
  inequality_jacobian_s?: number | null;
  lagrangian_assembly_s?: number | null;
  hessian_generation_s?: number | null;
  lowering_s?: number | null;
  llvm_cache_key_s?: number | null;
  llvm_jit_s?: number | null;
  llvm_module_build_s?: number | null;
  llvm_optimization_s?: number | null;
  llvm_object_emit_s?: number | null;
  llvm_ir_fingerprint_s?: number | null;
  jit_context_s?: number | null;
  llvm_cache_check_s?: number | null;
  llvm_cache_read_s?: number | null;
  llvm_cache_write_s?: number | null;
  llvm_cache_load_s?: number | null;
  llvm_cache_materialize_s?: number | null;
  llvm_cache_hits: number;
  llvm_cache_misses: number;
  symbolic_function_count: number;
  call_site_count: number;
  max_call_depth: number;
  inlines_at_call: number;
  inlines_at_lowering: number;
  llvm_root_instructions_emitted: number;
  llvm_total_instructions_emitted: number;
  llvm_subfunctions_emitted: number;
  llvm_call_instructions_emitted: number;
  kernels: WireCompileKernelSummary[];
  warnings: string[];
}

interface CompileCacheStatus {
  wire_problem_id: string;
  problem_id: ProblemIdCode;
  problem_name: string;
  variant_id: string;
  variant_label: string;
  state: CompileCacheStateCode;
  symbolic_build_s: number | null;
  symbolic_derivatives_s: number | null;
  symbolic_setup_s: number | null;
  jit_s: number | null;
  jit_disk_cache_hit: boolean;
  phase_details: SolverPhaseDetails;
  compile_report: WireCompileReportSummary | null;
}

interface Metric {
  key: MetricKeyCode;
  label: string;
  value: string;
  numeric_value?: number | null;
}

interface WireMetric extends Omit<Metric, "key"> {
  key: string | number;
}

interface TimeSeries {
  name: string;
  x: number[];
  y: number[];
  mode?: string | null;
  legend_group?: string | null;
  show_legend?: boolean;
  role: TimeSeriesRoleCode;
}

interface WireTimeSeries extends Omit<TimeSeries, "role"> {
  role?: string | number;
}

interface Chart {
  title: string;
  x_label: string;
  y_label: string;
  series: TimeSeries[];
}

interface WireChart extends Omit<Chart, "series"> {
  series?: WireTimeSeries[];
}

interface ScenePath {
  name: string;
  x: number[];
  y: number[];
}

interface SceneCircle {
  cx: number;
  cy: number;
  radius: number;
  label: string;
}

interface SceneArrow {
  x: number;
  y: number;
  dx: number;
  dy: number;
  label: string;
}

interface SceneFrame {
  points: Record<string, NumericPoint>;
  segments: Array<[NumericPoint, NumericPoint]>;
}

interface SceneAnimation {
  times: number[];
  frames: SceneFrame[];
}

interface Scene2D {
  title: string;
  x_label: string;
  y_label: string;
  paths: ScenePath[];
  circles: SceneCircle[];
  arrows: SceneArrow[];
  animation?: SceneAnimation | null;
}

interface ScenePath3D {
  name: string;
  x: number[];
  y: number[];
  z: number[];
}

interface Contour2DVisualization {
  kind: "contour_2d";
  title: string;
  x_label: string;
  y_label: string;
  x: number[];
  y: number[];
  z: number[][];
  paths: ScenePath[];
  circles: SceneCircle[];
}

interface Paths3DVisualization {
  kind: "paths_3d";
  title: string;
  x_label: string;
  y_label: string;
  z_label: string;
  paths: ScenePath3D[];
}

type ArtifactVisualization = Contour2DVisualization | Paths3DVisualization;

interface ConstraintPanelEntry {
  label: string;
  category: ConstraintPanelCategoryCode;
  worst_violation: number;
  violating_instances: number;
  total_instances: number;
  severity: ConstraintPanelSeverityCode;
  lower_bound?: number | null;
  upper_bound?: number | null;
  lower_severity?: ConstraintPanelSeverityCode | null;
  upper_severity?: ConstraintPanelSeverityCode | null;
  bound_side?: ConstraintPanelBoundSideCode | null;
  active_bound_side?: ConstraintPanelBoundSideCode | null;
  active_instances?: number | null;
  lower_active_instances?: number | null;
  upper_active_instances?: number | null;
  min_active_margin?: number | null;
  min_lower_margin?: number | null;
  min_upper_margin?: number | null;
}

interface WireConstraintPanelEntry extends Omit<
  ConstraintPanelEntry,
  | "category"
  | "severity"
  | "lower_severity"
  | "upper_severity"
  | "bound_side"
  | "active_bound_side"
> {
  category: string | number;
  severity: string | number;
  lower_severity?: string | number | null;
  upper_severity?: string | number | null;
  bound_side?: string | number | null;
  active_bound_side?: string | number | null;
}

interface ConstraintPanels {
  equalities: ConstraintPanelEntry[];
  inequalities: ConstraintPanelEntry[];
}

interface WireConstraintPanels {
  equalities?: WireConstraintPanelEntry[];
  inequalities?: WireConstraintPanelEntry[];
}

type FilterAcceptanceModeCode = EnumValue<typeof FILTER_ACCEPTANCE_MODE>;

interface FilterEntry {
  objective: number;
  violation: number;
}

interface FilterInfo {
  current: FilterEntry;
  entries: FilterEntry[];
  objective_label: string;
  title: string;
  accepted_mode?: FilterAcceptanceModeCode | null;
}

interface WireFilterInfo extends Omit<FilterInfo, "accepted_mode"> {
  accepted_mode?: string | number | null;
}

interface SolveTrustRegionInfo {
  radius: number;
  attempted_radius: number;
  step_norm: number;
  largest_attempted_step_norm?: number | null;
  contraction_count: number;
  qp_failure_retries: number;
  boundary_active: boolean;
  restoration_attempted: boolean;
  elastic_recovery_attempted: boolean;
}

interface WireSolveTrustRegionInfo extends Omit<SolveTrustRegionInfo, never> {}

interface SolveProgress {
  iteration: number;
  phase: SolvePhaseCode;
  objective: number;
  eq_inf?: number | null;
  ineq_inf?: number | null;
  dual_inf: number;
  step_inf?: number | null;
  penalty: number;
  alpha?: number | null;
  alpha_pr?: number | null;
  alpha_du?: number | null;
  line_search_iterations?: number | null;
  filter?: FilterInfo | null;
  trust_region?: SolveTrustRegionInfo | null;
}

interface WireSolveProgress extends Omit<SolveProgress, "phase" | "filter" | "trust_region"> {
  phase: string | number;
  filter?: WireFilterInfo | null;
  trust_region?: WireSolveTrustRegionInfo | null;
}

interface SolverPhaseDetail {
  label: string;
  value: string;
  count: number;
}

interface SolverPhaseDetails {
  symbolic_setup: SolverPhaseDetail[];
  jit: SolverPhaseDetail[];
  solve: SolverPhaseDetail[];
}

interface SolverReport {
  completed: boolean;
  status_label: string;
  status_kind: SolverStatusKindCode;
  iterations?: number | null;
  symbolic_build_s?: number | null;
  symbolic_derivatives_s?: number | null;
  symbolic_setup_s?: number | null;
  jit_s?: number | null;
  solve_s?: number | null;
  compile_cached: boolean;
  jit_disk_cache_hit: boolean;
  phase_details: SolverPhaseDetails;
  failure_message?: string;
}


interface WireSolverReport extends Omit<SolverReport, "status_kind"> {
  status_kind: string | number;
}

interface SolveArtifact {
  title: string;
  summary: Metric[];
  solver: SolverReport;
  compile_report: WireCompileReportSummary | null;
  constraint_panels: ConstraintPanels;
  charts: Chart[];
  scene: Scene2D;
  visualizations: ArtifactVisualization[];
  notes: string[];
}

interface WireSolveArtifact extends Omit<
  SolveArtifact,
  "summary" | "solver" | "compile_report" | "constraint_panels" | "charts" | "visualizations"
> {
  summary?: WireMetric[];
  solver: WireSolverReport;
  compile_report?: WireCompileReportSummary | null;
  constraint_panels?: WireConstraintPanels | null;
  charts?: WireChart[];
  visualizations?: ArtifactVisualization[];
}

interface SolveStatus {
  stage: SolveStageCode;
  solver_method?: SolverMethodCode | null;
  solver: SolverReport;
  compile_report?: WireCompileReportSummary | null;
}

interface StatusSolveEvent {
  kind: typeof STREAM_EVENT_KIND.status;
  status: SolveStatus;
}

interface LogSolveEvent {
  kind: typeof STREAM_EVENT_KIND.log;
  line: string;
  level: LogLevelCode;
}

interface IterationSolveEvent {
  kind: typeof STREAM_EVENT_KIND.iteration;
  progress: SolveProgress;
  artifact: SolveArtifact;
}

interface FinalSolveEvent {
  kind: typeof STREAM_EVENT_KIND.final;
  artifact: SolveArtifact;
}

interface ErrorSolveEvent {
  kind: typeof STREAM_EVENT_KIND.error;
  message: string;
}

type SolveEvent =
  | StatusSolveEvent
  | LogSolveEvent
  | IterationSolveEvent
  | FinalSolveEvent
  | ErrorSolveEvent;

interface WireStatusSolveEvent {
  kind: "status";
  status: WireSolveStatus;
}

interface WireSolveStatus extends Omit<SolveStatus, "stage" | "solver_method" | "solver"> {
  stage: string | number;
  solver_method?: string | number | null;
  solver: WireSolverReport;
  compile_report?: WireCompileReportSummary | null;
}

interface WireLogSolveEvent {
  kind: "log";
  line: string;
  level: string | number;
}

interface WireIterationSolveEvent {
  kind: "iteration";
  progress: WireSolveProgress;
  artifact: WireSolveArtifact;
}

interface WireFinalSolveEvent {
  kind: "final";
  artifact: WireSolveArtifact;
}

interface WireErrorSolveEvent {
  kind: "error";
  message: string;
}

type WireSolveEvent =
  | WireStatusSolveEvent
  | WireLogSolveEvent
  | WireIterationSolveEvent
  | WireFinalSolveEvent
  | WireErrorSolveEvent;

interface LogLine {
  text: string;
  level: LogLevelCode;
}

interface ChartView {
  plotEl: PlotlyHostElement;
  linkedRangeBound: boolean;
  linkXRange: boolean;
  chartTraceSignature: string | null;
}

interface PlotlyView {
  plotEl: PlotlyHostElement;
  sceneCamera?: PlotlyObject | null;
  sceneInteractionBound?: boolean;
  sceneInteracting?: boolean;
  scenePointerActive?: boolean;
  sceneIdleFrameHandle?: number | null;
  sceneTraceSignature?: string | null;
}

interface ChartPanelChart {
  kind: "chart";
  key: string;
  title: string;
  subtitle: string;
  chart: Chart;
}

interface ChartPanelVisualization {
  kind: "visualization";
  key: string;
  title: string;
  subtitle: string;
  visualization: ArtifactVisualization;
}

type ChartPanel = ChartPanelChart | ChartPanelVisualization;

interface Scene2DView {
  kind: "scene_2d";
  scene: Scene2D;
  shell: HTMLDivElement;
  meta: HTMLDivElement;
  playButton: HTMLButtonElement | null;
  slider: HTMLInputElement | null;
  plotEl: PlotlyHostElement;
}

interface Scene3DView {
  kind: "paths_3d";
  visualization: Paths3DVisualization;
  shell: HTMLDivElement;
  meta: HTMLDivElement;
  playButton: null;
  slider: null;
  plotEl: PlotlyHostElement;
  sceneCamera: PlotlyObject | null;
  sceneInteractionBound: boolean;
  sceneInteracting: boolean;
  scenePointerActive: boolean;
  sceneIdleFrameHandle: number | null;
  sceneTraceSignature: string | null;
}

type SceneView = Scene2DView | Scene3DView;

interface ControlSectionView {
  key: ControlSectionCode;
  title: string;
  subtitle: string;
  controls: ControlSpec[];
}

type ControlSectionCollapseState = Record<ControlSectionCode, boolean>;
type ControlPanelCollapseState = Record<ControlPanelCode, boolean>;
type ControlBlockCollapseState = Record<string, boolean>;

interface ControlPanelView {
  key: ControlPanelCode;
  title: string;
  subtitle: string;
  controls: ControlSpec[];
}

interface ControlBlockView {
  key: string;
  title: string;
  subtitle: string;
  defaultCollapsed: boolean;
  appendBody: (body: HTMLElement) => void;
}

interface FrontendState {
  specs: ProblemSpec[];
  selectedId: ProblemIdCode | null;
  values: Record<string, number>;
  compileCacheStatuses: CompileCacheStatus[];
  collapsedControlSections: ControlSectionCollapseState;
  collapsedControlPanels: ControlPanelCollapseState;
  collapsedControlBlocks: ControlBlockCollapseState;
  artifact: SolveArtifact | null;
  animationIndex: number;
  playing: boolean;
  playHandle: number | null;
  solving: boolean;
  renderScheduled: boolean;
  chartViews: Map<string, ChartView>;
  chartLayoutKey: string;
  progressPlotReady: boolean;
  filterPlotReady: boolean;
  trustRegionPlotReady: boolean;
  filterRecentPath: FilterEntry[];
  lastFilterPointKey: string | null;
  logLines: LogLine[];
  followSolverLog: boolean;
  latestProgress: SolveProgress | null;
  liveStatus: SolveStatus | null;
  liveSolver: SolverReport | null;
  terminalSolver: SolverReport | null;
  solveAbortController: AbortController | null;
  solveStopRequested: boolean;
  pendingIterationEvent: IterationSolveEvent | null;
  iterationFlushScheduled: boolean;
  sceneView: SceneView | null;
  linkedChartRange: NumericRange | null;
  linkedChartAutorange: boolean;
  linkingChartRange: boolean;
  artifactRenderFrameHandle: number | null;
  prewarmTimer: number | null;
  prewarmInFlightCount: number;
  compileStatusPollHandle: number | null;
}

type PlotlyTrace = PlotlyObject;
type PlotlyLayout = PlotlyObject;
type PlotlyConfig = PlotlyObject;
type ConstraintPanelKind = "eq" | "ineq" | "active";
type AnsiColor = "red" | "green" | "yellow" | "cyan";

interface ChartTraceBuild {
  data: PlotlyTrace[];
  signature: string;
}

interface AnsiState {
  bold: boolean;
  color: AnsiColor | null;
}

interface SceneBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

interface SolveSummaryItem {
  label: string;
  value: string;
}

type StatusClassName = "" | "success" | "warning" | "error" | "info";

interface StatusDisplay {
  eyebrow: string;
  title: string;
  detail: string;
  kind: StatusClassName;
  active: boolean;
}

const pendingMathRoots = new Set<Element>();
let mathTypesetRetryHandle: number | null = null;
let copyConsoleFeedbackHandle: number | null = null;
const PREWARM_DELAY_MS = 200;
const COMPILE_STATUS_POLL_MS = 250;
const COPY_CONSOLE_DEFAULT_LABEL = "Copy to Clipboard";
const COPY_CONSOLE_FEEDBACK_MS = 1400;

function requiredElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Missing required DOM element ${selector}`);
  }
  return element;
}

function requiredChild<T extends Element>(parent: ParentNode, selector: string): T {
  const element = parent.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Missing required child ${selector}`);
  }
  return element;
}

function parseJsonValue(text: string, context: string): JsonValue {
  try {
    return JSON.parse(text);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`${context} returned invalid JSON: ${message}`);
  }
}

function readJsonObject(value: JsonValue | undefined, context: string): JsonObject {
  if (value == null || Array.isArray(value) || typeof value !== "object") {
    throw new Error(`${context} must be an object.`);
  }
  return value;
}

function readJsonArray(value: JsonValue | undefined, context: string): JsonArray {
  if (!Array.isArray(value)) {
    throw new Error(`${context} must be an array.`);
  }
  return value;
}

function readJsonString(value: JsonValue | undefined, context: string): string {
  if (typeof value !== "string") {
    throw new Error(`${context} must be a string.`);
  }
  return value;
}

function readOptionalJsonString(value: JsonValue | undefined, context: string): string | undefined {
  if (value == null) {
    return undefined;
  }
  return readJsonString(value, context);
}

function readJsonNumber(value: JsonValue | undefined, context: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${context} must be a finite number.`);
  }
  return value;
}

function readOptionalJsonNumber(value: JsonValue | undefined, context: string): number | undefined {
  if (value == null) {
    return undefined;
  }
  return readJsonNumber(value, context);
}

function readOptionalJsonBoolean(value: JsonValue | undefined, context: string): boolean | undefined {
  if (value == null) {
    return undefined;
  }
  if (typeof value !== "boolean") {
    throw new Error(`${context} must be a boolean.`);
  }
  return value;
}

function readJsonStringOrNumber(value: JsonValue | undefined, context: string): string | number {
  if (typeof value === "string" || typeof value === "number") {
    return value;
  }
  throw new Error(`${context} must be a string or number.`);
}

function readOptionalJsonStringOrNumber(
  value: JsonValue | undefined,
  context: string,
): string | number | undefined {
  if (value == null) {
    return undefined;
  }
  return readJsonStringOrNumber(value, context);
}

function readJsonStringArray(value: JsonValue | undefined, context: string): string[] {
  return readJsonArray(value, context).map((item, index) =>
    readJsonString(item, `${context}[${index}]`));
}

function readJsonNumberArray(value: JsonValue | undefined, context: string): number[] {
  return readJsonArray(value, context).map((item, index) =>
    readJsonNumber(item, `${context}[${index}]`));
}

function readJsonNumberMatrix(value: JsonValue | undefined, context: string): number[][] {
  return readJsonArray(value, context).map((row, index) =>
    readJsonNumberArray(row, `${context}[${index}]`));
}

function readOptionalJsonArray(value: JsonValue | undefined, context: string): JsonArray | undefined {
  if (value == null) {
    return undefined;
  }
  return readJsonArray(value, context);
}

function readJsonValueAt(object: JsonObject, key: string): JsonValue | undefined {
  return object[key];
}

function readOptionalErrorMessage(value: JsonValue | null): string | null {
  if (value == null) {
    return null;
  }
  const object = readJsonObject(value, "error response");
  const error = readJsonValueAt(object, "error");
  return typeof error === "string" ? error : null;
}

function createPlotlyHostElement(className: string): PlotlyHostElement {
  const element: PlotlyHostElement = document.createElement("div");
  element.className = className;
  return element;
}

function readCurrentInputTarget(event: Event, context: string): HTMLInputElement {
  const target = event.currentTarget;
  if (!(target instanceof HTMLInputElement)) {
    throw new Error(`${context} target must be an input element.`);
  }
  return target;
}

function readCurrentSelectTarget(event: Event, context: string): HTMLSelectElement {
  const target = event.currentTarget;
  if (!(target instanceof HTMLSelectElement)) {
    throw new Error(`${context} target must be a select element.`);
  }
  return target;
}

function readControlChoice(value: JsonValue | undefined, context: string): ControlChoice {
  const object = readJsonObject(value, context);
  return {
    value: readJsonNumber(readJsonValueAt(object, "value"), `${context}.value`),
    label: readJsonString(readJsonValueAt(object, "label"), `${context}.label`),
  };
}

function readControlProfileDefault(
  value: JsonValue | undefined,
  context: string,
): ControlProfileDefault {
  const object = readJsonObject(value, context);
  return {
    profile: readJsonNumber(readJsonValueAt(object, "profile"), `${context}.profile`),
    value: readJsonNumber(readJsonValueAt(object, "value"), `${context}.value`),
  };
}

function readWireControlSpec(value: JsonValue | undefined, context: string): WireControlSpec {
  const object = readJsonObject(value, context);
  const choicesValue = readOptionalJsonArray(readJsonValueAt(object, "choices"), `${context}.choices`);
  const profileDefaultsValue = readOptionalJsonArray(
    readJsonValueAt(object, "profile_defaults"),
    `${context}.profile_defaults`,
  );
  return {
    id: readJsonString(readJsonValueAt(object, "id"), `${context}.id`),
    label: readJsonString(readJsonValueAt(object, "label"), `${context}.label`),
    min: readJsonNumber(readJsonValueAt(object, "min"), `${context}.min`),
    max: readJsonNumber(readJsonValueAt(object, "max"), `${context}.max`),
    step: readJsonNumber(readJsonValueAt(object, "step"), `${context}.step`),
    default: readJsonNumber(readJsonValueAt(object, "default"), `${context}.default`),
    unit: readJsonString(readJsonValueAt(object, "unit"), `${context}.unit`),
    help: readJsonString(readJsonValueAt(object, "help"), `${context}.help`),
    section: readOptionalJsonStringOrNumber(readJsonValueAt(object, "section"), `${context}.section`),
    panel: readOptionalJsonStringOrNumber(readJsonValueAt(object, "panel"), `${context}.panel`),
    editor: readOptionalJsonStringOrNumber(readJsonValueAt(object, "editor"), `${context}.editor`),
    visibility: readOptionalJsonStringOrNumber(
      readJsonValueAt(object, "visibility"),
      `${context}.visibility`,
    ),
    semantic: readOptionalJsonStringOrNumber(
      readJsonValueAt(object, "semantic"),
      `${context}.semantic`,
    ),
    value_display: readOptionalJsonStringOrNumber(
      readJsonValueAt(object, "value_display"),
      `${context}.value_display`,
    ),
    choices: choicesValue?.map((choice, index) =>
      readControlChoice(choice, `${context}.choices[${index}]`)),
    profile_defaults: profileDefaultsValue?.map((profileDefault, index) =>
      readControlProfileDefault(profileDefault, `${context}.profile_defaults[${index}]`)),
  };
}

function readLatexSection(value: JsonValue | undefined, context: string): LatexSection {
  const object = readJsonObject(value, context);
  return {
    title: readJsonString(readJsonValueAt(object, "title"), `${context}.title`),
    entries: readJsonStringArray(readJsonValueAt(object, "entries"), `${context}.entries`),
  };
}

function readWireProblemSpec(value: JsonValue | undefined, context: string): WireProblemSpec {
  const object = readJsonObject(value, context);
  const controlsValue = readOptionalJsonArray(readJsonValueAt(object, "controls"), `${context}.controls`);
  const mathSectionsValue = readOptionalJsonArray(
    readJsonValueAt(object, "math_sections"),
    `${context}.math_sections`,
  );
  return {
    id: readJsonStringOrNumber(readJsonValueAt(object, "id"), `${context}.id`),
    name: readJsonString(readJsonValueAt(object, "name"), `${context}.name`),
    description: readJsonString(readJsonValueAt(object, "description"), `${context}.description`),
    controls: controlsValue?.map((control, index) =>
      readWireControlSpec(control, `${context}.controls[${index}]`)),
    math_sections: mathSectionsValue?.map((section, index) =>
      readLatexSection(section, `${context}.math_sections[${index}]`)),
    notes: readOptionalJsonArray(readJsonValueAt(object, "notes"), `${context}.notes`)?.map((note, index) =>
      readJsonString(note, `${context}.notes[${index}]`)),
  };
}

function readWireCompileCacheStatus(
  value: JsonValue | undefined,
  context: string,
): WireCompileCacheStatus {
  const object = readJsonObject(value, context);
  const compileReportValue = readJsonValueAt(object, "compile_report");
  return {
    problem_id: readJsonStringOrNumber(readJsonValueAt(object, "problem_id"), `${context}.problem_id`),
    problem_name: readJsonString(readJsonValueAt(object, "problem_name"), `${context}.problem_name`),
    variant_id: readJsonString(readJsonValueAt(object, "variant_id"), `${context}.variant_id`),
    variant_label: readJsonString(readJsonValueAt(object, "variant_label"), `${context}.variant_label`),
    state: readJsonStringOrNumber(readJsonValueAt(object, "state"), `${context}.state`),
    symbolic_build_s:
      readOptionalJsonNumber(readJsonValueAt(object, "symbolic_build_s"), `${context}.symbolic_build_s`) ??
      null,
    symbolic_derivatives_s:
      readOptionalJsonNumber(
        readJsonValueAt(object, "symbolic_derivatives_s"),
        `${context}.symbolic_derivatives_s`,
      ) ?? null,
    symbolic_setup_s:
      readOptionalJsonNumber(readJsonValueAt(object, "symbolic_setup_s"), `${context}.symbolic_setup_s`) ??
      null,
    jit_s: readOptionalJsonNumber(readJsonValueAt(object, "jit_s"), `${context}.jit_s`) ?? null,
    jit_disk_cache_hit:
      readOptionalJsonBoolean(
        readJsonValueAt(object, "jit_disk_cache_hit"),
        `${context}.jit_disk_cache_hit`,
      ) ?? false,
    phase_details:
      readSolverPhaseDetails(readJsonValueAt(object, "phase_details"), `${context}.phase_details`),
    compile_report:
      compileReportValue == null
        ? null
        : readWireCompileReportSummary(compileReportValue, `${context}.compile_report`),
  };
}

function readWireCompileKernelSummary(
  value: JsonValue | undefined,
  context: string,
): WireCompileKernelSummary {
  const object = readJsonObject(value, context);
  return {
    name: readJsonString(readJsonValueAt(object, "name"), `${context}.name`),
    lowering_s: readOptionalJsonNumber(readJsonValueAt(object, "lowering_s"), `${context}.lowering_s`) ?? null,
    llvm_cache_key_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_cache_key_s"), `${context}.llvm_cache_key_s`) ??
      null,
    llvm_s: readOptionalJsonNumber(readJsonValueAt(object, "llvm_s"), `${context}.llvm_s`) ?? null,
    llvm_cache_hit:
      readOptionalJsonBoolean(readJsonValueAt(object, "llvm_cache_hit"), `${context}.llvm_cache_hit`) ??
      false,
    llvm_module_build_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_module_build_s"), `${context}.llvm_module_build_s`) ??
      null,
    llvm_optimization_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_optimization_s"), `${context}.llvm_optimization_s`) ??
      null,
    llvm_object_emit_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_object_emit_s"), `${context}.llvm_object_emit_s`) ??
      null,
    llvm_ir_fingerprint_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_ir_fingerprint_s"), `${context}.llvm_ir_fingerprint_s`) ??
      null,
    context_s: readOptionalJsonNumber(readJsonValueAt(object, "context_s"), `${context}.context_s`) ?? null,
    llvm_cache_check_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_cache_check_s"), `${context}.llvm_cache_check_s`) ??
      null,
    llvm_cache_read_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_cache_read_s"), `${context}.llvm_cache_read_s`) ??
      null,
    llvm_cache_write_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_cache_write_s"), `${context}.llvm_cache_write_s`) ??
      null,
    llvm_cache_load_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_cache_load_s"), `${context}.llvm_cache_load_s`) ??
      null,
    llvm_cache_materialize_s:
      readOptionalJsonNumber(
        readJsonValueAt(object, "llvm_cache_materialize_s"),
        `${context}.llvm_cache_materialize_s`,
      ) ?? null,
    object_size_bytes:
      readOptionalJsonNumber(readJsonValueAt(object, "object_size_bytes"), `${context}.object_size_bytes`) ??
      null,
    llvm_root_instructions_emitted: readJsonNumber(
      readJsonValueAt(object, "llvm_root_instructions_emitted"),
      `${context}.llvm_root_instructions_emitted`,
    ),
    llvm_total_instructions_emitted: readJsonNumber(
      readJsonValueAt(object, "llvm_total_instructions_emitted"),
      `${context}.llvm_total_instructions_emitted`,
    ),
    llvm_subfunctions_emitted: readJsonNumber(
      readJsonValueAt(object, "llvm_subfunctions_emitted"),
      `${context}.llvm_subfunctions_emitted`,
    ),
    llvm_call_instructions_emitted: readJsonNumber(
      readJsonValueAt(object, "llvm_call_instructions_emitted"),
      `${context}.llvm_call_instructions_emitted`,
    ),
  };
}

function readWireCompileReportSummary(
  value: JsonValue | undefined,
  context: string,
): WireCompileReportSummary {
  const object = readJsonObject(value, context);
  const kernels = readOptionalJsonArray(readJsonValueAt(object, "kernels"), `${context}.kernels`);
  const warnings = readOptionalJsonArray(readJsonValueAt(object, "warnings"), `${context}.warnings`);
  return {
    symbolic_construction_s:
      readOptionalJsonNumber(
        readJsonValueAt(object, "symbolic_construction_s"),
        `${context}.symbolic_construction_s`,
      ) ?? null,
    objective_gradient_s:
      readOptionalJsonNumber(readJsonValueAt(object, "objective_gradient_s"), `${context}.objective_gradient_s`) ??
      null,
    equality_jacobian_s:
      readOptionalJsonNumber(readJsonValueAt(object, "equality_jacobian_s"), `${context}.equality_jacobian_s`) ??
      null,
    inequality_jacobian_s:
      readOptionalJsonNumber(
        readJsonValueAt(object, "inequality_jacobian_s"),
        `${context}.inequality_jacobian_s`,
      ) ?? null,
    lagrangian_assembly_s:
      readOptionalJsonNumber(
        readJsonValueAt(object, "lagrangian_assembly_s"),
        `${context}.lagrangian_assembly_s`,
      ) ?? null,
    hessian_generation_s:
      readOptionalJsonNumber(
        readJsonValueAt(object, "hessian_generation_s"),
        `${context}.hessian_generation_s`,
      ) ?? null,
    lowering_s: readOptionalJsonNumber(readJsonValueAt(object, "lowering_s"), `${context}.lowering_s`) ?? null,
    llvm_cache_key_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_cache_key_s"), `${context}.llvm_cache_key_s`) ??
      null,
    llvm_jit_s: readOptionalJsonNumber(readJsonValueAt(object, "llvm_jit_s"), `${context}.llvm_jit_s`) ?? null,
    llvm_module_build_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_module_build_s"), `${context}.llvm_module_build_s`) ??
      null,
    llvm_optimization_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_optimization_s"), `${context}.llvm_optimization_s`) ??
      null,
    llvm_object_emit_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_object_emit_s"), `${context}.llvm_object_emit_s`) ??
      null,
    llvm_ir_fingerprint_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_ir_fingerprint_s"), `${context}.llvm_ir_fingerprint_s`) ??
      null,
    jit_context_s:
      readOptionalJsonNumber(readJsonValueAt(object, "jit_context_s"), `${context}.jit_context_s`) ??
      null,
    llvm_cache_check_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_cache_check_s"), `${context}.llvm_cache_check_s`) ??
      null,
    llvm_cache_read_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_cache_read_s"), `${context}.llvm_cache_read_s`) ??
      null,
    llvm_cache_write_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_cache_write_s"), `${context}.llvm_cache_write_s`) ??
      null,
    llvm_cache_load_s:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_cache_load_s"), `${context}.llvm_cache_load_s`) ??
      null,
    llvm_cache_materialize_s:
      readOptionalJsonNumber(
        readJsonValueAt(object, "llvm_cache_materialize_s"),
        `${context}.llvm_cache_materialize_s`,
      ) ?? null,
    llvm_cache_hits: readOptionalJsonNumber(readJsonValueAt(object, "llvm_cache_hits"), `${context}.llvm_cache_hits`) ?? 0,
    llvm_cache_misses:
      readOptionalJsonNumber(readJsonValueAt(object, "llvm_cache_misses"), `${context}.llvm_cache_misses`) ?? 0,
    symbolic_function_count:
      readOptionalJsonNumber(readJsonValueAt(object, "symbolic_function_count"), `${context}.symbolic_function_count`) ??
      0,
    call_site_count: readOptionalJsonNumber(readJsonValueAt(object, "call_site_count"), `${context}.call_site_count`) ?? 0,
    max_call_depth: readOptionalJsonNumber(readJsonValueAt(object, "max_call_depth"), `${context}.max_call_depth`) ?? 0,
    inlines_at_call: readOptionalJsonNumber(readJsonValueAt(object, "inlines_at_call"), `${context}.inlines_at_call`) ?? 0,
    inlines_at_lowering:
      readOptionalJsonNumber(readJsonValueAt(object, "inlines_at_lowering"), `${context}.inlines_at_lowering`) ?? 0,
    llvm_root_instructions_emitted:
      readOptionalJsonNumber(
        readJsonValueAt(object, "llvm_root_instructions_emitted"),
        `${context}.llvm_root_instructions_emitted`,
      ) ?? 0,
    llvm_total_instructions_emitted:
      readOptionalJsonNumber(
        readJsonValueAt(object, "llvm_total_instructions_emitted"),
        `${context}.llvm_total_instructions_emitted`,
      ) ?? 0,
    llvm_subfunctions_emitted:
      readOptionalJsonNumber(
        readJsonValueAt(object, "llvm_subfunctions_emitted"),
        `${context}.llvm_subfunctions_emitted`,
      ) ?? 0,
    llvm_call_instructions_emitted:
      readOptionalJsonNumber(
        readJsonValueAt(object, "llvm_call_instructions_emitted"),
        `${context}.llvm_call_instructions_emitted`,
      ) ?? 0,
    kernels: kernels?.map((kernel, index) =>
      readWireCompileKernelSummary(kernel, `${context}.kernels[${index}]`)) ?? [],
    warnings: warnings?.map((warning, index) =>
      readJsonString(warning, `${context}.warnings[${index}]`)) ?? [],
  };
}

function readWireCompileCacheSnapshot(
  value: JsonValue | undefined,
  context: string,
): WireCompileCacheSnapshot {
  const object = readJsonObject(value, context);
  const entriesValue = readOptionalJsonArray(readJsonValueAt(object, "entries"), `${context}.entries`);
  return {
    entries: entriesValue?.map((entry, index) =>
      readWireCompileCacheStatus(entry, `${context}.entries[${index}]`)),
  };
}

function readWireMetric(value: JsonValue | undefined, context: string): WireMetric {
  const object = readJsonObject(value, context);
  const numericValue = readOptionalJsonNumber(
    readJsonValueAt(object, "numeric_value"),
    `${context}.numeric_value`,
  );
  return {
    key: readJsonStringOrNumber(readJsonValueAt(object, "key"), `${context}.key`),
    label: readJsonString(readJsonValueAt(object, "label"), `${context}.label`),
    value: readJsonString(readJsonValueAt(object, "value"), `${context}.value`),
    numeric_value: numericValue ?? null,
  };
}

function readWireTimeSeries(value: JsonValue | undefined, context: string): WireTimeSeries {
  const object = readJsonObject(value, context);
  return {
    name: readJsonString(readJsonValueAt(object, "name"), `${context}.name`),
    x: readJsonNumberArray(readJsonValueAt(object, "x"), `${context}.x`),
    y: readJsonNumberArray(readJsonValueAt(object, "y"), `${context}.y`),
    mode: readOptionalJsonString(readJsonValueAt(object, "mode"), `${context}.mode`) ?? null,
    legend_group:
      readOptionalJsonString(readJsonValueAt(object, "legend_group"), `${context}.legend_group`) ??
      null,
    show_legend: readOptionalJsonBoolean(
      readJsonValueAt(object, "show_legend"),
      `${context}.show_legend`,
    ),
    role: readOptionalJsonStringOrNumber(readJsonValueAt(object, "role"), `${context}.role`),
  };
}

function readWireChart(value: JsonValue | undefined, context: string): WireChart {
  const object = readJsonObject(value, context);
  const seriesValue = readOptionalJsonArray(readJsonValueAt(object, "series"), `${context}.series`);
  return {
    title: readJsonString(readJsonValueAt(object, "title"), `${context}.title`),
    x_label: readJsonString(readJsonValueAt(object, "x_label"), `${context}.x_label`),
    y_label: readJsonString(readJsonValueAt(object, "y_label"), `${context}.y_label`),
    series: seriesValue?.map((series, index) => readWireTimeSeries(series, `${context}.series[${index}]`)),
  };
}

function readNumericPoint(value: JsonValue | undefined, context: string): NumericPoint {
  const point = readJsonArray(value, context);
  if (point.length !== 2) {
    throw new Error(`${context} must have exactly two coordinates.`);
  }
  return [
    readJsonNumber(point[0], `${context}[0]`),
    readJsonNumber(point[1], `${context}[1]`),
  ];
}

function readScenePath(value: JsonValue | undefined, context: string): ScenePath {
  const object = readJsonObject(value, context);
  return {
    name: readJsonString(readJsonValueAt(object, "name"), `${context}.name`),
    x: readJsonNumberArray(readJsonValueAt(object, "x"), `${context}.x`),
    y: readJsonNumberArray(readJsonValueAt(object, "y"), `${context}.y`),
  };
}

function readScenePath3D(value: JsonValue | undefined, context: string): ScenePath3D {
  const object = readJsonObject(value, context);
  return {
    name: readJsonString(readJsonValueAt(object, "name"), `${context}.name`),
    x: readJsonNumberArray(readJsonValueAt(object, "x"), `${context}.x`),
    y: readJsonNumberArray(readJsonValueAt(object, "y"), `${context}.y`),
    z: readJsonNumberArray(readJsonValueAt(object, "z"), `${context}.z`),
  };
}

function readSceneCircle(value: JsonValue | undefined, context: string): SceneCircle {
  const object = readJsonObject(value, context);
  return {
    cx: readJsonNumber(readJsonValueAt(object, "cx"), `${context}.cx`),
    cy: readJsonNumber(readJsonValueAt(object, "cy"), `${context}.cy`),
    radius: readJsonNumber(readJsonValueAt(object, "radius"), `${context}.radius`),
    label: readJsonString(readJsonValueAt(object, "label"), `${context}.label`),
  };
}

function readSceneArrow(value: JsonValue | undefined, context: string): SceneArrow {
  const object = readJsonObject(value, context);
  return {
    x: readJsonNumber(readJsonValueAt(object, "x"), `${context}.x`),
    y: readJsonNumber(readJsonValueAt(object, "y"), `${context}.y`),
    dx: readJsonNumber(readJsonValueAt(object, "dx"), `${context}.dx`),
    dy: readJsonNumber(readJsonValueAt(object, "dy"), `${context}.dy`),
    label: readJsonString(readJsonValueAt(object, "label"), `${context}.label`),
  };
}

function readSceneFrame(value: JsonValue | undefined, context: string): SceneFrame {
  const object = readJsonObject(value, context);
  const pointsObject = readJsonObject(readJsonValueAt(object, "points"), `${context}.points`);
  const segmentsValue = readJsonArray(readJsonValueAt(object, "segments"), `${context}.segments`);
  const points: Record<string, NumericPoint> = {};
  for (const [label, pointValue] of Object.entries(pointsObject)) {
    points[label] = readNumericPoint(pointValue, `${context}.points.${label}`);
  }
  const segments = segmentsValue.map((segmentValue, index) => {
    const segment = readJsonArray(segmentValue, `${context}.segments[${index}]`);
    if (segment.length !== 2) {
      throw new Error(`${context}.segments[${index}] must have exactly two endpoints.`);
    }
    return [
      readNumericPoint(segment[0], `${context}.segments[${index}][0]`),
      readNumericPoint(segment[1], `${context}.segments[${index}][1]`),
    ] as [NumericPoint, NumericPoint];
  });
  return { points, segments };
}

function readSceneAnimation(value: JsonValue | undefined, context: string): SceneAnimation {
  const object = readJsonObject(value, context);
  const framesValue = readJsonArray(readJsonValueAt(object, "frames"), `${context}.frames`);
  return {
    times: readJsonNumberArray(readJsonValueAt(object, "times"), `${context}.times`),
    frames: framesValue.map((frame, index) => readSceneFrame(frame, `${context}.frames[${index}]`)),
  };
}

function readScene2D(value: JsonValue | undefined, context: string): Scene2D {
  const object = readJsonObject(value, context);
  const pathsValue = readJsonArray(readJsonValueAt(object, "paths"), `${context}.paths`);
  const circlesValue = readJsonArray(readJsonValueAt(object, "circles"), `${context}.circles`);
  const arrowsValue = readJsonArray(readJsonValueAt(object, "arrows"), `${context}.arrows`);
  const animationValue = readJsonValueAt(object, "animation");
  return {
    title: readJsonString(readJsonValueAt(object, "title"), `${context}.title`),
    x_label: readJsonString(readJsonValueAt(object, "x_label"), `${context}.x_label`),
    y_label: readJsonString(readJsonValueAt(object, "y_label"), `${context}.y_label`),
    paths: pathsValue.map((path, index) => readScenePath(path, `${context}.paths[${index}]`)),
    circles: circlesValue.map((circle, index) =>
      readSceneCircle(circle, `${context}.circles[${index}]`)),
    arrows: arrowsValue.map((arrow, index) => readSceneArrow(arrow, `${context}.arrows[${index}]`)),
    animation:
      animationValue == null ? null : readSceneAnimation(animationValue, `${context}.animation`),
  };
}

function readArtifactVisualization(
  value: JsonValue | undefined,
  context: string,
): ArtifactVisualization {
  const object = readJsonObject(value, context);
  const kind = readJsonString(readJsonValueAt(object, "kind"), `${context}.kind`);
  if (kind === "contour_2d") {
    const pathsValue = readOptionalJsonArray(readJsonValueAt(object, "paths"), `${context}.paths`);
    const circlesValue = readOptionalJsonArray(
      readJsonValueAt(object, "circles"),
      `${context}.circles`,
    );
    return {
      kind,
      title: readJsonString(readJsonValueAt(object, "title"), `${context}.title`),
      x_label: readJsonString(readJsonValueAt(object, "x_label"), `${context}.x_label`),
      y_label: readJsonString(readJsonValueAt(object, "y_label"), `${context}.y_label`),
      x: readJsonNumberArray(readJsonValueAt(object, "x"), `${context}.x`),
      y: readJsonNumberArray(readJsonValueAt(object, "y"), `${context}.y`),
      z: readJsonNumberMatrix(readJsonValueAt(object, "z"), `${context}.z`),
      paths: (pathsValue ?? []).map((path, index) => readScenePath(path, `${context}.paths[${index}]`)),
      circles: (circlesValue ?? []).map((circle, index) =>
        readSceneCircle(circle, `${context}.circles[${index}]`)),
    };
  }
  if (kind === "paths_3d") {
    const pathsValue = readOptionalJsonArray(readJsonValueAt(object, "paths"), `${context}.paths`);
    return {
      kind,
      title: readJsonString(readJsonValueAt(object, "title"), `${context}.title`),
      x_label: readJsonString(readJsonValueAt(object, "x_label"), `${context}.x_label`),
      y_label: readJsonString(readJsonValueAt(object, "y_label"), `${context}.y_label`),
      z_label: readJsonString(readJsonValueAt(object, "z_label"), `${context}.z_label`),
      paths: (pathsValue ?? []).map((path, index) => readScenePath3D(path, `${context}.paths[${index}]`)),
    };
  }
  throw new Error(`${context}.kind has unsupported visualization kind ${JSON.stringify(kind)}.`);
}

function readWireConstraintPanelEntry(
  value: JsonValue | undefined,
  context: string,
): WireConstraintPanelEntry {
  const object = readJsonObject(value, context);
  return {
    label: readJsonString(readJsonValueAt(object, "label"), `${context}.label`),
    category: readJsonStringOrNumber(readJsonValueAt(object, "category"), `${context}.category`),
    worst_violation: readJsonNumber(
      readJsonValueAt(object, "worst_violation"),
      `${context}.worst_violation`,
    ),
    violating_instances: readJsonNumber(
      readJsonValueAt(object, "violating_instances"),
      `${context}.violating_instances`,
    ),
    total_instances: readJsonNumber(
      readJsonValueAt(object, "total_instances"),
      `${context}.total_instances`,
    ),
    severity: readJsonStringOrNumber(readJsonValueAt(object, "severity"), `${context}.severity`),
    lower_bound: readOptionalJsonNumber(
      readJsonValueAt(object, "lower_bound"),
      `${context}.lower_bound`,
    ),
    upper_bound: readOptionalJsonNumber(
      readJsonValueAt(object, "upper_bound"),
      `${context}.upper_bound`,
    ),
    lower_severity: readOptionalJsonStringOrNumber(
      readJsonValueAt(object, "lower_severity"),
      `${context}.lower_severity`,
    ),
    upper_severity: readOptionalJsonStringOrNumber(
      readJsonValueAt(object, "upper_severity"),
      `${context}.upper_severity`,
    ),
    bound_side: readOptionalJsonStringOrNumber(
      readJsonValueAt(object, "bound_side"),
      `${context}.bound_side`,
    ),
    active_bound_side: readOptionalJsonStringOrNumber(
      readJsonValueAt(object, "active_bound_side"),
      `${context}.active_bound_side`,
    ),
    active_instances: readOptionalJsonNumber(
      readJsonValueAt(object, "active_instances"),
      `${context}.active_instances`,
    ),
    lower_active_instances: readOptionalJsonNumber(
      readJsonValueAt(object, "lower_active_instances"),
      `${context}.lower_active_instances`,
    ),
    upper_active_instances: readOptionalJsonNumber(
      readJsonValueAt(object, "upper_active_instances"),
      `${context}.upper_active_instances`,
    ),
    min_active_margin: readOptionalJsonNumber(
      readJsonValueAt(object, "min_active_margin"),
      `${context}.min_active_margin`,
    ),
    min_lower_margin: readOptionalJsonNumber(
      readJsonValueAt(object, "min_lower_margin"),
      `${context}.min_lower_margin`,
    ),
    min_upper_margin: readOptionalJsonNumber(
      readJsonValueAt(object, "min_upper_margin"),
      `${context}.min_upper_margin`,
    ),
  };
}

function readWireConstraintPanels(value: JsonValue | undefined, context: string): WireConstraintPanels {
  const object = readJsonObject(value, context);
  const equalities = readOptionalJsonArray(
    readJsonValueAt(object, "equalities"),
    `${context}.equalities`,
  );
  const inequalities = readOptionalJsonArray(
    readJsonValueAt(object, "inequalities"),
    `${context}.inequalities`,
  );
  return {
    equalities: equalities?.map((entry, index) =>
      readWireConstraintPanelEntry(entry, `${context}.equalities[${index}]`)),
    inequalities: inequalities?.map((entry, index) =>
      readWireConstraintPanelEntry(entry, `${context}.inequalities[${index}]`)),
  };
}

function readFilterEntry(value: JsonValue | undefined, context: string): FilterEntry {
  const object = readJsonObject(value, context);
  return {
    objective: readJsonNumber(readJsonValueAt(object, "objective"), `${context}.objective`),
    violation: readJsonNumber(readJsonValueAt(object, "violation"), `${context}.violation`),
  };
}

function readWireFilterInfo(value: JsonValue | undefined, context: string): WireFilterInfo {
  const object = readJsonObject(value, context);
  const entries = readOptionalJsonArray(readJsonValueAt(object, "entries"), `${context}.entries`);
  return {
    current: readFilterEntry(readJsonValueAt(object, "current"), `${context}.current`),
    entries: entries?.map((entry, index) => readFilterEntry(entry, `${context}.entries[${index}]`)) ?? [],
    objective_label: readJsonString(readJsonValueAt(object, "objective_label"), `${context}.objective_label`),
    title: readJsonString(readJsonValueAt(object, "title"), `${context}.title`),
    accepted_mode:
      readOptionalJsonStringOrNumber(
        readJsonValueAt(object, "accepted_mode"),
        `${context}.accepted_mode`,
      ) ?? null,
  };
}

function readWireSolveProgress(value: JsonValue | undefined, context: string): WireSolveProgress {
  const object = readJsonObject(value, context);
  return {
    iteration: readJsonNumber(readJsonValueAt(object, "iteration"), `${context}.iteration`),
    phase: readJsonStringOrNumber(readJsonValueAt(object, "phase"), `${context}.phase`),
    objective: readJsonNumber(readJsonValueAt(object, "objective"), `${context}.objective`),
    eq_inf: readOptionalJsonNumber(readJsonValueAt(object, "eq_inf"), `${context}.eq_inf`) ?? null,
    ineq_inf:
      readOptionalJsonNumber(readJsonValueAt(object, "ineq_inf"), `${context}.ineq_inf`) ?? null,
    dual_inf: readJsonNumber(readJsonValueAt(object, "dual_inf"), `${context}.dual_inf`),
    step_inf:
      readOptionalJsonNumber(readJsonValueAt(object, "step_inf"), `${context}.step_inf`) ?? null,
    penalty: readJsonNumber(readJsonValueAt(object, "penalty"), `${context}.penalty`),
    alpha: readOptionalJsonNumber(readJsonValueAt(object, "alpha"), `${context}.alpha`) ?? null,
    alpha_pr:
      readOptionalJsonNumber(readJsonValueAt(object, "alpha_pr"), `${context}.alpha_pr`) ?? null,
    alpha_du:
      readOptionalJsonNumber(readJsonValueAt(object, "alpha_du"), `${context}.alpha_du`) ?? null,
    line_search_iterations:
      readOptionalJsonNumber(
        readJsonValueAt(object, "line_search_iterations"),
        `${context}.line_search_iterations`,
      ) ?? null,
    filter:
      readJsonValueAt(object, "filter") == null
        ? null
        : readWireFilterInfo(readJsonValueAt(object, "filter"), `${context}.filter`),
    trust_region:
      readJsonValueAt(object, "trust_region") == null
        ? null
        : readWireSolveTrustRegionInfo(
            readJsonValueAt(object, "trust_region"),
            `${context}.trust_region`,
          ),
  };
}

function readWireSolveTrustRegionInfo(
  value: JsonValue | undefined,
  context: string,
): WireSolveTrustRegionInfo {
  const object = readJsonObject(value, context);
  return {
    radius: readJsonNumber(readJsonValueAt(object, "radius"), `${context}.radius`),
    attempted_radius: readJsonNumber(
      readJsonValueAt(object, "attempted_radius"),
      `${context}.attempted_radius`,
    ),
    step_norm: readJsonNumber(readJsonValueAt(object, "step_norm"), `${context}.step_norm`),
    largest_attempted_step_norm:
      readOptionalJsonNumber(
        readJsonValueAt(object, "largest_attempted_step_norm"),
        `${context}.largest_attempted_step_norm`,
      ) ?? null,
    contraction_count: readJsonNumber(
      readJsonValueAt(object, "contraction_count"),
      `${context}.contraction_count`,
    ),
    qp_failure_retries: readJsonNumber(
      readJsonValueAt(object, "qp_failure_retries"),
      `${context}.qp_failure_retries`,
    ),
    boundary_active: readOptionalJsonBoolean(
      readJsonValueAt(object, "boundary_active"),
      `${context}.boundary_active`,
    ) ?? false,
    restoration_attempted: readOptionalJsonBoolean(
      readJsonValueAt(object, "restoration_attempted"),
      `${context}.restoration_attempted`,
    ) ?? false,
    elastic_recovery_attempted: readOptionalJsonBoolean(
      readJsonValueAt(object, "elastic_recovery_attempted"),
      `${context}.elastic_recovery_attempted`,
    ) ?? false,
  };
}

function readSolverPhaseDetail(value: JsonValue | undefined, context: string): SolverPhaseDetail {
  const object = readJsonObject(value, context);
  return {
    label: readJsonString(readJsonValueAt(object, "label"), `${context}.label`),
    value: readJsonString(readJsonValueAt(object, "value"), `${context}.value`),
    count: readJsonNumber(readJsonValueAt(object, "count"), `${context}.count`),
  };
}

function readSolverPhaseDetails(value: JsonValue | undefined, context: string): SolverPhaseDetails {
  if (value == null) {
    return emptySolverPhaseDetails();
  }
  const object = readJsonObject(value, context);
  const symbolicSetup = readOptionalJsonArray(
    readJsonValueAt(object, "symbolic_setup"),
    `${context}.symbolic_setup`,
  );
  const jit = readOptionalJsonArray(readJsonValueAt(object, "jit"), `${context}.jit`);
  const solve = readOptionalJsonArray(readJsonValueAt(object, "solve"), `${context}.solve`);
  return {
    symbolic_setup: (symbolicSetup ?? []).map((entry, index) =>
      readSolverPhaseDetail(entry, `${context}.symbolic_setup[${index}]`)),
    jit: (jit ?? []).map((entry, index) => readSolverPhaseDetail(entry, `${context}.jit[${index}]`)),
    solve: (solve ?? []).map((entry, index) =>
      readSolverPhaseDetail(entry, `${context}.solve[${index}]`)),
  };
}

function readWireSolverReport(value: JsonValue | undefined, context: string): WireSolverReport {
  const object = readJsonObject(value, context);
  return {
    completed:
      (() => {
        const completed = readJsonValueAt(object, "completed");
        if (typeof completed !== "boolean") {
          throw new Error(`${context}.completed must be a boolean.`);
        }
        return completed;
      })(),
    status_label: readJsonString(readJsonValueAt(object, "status_label"), `${context}.status_label`),
    status_kind: readJsonStringOrNumber(
      readJsonValueAt(object, "status_kind"),
      `${context}.status_kind`,
    ),
    iterations: readOptionalJsonNumber(readJsonValueAt(object, "iterations"), `${context}.iterations`) ?? null,
    symbolic_build_s:
      readOptionalJsonNumber(readJsonValueAt(object, "symbolic_build_s"), `${context}.symbolic_build_s`) ??
      null,
    symbolic_derivatives_s:
      readOptionalJsonNumber(
        readJsonValueAt(object, "symbolic_derivatives_s"),
        `${context}.symbolic_derivatives_s`,
      ) ?? null,
    symbolic_setup_s:
      readOptionalJsonNumber(readJsonValueAt(object, "symbolic_setup_s"), `${context}.symbolic_setup_s`) ??
      null,
    jit_s: readOptionalJsonNumber(readJsonValueAt(object, "jit_s"), `${context}.jit_s`) ?? null,
    solve_s: readOptionalJsonNumber(readJsonValueAt(object, "solve_s"), `${context}.solve_s`) ?? null,
    compile_cached:
      readOptionalJsonBoolean(readJsonValueAt(object, "compile_cached"), `${context}.compile_cached`) ??
      false,
    jit_disk_cache_hit:
      readOptionalJsonBoolean(
        readJsonValueAt(object, "jit_disk_cache_hit"),
        `${context}.jit_disk_cache_hit`,
      ) ??
      false,
    phase_details:
      readSolverPhaseDetails(readJsonValueAt(object, "phase_details"), `${context}.phase_details`),
    failure_message:
      readOptionalJsonString(readJsonValueAt(object, "failure_message"), `${context}.failure_message`),
  };
}

function readWireSolveArtifact(value: JsonValue | undefined, context: string): WireSolveArtifact {
  const object = readJsonObject(value, context);
  const summary = readOptionalJsonArray(readJsonValueAt(object, "summary"), `${context}.summary`);
  const compileReportValue = readJsonValueAt(object, "compile_report");
  const charts = readOptionalJsonArray(readJsonValueAt(object, "charts"), `${context}.charts`);
  const visualizations = readOptionalJsonArray(
    readJsonValueAt(object, "visualizations"),
    `${context}.visualizations`,
  );
  const notes = readOptionalJsonArray(readJsonValueAt(object, "notes"), `${context}.notes`);
  const constraintPanels = readJsonValueAt(object, "constraint_panels");
  return {
    title: readJsonString(readJsonValueAt(object, "title"), `${context}.title`),
    summary: summary?.map((metric, index) => readWireMetric(metric, `${context}.summary[${index}]`)),
    solver: readWireSolverReport(readJsonValueAt(object, "solver"), `${context}.solver`),
    compile_report:
      compileReportValue == null
        ? null
        : readWireCompileReportSummary(compileReportValue, `${context}.compile_report`),
    constraint_panels:
      constraintPanels == null
        ? null
        : readWireConstraintPanels(constraintPanels, `${context}.constraint_panels`),
    charts: charts?.map((chart, index) => readWireChart(chart, `${context}.charts[${index}]`)),
    visualizations: visualizations?.map((visualization, index) =>
      readArtifactVisualization(visualization, `${context}.visualizations[${index}]`)),
    scene: readScene2D(readJsonValueAt(object, "scene"), `${context}.scene`),
    notes: notes?.map((note, index) => readJsonString(note, `${context}.notes[${index}]`)) ?? [],
  };
}

function readWireSolveStatus(value: JsonValue | undefined, context: string): WireSolveStatus {
  const object = readJsonObject(value, context);
  const compileReportValue = readJsonValueAt(object, "compile_report");
  return {
    stage: readJsonStringOrNumber(readJsonValueAt(object, "stage"), `${context}.stage`),
    solver_method:
      readOptionalJsonStringOrNumber(
        readJsonValueAt(object, "solver_method"),
        `${context}.solver_method`,
      ) ?? null,
    solver: readWireSolverReport(readJsonValueAt(object, "solver"), `${context}.solver`),
    compile_report:
      compileReportValue == null
        ? null
        : readWireCompileReportSummary(compileReportValue, `${context}.compile_report`),
  };
}

function readWireSolveEvent(value: JsonValue | undefined): WireSolveEvent {
  const object = readJsonObject(value, "solve stream event");
  const kind = readJsonStringOrNumber(readJsonValueAt(object, "kind"), "solve stream event.kind");
  const decodedKind = decodeWireEnum(STREAM_EVENT_KIND_FROM_WIRE, kind, STREAM_EVENT_KIND.error);
  switch (decodedKind) {
    case STREAM_EVENT_KIND.status:
      return {
        kind: "status",
        status: readWireSolveStatus(
          readJsonValueAt(object, "status"),
          "solve stream event.status",
        ),
      };
    case STREAM_EVENT_KIND.log:
      return {
        kind: "log",
        line: readJsonString(readJsonValueAt(object, "line"), "solve stream event.line"),
        level: readJsonStringOrNumber(
          readJsonValueAt(object, "level"),
          "solve stream event.level",
        ),
      };
    case STREAM_EVENT_KIND.iteration:
      return {
        kind: "iteration",
        progress: readWireSolveProgress(
          readJsonValueAt(object, "progress"),
          "solve stream event.progress",
        ),
        artifact: readWireSolveArtifact(
          readJsonValueAt(object, "artifact"),
          "solve stream event.artifact",
        ),
      };
    case STREAM_EVENT_KIND.final:
      return {
        kind: "final",
        artifact: readWireSolveArtifact(
          readJsonValueAt(object, "artifact"),
          "solve stream event.artifact",
        ),
      };
    case STREAM_EVENT_KIND.error:
    default:
      return {
        kind: "error",
        message: readJsonString(readJsonValueAt(object, "message"), "solve stream event.message"),
      };
  }
}

const state: FrontendState = {
  specs: [],
  selectedId: null,
  values: {},
  compileCacheStatuses: [],
  collapsedControlSections: {
    [CONTROL_SECTION.transcription]: false,
    [CONTROL_SECTION.solver]: false,
    [CONTROL_SECTION.problem]: false,
  },
  collapsedControlPanels: {
    [CONTROL_PANEL.sxFunctions]: true,
  },
  collapsedControlBlocks: {},
  artifact: null,
  animationIndex: 0,
  playing: false,
  playHandle: null,
  solving: false,
  renderScheduled: false,
  chartViews: new Map<string, ChartView>(),
  chartLayoutKey: "",
  progressPlotReady: false,
  filterPlotReady: false,
  trustRegionPlotReady: false,
  filterRecentPath: [],
  lastFilterPointKey: null,
  logLines: [],
  followSolverLog: true,
  latestProgress: null,
  liveStatus: null,
  liveSolver: null,
  terminalSolver: null,
  solveAbortController: null,
  solveStopRequested: false,
  pendingIterationEvent: null,
  iterationFlushScheduled: false,
  sceneView: null,
  linkedChartRange: null,
  linkedChartAutorange: true,
  linkingChartRange: false,
  artifactRenderFrameHandle: null,
  prewarmTimer: null,
  prewarmInFlightCount: 0,
  compileStatusPollHandle: null,
};

const problemList = requiredElement<HTMLDivElement>("#problem-list");
const controls = requiredElement<HTMLDivElement>("#controls");
const controlsForm = requiredElement<HTMLFormElement>("#controls-form");
const solveButton = requiredElement<HTMLButtonElement>("#solve-button");
const stopButton = requiredElement<HTMLButtonElement>("#stop-button");
const statusEl = requiredElement<HTMLDivElement>("#status");
const problemNameEl = requiredElement<HTMLDivElement>("#problem-name");
const problemDescriptionEl = requiredElement<HTMLParagraphElement>("#problem-description");
const metricsEl = requiredElement<HTMLDivElement>("#metrics");
const sceneEl = requiredElement<HTMLDivElement>("#scene");
const sceneSubtitleEl = requiredElement<HTMLDivElement>("#scene-subtitle");
const chartsEl = requiredElement<HTMLDivElement>("#charts");
const modelEl = requiredElement<HTMLDivElement>("#model");
const notesEl = requiredElement<HTMLDivElement>("#notes");
const solverSummaryEl = requiredElement<HTMLDivElement>("#solver-summary");
const progressPlotEl = requiredElement<PlotlyHostElement>("#progress-plot");
const filterPlotEl = requiredElement<PlotlyHostElement>("#filter-plot");
const trustRegionPlotEl = requiredElement<PlotlyHostElement>("#trust-region-plot");
const copyConsoleButton = requiredElement<HTMLButtonElement>("#copy-console-button");
const consoleFollowCheckbox = requiredElement<HTMLInputElement>("#console-follow-checkbox");
const clearJitCacheButton = requiredElement<HTMLButtonElement>("#clear-jit-cache-button");
const solverLogEl = requiredElement<HTMLPreElement>("#solver-log");
const prewarmStatusEl = requiredElement<HTMLDivElement>("#prewarm-status");
const eqViolationsEl = requiredElement<HTMLDivElement>("#eq-violations");
const ineqViolationsEl = requiredElement<HTMLDivElement>("#ineq-violations");
const activeConstraintsEl = requiredElement<HTMLDivElement>("#active-constraints");

const SECTION_META: ReadonlyArray<Omit<ControlSectionView, "controls">> = [
  {
    key: CONTROL_SECTION.transcription,
    title: "Transcription",
    subtitle: "Discretization, mesh, and collocation settings.",
  },
  {
    key: CONTROL_SECTION.solver,
    title: "Solver",
    subtitle: "Runtime NLP solver selection and termination thresholds.",
  },
  {
    key: CONTROL_SECTION.problem,
    title: "Problem",
    subtitle: "Problem-specific physical parameters and scenario settings.",
  },
];
const CONTROL_PANEL_META: ReadonlyArray<Omit<ControlPanelView, "controls">> = [
  {
    key: CONTROL_PANEL.sxFunctions,
    title: "SX Functions",
    subtitle: "Reusable symbolic-function strategy for repeated OCP kernels, including global call policy and per-kernel overrides.",
  },
];
const PHASE_LABEL = new Map<SolvePhaseCode, string>([
  [SOLVE_PHASE.initial, "initial"],
  [SOLVE_PHASE.acceptedStep, "accepted step"],
  [SOLVE_PHASE.postConvergence, "post convergence"],
  [SOLVE_PHASE.converged, "converged"],
  [SOLVE_PHASE.regular, "regular"],
  [SOLVE_PHASE.restoration, "restoration"],
]);
const DIRECT_COLLOCATION_VALUE = 1;
const MULTIPLE_SHOOTING_VALUE = 0;

function setStatus(message: string, kind: StatusClassName = "info"): void {
  setStatusDisplay({
    eyebrow: "Run Status",
    title: message,
    detail: "",
    kind,
    active: false,
  });
}

function setStatusDisplay(display: StatusDisplay): void {
  statusEl.className = `status ${display.kind} ${display.active ? "status-active" : ""}`.trim();

  const eyebrow = document.createElement("div");
  eyebrow.className = "status-eyebrow";
  eyebrow.textContent = display.eyebrow;

  const main = document.createElement("div");
  main.className = "status-main";
  if (display.active) {
    const spinner = document.createElement("span");
    spinner.className = "status-spinner";
    spinner.setAttribute("aria-hidden", "true");
    main.appendChild(spinner);
  }
  main.append(document.createTextNode(display.title));

  statusEl.replaceChildren(eyebrow, main);

  if (display.detail) {
    const detail = document.createElement("div");
    detail.className = "status-detail";
    detail.textContent = display.detail;
    statusEl.appendChild(detail);
  }
}

function solverMethodLabel(method: SolverMethodCode | null | undefined): string {
  switch (method) {
    case SOLVER_METHOD.nlip:
      return "NLIP";
    case SOLVER_METHOD.ipopt:
      return "IPOPT";
    case SOLVER_METHOD.sqp:
    default:
      return "SQP";
  }
}

function statusDisplayForSolveStatus(
  status: SolveStatus,
  iteration: number | null = null,
): StatusDisplay {
  switch (status.stage) {
    case SOLVE_STAGE.jitCompilation:
      return {
        eyebrow: "Compile",
        title: "JIT",
        detail: "Compiling numeric evaluation kernels.",
        kind: statusClass(status.solver.status_kind),
        active: true,
      };
    case SOLVE_STAGE.solving:
      return {
        eyebrow: solverMethodLabel(status.solver_method),
        title: "Solving",
        detail:
          iteration == null
            ? "Runtime NLP iterations are in progress."
            : `Iteration ${iteration}`,
        kind: statusClass(status.solver.status_kind),
        active: true,
      };
    case SOLVE_STAGE.symbolicSetup:
    default:
      return {
        eyebrow: "Compile",
        title: "Symbolic Setup",
        detail: "Building symbolic model and derivatives.",
        kind: statusClass(status.solver.status_kind),
        active: true,
      };
  }
}

function statusDisplayForSolverReport(solver: SolverReport): StatusDisplay {
  return {
    eyebrow: "Result",
    title: solver.status_label,
    detail: solver.failure_message ?? "",
    kind: statusClass(solver.status_kind),
    active: false,
  };
}

function pendingSolveRequestStatusDisplay(): StatusDisplay {
  return {
    eyebrow: "Run Status",
    title: "Starting solve",
    detail: "Waiting for backend solver status.",
    kind: "info",
    active: true,
  };
}

function describeThrownValue(
  value: object | string | number | boolean | null | undefined,
): string {
  if (value != null && typeof value === "object") {
    const candidate = value as { message?: string; name?: string };
    if (typeof candidate.message === "string" && candidate.message) {
      return candidate.message;
    }
    if (typeof candidate.name === "string" && candidate.name) {
      return candidate.name;
    }
  }
  if (typeof value === "string") {
    return value || "Unexpected frontend error.";
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return "Unexpected frontend error.";
}

function reportFrontendError(
  value: object | string | number | boolean | null | undefined,
  context = "frontend",
): void {
  const detail = describeThrownValue(value);
  const message = context === "frontend" ? detail : `${context}: ${detail}`;
  if (state.solving) {
    applySolveFailure(`Frontend error: ${message}`);
    return;
  }
  appendLogLine(`frontend error: ${message}`, LOG_LEVEL.error);
  setStatusDisplay({
    eyebrow: "App Error",
    title: "Frontend Error",
    detail: message,
    kind: "error",
    active: false,
  });
}

async function readResponseJsonValue(response: Response, context: string): Promise<JsonValue | null> {
  const body = await response.text();
  if (!body.trim()) {
    return null;
  }
  return parseJsonValue(body, context);
}

async function fetchJson<T>(
  url: string,
  decode: (value: JsonValue) => T,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(url, options);
  const payload = await readResponseJsonValue(response, url);
  if (!response.ok) {
    throw new Error(readOptionalErrorMessage(payload) ?? `Request failed with ${response.status}`);
  }
  if (payload == null) {
    throw new Error(`${url} returned an empty response body.`);
  }
  return decode(payload);
}

function fmt(value: number | string | null | undefined, digits = 2): string {
  return Number(value).toFixed(digits).replace(/\.00$/, "");
}

function escapeHtml(value: string): string {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll(" ", "&nbsp;");
}

function isTextEntryControl(control: ControlSpec): boolean {
  return control.editor === CONTROL_EDITOR.text;
}

function isCheckboxControl(control: ControlSpec): boolean {
  return control.editor === CONTROL_EDITOR.checkbox;
}

function formatControlValue(control: ControlSpec, numeric: number): string {
  switch (control.value_display) {
    case CONTROL_VALUE_DISPLAY.integer:
      return String(Math.round(numeric));
    case CONTROL_VALUE_DISPLAY.scientific:
      return Number(numeric).toExponential(1);
    default:
      return `${fmt(numeric, 3)} ${control.unit}`.trim();
  }
}

const LOG_RANGE_CONTROL_IDS = new Set<string>([
  "alpha_rate_regularization",
  "roll_rate_regularization",
]);

function usesLogRangeControl(control: ControlSpec): boolean {
  return LOG_RANGE_CONTROL_IDS.has(control.id)
    && Number.isFinite(control.min)
    && Number.isFinite(control.max)
    && control.min > 0
    && control.max > control.min;
}

function controlValueToRangeValue(control: ControlSpec, numeric: number): number {
  if (!usesLogRangeControl(control)) {
    return numeric;
  }
  return Math.log10(Math.max(control.min, Math.min(control.max, numeric)));
}

function rangeValueToControlValue(control: ControlSpec, numeric: number): number {
  if (!usesLogRangeControl(control)) {
    return numeric;
  }
  return 10 ** numeric;
}

function formatControlInputValue(control: ControlSpec, numeric: number): string {
  if (control.value_display === CONTROL_VALUE_DISPLAY.scientific) {
    return Number(numeric).toExponential(1);
  }
  if (!usesLogRangeControl(control)) {
    return String(numeric);
  }
  return Number(numeric.toPrecision(6)).toString();
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) {
    return "--";
  }
  if (seconds >= 1.0) {
    return `${seconds.toFixed(seconds >= 10 ? 1 : 2)} s`;
  }
  if (seconds >= 1.0e-3) {
    return `${(seconds * 1.0e3).toFixed(1)} ms`;
  }
  return `${(seconds * 1.0e6).toFixed(1)} us`;
}

function formatCompileDuration(seconds: number | null | undefined): string {
  return formatDuration(seconds);
}

type CompileTimingTotals = {
  symbolic_setup_s?: number | null;
  jit_s?: number | null;
};

function sumKnownDurations(values: Array<number | null | undefined>): number | null {
  let total = 0.0;
  let sawValue = false;
  for (const value of values) {
    if (value == null || !Number.isFinite(value)) {
      continue;
    }
    total += value;
    sawValue = true;
  }
  return sawValue ? total : null;
}

function preSolveSeconds(timing: CompileTimingTotals | null | undefined): number | null {
  if (!timing) {
    return null;
  }
  if (
    timing.symbolic_setup_s == null ||
    timing.jit_s == null ||
    !Number.isFinite(timing.symbolic_setup_s) ||
    !Number.isFinite(timing.jit_s)
  ) {
    return null;
  }
  return sumKnownDurations([timing.symbolic_setup_s, timing.jit_s]);
}

type JitCacheOutcome = "unknown" | "disk_hit" | "disk_miss" | "mixed";

type JitCacheCounts = {
  hits: number;
  misses: number;
};

function cacheOutcomeFromCounts(counts: JitCacheCounts | null): JitCacheOutcome {
  if (!counts) {
    return "unknown";
  }
  if (counts.hits > 0 && counts.misses === 0) {
    return "disk_hit";
  }
  if (counts.misses > 0 && counts.hits === 0) {
    return "disk_miss";
  }
  if (counts.hits > 0 && counts.misses > 0) {
    return "mixed";
  }
  return "unknown";
}

function cacheOutcomeFromFlag(jitDiskCacheHit: boolean): JitCacheOutcome {
  return jitDiskCacheHit ? "disk_hit" : "disk_miss";
}

function cacheOutcomeSuffix(outcome: JitCacheOutcome): string {
  switch (outcome) {
    case "disk_hit":
      return "disk hit";
    case "disk_miss":
      return "disk miss";
    case "mixed":
      return "mixed cache";
    default:
      return "cache unknown";
  }
}

function formatJitDurationWithOutcome(
  seconds: number | null | undefined,
  outcome: JitCacheOutcome,
): string {
  const formatted = formatDuration(seconds);
  if (formatted === "--") {
    return formatted;
  }
  if (outcome === "unknown") {
    return formatted;
  }
  return `${formatted} (${cacheOutcomeSuffix(outcome)})`;
}

function emptySolverPhaseDetails(): SolverPhaseDetails {
  return {
    symbolic_setup: [],
    jit: [],
    solve: [],
  };
}

function normalizeSolverPhaseDetails(
  details: SolverPhaseDetails | null | undefined,
): SolverPhaseDetails {
  if (!details) {
    return emptySolverPhaseDetails();
  }
  return {
    symbolic_setup: details.symbolic_setup ?? [],
    jit: details.jit ?? [],
    solve: details.solve ?? [],
  };
}

function selectSolverPhaseDetails(
  primary: SolverPhaseDetail[] | null | undefined,
  fallback: SolverPhaseDetail[] | null | undefined,
): SolverPhaseDetail[] {
  if (primary && primary.length > 0) {
    return primary;
  }
  return fallback ?? [];
}

function mergeSolverPhaseDetails(
  primary: SolverPhaseDetails | null | undefined,
  fallback: SolverPhaseDetails | null | undefined,
): SolverPhaseDetails {
  const next = normalizeSolverPhaseDetails(primary);
  const previous = normalizeSolverPhaseDetails(fallback);
  return {
    symbolic_setup: selectSolverPhaseDetails(next.symbolic_setup, previous.symbolic_setup),
    jit: selectSolverPhaseDetails(next.jit, previous.jit),
    solve: selectSolverPhaseDetails(next.solve, previous.solve),
  };
}

function statusClass(kind: SolverStatusKindCode): StatusClassName {
  switch (kind) {
    case SOLVER_STATUS_KIND.success:
      return "success";
    case SOLVER_STATUS_KIND.warning:
      return "warning";
    case SOLVER_STATUS_KIND.error:
      return "error";
    default:
      return "info";
  }
}

function constraintSeverityClass(
  severity: ConstraintPanelSeverityCode,
): "success" | "warning" | "error" {
  switch (severity) {
    case CONSTRAINT_PANEL_SEVERITY.fullAccuracy:
      return "success";
    case CONSTRAINT_PANEL_SEVERITY.reducedAccuracy:
      return "warning";
    default:
      return "error";
  }
}

function boundSeverityClass(
  severity: ConstraintPanelSeverityCode | null | undefined,
): "error" | "warning" | "neutral" {
  switch (severity) {
    case CONSTRAINT_PANEL_SEVERITY.violated:
      return "error";
    case CONSTRAINT_PANEL_SEVERITY.reducedAccuracy:
      return "warning";
    default:
      return "neutral";
  }
}

function logLevelClass(level: LogLevelCode): string {
  switch (level) {
    case LOG_LEVEL.info:
      return "log-info";
    case LOG_LEVEL.warning:
      return "log-warning";
    case LOG_LEVEL.error:
      return "log-error";
    default:
      return "";
  }
}

function currentSolverReport(): SolverReport | null {
  if (state.terminalSolver) {
    return state.terminalSolver;
  }
  return state.liveSolver;
}

function currentSolveStage(): SolveStageCode | null {
  if (state.terminalSolver?.completed) {
    return null;
  }
  return state.liveStatus?.stage ?? null;
}

function buildStatusSolverReport(status: SolveStatus): SolverReport {
  const liveSolver = state.liveSolver;
  const nextSolver = status.solver;
  return {
    ...nextSolver,
    symbolic_build_s: nextSolver.symbolic_build_s ?? liveSolver?.symbolic_build_s ?? null,
    symbolic_derivatives_s:
      nextSolver.symbolic_derivatives_s ?? liveSolver?.symbolic_derivatives_s ?? null,
    symbolic_setup_s: nextSolver.symbolic_setup_s ?? liveSolver?.symbolic_setup_s ?? null,
    jit_s: nextSolver.jit_s ?? liveSolver?.jit_s ?? null,
    solve_s: nextSolver.solve_s ?? liveSolver?.solve_s ?? null,
    compile_cached: nextSolver.compile_cached || liveSolver?.compile_cached === true,
    jit_disk_cache_hit:
      nextSolver.jit_disk_cache_hit || liveSolver?.jit_disk_cache_hit === true,
    phase_details: mergeSolverPhaseDetails(nextSolver.phase_details, liveSolver?.phase_details),
  };
}

function buildFailureSolverReport(message: string): SolverReport {
  const liveSolver = state.liveSolver;
  return {
    completed: true,
    status_label:
      liveSolver?.completed && liveSolver.status_kind === SOLVER_STATUS_KIND.error
        ? liveSolver.status_label
        : "Failed",
    status_kind: SOLVER_STATUS_KIND.error,
    iterations: state.latestProgress?.iteration ?? null,
    symbolic_build_s: liveSolver?.symbolic_build_s ?? null,
    symbolic_derivatives_s: liveSolver?.symbolic_derivatives_s ?? null,
    symbolic_setup_s: liveSolver?.symbolic_setup_s ?? null,
    jit_s: liveSolver?.jit_s ?? null,
    solve_s: liveSolver?.solve_s ?? null,
    compile_cached: liveSolver?.compile_cached ?? false,
    jit_disk_cache_hit: liveSolver?.jit_disk_cache_hit ?? false,
    phase_details: normalizeSolverPhaseDetails(liveSolver?.phase_details),
    failure_message: message,
  };
}

function buildStoppedSolverReport(): SolverReport {
  const liveSolver = state.liveSolver;
  return {
    completed: true,
    status_label: "Stopped",
    status_kind: SOLVER_STATUS_KIND.warning,
    iterations: state.latestProgress?.iteration ?? null,
    symbolic_build_s: liveSolver?.symbolic_build_s ?? null,
    symbolic_derivatives_s: liveSolver?.symbolic_derivatives_s ?? null,
    symbolic_setup_s: liveSolver?.symbolic_setup_s ?? null,
    jit_s: liveSolver?.jit_s ?? null,
    solve_s: liveSolver?.solve_s ?? null,
    compile_cached: liveSolver?.compile_cached ?? false,
    jit_disk_cache_hit: liveSolver?.jit_disk_cache_hit ?? false,
    phase_details: normalizeSolverPhaseDetails(liveSolver?.phase_details),
    failure_message: "Solve stopped by user.",
  };
}

function mergeSolverReport(
  next: SolverReport,
  fallback: SolverReport | null | undefined,
): SolverReport {
  return {
    ...next,
    iterations: next.iterations ?? fallback?.iterations ?? null,
    symbolic_build_s: next.symbolic_build_s ?? fallback?.symbolic_build_s ?? null,
    symbolic_derivatives_s:
      next.symbolic_derivatives_s ?? fallback?.symbolic_derivatives_s ?? null,
    symbolic_setup_s: next.symbolic_setup_s ?? fallback?.symbolic_setup_s ?? null,
    jit_s: next.jit_s ?? fallback?.jit_s ?? null,
    solve_s: next.solve_s ?? fallback?.solve_s ?? null,
    compile_cached: next.compile_cached || fallback?.compile_cached === true,
    jit_disk_cache_hit: next.jit_disk_cache_hit || fallback?.jit_disk_cache_hit === true,
    phase_details: mergeSolverPhaseDetails(next.phase_details, fallback?.phase_details),
    failure_message: next.failure_message ?? fallback?.failure_message,
  };
}

const ANSI_SGR_RE = /\[([0-9;]*)m/g;

function ansiClassName(state: AnsiState): string {
  const classes: string[] = [];
  if (state.bold) {
    classes.push("ansi-bold");
  }
  if (state.color) {
    classes.push(`ansi-${state.color}`);
  }
  return classes.join(" ");
}

function pushAnsiSegment(parts: string[], text: string, state: AnsiState): void {
  if (!text) {
    return;
  }
  const escaped = escapeHtml(text);
  const className = ansiClassName(state);
  if (className) {
    parts.push(`<span class="${className}">${escaped}</span>`);
  } else {
    parts.push(escaped);
  }
}

function applyAnsiCodes(state: AnsiState, codesText: string): void {
  const codes = (codesText === "" ? [0] : codesText.split(";").map((value) => Number(value)))
    .filter((value) => Number.isFinite(value));
  const normalized = codes.length > 0 ? codes : [0];
  for (const code of normalized) {
    switch (code) {
      case 0:
        state.bold = false;
        state.color = null;
        break;
      case 1:
        state.bold = true;
        break;
      case 22:
        state.bold = false;
        break;
      case 31:
        state.color = "red";
        break;
      case 32:
        state.color = "green";
        break;
      case 33:
        state.color = "yellow";
        break;
      case 36:
        state.color = "cyan";
        break;
      case 39:
        state.color = null;
        break;
      default:
        break;
      }
  }
}

function ansiToHtml(raw: string): string {
  const input = raw;
  const parts: string[] = [];
  const state: AnsiState = { bold: false, color: null };
  ANSI_SGR_RE.lastIndex = 0;
  let lastIndex = 0;
  for (const match of input.matchAll(ANSI_SGR_RE)) {
    pushAnsiSegment(parts, input.slice(lastIndex, match.index), state);
    applyAnsiCodes(state, match[1] ?? "");
    lastIndex = match.index + match[0].length;
  }
  pushAnsiSegment(parts, input.slice(lastIndex), state);
  return parts.join("");
}

function clearCopyConsoleFeedbackTimer(): void {
  if (copyConsoleFeedbackHandle != null) {
    window.clearTimeout(copyConsoleFeedbackHandle);
    copyConsoleFeedbackHandle = null;
  }
}

function hasCopyableConsoleTranscript(): boolean {
  return state.logLines.some((entry) => entry.text.length > 0);
}

function solverConsoleTranscriptText(): string {
  return state.logLines.map((entry) => entry.text).join("\n");
}

function setCopyConsoleButtonState(
  label: string,
  stateValue: "idle" | "success" | "error" = "idle",
): void {
  copyConsoleButton.textContent = label;
  if (stateValue === "idle") {
    delete copyConsoleButton.dataset.copyState;
  } else {
    copyConsoleButton.dataset.copyState = stateValue;
  }
}

function syncCopyConsoleButtonAvailability(): void {
  copyConsoleButton.disabled = !hasCopyableConsoleTranscript();
}

function resetCopyConsoleButton(): void {
  clearCopyConsoleFeedbackTimer();
  setCopyConsoleButtonState(COPY_CONSOLE_DEFAULT_LABEL);
  syncCopyConsoleButtonAvailability();
}

async function writeClipboardText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch (error) {
      console.warn("clipboard writeText failed; falling back to execCommand copy", error);
    }
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "true");
  textarea.style.position = "fixed";
  textarea.style.top = "0";
  textarea.style.left = "-9999px";
  textarea.style.opacity = "0";
  textarea.style.pointerEvents = "none";
  document.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  textarea.setSelectionRange(0, textarea.value.length);

  try {
    if (!document.execCommand("copy")) {
      throw new Error("document.execCommand('copy') returned false");
    }
  } finally {
    document.body.removeChild(textarea);
  }
}

async function copyConsoleTranscript(): Promise<void> {
  const text = solverConsoleTranscriptText();
  if (!text.trim()) {
    resetCopyConsoleButton();
    return;
  }

  clearCopyConsoleFeedbackTimer();
  copyConsoleButton.disabled = true;
  setCopyConsoleButtonState("Copying...");

  try {
    await writeClipboardText(text);
    setCopyConsoleButtonState("Copied", "success");
  } catch (error) {
    console.warn("copy console transcript failed", error);
    setCopyConsoleButtonState("Copy Failed", "error");
  } finally {
    copyConsoleFeedbackHandle = window.setTimeout(() => {
      copyConsoleFeedbackHandle = null;
      resetCopyConsoleButton();
    }, COPY_CONSOLE_FEEDBACK_MS);
  }
}

function setConsoleFollowState(follow: boolean): void {
  state.followSolverLog = follow;
  consoleFollowCheckbox.checked = follow;
}

function scrollConsoleToBottom(): void {
  solverLogEl.scrollTop = solverLogEl.scrollHeight;
}

function enableConsoleFollow(): void {
  setConsoleFollowState(true);
  scrollConsoleToBottom();
}

function disableConsoleFollowForManualScroll(): void {
  if (state.followSolverLog) {
    setConsoleFollowState(false);
  }
}

function buildLogLineElements(entries: readonly LogLine[]): DocumentFragment {
  const fragment = document.createDocumentFragment();
  for (const entry of entries) {
    const lineEl = document.createElement("span");
    lineEl.className = `log-line ${logLevelClass(entry.level)}`;
    lineEl.innerHTML = ansiToHtml(entry.text) || "&nbsp;";
    fragment.appendChild(lineEl);
  }
  return fragment;
}

function renderLog(): void {
  const previousScrollTop = solverLogEl.scrollTop;
  solverLogEl.replaceChildren(buildLogLineElements(state.logLines));
  syncCopyConsoleButtonAvailability();
  if (state.followSolverLog) {
    scrollConsoleToBottom();
    return;
  }
  const maxScrollTop = Math.max(0, solverLogEl.scrollHeight - solverLogEl.clientHeight);
  solverLogEl.scrollTop = Math.min(previousScrollTop, maxScrollTop);
}

function renderCompileCacheStatus(): void {
  const rows = [...state.compileCacheStatuses];
  rows.sort((left, right) => {
      const problemOrder = left.problem_name.localeCompare(right.problem_name);
      if (problemOrder !== 0) {
        return problemOrder;
      }
      return left.variant_label.localeCompare(right.variant_label);
    });

  if (rows.length === 0) {
    prewarmStatusEl.innerHTML = `<div class="placeholder">Warm compile entries will appear here.</div>`;
    return;
  }

  const table = document.createElement("div");
  table.className = "compile-cache-table";

  const header = document.createElement("div");
  header.className = "compile-cache-row compile-cache-row-header";
  header.innerHTML = `
    <div class="compile-cache-problem">Problem</div>
    <div class="compile-cache-variant">Variant</div>
    <div class="compile-cache-status">Status</div>
  `;
  table.appendChild(header);

  for (const row of rows) {
    const statusLabel = row.state === COMPILE_CACHE_STATE.warming ? "warming" : "ready";
    const report = row.compile_report;
    const rowEl = document.createElement("div");
    rowEl.className = "compile-cache-row";
    const timingGroups = renderCompileCacheTimingGroups(row, report);
    const problemTitle = escapeHtml(row.problem_name);
    const variantTitle = escapeHtml(row.variant_label);
    const cacheCounts = cacheCountsForCompileRow(row, report);
    const cacheOutcome = cacheOutcomeForCompileRow(row, report);
    const statusTitle = report == null
      ? "Compile report pending"
      : `cache ${cacheCountsText(cacheCounts)}; kernels: ${report.kernels.length}`;
    const statusText = statusLabel === "ready" && row.jit_s != null
      ? `${statusLabel}${cacheOutcome === "unknown" ? "" : ` · ${cacheOutcomeSuffix(cacheOutcome)}`}`
      : statusLabel;
    const timingHtml = timingGroups
      .join("");
    rowEl.innerHTML = `
      <div class="compile-cache-problem" title="${problemTitle}">${problemTitle}</div>
      <div class="compile-cache-variant" title="${variantTitle}">${variantTitle}</div>
      <div class="compile-cache-status" title="${escapeHtml(statusTitle)}"><span class="compile-cache-badge compile-cache-badge-${statusLabel}">${escapeHtml(statusText)}</span></div>
      <div class="compile-cache-metrics">${timingHtml}</div>
    `;
    const kernelCount = report?.kernels.length ?? 0;
    rowEl.title = `pre-solve: ${formatCompileDuration(preSolveSeconds(row))}; symbolic total: ${formatCompileDuration(row.symbolic_setup_s)}; ${cacheCountsText(cacheCounts)}; kernels: ${kernelCount}`;
    table.appendChild(rowEl);
  }

  prewarmStatusEl.replaceChildren(table);
}

type CompileCacheMetric = {
  label: string;
  title: string;
  value: string;
};

type CompileMetricSource = {
  row: CompileCacheStatus;
  report: WireCompileReportSummary | null;
  details: SolverPhaseDetail[];
};

type CompileMetricSpec = {
  label: string;
  title: string;
  phaseLabel?: string;
  seconds?: (source: CompileMetricSource) => number | null | undefined;
  count?: (source: CompileMetricSource) => number | null | undefined;
  value?: (source: CompileMetricSource) => string | null | undefined;
  always?: boolean;
  preferNumeric?: boolean;
};

const SYMBOLIC_PHASE_LABEL = {
  build: "Build Problem",
  objectiveGradient: "Objective Gradient",
  equalityJacobian: "Equality Jacobian",
  inequalityJacobian: "Inequality Jacobian",
  lagrangianAssembly: "Lagrangian Assembly",
  hessianGeneration: "Hessian Generation",
} as const;

const JIT_PHASE_LABEL = {
  sxLowering: "SX Lowering",
  llvmCacheKey: "LLVM Cache Key",
  llvmCacheHits: "LLVM Cache Hits",
  llvmCacheMisses: "LLVM Cache Misses",
  llvmCacheCheck: "LLVM Cache Check",
  llvmCacheRead: "LLVM Cache Object Read",
  llvmCacheWrite: "LLVM Cache Object Write",
  llvmCacheHitTotal: "LLVM Cache Hit Total",
  llvmObjectMaterialization: "LLVM Object Materialization",
  llvmModuleBuild: "LLVM Module Build",
  llvmOptimize: "LLVM Optimize",
  llvmObjectEmit: "LLVM Object Emit",
  llvmIrFingerprint: "LLVM IR Fingerprint",
  llvmCompileLoad: "LLVM Compile / Load",
  jitContextAllocation: "JIT Context Allocation",
  xdotHelper: "Xdot Helper",
  rk4ArcHelper: "RK4 Arc Helper",
} as const;

const SYMBOLIC_COMPONENT_METRICS: CompileMetricSpec[] = [
  {
    label: "Build",
    title: "Symbolic model construction",
    phaseLabel: SYMBOLIC_PHASE_LABEL.build,
    seconds: ({ row, report }) => report?.symbolic_construction_s ?? row.symbolic_build_s,
    always: true,
    preferNumeric: true,
  },
  {
    label: "Objective grad",
    title: "Objective gradient generation",
    phaseLabel: SYMBOLIC_PHASE_LABEL.objectiveGradient,
    seconds: ({ report }) => report?.objective_gradient_s,
    preferNumeric: true,
  },
  {
    label: "Eq Jacobian",
    title: "Equality Jacobian generation",
    phaseLabel: SYMBOLIC_PHASE_LABEL.equalityJacobian,
    seconds: ({ report }) => report?.equality_jacobian_s,
    preferNumeric: true,
  },
  {
    label: "Ineq Jacobian",
    title: "Inequality Jacobian generation",
    phaseLabel: SYMBOLIC_PHASE_LABEL.inequalityJacobian,
    seconds: ({ report }) => report?.inequality_jacobian_s,
    preferNumeric: true,
  },
  {
    label: "Lagrangian",
    title: "Lagrangian assembly",
    phaseLabel: SYMBOLIC_PHASE_LABEL.lagrangianAssembly,
    seconds: ({ report }) => report?.lagrangian_assembly_s,
    preferNumeric: true,
  },
  {
    label: "Hessian",
    title: "Lagrangian Hessian generation",
    phaseLabel: SYMBOLIC_PHASE_LABEL.hessianGeneration,
    seconds: ({ report }) => report?.hessian_generation_s,
    preferNumeric: true,
  },
];

const SYMBOLIC_FALLBACK_METRICS: CompileMetricSpec[] = [
  {
    label: "Build",
    title: "Symbolic model construction",
    seconds: ({ row }) => row.symbolic_build_s,
    always: true,
  },
  {
    label: "Symbolic work",
    title: "Elapsed symbolic setup work after model construction",
    seconds: ({ row }) => row.symbolic_derivatives_s,
    always: true,
  },
  {
    label: "Elapsed",
    title: "Elapsed symbolic setup time",
    seconds: ({ row }) => row.symbolic_setup_s,
    always: true,
  },
];

const JIT_CACHE_METRICS: CompileMetricSpec[] = [
  {
    label: "Hits",
    title: "LLVM JIT disk-cache hits across NLP and helper kernels",
    phaseLabel: JIT_PHASE_LABEL.llvmCacheHits,
    count: ({ report }) => report?.llvm_cache_hits,
  },
  {
    label: "Misses",
    title: "LLVM JIT disk-cache misses across NLP and helper kernels",
    phaseLabel: JIT_PHASE_LABEL.llvmCacheMisses,
    count: ({ report }) => report?.llvm_cache_misses,
  },
  {
    label: "Key",
    title: "Cache-key and lowered-function fingerprint generation",
    phaseLabel: JIT_PHASE_LABEL.llvmCacheKey,
    seconds: ({ report }) => report?.llvm_cache_key_s,
  },
  {
    label: "Check",
    title: "Cache manifest lookup and validation across NLP and helper kernels",
    phaseLabel: JIT_PHASE_LABEL.llvmCacheCheck,
    seconds: ({ report }) => report?.llvm_cache_check_s,
  },
  {
    label: "Read",
    title: "Cached object file read across NLP and helper kernels",
    phaseLabel: JIT_PHASE_LABEL.llvmCacheRead,
    seconds: ({ report }) => report?.llvm_cache_read_s,
  },
  {
    label: "Write",
    title: "Cached object file write across NLP and helper kernels",
    phaseLabel: JIT_PHASE_LABEL.llvmCacheWrite,
    seconds: ({ report }) => report?.llvm_cache_write_s,
  },
  {
    label: "Hit path",
    title: "Total cache-hit load path across NLP and helper kernels",
    phaseLabel: JIT_PHASE_LABEL.llvmCacheHitTotal,
    seconds: ({ report }) => report?.llvm_cache_load_s,
  },
  {
    label: "Materialize",
    title: "LLVM object materialization across NLP and helper kernels",
    phaseLabel: JIT_PHASE_LABEL.llvmObjectMaterialization,
    seconds: ({ report }) => report?.llvm_cache_materialize_s,
  },
];

const JIT_NLP_LLVM_METRICS: CompileMetricSpec[] = [
  {
    label: "SX lowering",
    title: "SX lowering",
    phaseLabel: JIT_PHASE_LABEL.sxLowering,
    seconds: ({ report }) => report?.lowering_s,
    preferNumeric: true,
  },
  {
    label: "Module",
    title: "LLVM module construction",
    phaseLabel: JIT_PHASE_LABEL.llvmModuleBuild,
    seconds: ({ report }) => report?.llvm_module_build_s,
    preferNumeric: true,
  },
  {
    label: "Optimize",
    title: "LLVM optimization passes",
    phaseLabel: JIT_PHASE_LABEL.llvmOptimize,
    seconds: ({ report }) => report?.llvm_optimization_s,
    preferNumeric: true,
  },
  {
    label: "Emit object",
    title: "LLVM object-code emission",
    phaseLabel: JIT_PHASE_LABEL.llvmObjectEmit,
    seconds: ({ report }) => report?.llvm_object_emit_s,
    preferNumeric: true,
  },
  {
    label: "IR fingerprint",
    title: "Optimized LLVM IR fingerprint generation",
    phaseLabel: JIT_PHASE_LABEL.llvmIrFingerprint,
    seconds: ({ report }) => report?.llvm_ir_fingerprint_s,
    preferNumeric: true,
  },
  {
    label: "LLVM total",
    title: "LLVM compile/load total",
    phaseLabel: JIT_PHASE_LABEL.llvmCompileLoad,
    seconds: ({ report }) => report?.llvm_jit_s,
    preferNumeric: true,
  },
  {
    label: "Context",
    title: "JIT execution-context allocation",
    phaseLabel: JIT_PHASE_LABEL.jitContextAllocation,
    seconds: ({ report }) => report?.jit_context_s,
    preferNumeric: true,
  },
];

const JIT_HELPER_METRICS: CompileMetricSpec[] = [
  {
    label: "Xdot helper",
    title: "OCP dynamics helper compile",
    phaseLabel: JIT_PHASE_LABEL.xdotHelper,
  },
  {
    label: "RK4 helper",
    title: "Multiple-shooting arc helper compile",
    phaseLabel: JIT_PHASE_LABEL.rk4ArcHelper,
  },
];

function compileDurationMetric(
  label: string,
  title: string,
  seconds: number | null | undefined,
  options: { always?: boolean } = {},
): CompileCacheMetric | null {
  if (!options.always && (seconds == null || seconds === 0)) {
    return null;
  }
  return {
    label,
    title,
    value: formatCompileDuration(seconds),
  };
}

function compileCountMetric(
  label: string,
  title: string,
  value: number | null | undefined,
): CompileCacheMetric | null {
  if (value == null) {
    return null;
  }
  return {
    label,
    title,
    value: String(value),
  };
}

function renderCompileCacheMetric(metric: CompileCacheMetric): string {
  return `
    <div class="compile-cache-metric" title="${escapeHtml(metric.title)}">
      <span class="compile-cache-metric-label">${escapeHtml(metric.label)}</span>
      <span class="compile-cache-metric-value">${escapeHtml(metric.value)}</span>
    </div>
  `;
}

function renderCompileCacheTimingGroup(
  title: string,
  metrics: Array<CompileCacheMetric | null>,
): string {
  const visibleMetrics = metrics.filter((metric): metric is CompileCacheMetric => metric != null);
  if (visibleMetrics.length === 0) {
    return "";
  }
  return `
    <section class="compile-cache-metric-group" aria-label="${escapeHtml(title)} timings">
      <div class="compile-cache-metric-group-title">${escapeHtml(title)}</div>
      <div class="compile-cache-metric-list">
        ${visibleMetrics.map(renderCompileCacheMetric).join("")}
      </div>
    </section>
  `;
}

function findSolverPhaseDetail(
  details: SolverPhaseDetail[],
  label: string,
): SolverPhaseDetail | null {
  return details.find((detail) => detail.label === label) ?? null;
}

function parseCompileDurationSeconds(value: string): number | null {
  const match = value.trim().match(/^([0-9]+(?:\.[0-9]+)?)\s*(us|µs|ms|s)$/i);
  if (!match) {
    return null;
  }
  const amount = Number(match[1]);
  if (!Number.isFinite(amount)) {
    return null;
  }
  switch (match[2].toLowerCase()) {
    case "s":
      return amount;
    case "ms":
      return amount / 1.0e3;
    case "us":
    case "µs":
      return amount / 1.0e6;
    default:
      return null;
  }
}

function phaseDetailSeconds(details: SolverPhaseDetail[], label: string): number | null {
  const detail = findSolverPhaseDetail(details, label);
  return detail ? parseCompileDurationSeconds(detail.value) : null;
}

function phaseDetailInteger(details: SolverPhaseDetail[], label: string): number | null {
  const detail = findSolverPhaseDetail(details, label);
  if (!detail) {
    return null;
  }
  const parsed = Number(detail.value);
  return Number.isFinite(parsed) ? parsed : null;
}

function compileMetricFromPhaseDetail(
  details: SolverPhaseDetail[],
  spec: CompileMetricSpec,
): CompileCacheMetric | null {
  if (!spec.phaseLabel) {
    return null;
  }
  const detail = findSolverPhaseDetail(details, spec.phaseLabel);
  if (!detail) {
    return null;
  }
  return {
    label: spec.label,
    title: spec.title,
    value: detail.value,
  };
}

function compileMetricFromNumericSource(
  source: CompileMetricSource,
  spec: CompileMetricSpec,
): CompileCacheMetric | null {
  if (spec.value) {
    const value = spec.value(source);
    if (value == null || (!spec.always && value === "--")) {
      return null;
    }
    return {
      label: spec.label,
      title: spec.title,
      value,
    };
  }
  if (spec.count) {
    return compileCountMetric(spec.label, spec.title, spec.count(source));
  }
  if (spec.seconds) {
    return compileDurationMetric(spec.label, spec.title, spec.seconds(source), {
      always: spec.always,
    });
  }
  return null;
}

function compileMetricFromSpec(
  source: CompileMetricSource,
  spec: CompileMetricSpec,
): CompileCacheMetric | null {
  if (spec.preferNumeric) {
    const numericMetric = compileMetricFromNumericSource(source, spec);
    if (numericMetric && numericMetric.value !== "--") {
      return numericMetric;
    }
    return compileMetricFromPhaseDetail(source.details, spec) ?? numericMetric;
  }
  return (
    compileMetricFromPhaseDetail(source.details, spec) ??
    compileMetricFromNumericSource(source, spec)
  );
}

function compileMetricsFromSpecs(
  source: CompileMetricSource,
  specs: CompileMetricSpec[],
): Array<CompileCacheMetric | null> {
  return specs.map((spec) => compileMetricFromSpec(source, spec));
}

function compileMetricSecondsFromSpec(
  source: CompileMetricSource,
  spec: CompileMetricSpec,
): number | null {
  if (spec.preferNumeric) {
    const numeric = spec.seconds?.(source);
    if (numeric != null) {
      return numeric;
    }
  }
  if (spec.phaseLabel) {
    const phaseSeconds = phaseDetailSeconds(source.details, spec.phaseLabel);
    if (phaseSeconds != null) {
      return phaseSeconds;
    }
  }
  return spec.seconds?.(source) ?? null;
}

function hasAnyPhaseDetail(details: SolverPhaseDetail[], specs: CompileMetricSpec[]): boolean {
  return specs.some((spec) => spec.phaseLabel && findSolverPhaseDetail(details, spec.phaseLabel));
}

function cacheCountsFromReport(report: WireCompileReportSummary | null | undefined): JitCacheCounts | null {
  if (!report) {
    return null;
  }
  return {
    hits: report.llvm_cache_hits,
    misses: report.llvm_cache_misses,
  };
}

function cacheCountsFromPhaseDetails(details: SolverPhaseDetail[]): JitCacheCounts | null {
  const hits = phaseDetailInteger(details, JIT_PHASE_LABEL.llvmCacheHits);
  const misses = phaseDetailInteger(details, JIT_PHASE_LABEL.llvmCacheMisses);
  if (hits == null && misses == null) {
    return null;
  }
  return {
    hits: hits ?? 0,
    misses: misses ?? 0,
  };
}

function cacheCountsForCompileRow(
  row: CompileCacheStatus,
  report: WireCompileReportSummary | null,
): JitCacheCounts | null {
  return cacheCountsFromPhaseDetails(row.phase_details.jit) ?? cacheCountsFromReport(report);
}

function cacheCountsText(counts: JitCacheCounts | null): string {
  if (!counts) {
    return "cache counts pending";
  }
  return `${counts.hits} hit${counts.hits === 1 ? "" : "s"} / ${counts.misses} miss${counts.misses === 1 ? "" : "es"}`;
}

function cacheOutcomeForCompileRow(
  row: CompileCacheStatus,
  report: WireCompileReportSummary | null,
): JitCacheOutcome {
  const counted = cacheOutcomeFromCounts(cacheCountsForCompileRow(row, report));
  if (counted !== "unknown") {
    return counted;
  }
  if (row.jit_s != null) {
    return cacheOutcomeFromFlag(row.jit_disk_cache_hit);
  }
  return "unknown";
}

function currentCompileReport(): WireCompileReportSummary | null {
  return state.liveStatus?.compile_report ?? state.artifact?.compile_report ?? null;
}

function cacheOutcomeForSolver(solver: SolverReport | null | undefined): JitCacheOutcome {
  const fromReport = cacheOutcomeFromCounts(cacheCountsFromReport(currentCompileReport()));
  if (fromReport !== "unknown") {
    return fromReport;
  }
  if (solver?.jit_s != null) {
    return cacheOutcomeFromFlag(solver.jit_disk_cache_hit || solver.compile_cached);
  }
  return "unknown";
}

function sumKnownSymbolicSeconds(values: Array<number | null | undefined>): number {
  return values.reduce<number>((sum, value) => sum + (value ?? 0), 0);
}

function knownSymbolicSeconds(source: CompileMetricSource): number {
  return sumKnownSymbolicSeconds(
    SYMBOLIC_COMPONENT_METRICS.map((spec) => compileMetricSecondsFromSpec(source, spec)),
  );
}

function residualCompileMetric(
  label: string,
  title: string,
  totalSeconds: number | null | undefined,
  knownSeconds: number,
): CompileCacheMetric | null {
  if (totalSeconds == null) {
    return null;
  }
  const residualSeconds = totalSeconds - knownSeconds;
  if (residualSeconds <= 1.0e-3) {
    return null;
  }
  return {
    label,
    title,
    value: formatCompileDuration(residualSeconds),
  };
}

function symbolicCompileMetrics(
  row: CompileCacheStatus,
  report: WireCompileReportSummary | null,
): Array<CompileCacheMetric | null> {
  const source = {
    row,
    report,
    details: row.phase_details.symbolic_setup,
  };
  const hasDetailedSymbolicMetrics = hasAnyPhaseDetail(
    source.details,
    SYMBOLIC_COMPONENT_METRICS.filter((spec) => spec.phaseLabel !== SYMBOLIC_PHASE_LABEL.build),
  );

  if (!report && !hasDetailedSymbolicMetrics) {
    return compileMetricsFromSpecs(source, SYMBOLIC_FALLBACK_METRICS);
  }

  return [
    ...compileMetricsFromSpecs(source, SYMBOLIC_COMPONENT_METRICS),
    residualCompileMetric(
      !report && row.state === COMPILE_CACHE_STATE.warming ? "Current / other" : "Other",
      report
        ? "Symbolic setup time not covered by visible symbolic buckets"
        : "Elapsed symbolic setup time not covered by completed visible symbolic buckets",
      row.symbolic_setup_s,
      knownSymbolicSeconds(source),
    ),
    compileDurationMetric(
      !report && row.state === COMPILE_CACHE_STATE.warming ? "Elapsed" : "Total",
      "Symbolic setup total",
      row.symbolic_setup_s,
      { always: true },
    ),
  ];
}

function renderCompileCacheTimingGroups(
  row: CompileCacheStatus,
  report: WireCompileReportSummary | null,
): string[] {
  const symbolicMetrics = symbolicCompileMetrics(row, report);
  const jitSource = {
    row,
    report,
    details: row.phase_details.jit,
  };
  const cacheOutcome = cacheOutcomeForCompileRow(row, report);
  return [
    renderCompileCacheTimingGroup("Symbolic", symbolicMetrics),
    renderCompileCacheTimingGroup("JIT Cache", compileMetricsFromSpecs(jitSource, JIT_CACHE_METRICS)),
    renderCompileCacheTimingGroup("NLP Lowering / LLVM", compileMetricsFromSpecs(jitSource, JIT_NLP_LLVM_METRICS)),
    renderCompileCacheTimingGroup("OCP Helpers", compileMetricsFromSpecs(jitSource, JIT_HELPER_METRICS)),
    renderCompileCacheTimingGroup(
      "Overall",
      compileMetricsFromSpecs(jitSource, [
        {
          label: "Pre-solve",
          title: "Symbolic setup plus JIT stage before calling the NLP solver",
          value: ({ row: sourceRow }) => formatCompileDuration(preSolveSeconds(sourceRow)),
          always: true,
        },
        {
          label: "JIT stage",
          title: "JIT stage total, including OCP helper kernels when reported",
          value: ({ row: sourceRow }) => formatJitDurationWithOutcome(sourceRow.jit_s, cacheOutcome),
          always: true,
        },
      ]),
    ),
  ].filter((group) => group.length > 0);
}

function decodeWireEnum<T extends number>(
  map: Readonly<Record<string, T>>,
  wireValue: string | number | null | undefined,
  fallback: T,
): T {
  if (wireValue == null) {
    return fallback;
  }
  return Object.prototype.hasOwnProperty.call(map, wireValue) ? map[String(wireValue)] : fallback;
}

function normalizeControl(control: WireControlSpec): ControlSpec {
  return {
    ...control,
    section: decodeWireEnum(CONTROL_SECTION_FROM_WIRE, control.section, CONTROL_SECTION.problem),
    panel:
      control.panel == null
        ? null
        : decodeWireEnum(CONTROL_PANEL_FROM_WIRE, control.panel, CONTROL_PANEL.sxFunctions),
    editor: decodeWireEnum(CONTROL_EDITOR_FROM_WIRE, control.editor, CONTROL_EDITOR.slider),
    visibility: decodeWireEnum(
      CONTROL_VISIBILITY_FROM_WIRE,
      control.visibility,
      CONTROL_VISIBILITY.always,
    ),
    semantic: decodeWireEnum(
      CONTROL_SEMANTIC_FROM_WIRE,
      control.semantic,
      CONTROL_SEMANTIC.problemParameter,
    ),
    value_display: decodeWireEnum(
      CONTROL_VALUE_DISPLAY_FROM_WIRE,
      control.value_display,
      CONTROL_VALUE_DISPLAY.scalar,
    ),
    choices: control.choices ?? [],
    profile_defaults: control.profile_defaults ?? [],
  };
}

function normalizeProblemSpec(spec: WireProblemSpec): ProblemSpec {
  return {
    ...spec,
    wire_id: String(spec.id),
    id: decodeWireEnum(PROBLEM_ID_FROM_WIRE, spec.id, PROBLEM_ID.optimalDistanceGlider),
    controls: (spec.controls ?? []).map(normalizeControl),
    math_sections: spec.math_sections ?? [],
    notes: spec.notes ?? [],
  };
}

function normalizeCompileCacheStatus(status: WireCompileCacheStatus): CompileCacheStatus {
  return {
    ...status,
    wire_problem_id: String(status.problem_id),
    problem_id: decodeWireEnum(PROBLEM_ID_FROM_WIRE, status.problem_id, PROBLEM_ID.optimalDistanceGlider),
    state: decodeWireEnum(COMPILE_CACHE_STATE_FROM_WIRE, status.state, COMPILE_CACHE_STATE.ready),
    symbolic_build_s: status.symbolic_build_s ?? null,
    symbolic_derivatives_s: status.symbolic_derivatives_s ?? null,
    symbolic_setup_s: status.symbolic_setup_s ?? null,
    jit_s: status.jit_s ?? null,
    jit_disk_cache_hit: status.jit_disk_cache_hit ?? false,
    phase_details: normalizeSolverPhaseDetails(status.phase_details),
    compile_report: status.compile_report ?? null,
  };
}

function normalizeSolverReport(solver: WireSolverReport): SolverReport {
  if (!solver) {
    throw new Error("Solver report missing from solve artifact.");
  }
  return {
    ...solver,
    status_kind: decodeWireEnum(
      SOLVER_STATUS_KIND_FROM_WIRE,
      solver.status_kind,
      SOLVER_STATUS_KIND.info,
    ),
    compile_cached: solver.compile_cached ?? false,
    jit_disk_cache_hit: solver.jit_disk_cache_hit ?? solver.compile_cached ?? false,
    phase_details: normalizeSolverPhaseDetails(solver.phase_details),
  };
}

function normalizeTimeSeries(series: WireTimeSeries): TimeSeries {
  return {
    ...series,
    role: decodeWireEnum(TIME_SERIES_ROLE_FROM_WIRE, series.role, TIME_SERIES_ROLE.data),
  };
}

function normalizeChart(chart: WireChart): Chart {
  return {
    ...chart,
    series: (chart.series ?? []).map(normalizeTimeSeries),
  };
}

function normalizeMetric(metric: WireMetric): Metric {
  return {
    ...metric,
    key: decodeWireEnum(METRIC_KEY_FROM_WIRE, metric.key, METRIC_KEY.custom),
  };
}

function normalizeConstraintPanelEntry(entry: WireConstraintPanelEntry): ConstraintPanelEntry {
  return {
    ...entry,
    category: decodeWireEnum(
      CONSTRAINT_PANEL_CATEGORY_FROM_WIRE,
      entry.category,
      CONSTRAINT_PANEL_CATEGORY.path,
    ),
    severity: decodeWireEnum(
      CONSTRAINT_PANEL_SEVERITY_FROM_WIRE,
      entry.severity,
      CONSTRAINT_PANEL_SEVERITY.violated,
    ),
    lower_severity:
      entry.lower_severity == null
        ? null
        : decodeWireEnum(
            CONSTRAINT_PANEL_SEVERITY_FROM_WIRE,
            entry.lower_severity,
            CONSTRAINT_PANEL_SEVERITY.fullAccuracy,
          ),
    upper_severity:
      entry.upper_severity == null
        ? null
        : decodeWireEnum(
            CONSTRAINT_PANEL_SEVERITY_FROM_WIRE,
            entry.upper_severity,
            CONSTRAINT_PANEL_SEVERITY.fullAccuracy,
          ),
    bound_side:
      entry.bound_side == null
        ? null
        : decodeWireEnum(
            CONSTRAINT_PANEL_BOUND_SIDE_FROM_WIRE,
            entry.bound_side,
            CONSTRAINT_PANEL_BOUND_SIDE.none,
          ),
    active_bound_side:
      entry.active_bound_side == null
        ? null
        : decodeWireEnum(
            CONSTRAINT_PANEL_BOUND_SIDE_FROM_WIRE,
            entry.active_bound_side,
            CONSTRAINT_PANEL_BOUND_SIDE.none,
          ),
  };
}

function normalizeConstraintPanels(panels: WireConstraintPanels | null | undefined): ConstraintPanels {
  if (!panels) {
    return { equalities: [], inequalities: [] };
  }
  return {
    equalities: (panels.equalities ?? []).map(normalizeConstraintPanelEntry),
    inequalities: (panels.inequalities ?? []).map(normalizeConstraintPanelEntry),
  };
}

function normalizeProgress(progress: WireSolveProgress): SolveProgress {
  return {
    ...progress,
    phase: decodeWireEnum(SOLVE_PHASE_FROM_WIRE, progress.phase, SOLVE_PHASE.initial),
    filter:
      progress.filter == null
        ? null
        : {
            ...progress.filter,
            accepted_mode:
              progress.filter.accepted_mode == null
                ? null
                : decodeWireEnum(
                    FILTER_ACCEPTANCE_MODE_FROM_WIRE,
                    progress.filter.accepted_mode,
                    FILTER_ACCEPTANCE_MODE.objectiveArmijo,
                  ),
          },
    trust_region: progress.trust_region ?? null,
  };
}

function normalizeSolveStatus(status: WireSolveStatus): SolveStatus {
  return {
    stage: decodeWireEnum(SOLVE_STAGE_FROM_WIRE, status.stage, SOLVE_STAGE.symbolicSetup),
    solver_method:
      status.solver_method == null
        ? null
        : decodeWireEnum(SOLVER_METHOD_FROM_WIRE, status.solver_method, SOLVER_METHOD.nlip),
    solver: normalizeSolverReport(status.solver),
    compile_report: status.compile_report ?? null,
  };
}

function normalizeArtifact(artifact: WireSolveArtifact): SolveArtifact {
  return {
    ...artifact,
    solver: normalizeSolverReport(artifact.solver),
    compile_report: artifact.compile_report ?? null,
    summary: (artifact.summary ?? []).map(normalizeMetric),
    constraint_panels: normalizeConstraintPanels(artifact.constraint_panels),
    charts: (artifact.charts ?? []).map(normalizeChart),
    visualizations: artifact.visualizations ?? [],
  };
}

function findMetric(artifact: SolveArtifact | null | undefined, key: MetricKeyCode): Metric | null {
  return artifact?.summary?.find((metric) => metric.key === key) ?? null;
}

function normalizeSolveEvent(event: WireSolveEvent): SolveEvent {
  switch (event.kind) {
    case "status":
      return {
        kind: STREAM_EVENT_KIND.status,
        status: normalizeSolveStatus(event.status),
      };
    case "log":
      return {
        kind: STREAM_EVENT_KIND.log,
        line: event.line,
        level: decodeWireEnum(LOG_LEVEL_FROM_WIRE, event.level, LOG_LEVEL.console),
      };
    case "iteration":
      return {
        kind: STREAM_EVENT_KIND.iteration,
        progress: normalizeProgress(event.progress),
        artifact: normalizeArtifact(event.artifact),
      };
    case "final":
      return {
        kind: STREAM_EVENT_KIND.final,
        artifact: normalizeArtifact(event.artifact),
      };
    case "error":
    default:
      return {
        kind: STREAM_EVENT_KIND.error,
        message: event.message,
      };
  }
}

function currentSpec(): ProblemSpec | undefined {
  return state.specs.find((spec) => spec.id === state.selectedId);
}

function findControlById(spec: ProblemSpec | undefined, id: string): ControlSpec | null {
  return spec?.controls.find((control) => control.id === id) ?? null;
}

function findControlBySemantic(
  spec: ProblemSpec | undefined,
  semantic: ControlSemanticCode,
): ControlSpec | null {
  return spec?.controls.find((control) => control.semantic === semantic) ?? null;
}

function hasControlOverride(control: ControlSpec): boolean {
  return Object.prototype.hasOwnProperty.call(state.values, control.id);
}

function currentSolverProfileValue(): number {
  const control = findControlBySemantic(currentSpec(), CONTROL_SEMANTIC.solverProfile);
  if (!control) {
    return 0;
  }
  return Number(hasControlOverride(control) ? state.values[control.id] : control.default);
}

function profileDefaultForControl(control: ControlSpec): number {
  const profile = currentSolverProfileValue();
  const profileDefault = control.profile_defaults.find(
    (entry) => Number(entry.profile) === Number(profile),
  );
  return Number(profileDefault?.value ?? control.default);
}

function effectiveControlValue(control: ControlSpec): number {
  return Number(hasControlOverride(control) ? state.values[control.id] : profileDefaultForControl(control));
}

function albatrossDesignPrefix(controlId: string): AlbatrossDesignPrefix | null {
  return ALBATROSS_DESIGN_PREFIXES.find((prefix) => controlId.startsWith(`${prefix}_`)) ?? null;
}

function albatrossDesignLabel(prefix: AlbatrossDesignPrefix): string {
  switch (prefix) {
    case "delta_l":
      return "Delta L";
    case "h0":
      return "h0";
    case "vx0":
      return "vx0";
    case "tf":
      return "T";
  }
}

function isAlbatrossDesignModeControl(control: ControlSpec): boolean {
  return currentSpec()?.id === PROBLEM_ID.albatrossDynamicSoaring
    && ALBATROSS_DESIGN_PREFIXES.some((prefix) => control.id === `${prefix}_free`);
}

function isAlbatrossDesignBoundsControl(control: ControlSpec): boolean {
  return currentSpec()?.id === PROBLEM_ID.albatrossDynamicSoaring
    && (control.id.endsWith("_lower") || control.id.endsWith("_upper"))
    && albatrossDesignPrefix(control.id) !== null;
}

function albatrossDesignBoundsVisible(control: ControlSpec): boolean {
  const prefix = albatrossDesignPrefix(control.id);
  if (!prefix) {
    return true;
  }
  const modeControl = findControlById(currentSpec(), `${prefix}_free`);
  return modeControl ? effectiveControlValue(modeControl) >= 0.5 : true;
}

function currentSharedControlValue(semantic: ControlSemanticCode, fallback = 0): number {
  const control = findControlBySemantic(currentSpec(), semantic);
  if (!control) {
    return fallback;
  }
  const numeric = effectiveControlValue(control);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function formatSharedControlValue(
  semantic: ControlSemanticCode,
  fallback = "--",
): string {
  const control = findControlBySemantic(currentSpec(), semantic);
  if (!control) {
    return fallback;
  }
  const numeric = effectiveControlValue(control);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return formatControlValue(control, numeric);
}

function currentTranscriptionMethodValue(): number {
  return currentSharedControlValue(CONTROL_SEMANTIC.transcriptionMethod, 0);
}

function currentTimeGridValue(): number {
  return currentSharedControlValue(CONTROL_SEMANTIC.timeGrid, TIME_GRID.uniform);
}

function currentTimeGridUsesStrength(): boolean {
  const value = currentTimeGridValue();
  return value === TIME_GRID.cosine
    || value === TIME_GRID.tanh
    || value === TIME_GRID.geometricStart
    || value === TIME_GRID.geometricEnd
    || value === TIME_GRID.focus;
}

function currentTimeGridIsFocus(): boolean {
  return currentTimeGridValue() === TIME_GRID.focus;
}

function currentTimeGridIsPiecewise(): boolean {
  return currentTimeGridValue() === TIME_GRID.piecewise;
}

function currentSolverMethodValue(): number {
  return currentSharedControlValue(CONTROL_SEMANTIC.solverMethod, SOLVER_METHOD.nlip);
}

function currentNlipLinearSolverValue(): number {
  return currentSharedControlValue(
    CONTROL_SEMANTIC.solverNlipLinearSolver,
    NLIP_LINEAR_SOLVER.spralSrc,
  );
}

function currentNlipLinearSolverUsesSpralControls(): boolean {
  const value = currentNlipLinearSolverValue();
  return value !== NLIP_LINEAR_SOLVER.sparseQdldl;
}

function currentSqpGlobalizationValue(): number {
  return currentSharedControlValue(
    CONTROL_SEMANTIC.solverGlobalization,
    SQP_GLOBALIZATION.lineSearchFilter,
  );
}

function isLineSearchGlobalizationSelected(): boolean {
  const value = currentSqpGlobalizationValue();
  return value === SQP_GLOBALIZATION.lineSearchFilter || value === SQP_GLOBALIZATION.lineSearchMerit;
}

function isFilterGlobalizationSelected(): boolean {
  const value = currentSqpGlobalizationValue();
  return value === SQP_GLOBALIZATION.lineSearchFilter || value === SQP_GLOBALIZATION.trustRegionFilter;
}

function isTrustRegionGlobalizationSelected(): boolean {
  const value = currentSqpGlobalizationValue();
  return value === SQP_GLOBALIZATION.trustRegionFilter || value === SQP_GLOBALIZATION.trustRegionMerit;
}

function isTrustRegionMeritSelected(): boolean {
  return currentSqpGlobalizationValue() === SQP_GLOBALIZATION.trustRegionMerit;
}

function isLineSearchMeritSelected(): boolean {
  return currentSqpGlobalizationValue() === SQP_GLOBALIZATION.lineSearchMerit;
}

function isStructuralControl(control: ControlSpec): boolean {
  if (currentSpec()?.id === PROBLEM_ID.albatrossDynamicSoaring && control.id === "objective") {
    return true;
  }
  switch (control.semantic) {
    case CONTROL_SEMANTIC.transcriptionMethod:
    case CONTROL_SEMANTIC.transcriptionIntervals:
    case CONTROL_SEMANTIC.collocationFamily:
    case CONTROL_SEMANTIC.collocationDegree:
    case CONTROL_SEMANTIC.timeGrid:
    case CONTROL_SEMANTIC.timeGridStrength:
    case CONTROL_SEMANTIC.timeGridFocusCenter:
    case CONTROL_SEMANTIC.timeGridFocusWidth:
    case CONTROL_SEMANTIC.timeGridBreakpoint:
    case CONTROL_SEMANTIC.timeGridFirstIntervalFraction:
    case CONTROL_SEMANTIC.sxFunctionOption:
      return true;
    default:
      return false;
  }
}

function clearScheduledPrewarm(): void {
  if (state.prewarmTimer !== null) {
    window.clearTimeout(state.prewarmTimer);
    state.prewarmTimer = null;
  }
}

function syncCompileStatusPolling(): void {
  const shouldPoll = state.solving || state.prewarmInFlightCount > 0;
  if (shouldPoll) {
    if (state.compileStatusPollHandle === null) {
      state.compileStatusPollHandle = window.setInterval(() => {
        void refreshCompileCacheStatus();
      }, COMPILE_STATUS_POLL_MS);
    }
    return;
  }
  if (state.compileStatusPollHandle !== null) {
    window.clearInterval(state.compileStatusPollHandle);
    state.compileStatusPollHandle = null;
  }
}

async function refreshCompileCacheStatus(): Promise<void> {
  try {
    const snapshot = await fetchJson("/api/prewarm_status", (value) =>
      readWireCompileCacheSnapshot(value, "/api/prewarm_status"),
    );
    state.compileCacheStatuses = (snapshot.entries ?? []).map(normalizeCompileCacheStatus);
    renderCompileCacheStatus();
  } catch (error) {
    console.warn("compile cache status refresh failed", error);
  }
}

async function clearJitCache(): Promise<void> {
  clearJitCacheButton.disabled = true;
  try {
    const response = await fetch("/api/clear_jit_cache", { method: "POST" });
    const payload = await readResponseJsonValue(response, "/api/clear_jit_cache");
    if (!response.ok) {
      throw new Error(readOptionalErrorMessage(payload) ?? `Request failed with ${response.status}`);
    }
    state.compileCacheStatuses = [];
    renderCompileCacheStatus();
    appendLogLine("Cleared on-disk LLVM JIT cache and in-process compile statuses.", LOG_LEVEL.info);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn("failed to clear LLVM JIT cache", error);
    appendLogLine(`Failed to clear LLVM JIT cache: ${message}`, LOG_LEVEL.error);
  } finally {
    clearJitCacheButton.disabled = false;
    void refreshCompileCacheStatus();
  }
}

async function runPrewarm(): Promise<void> {
  const spec = currentSpec();
  if (!spec || state.solving) {
    return;
  }
  state.prewarmInFlightCount += 1;
  syncCompileStatusPolling();
  void refreshCompileCacheStatus();
  try {
    const response = await fetch(`/api/prewarm/${spec.wire_id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values: state.values }),
    });
    const payload = await readResponseJsonValue(response, `/api/prewarm/${spec.wire_id}`);
    if (!response.ok) {
      throw new Error(readOptionalErrorMessage(payload) ?? `Request failed with ${response.status}`);
    }
  } catch (error) {
    console.warn("prewarm failed", error);
  } finally {
    state.prewarmInFlightCount = Math.max(0, state.prewarmInFlightCount - 1);
    syncCompileStatusPolling();
    void refreshCompileCacheStatus();
  }
}

function schedulePrewarm(): void {
  if (state.solving) {
    return;
  }
  if (!currentSpec()) {
    return;
  }
  clearScheduledPrewarm();
  state.prewarmTimer = window.setTimeout(() => {
    state.prewarmTimer = null;
    void runPrewarm();
  }, PREWARM_DELAY_MS);
}

function handleControlUpdate(control: ControlSpec): void {
  if (
    control.semantic === CONTROL_SEMANTIC.transcriptionMethod
    || control.semantic === CONTROL_SEMANTIC.timeGrid
    || control.semantic === CONTROL_SEMANTIC.solverMethod
    || control.semantic === CONTROL_SEMANTIC.solverProfile
    || control.semantic === CONTROL_SEMANTIC.solverGlobalization
    || control.semantic === CONTROL_SEMANTIC.solverNlipLinearSolver
    || isAlbatrossDesignModeControl(control)
  ) {
    renderControls();
    renderTrustRegionPlotVisibility();
  }
  if (isStructuralControl(control)) {
    renderCompileCacheStatus();
    schedulePrewarm();
  }
}

function isControlVisible(control: ControlSpec): boolean {
  if (isAlbatrossDesignBoundsControl(control)) {
    return albatrossDesignBoundsVisible(control);
  }
  if (control.semantic === CONTROL_SEMANTIC.timeGridStrength) {
    return currentTranscriptionMethodValue() === DIRECT_COLLOCATION_VALUE
      && currentTimeGridUsesStrength();
  }
  if (
    control.semantic === CONTROL_SEMANTIC.timeGridFocusCenter
    || control.semantic === CONTROL_SEMANTIC.timeGridFocusWidth
  ) {
    return currentTranscriptionMethodValue() === DIRECT_COLLOCATION_VALUE
      && currentTimeGridIsFocus();
  }
  if (
    control.semantic === CONTROL_SEMANTIC.timeGridBreakpoint
    || control.semantic === CONTROL_SEMANTIC.timeGridFirstIntervalFraction
  ) {
    return currentTranscriptionMethodValue() === DIRECT_COLLOCATION_VALUE
      && currentTimeGridIsPiecewise();
  }
  if (
    control.semantic === CONTROL_SEMANTIC.solverHessianRegularization
    || control.semantic === CONTROL_SEMANTIC.solverGlobalization
  ) {
    return currentSolverMethodValue() === SOLVER_METHOD.sqp;
  }
  if (control.semantic === CONTROL_SEMANTIC.solverNlipLinearSolver) {
    return currentSolverMethodValue() === SOLVER_METHOD.nlip;
  }
  if (
    control.semantic === CONTROL_SEMANTIC.solverNlipSpralPivotMethod
    || control.semantic === CONTROL_SEMANTIC.solverNlipSpralZeroPivotAction
    || control.semantic === CONTROL_SEMANTIC.solverNlipSpralSmallPivot
    || control.semantic === CONTROL_SEMANTIC.solverNlipSpralPivotU
  ) {
    return currentSolverMethodValue() === SOLVER_METHOD.nlip
      && currentNlipLinearSolverUsesSpralControls();
  }
  if (
    control.semantic === CONTROL_SEMANTIC.solverExactMeritPenalty
    || control.semantic === CONTROL_SEMANTIC.solverTrustRegionFixedPenalty
    || control.semantic === CONTROL_SEMANTIC.solverPenaltyIncreaseFactor
    || control.semantic === CONTROL_SEMANTIC.solverMaxPenaltyUpdates
    || control.semantic === CONTROL_SEMANTIC.solverArmijoC1
    || control.semantic === CONTROL_SEMANTIC.solverWolfeC2
    || control.semantic === CONTROL_SEMANTIC.solverLineSearchBeta
    || control.semantic === CONTROL_SEMANTIC.solverLineSearchMaxSteps
    || control.semantic === CONTROL_SEMANTIC.solverMinStep
    || control.semantic === CONTROL_SEMANTIC.solverFilterGammaObjective
    || control.semantic === CONTROL_SEMANTIC.solverFilterGammaViolation
    || control.semantic === CONTROL_SEMANTIC.solverFilterThetaMaxFactor
    || control.semantic === CONTROL_SEMANTIC.solverFilterSwitchingReferenceMin
    || control.semantic === CONTROL_SEMANTIC.solverFilterSwitchingViolationFactor
    || control.semantic === CONTROL_SEMANTIC.solverFilterSwitchingLinearizedReductionFactor
    || control.semantic === CONTROL_SEMANTIC.solverTrustRegionInitialRadius
    || control.semantic === CONTROL_SEMANTIC.solverTrustRegionMaxRadius
    || control.semantic === CONTROL_SEMANTIC.solverTrustRegionMinRadius
    || control.semantic === CONTROL_SEMANTIC.solverTrustRegionShrinkFactor
    || control.semantic === CONTROL_SEMANTIC.solverTrustRegionGrowFactor
    || control.semantic === CONTROL_SEMANTIC.solverTrustRegionAcceptRatio
    || control.semantic === CONTROL_SEMANTIC.solverTrustRegionExpandRatio
    || control.semantic === CONTROL_SEMANTIC.solverTrustRegionBoundaryFraction
    || control.semantic === CONTROL_SEMANTIC.solverTrustRegionMaxContractions
  ) {
    if (currentSolverMethodValue() !== SOLVER_METHOD.sqp) {
      return false;
    }
    if (
      control.semantic === CONTROL_SEMANTIC.solverPenaltyIncreaseFactor
      || control.semantic === CONTROL_SEMANTIC.solverMaxPenaltyUpdates
    ) {
      return isLineSearchMeritSelected();
    }
    if (
      control.semantic === CONTROL_SEMANTIC.solverArmijoC1
      || control.semantic === CONTROL_SEMANTIC.solverWolfeC2
      || control.semantic === CONTROL_SEMANTIC.solverLineSearchBeta
      || control.semantic === CONTROL_SEMANTIC.solverLineSearchMaxSteps
      || control.semantic === CONTROL_SEMANTIC.solverMinStep
    ) {
      return isLineSearchGlobalizationSelected();
    }
    if (
      control.semantic === CONTROL_SEMANTIC.solverFilterGammaObjective
      || control.semantic === CONTROL_SEMANTIC.solverFilterGammaViolation
      || control.semantic === CONTROL_SEMANTIC.solverFilterThetaMaxFactor
      || control.semantic === CONTROL_SEMANTIC.solverFilterSwitchingReferenceMin
      || control.semantic === CONTROL_SEMANTIC.solverFilterSwitchingViolationFactor
      || control.semantic === CONTROL_SEMANTIC.solverFilterSwitchingLinearizedReductionFactor
    ) {
      return isFilterGlobalizationSelected();
    }
    if (
      control.semantic === CONTROL_SEMANTIC.solverTrustRegionInitialRadius
      || control.semantic === CONTROL_SEMANTIC.solverTrustRegionMaxRadius
      || control.semantic === CONTROL_SEMANTIC.solverTrustRegionMinRadius
      || control.semantic === CONTROL_SEMANTIC.solverTrustRegionShrinkFactor
      || control.semantic === CONTROL_SEMANTIC.solverTrustRegionGrowFactor
      || control.semantic === CONTROL_SEMANTIC.solverTrustRegionAcceptRatio
      || control.semantic === CONTROL_SEMANTIC.solverTrustRegionExpandRatio
      || control.semantic === CONTROL_SEMANTIC.solverTrustRegionBoundaryFraction
      || control.semantic === CONTROL_SEMANTIC.solverTrustRegionMaxContractions
    ) {
      return isTrustRegionGlobalizationSelected();
    }
    if (control.semantic === CONTROL_SEMANTIC.solverTrustRegionFixedPenalty) {
      return isTrustRegionMeritSelected();
    }
    return true;
  }
  switch (control.visibility) {
    case CONTROL_VISIBILITY.directCollocationOnly:
      return currentTranscriptionMethodValue() === DIRECT_COLLOCATION_VALUE;
    case CONTROL_VISIBILITY.multipleShootingOnly:
      return currentTranscriptionMethodValue() === MULTIPLE_SHOOTING_VALUE;
    default:
      return true;
  }
}

function controlSections(spec: ProblemSpec): ControlSectionView[] {
  const sections: ControlSectionView[] = SECTION_META.map((meta) => ({
    key: meta.key,
    title: meta.title,
    subtitle: meta.subtitle,
    controls: [],
  }));
  const byKey = new Map<ControlSectionCode, ControlSectionView>(
    sections.map((section) => [section.key, section]),
  );
  for (const control of spec.controls) {
    if (!isControlVisible(control)) {
      continue;
    }
    const sectionKey = control.section ?? CONTROL_SECTION.problem;
    const section = byKey.get(sectionKey);
    if (section) {
      section.controls.push(control);
    }
  }
  return sections.filter((section) => section.controls.length > 0);
}

function groupedControls(
  controlsForSection: readonly ControlSpec[],
): { ungrouped: ControlSpec[]; panels: ControlPanelView[] } {
  const ungrouped: ControlSpec[] = [];
  const panels: ControlPanelView[] = CONTROL_PANEL_META.map((meta) => ({
    key: meta.key,
    title: meta.title,
    subtitle: meta.subtitle,
    controls: [],
  }));
  const byKey = new Map<ControlPanelCode, ControlPanelView>(
    panels.map((panel) => [panel.key, panel]),
  );
  for (const control of controlsForSection) {
    if (control.panel == null) {
      ungrouped.push(control);
      continue;
    }
    const panel = byKey.get(control.panel);
    if (panel) {
      panel.controls.push(control);
    } else {
      ungrouped.push(control);
    }
  }
  return {
    ungrouped,
    panels: panels.filter((panel) => panel.controls.length > 0),
  };
}

function isControlSectionCollapsed(section: ControlSectionCode): boolean {
  return state.collapsedControlSections[section];
}

function toggleControlSection(section: ControlSectionCode): void {
  state.collapsedControlSections[section] = !state.collapsedControlSections[section];
  renderControls();
}

function isControlPanelCollapsed(panel: ControlPanelCode): boolean {
  return state.collapsedControlPanels[panel];
}

function toggleControlPanel(panel: ControlPanelCode): void {
  state.collapsedControlPanels[panel] = !state.collapsedControlPanels[panel];
  renderControls();
}

function isControlBlockCollapsed(block: ControlBlockView): boolean {
  return state.collapsedControlBlocks[block.key] ?? block.defaultCollapsed;
}

function toggleControlBlock(block: ControlBlockView): void {
  state.collapsedControlBlocks[block.key] = !isControlBlockCollapsed(block);
  renderControls();
}

function phaseLabel(phase: SolvePhaseCode): string {
  return PHASE_LABEL.get(phase) ?? "--";
}

interface AppendControlOptions {
  className?: string;
  label?: string;
  help?: string;
  checkboxText?: string;
}

function appendControl(
  wrapperParent: HTMLElement,
  control: ControlSpec,
  options: AppendControlOptions = {},
): void {
  const wrapper = document.createElement("section");
  wrapper.className = options.className ?? "control-group";
  const value = effectiveControlValue(control);
  const initialHasOverride = hasControlOverride(control);
  const label = options.label ?? control.label;
  const help = options.help ?? control.help;
  const choiceMap = new Map<number, string>(
    (control.choices ?? []).map((choice) => [Number(choice.value), choice.label]),
  );
  const formatValue = (numeric: number): string => {
    const choiceLabel = choiceMap.get(numeric);
    if (choiceLabel) {
      return control.unit ? `${choiceLabel} ${control.unit}`.trim() : choiceLabel;
    }
    return formatControlValue(control, numeric);
  };

  if ((control.choices ?? []).length > 0) {
    const options = control.choices
      .map(
        (choice) =>
          `<option value="${choice.value}"${Number(choice.value) === value ? " selected" : ""}>${choice.label}</option>`,
      )
      .join("");
    wrapper.innerHTML = `
      <div class="control-header">
        <div>
          <div class="control-label">${label}</div>
          <div class="control-help">${help}</div>
        </div>
        <div class="value-pill">${formatValue(value)}</div>
      </div>
      <div class="control-inputs control-inputs-select">
        <select>${options}</select>
      </div>
    `;
    const selectInput = requiredChild<HTMLSelectElement>(wrapper, "select");
    const pill = requiredChild<HTMLDivElement>(wrapper, ".value-pill");
    selectInput.addEventListener("input", (event) => {
      const target = readCurrentSelectTarget(event, `${control.id} select input`);
      const numeric = Number(target.value);
      state.values[control.id] = numeric;
      pill.textContent = formatValue(numeric);
      handleControlUpdate(control);
    });
    wrapperParent.appendChild(wrapper);
    return;
  }

  if (isTextEntryControl(control)) {
    const placeholderValue = profileDefaultForControl(control);
    const inputValue = initialHasOverride ? formatValue(Number(state.values[control.id])) : "";
    wrapper.innerHTML = `
      <div class="control-header">
        <div>
          <div class="control-label">${label}</div>
          <div class="control-help">${help}</div>
        </div>
        <div class="value-pill">${formatValue(value)}</div>
      </div>
      <div class="control-inputs control-inputs-select">
        <input type="text" value="${inputValue}" placeholder="${formatValue(placeholderValue)}" spellcheck="false" />
      </div>
    `;
    const textInput = requiredChild<HTMLInputElement>(wrapper, "input");
    const pill = requiredChild<HTMLDivElement>(wrapper, ".value-pill");
    const sync = (raw: string): void => {
      const trimmed = raw.trim();
      if (trimmed.length === 0) {
        if (hasControlOverride(control)) {
          delete state.values[control.id];
        }
        pill.textContent = formatValue(profileDefaultForControl(control));
        handleControlUpdate(control);
        return;
      }
      const numeric = Number(trimmed);
      if (!Number.isFinite(numeric)) {
        return;
      }
      if (Number.isFinite(control.min) && numeric < control.min) {
        return;
      }
      if (Number.isFinite(control.max) && numeric > control.max) {
        return;
      }
      state.values[control.id] = numeric;
      pill.textContent = formatValue(numeric);
      handleControlUpdate(control);
    };
    textInput.addEventListener("input", (event) => {
      const target = readCurrentInputTarget(event, `${control.id} text input`);
      sync(target.value);
    });
    textInput.addEventListener("blur", () => {
      textInput.value = hasControlOverride(control)
        ? formatValue(Number(state.values[control.id]))
        : "";
    });
    wrapperParent.appendChild(wrapper);
    return;
  }

  if (isCheckboxControl(control)) {
    const checked = value >= 0.5;
    const checkboxText = options.checkboxText ?? (checked ? "Enabled" : "Disabled");
    wrapper.innerHTML = `
      <div class="control-header">
        <div>
          <div class="control-label">${label}</div>
          <div class="control-help">${help}</div>
        </div>
        <div class="value-pill">${checked ? "On" : "Off"}</div>
      </div>
      <label class="control-inputs control-inputs-checkbox">
        <input type="checkbox" aria-label="${label}"${checked ? " checked" : ""} />
        <span>${checkboxText}</span>
      </label>
    `;
    const checkboxInput = requiredChild<HTMLInputElement>(wrapper, 'input[type="checkbox"]');
    const checkboxLabel = requiredChild<HTMLSpanElement>(wrapper, ".control-inputs-checkbox span");
    const pill = requiredChild<HTMLDivElement>(wrapper, ".value-pill");
    checkboxInput.addEventListener("input", (event) => {
      const target = readCurrentInputTarget(event, `${control.id} checkbox input`);
      const numeric = target.checked ? 1 : 0;
      state.values[control.id] = numeric;
      pill.textContent = target.checked ? "On" : "Off";
      checkboxLabel.textContent = options.checkboxText ?? (target.checked ? "Enabled" : "Disabled");
      handleControlUpdate(control);
    });
    wrapperParent.appendChild(wrapper);
    return;
  }

  const rangeMin = usesLogRangeControl(control) ? Math.log10(control.min) : control.min;
  const rangeMax = usesLogRangeControl(control) ? Math.log10(control.max) : control.max;
  const rangeStep = usesLogRangeControl(control) ? control.step : control.step;
  const rangeValue = controlValueToRangeValue(control, value);
  wrapper.innerHTML = `
    <div class="control-header">
      <div>
        <div class="control-label">${label}</div>
        <div class="control-help">${help}</div>
      </div>
      <div class="value-pill">${formatValue(value)}</div>
    </div>
    <div class="control-inputs">
      <input type="range" min="${rangeMin}" max="${rangeMax}" step="${rangeStep}" value="${rangeValue}" />
      <input
        class="control-number-input"
        type="text"
        inputmode="decimal"
        value="${formatControlInputValue(control, value)}"
        placeholder="${control.default}"
        spellcheck="false"
      />
    </div>
  `;
  const rangeInput = requiredChild<HTMLInputElement>(wrapper, 'input[type="range"]');
  const numberInput = requiredChild<HTMLInputElement>(wrapper, ".control-number-input");
  const pill = requiredChild<HTMLDivElement>(wrapper, ".value-pill");

  const parseBufferedNumeric = (raw: string): number | null => {
    const trimmed = raw.trim();
    if (trimmed.length === 0) {
      return null;
    }
    const numeric = Number(trimmed);
    if (!Number.isFinite(numeric)) {
      return null;
    }
    if (Number.isFinite(control.min) && numeric < control.min) {
      return null;
    }
    if (Number.isFinite(control.max) && numeric > control.max) {
      return null;
    }
    return numeric;
  };

  const syncCommittedValue = (numeric: number, writeNumberInput = true): void => {
    state.values[control.id] = numeric;
    rangeInput.value = String(controlValueToRangeValue(control, numeric));
    if (writeNumberInput) {
      numberInput.value = formatControlInputValue(control, numeric);
    }
    pill.textContent = formatValue(numeric);
    handleControlUpdate(control);
  };

  rangeInput.addEventListener("input", (event) => {
    const target = readCurrentInputTarget(event, `${control.id} range input`);
    syncCommittedValue(rangeValueToControlValue(control, Number(target.value)));
  });

  numberInput.addEventListener("input", (event) => {
    const target = readCurrentInputTarget(event, `${control.id} number input`);
    const numeric = parseBufferedNumeric(target.value);
    if (numeric == null) {
      return;
    }
    syncCommittedValue(numeric, false);
  });

  const normalizeNumberInput = (): void => {
    numberInput.value = formatControlInputValue(control, effectiveControlValue(control));
  };

  numberInput.addEventListener("blur", () => {
    const numeric = parseBufferedNumeric(numberInput.value);
    if (numeric != null) {
      syncCommittedValue(numeric);
      return;
    }
    normalizeNumberInput();
  });

  numberInput.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") {
      return;
    }
    const numeric = parseBufferedNumeric(numberInput.value);
    if (numeric != null) {
      syncCommittedValue(numeric);
    } else {
      normalizeNumberInput();
    }
    numberInput.blur();
  });
  wrapperParent.appendChild(wrapper);
}

function appendControlPanel(wrapperParent: HTMLElement, panel: ControlPanelView): void {
  const shell = document.createElement("section");
  shell.className = "control-panel";
  const collapsed = isControlPanelCollapsed(panel.key);
  shell.dataset.collapsed = collapsed ? "true" : "false";

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "control-panel-toggle";
  toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");

  const text = document.createElement("div");
  text.className = "control-panel-header-text";

  const title = document.createElement("div");
  title.className = "control-panel-title";
  title.textContent = panel.title;

  const help = document.createElement("div");
  help.className = "control-panel-help";
  help.textContent = panel.subtitle;

  text.append(title, help);

  const chevron = document.createElement("span");
  chevron.className = "control-panel-chevron";
  chevron.setAttribute("aria-hidden", "true");
  chevron.textContent = "⌄";

  toggle.append(text, chevron);

  const body = document.createElement("div");
  body.className = "control-panel-body";
  body.id = `control-panel-body-${panel.key}`;
  toggle.setAttribute("aria-controls", body.id);
  toggle.addEventListener("click", () => {
    toggleControlPanel(panel.key);
  });

  shell.append(toggle, body);
  if (!collapsed) {
    for (const control of panel.controls) {
      appendControl(body, control);
    }
  }
  wrapperParent.appendChild(shell);
}

function controlBlockKey(section: ControlSectionCode, slug: string): string {
  const spec = currentSpec();
  return `${spec?.wire_id ?? "unknown"}:${section}:${slug}`;
}

function appendControlBlock(wrapperParent: HTMLElement, block: ControlBlockView): void {
  const shell = document.createElement("section");
  shell.className = "control-panel control-block";
  const collapsed = isControlBlockCollapsed(block);
  shell.dataset.collapsed = collapsed ? "true" : "false";

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "control-panel-toggle";
  toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");

  const text = document.createElement("div");
  text.className = "control-panel-header-text";

  const title = document.createElement("div");
  title.className = "control-panel-title";
  title.textContent = block.title;

  const help = document.createElement("div");
  help.className = "control-panel-help";
  help.textContent = block.subtitle;

  text.append(title, help);

  const chevron = document.createElement("span");
  chevron.className = "control-panel-chevron";
  chevron.setAttribute("aria-hidden", "true");
  chevron.textContent = "⌄";

  toggle.append(text, chevron);

  const body = document.createElement("div");
  body.className = "control-panel-body";
  body.id = `control-block-body-${block.key.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  toggle.setAttribute("aria-controls", body.id);
  toggle.addEventListener("click", () => {
    toggleControlBlock(block);
  });

  shell.append(toggle, body);
  if (!collapsed) {
    block.appendBody(body);
  }
  wrapperParent.appendChild(shell);
}

function appendAlbatrossDesignGroup(
  wrapperParent: HTMLElement,
  controlsForSection: readonly ControlSpec[],
  prefix: AlbatrossDesignPrefix,
): void {
  const modeControl = controlsForSection.find((control) => control.id === `${prefix}_free`);
  const valueControl = controlsForSection.find((control) => control.id === `${prefix}_value`);
  if (!modeControl || !valueControl) {
    return;
  }
  const lowerControl = controlsForSection.find((control) => control.id === `${prefix}_lower`);
  const upperControl = controlsForSection.find((control) => control.id === `${prefix}_upper`);
  const isFree = effectiveControlValue(modeControl) >= 0.5;
  const label = albatrossDesignLabel(prefix);
  const shell = document.createElement("section");
  shell.className = "design-variable-group";
  shell.dataset.designMode = isFree ? "free" : "fixed";

  const header = document.createElement("div");
  header.className = "design-variable-header";
  header.innerHTML = `
    <div>
      <div class="design-variable-title">${label}</div>
      <div class="design-variable-help">${isFree
        ? "Optimized as a global design variable."
        : "Held fixed by equal lower and upper global bounds."}</div>
    </div>
    <div class="value-pill">${isFree ? "Free" : "Fixed"}</div>
  `;

  const body = document.createElement("div");
  body.className = "design-variable-body";
  appendControl(body, modeControl, {
    className: "design-control-row design-control-row-mode",
    label: "Mode",
    help: "Check Free to optimize this design variable; leave it unchecked to hold the fixed value.",
    checkboxText: "Free",
  });
  appendControl(body, valueControl, {
    className: "design-control-row",
    label: `${label} ${isFree ? "Guess" : "Fixed"}`,
    help: isFree
      ? "Initial guess for the free design variable. It must satisfy the active bounds."
      : "Fixed value used as equal lower and upper bounds.",
  });
  if (isFree && lowerControl) {
    appendControl(body, lowerControl, {
      className: "design-control-row",
      label: "Lower Bound",
      help: `Lower bound for free ${label}.`,
    });
  }
  if (isFree && upperControl) {
    appendControl(body, upperControl, {
      className: "design-control-row",
      label: "Upper Bound",
      help: `Upper bound for free ${label}.`,
    });
  }

  shell.append(header, body);
  wrapperParent.appendChild(shell);
}

function appendControlList(wrapperParent: HTMLElement, controlsForSection: readonly ControlSpec[]): void {
  for (const control of controlsForSection) {
    appendControl(wrapperParent, control);
  }
}

interface ControlSubgroupView {
  title: string;
  subtitle: string;
  controls: readonly ControlSpec[];
}

function appendControlSubgroup(wrapperParent: HTMLElement, subgroup: ControlSubgroupView): void {
  if (subgroup.controls.length === 0) {
    return;
  }
  const shell = document.createElement("section");
  shell.className = "control-subgroup";

  const header = document.createElement("div");
  header.className = "control-subgroup-header";

  const title = document.createElement("div");
  title.className = "control-subgroup-title";
  title.textContent = subgroup.title;

  const help = document.createElement("div");
  help.className = "control-subgroup-help";
  help.textContent = subgroup.subtitle;

  header.append(title, help);

  const body = document.createElement("div");
  body.className = "control-subgroup-body";
  appendControlList(body, subgroup.controls);

  shell.append(header, body);
  wrapperParent.appendChild(shell);
}

function appendControlListWithSubgroups(
  wrapperParent: HTMLElement,
  controlsForBlock: readonly ControlSpec[],
  subgroups: readonly ControlSubgroupView[],
): void {
  appendControlList(wrapperParent, controlsForBlock);
  for (const subgroup of subgroups) {
    appendControlSubgroup(wrapperParent, subgroup);
  }
}

function controlBlockFromControls(
  section: ControlSectionCode,
  slug: string,
  title: string,
  subtitle: string,
  controlsForBlock: readonly ControlSpec[],
  defaultCollapsed = true,
): ControlBlockView | null {
  if (controlsForBlock.length === 0) {
    return null;
  }
  return {
    key: controlBlockKey(section, slug),
    title,
    subtitle,
    defaultCollapsed,
    appendBody: (body) => appendControlList(body, controlsForBlock),
  };
}

function controlBlockFromGroups(
  section: ControlSectionCode,
  slug: string,
  title: string,
  subtitle: string,
  controlsForBlock: readonly ControlSpec[],
  subgroups: readonly ControlSubgroupView[],
  defaultCollapsed = true,
): ControlBlockView | null {
  const nonEmptySubgroups = subgroups.filter((subgroup) => subgroup.controls.length > 0);
  if (controlsForBlock.length === 0 && nonEmptySubgroups.length === 0) {
    return null;
  }
  return {
    key: controlBlockKey(section, slug),
    title,
    subtitle,
    defaultCollapsed,
    appendBody: (body) => appendControlListWithSubgroups(body, controlsForBlock, nonEmptySubgroups),
  };
}

function makeControlTaker(controlsForSection: readonly ControlSpec[]): {
  remainingControls: () => ControlSpec[];
  takeIds: (ids: readonly string[]) => ControlSpec[];
  takePrefix: (prefix: string) => ControlSpec[];
} {
  const remaining = new Set(controlsForSection.map((control) => control.id));
  const takeWhere = (predicate: (control: ControlSpec) => boolean): ControlSpec[] => {
    const taken: ControlSpec[] = [];
    for (const control of controlsForSection) {
      if (!remaining.has(control.id) || !predicate(control)) {
        continue;
      }
      remaining.delete(control.id);
      taken.push(control);
    }
    return taken;
  };
  return {
    remainingControls: () => controlsForSection.filter((control) => remaining.has(control.id)),
    takeIds: (ids) => {
      const idSet = new Set(ids);
      return takeWhere((control) => idSet.has(control.id));
    },
    takePrefix: (prefix) => takeWhere((control) => control.id.startsWith(prefix)),
  };
}

function albatrossProblemControlBlocks(controlsForSection: readonly ControlSpec[]): ControlBlockView[] {
  const section = CONTROL_SECTION.problem;
  const taker = makeControlTaker(controlsForSection);
  const blocks: ControlBlockView[] = [];
  const push = (block: ControlBlockView | null): void => {
    if (block) {
      blocks.push(block);
    }
  };

  push(controlBlockFromGroups(
    section,
    "objective_weights",
    "Objective & Weights",
    "Objective variant and rate-penalty tuning.",
    taker.takeIds(["objective"]),
    [
      {
        title: "Rate Penalties",
        subtitle: "Separate alpha-rate and roll-rate regularization weights.",
        controls: taker.takeIds(["alpha_rate_regularization", "roll_rate_regularization"]),
      },
    ],
  ));

  const designPrefixes = ALBATROSS_DESIGN_PREFIXES.filter((prefix) => (
    controlsForSection.some((control) => control.id === `${prefix}_free`)
    || controlsForSection.some((control) => control.id === `${prefix}_value`)
  ));
  for (const prefix of designPrefixes) {
    taker.takePrefix(`${prefix}_`);
  }
  const boundaryAnchorControls = taker.takeIds(["constrain_vy0_zero"]);
  if (designPrefixes.length > 0) {
    blocks.push({
      key: controlBlockKey(section, "design_variables"),
      title: "Design Variables",
      subtitle: "Fixed values, guesses, free-variable bounds, and boundary anchors.",
      defaultCollapsed: true,
      appendBody: (body) => {
        for (const prefix of designPrefixes) {
          appendAlbatrossDesignGroup(body, controlsForSection, prefix);
        }
        appendControlSubgroup(body, {
          title: "Boundary Anchors",
          subtitle: "Optional fixed initial lateral-velocity anchor.",
          controls: boundaryAnchorControls,
        });
      },
    });
  }

  push(controlBlockFromGroups(
    section,
    "aircraft_aero",
    "Aircraft & Aero",
    "Runtime aircraft, atmosphere, and aerodynamic polar parameters.",
    [],
    [
      {
        title: "Aircraft & Atmosphere",
        subtitle: "Gravity, density, mass, and reference area.",
        controls: taker.takeIds([
          "gravity_mps2",
          "air_density_kg_m3",
          "mass_kg",
          "reference_area_m2",
        ]),
      },
      {
        title: "Aero Polar",
        subtitle: "Lift slope and induced-drag model parameters.",
        controls: taker.takeIds([
          "cl_slope_per_rad",
          "cd0",
          "aspect_ratio",
          "oswald_efficiency",
        ]),
      },
    ],
  ));

  push(controlBlockFromControls(
    section,
    "wind_shear",
    "Wind Shear",
    "Wind direction and smooth tanh shear profile.",
    taker.takeIds([
      "wind_azimuth_deg",
      "wind_low_mps",
      "wind_high_mps",
      "wind_mid_altitude_m",
      "wind_transition_height_m",
    ]),
  ));

  push(controlBlockFromControls(
    section,
    "path_limits",
    "Bounds & Limits",
    "Altitude, airspeed, load-factor, and control-rate bounds.",
    taker.takeIds([
      "min_altitude_m",
      "min_airspeed_mps",
      "max_airspeed_mps",
      "max_load_factor",
      "max_alpha_rate_deg_s",
      "max_roll_rate_deg_s",
    ]),
  ));

  push(controlBlockFromControls(
    section,
    "initial_guess",
    "Initial Guess",
    "Trajectory wave and periodic control seed.",
    taker.takeIds([
      "initial_wave_amplitude_m",
      "initial_wave_rotation_deg",
      "initial_alpha_deg",
      "initial_roll_amplitude_deg",
    ]),
  ));

  push(controlBlockFromGroups(
    section,
    "numerics_scaling",
    "Numerics & Scaling",
    "Smooth-norm regularization and OCP scaling controls.",
    [],
    [
      {
        title: "Smooth Guards",
        subtitle: "Epsilon values used in smooth airspeed and lift-frame norms.",
        controls: taker.takeIds(["speed_eps_mps", "frame_eps"]),
      },
      {
        title: "Scaling",
        subtitle: "Numerical scaling for the compiled OCP.",
        controls: taker.takeIds(["scaling_enabled"]),
      },
    ],
  ));

  push(controlBlockFromControls(
    section,
    "other_problem",
    "Other Problem Settings",
    "Additional problem-specific controls.",
    taker.remainingControls(),
  ));
  return blocks;
}

function transcriptionControlBlocks(controlsForSection: readonly ControlSpec[]): ControlBlockView[] {
  const section = CONTROL_SECTION.transcription;
  const taker = makeControlTaker(controlsForSection);
  return [
    controlBlockFromGroups(
      section,
      "mesh",
      "Mesh",
      "Transcription size, collocation nodes, and mesh spacing.",
      taker.takeIds(["transcription_intervals", "transcription_method"]),
      [
        {
          title: "Collocation",
          subtitle: "Direct-collocation family and node count.",
          controls: taker.takePrefix("collocation_"),
        },
        {
          title: "Time Grid",
          subtitle: "Direct-collocation interval spacing controls.",
          controls: taker.takePrefix("time_grid"),
        },
      ],
    ),
    controlBlockFromControls(
      section,
      "other_transcription",
      "Other Transcription Settings",
      "Additional transcription controls.",
      taker.remainingControls(),
      true,
    ),
  ].filter((block): block is ControlBlockView => block !== null);
}

function solverControlBlocks(controlsForSection: readonly ControlSpec[]): ControlBlockView[] {
  const section = CONTROL_SECTION.solver;
  const taker = makeControlTaker(controlsForSection);
  const globalizationIds = [
    "solver_globalization",
    "solver_exact_merit_penalty",
    "solver_armijo_c1",
    "solver_wolfe_c2",
    "solver_line_search_beta",
    "solver_line_search_max_steps",
    "solver_min_step",
    "solver_penalty_increase_factor",
    "solver_max_penalty_updates",
  ];
  const globalizationControls = [
    ...taker.takeIds(globalizationIds),
    ...taker.takePrefix("solver_filter_"),
    ...taker.takePrefix("solver_trust_region_"),
  ];
  return [
    controlBlockFromGroups(
      section,
      "core",
      "NLP Solver",
      "Solver method, profile, iteration budget, and method-specific options.",
      taker.takeIds([
        "solver_method",
        "solver_profile",
        "solver_max_iters",
        "solver_overall_tol",
      ]),
      [
        {
          title: "NLIP Linear Solver",
          subtitle: "Sparse KKT backend and SPRAL pivot controls.",
          controls: [
            ...taker.takeIds(["solver_nlip_linear_solver"]),
            ...taker.takePrefix("solver_nlip_spral_"),
          ],
        },
        {
          title: "Termination Tolerances",
          subtitle: "Dual, constraint, and complementarity thresholds.",
          controls: taker.takeIds([
            "solver_dual_tol",
            "solver_constraint_tol",
            "solver_complementarity_tol",
          ]),
        },
        {
          title: "SQP Globalization",
          subtitle: "SQP Hessian regularization, line-search, filter, trust-region, and merit controls.",
          controls: [
            ...taker.takeIds(["solver_hessian_regularization"]),
            ...globalizationControls,
          ],
        },
      ],
    ),
    controlBlockFromControls(
      section,
      "other_solver",
      "Other Solver Settings",
      "Additional solver controls.",
      taker.remainingControls(),
    ),
  ].filter((block): block is ControlBlockView => block !== null);
}

function defaultProblemControlBlocks(controlsForSection: readonly ControlSpec[]): ControlBlockView[] {
  return [
    controlBlockFromControls(
      CONTROL_SECTION.problem,
      "problem_parameters",
      "Problem Parameters",
      "Problem-specific physical parameters and scenario settings.",
      controlsForSection,
      true,
    ),
  ].filter((block): block is ControlBlockView => block !== null);
}

function controlBlocksForSection(
  section: ControlSectionCode,
  controlsForSection: readonly ControlSpec[],
): ControlBlockView[] {
  switch (section) {
    case CONTROL_SECTION.transcription:
      return transcriptionControlBlocks(controlsForSection);
    case CONTROL_SECTION.solver:
      return solverControlBlocks(controlsForSection);
    case CONTROL_SECTION.problem:
      return currentSpec()?.id === PROBLEM_ID.albatrossDynamicSoaring
        ? albatrossProblemControlBlocks(controlsForSection)
        : defaultProblemControlBlocks(controlsForSection);
  }
  return defaultProblemControlBlocks(controlsForSection);
}

function appendControlBlocksForSection(
  wrapperParent: HTMLElement,
  section: ControlSectionCode,
  controlsForSection: readonly ControlSpec[],
): void {
  const blocks = controlBlocksForSection(section, controlsForSection);
  for (const block of blocks) {
    appendControlBlock(wrapperParent, block);
  }
}

function resetChartViews(): void {
  if (window.Plotly) {
    for (const view of state.chartViews.values()) {
      window.Plotly.purge(view.plotEl);
    }
  }
  state.chartViews = new Map();
  state.chartLayoutKey = "";
  state.linkedChartRange = null;
  state.linkedChartAutorange = true;
  state.linkingChartRange = false;
}

function linkedChartRelayoutPayload(): PlotlyRelayoutPayload {
  if (state.linkedChartAutorange || !state.linkedChartRange) {
    return { "xaxis.autorange": true };
  }
  return {
    "xaxis.autorange": false,
    "xaxis.range": state.linkedChartRange.slice(),
  };
}

function extractLinkedChartRange(
  eventData: PlotlyRelayoutPayload,
): { autorange: boolean; range: NumericRange | null } | null {
  const payload = eventData;
  if (payload["xaxis.autorange"]) {
    return { autorange: true, range: null };
  }
  if (Array.isArray(payload["xaxis.range"]) && payload["xaxis.range"].length === 2) {
    return {
      autorange: false,
      range: [Number(payload["xaxis.range"][0]), Number(payload["xaxis.range"][1])],
    };
  }
  if ("xaxis.range[0]" in payload && "xaxis.range[1]" in payload) {
    return {
      autorange: false,
      range: [Number(payload["xaxis.range[0]"]), Number(payload["xaxis.range[1]"])],
    };
  }
  return null;
}

function syncLinkedChartRange(sourceView: ChartView, eventData: PlotlyRelayoutPayload): void {
  if (state.linkingChartRange || !window.Plotly) {
    return;
  }
  const next = extractLinkedChartRange(eventData);
  if (!next) {
    return;
  }
  state.linkedChartAutorange = next.autorange;
  state.linkedChartRange = next.range;
  const payload = linkedChartRelayoutPayload();
  const tasks: Promise<void>[] = [];
  state.linkingChartRange = true;
  for (const view of state.chartViews.values()) {
    if (view === sourceView || !view.linkXRange) {
      continue;
    }
    tasks.push(window.Plotly.relayout(view.plotEl, payload));
  }
  Promise.allSettled(tasks).finally(() => {
    state.linkingChartRange = false;
  });
}

function createSceneView(scene: Scene2D): Scene2DView {
  const shell = document.createElement("div");
  shell.className = "scene-shell";

  const toolbar = document.createElement("div");
  toolbar.className = "scene-toolbar";
  const meta = document.createElement("div");
  meta.className = "scene-meta";
  meta.textContent = `${scene.x_label} · ${scene.y_label}`;
  toolbar.appendChild(meta);

  let playButton: HTMLButtonElement | null = null;
  let slider: HTMLInputElement | null = null;
  if (scene.animation) {
    const controlsEl = document.createElement("div");
    controlsEl.className = "scene-controls";
    playButton = document.createElement("button");
    playButton.type = "button";
    playButton.className = "mini-button";
    playButton.addEventListener("click", () => {
      if (state.playing) {
        stopAnimation();
      } else {
        startAnimation();
      }
      renderScene();
    });

    slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = String(Math.max((scene.animation?.frames.length ?? 1) - 1, 0));
    slider.step = "1";
    slider.addEventListener("input", (event) => {
      stopAnimation();
      const target = readCurrentInputTarget(event, "scene animation slider");
      state.animationIndex = Number(target.value);
      renderScene();
    });
    controlsEl.append(playButton, slider);
    toolbar.appendChild(controlsEl);
  }

  const plotEl = createPlotlyHostElement("plot-surface scene-plot-surface");
  shell.append(toolbar, plotEl);

  return {
    kind: "scene_2d",
    scene,
    shell,
    meta,
    playButton,
    slider,
    plotEl,
  };
}

function createScene3DView(visualization: Paths3DVisualization): Scene3DView {
  const shell = document.createElement("div");
  shell.className = "scene-shell scene-shell-3d";

  const toolbar = document.createElement("div");
  toolbar.className = "scene-toolbar";
  const meta = document.createElement("div");
  meta.className = "scene-meta";
  meta.textContent = visualizationSubtitle(visualization);
  toolbar.appendChild(meta);

  const plotEl = createPlotlyHostElement("plot-surface scene-plot-surface plot-surface-3d");
  shell.append(toolbar, plotEl);

  return {
    kind: "paths_3d",
    visualization,
    shell,
    meta,
    playButton: null,
    slider: null,
    plotEl,
    sceneCamera: null,
    sceneInteractionBound: false,
    sceneInteracting: false,
    scenePointerActive: false,
    sceneIdleFrameHandle: null,
    sceneTraceSignature: null,
  };
}

function scenePlotBounds(scene: Scene2D): SceneBounds {
  return collectSceneBounds(scene);
}

function scenePathTraces(scene: Scene2D): PlotlyTrace[] {
  return scene.paths.map((path, index) => ({
    type: "scatter",
    mode: "lines",
    name: path.name,
    x: path.x,
    y: path.y,
    line: {
      color: PALETTE[index % PALETTE.length],
      width: index === 0 ? PLOT_LINE_WIDTH.scenePrimary : PLOT_LINE_WIDTH.sceneSecondary,
      shape: "linear",
    },
    hovertemplate: `${scene.x_label}: %{x:.3f}<br>${scene.y_label}: %{y:.3f}<extra>${path.name}</extra>`,
  }));
}

function sceneFrameTraces(scene: Scene2D, frameIndex: number): PlotlyTrace[] {
  const frames = scene.animation?.frames ?? [];
  if (frames.length === 0) {
    return [];
  }
  const frame = frames[Math.min(frameIndex, frames.length - 1)]!;
  const traces: PlotlyTrace[] = [];

  if ((frame.segments ?? []).length > 0) {
    const x: Array<number | null> = [];
    const y: Array<number | null> = [];
    for (const [start, end] of frame.segments) {
      x.push(start[0], end[0], null);
      y.push(start[1], end[1], null);
    }
    traces.push({
      type: "scatter",
      mode: "lines",
      name: "Animated Geometry",
      x,
      y,
      showlegend: false,
      hoverinfo: "skip",
      line: {
        color: "#e5f1f4",
        width: PLOT_LINE_WIDTH.sceneSecondary,
      },
    });
  }

  const entries = Object.entries(frame.points);
  if (entries.length > 0) {
    traces.push({
      type: "scatter",
      mode: "markers+text",
      name: "Animated Points",
      x: entries.map(([, point]) => point[0]),
      y: entries.map(([, point]) => point[1]),
      text: entries.map(([label]) => label),
      textposition: "top center",
      textfont: {
        color: "#94b6bd",
        size: 11,
      },
      showlegend: false,
      marker: {
        color: "#5bd1b5",
        size: 4,
        line: {
          color: "#e5f1f4",
          width: PLOT_LINE_WIDTH.tolerance,
        },
      },
      hovertemplate: `${scene.x_label}: %{x:.3f}<br>${scene.y_label}: %{y:.3f}<extra>%{text}</extra>`,
    });
  }

  return traces;
}

function sceneShapes(scene: Scene2D): PlotlyTrace[] {
  return scene.circles.map((circle) => ({
    type: "circle",
    xref: "x",
    yref: "y",
    x0: circle.cx - circle.radius,
    x1: circle.cx + circle.radius,
    y0: circle.cy - circle.radius,
    y1: circle.cy + circle.radius,
    line: {
      color: "#f25f5c",
      width: PLOT_LINE_WIDTH.shape,
      dash: "dash",
    },
    fillcolor: "rgba(242, 95, 92, 0.13)",
  }));
}

function sceneAnnotations(scene: Scene2D): PlotlyTrace[] {
  const annotations: PlotlyTrace[] = [];
  for (const circle of scene.circles) {
    if (!circle.label) {
      continue;
    }
    annotations.push({
      x: circle.cx,
      y: circle.cy,
      text: circle.label,
      showarrow: false,
      font: {
        color: "#f25f5c",
        size: 11,
      },
      bgcolor: "rgba(4, 15, 22, 0.75)",
      bordercolor: "rgba(242, 95, 92, 0.28)",
      borderwidth: 1,
      borderpad: 3,
    });
  }
  for (const arrow of scene.arrows) {
    annotations.push({
      x: arrow.x + arrow.dx,
      y: arrow.y + arrow.dy,
      ax: arrow.x,
      ay: arrow.y,
      xref: "x",
      yref: "y",
      axref: "x",
      ayref: "y",
      text: arrow.label,
      showarrow: true,
      arrowhead: 3,
      arrowwidth: PLOT_LINE_WIDTH.secondary,
      arrowcolor: "#f7b267",
      font: {
        color: "#f7b267",
        size: 11,
      },
      bgcolor: arrow.label ? "rgba(4, 15, 22, 0.72)" : "rgba(0,0,0,0)",
      bordercolor: arrow.label ? "rgba(247, 178, 103, 0.24)" : "rgba(0,0,0,0)",
      borderwidth: arrow.label ? 1 : 0,
      borderpad: arrow.label ? 3 : 0,
    });
  }
  return annotations;
}

function updateScenePlot(view: SceneView): void {
  if (!window.Plotly) {
    return;
  }
  if (view.kind !== "scene_2d") {
    return;
  }
  const scene = view.scene;
  const bounds = scenePlotBounds(scene);
  const data = [
    ...scenePathTraces(scene),
    ...sceneFrameTraces(scene, state.animationIndex),
  ];
  const layout = {
    uirevision: `${scene.title}:${scene.x_label}:${scene.y_label}`,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(4, 15, 22, 0.92)",
    font: {
      color: "#e5f1f4",
      family: '"Avenir Next", Futura, "Trebuchet MS", sans-serif',
      size: 12,
    },
    margin: { l: 74, r: 24, t: 18, b: 52 },
    legend: {
      orientation: "h",
      y: -0.22,
      x: 0,
      font: { color: "#94b6bd", size: 11 },
    },
    dragmode: "zoom",
    hovermode: "closest",
    shapes: sceneShapes(scene),
    annotations: sceneAnnotations(scene),
    xaxis: {
      title: scene.x_label,
      range: [bounds.minX, bounds.maxX],
      gridcolor: "rgba(229, 241, 244, 0.08)",
      linecolor: "rgba(177, 214, 222, 0.18)",
      zeroline: false,
      ticks: "outside",
      titlefont: { color: "#94b6bd" },
    },
    yaxis: {
      title: scene.y_label,
      range: [bounds.minY, bounds.maxY],
      gridcolor: "rgba(229, 241, 244, 0.08)",
      linecolor: "rgba(177, 214, 222, 0.18)",
      zeroline: false,
      ticks: "outside",
      titlefont: { color: "#94b6bd" },
      scaleanchor: "x",
      scaleratio: 1,
    },
  };
  const config = {
    responsive: true,
    displaylogo: false,
    displayModeBar: "hover",
    scrollZoom: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toImage"],
  };
  window.Plotly.react(view.plotEl, data, layout, config);
}

function shouldShowTrustRegionPlot(): boolean {
  return currentSolverMethodValue() === SOLVER_METHOD.sqp
    && (
      Boolean(state.latestProgress?.trust_region)
      || isTrustRegionGlobalizationSelected()
    );
}

function renderTrustRegionPlotVisibility(): void {
  const visible = shouldShowTrustRegionPlot();
  trustRegionPlotEl.hidden = !visible;
  if (!visible) {
    if (window.Plotly && state.trustRegionPlotReady) {
      window.Plotly.purge(trustRegionPlotEl);
    }
    trustRegionPlotEl.replaceChildren();
    state.trustRegionPlotReady = false;
    return;
  }
  if (!state.trustRegionPlotReady && trustRegionPlotEl.childElementCount === 0) {
    trustRegionPlotEl.innerHTML =
      `<div class="placeholder">Trust-region telemetry will appear here during trust-region SQP solves.</div>`;
  }
}

function resetSolverPanel(): void {
  clearScheduledArtifactRender();
  state.renderScheduled = false;
  state.latestProgress = null;
  state.liveStatus = null;
  state.liveSolver = null;
  state.terminalSolver = null;
  state.pendingIterationEvent = null;
  state.iterationFlushScheduled = false;
  state.logLines = [];
  setConsoleFollowState(true);
  renderSolverSummary();
  renderConstraintPanels();
  solverLogEl.replaceChildren();
  resetCopyConsoleButton();
  if (window.Plotly && state.progressPlotReady) {
    window.Plotly.purge(progressPlotEl);
  }
  if (window.Plotly && state.filterPlotReady) {
    window.Plotly.purge(filterPlotEl);
  }
  if (window.Plotly && state.trustRegionPlotReady) {
    window.Plotly.purge(trustRegionPlotEl);
  }
  progressPlotEl.innerHTML = `<div class="placeholder">Solve a problem to populate the live convergence history.</div>`;
  filterPlotEl.innerHTML = `<div class="placeholder">SQP filter telemetry will appear here during the solve.</div>`;
  state.progressPlotReady = false;
  state.filterPlotReady = false;
  state.trustRegionPlotReady = false;
  renderTrustRegionPlotVisibility();
  state.lastFilterPointKey = null;
  state.filterRecentPath = [];
  renderCompileCacheStatus();
}

function selectProblem(problemId: ProblemIdCode): void {
  const spec = state.specs.find((item) => item.id === problemId);
  if (!spec) {
    return;
  }
  clearScheduledPrewarm();
  stopAnimation();
  state.selectedId = problemId;
  state.values = Object.fromEntries(
    spec.controls
      .filter((control) => !isTextEntryControl(control) || control.profile_defaults.length === 0)
      .map((control) => [control.id, control.default]),
  );
  state.artifact = null;
  state.animationIndex = 0;
  state.sceneView = null;
  resetChartViews();
  resetSolverPanel();
  renderProblemList();
  renderOverview();
  renderControls();
  renderMetrics();
  renderConstraintPanels();
  renderScene();
  renderCharts();
  renderModel(spec);
  renderNotes(spec.notes);
  setStatus("Ready to solve.", "success");
  renderCompileCacheStatus();
  schedulePrewarm();
  void refreshCompileCacheStatus();
}

function renderProblemList(): void {
  problemList.innerHTML = "";
  for (const spec of state.specs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `problem-card ${spec.id === state.selectedId ? "active" : ""}`.trim();
    button.innerHTML = `<strong>${spec.name}</strong><p>${spec.description}</p>`;
    button.addEventListener("click", () => selectProblem(spec.id));
    problemList.appendChild(button);
  }
}

function renderOverview(): void {
  const spec = currentSpec();
  problemNameEl.textContent = spec?.name ?? "";
  problemDescriptionEl.textContent = spec?.description ?? "";
}

function renderControls(): void {
  const spec = currentSpec();
  controls.innerHTML = "";
  if (!spec) {
    return;
  }

  const sections = controlSections(spec);
  for (const section of sections) {
    const shell = document.createElement("section");
    shell.className = "control-section";
    const collapsed = isControlSectionCollapsed(section.key);
    shell.dataset.collapsed = collapsed ? "true" : "false";

    const header = document.createElement("div");
    header.className = "control-section-header";

    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "control-section-toggle";
    toggle.setAttribute("aria-expanded", collapsed ? "false" : "true");

    const titleWrap = document.createElement("div");
    titleWrap.className = "control-section-header-text";

    const title = document.createElement("div");
    title.className = "control-section-title";
    title.textContent = section.title;

    const help = document.createElement("div");
    help.className = "control-section-help";
    help.textContent = section.subtitle;

    titleWrap.append(title, help);

    const chevron = document.createElement("span");
    chevron.className = "control-section-chevron";
    chevron.setAttribute("aria-hidden", "true");
    chevron.textContent = "⌄";

    toggle.append(titleWrap, chevron);

    const body = document.createElement("div");
    body.className = "control-section-body";
    body.id = `control-section-body-${section.key}`;
    toggle.setAttribute("aria-controls", body.id);

    toggle.addEventListener("click", () => {
      toggleControlSection(section.key);
    });

    header.appendChild(toggle);
    shell.append(header, body);

    if (!collapsed) {
      const grouped = groupedControls(section.controls);
      appendControlBlocksForSection(body, section.key, grouped.ungrouped);
      for (const panel of grouped.panels) {
        appendControlPanel(body, panel);
      }
    }
    controls.appendChild(shell);
  }
}

function renderMetrics(): void {
  metricsEl.innerHTML = "";
  const solver = currentSolverReport();
  const activeStage = currentSolveStage();
  if (!solver && !state.solving) {
    metricsEl.className = "metrics empty";
    return;
  }

  const singleCards = [
    {
      label: "Symbolic Setup",
      value: formatCompileDuration(solver?.symbolic_setup_s ?? null),
      active: activeStage === SOLVE_STAGE.symbolicSetup,
    },
    {
      label: "JIT",
      value: formatJitDurationWithOutcome(solver?.jit_s ?? null, cacheOutcomeForSolver(solver)),
      active: activeStage === SOLVE_STAGE.jitCompilation,
    },
    {
      label: "Pre-solve",
      value: formatCompileDuration(preSolveSeconds(solver)),
      active: activeStage === SOLVE_STAGE.symbolicSetup || activeStage === SOLVE_STAGE.jitCompilation,
    },
  ];
  metricsEl.className = "metrics";
  const appendMetricValue = (
    target: HTMLElement,
    valueText: string,
    active: boolean,
  ): void => {
    const value = document.createElement("div");
    value.className = "metric-value";
    if (active) {
      value.classList.add("metric-value-progress");
      const spinner = document.createElement("span");
      spinner.className = "metric-spinner";
      spinner.setAttribute("aria-hidden", "true");
      value.appendChild(spinner);
    }
    value.append(document.createTextNode(valueText));
    target.appendChild(value);
  };

  for (const metric of singleCards) {
    const card = document.createElement("article");
    const activeClass = metric.active ? "metric-card-active" : "";
    card.className = `metric-card ${activeClass}`.trim();

    const label = document.createElement("div");
    label.className = "metric-label";
    label.textContent = metric.label;
    card.appendChild(label);

    appendMetricValue(card, metric.value, metric.active);
    metricsEl.appendChild(card);
  }

  const solveGroup = document.createElement("article");
  const solveGroupActive = activeStage === SOLVE_STAGE.solving;
  solveGroup.className = `metric-card metric-card-group ${
    solveGroupActive ? "metric-card-active" : ""
  }`.trim();

  const solveTimeItem = document.createElement("div");
  solveTimeItem.className = "metric-group-item";
  const solveTimeLabel = document.createElement("div");
  solveTimeLabel.className = "metric-label";
  solveTimeLabel.textContent = "Solve";
  solveTimeItem.appendChild(solveTimeLabel);
  appendMetricValue(solveTimeItem, formatDuration(solver?.solve_s ?? null), solveGroupActive);

  const iterationItem = document.createElement("div");
  iterationItem.className = "metric-group-item";
  const iterationLabel = document.createElement("div");
  iterationLabel.className = "metric-label";
  iterationLabel.textContent = "Iterations";
  iterationItem.appendChild(iterationLabel);
  appendMetricValue(
    iterationItem,
    solver?.iterations == null ? "--" : String(solver.iterations),
    false,
  );

  solveGroup.append(solveTimeItem, iterationItem);
  metricsEl.appendChild(solveGroup);
}

function renderNotes(notes: string[]): void {
  const list = document.createElement("ul");
  list.className = "notes";
  for (const note of notes) {
    const item = document.createElement("li");
    item.textContent = note;
    list.appendChild(item);
  }
  notesEl.innerHTML = "";
  notesEl.appendChild(list);
}

function createSolverSummaryChip(item: SolveSummaryItem): HTMLElement {
  const chip = document.createElement("article");
  chip.className = "solver-chip";

  const label = document.createElement("div");
  label.className = "solver-chip-label";
  label.textContent = item.label;
  chip.appendChild(label);

  const value = document.createElement("div");
  value.className = "solver-chip-value";
  value.textContent = item.value;
  chip.appendChild(value);

  return chip;
}

function parseDurationLabelToMs(value: string): number | null {
  const match = value.trim().match(/^([0-9]+(?:\.[0-9]+)?)\s*(us|ms|s)$/i);
  if (!match) {
    return null;
  }
  const numeric = Number(match[1]);
  if (!Number.isFinite(numeric)) {
    return null;
  }
  switch ((match[2] ?? "").toLowerCase()) {
    case "us":
      return numeric / 1000;
    case "ms":
      return numeric;
    case "s":
      return numeric * 1000;
    default:
      return null;
  }
}

function profilingShareHeatScore(
  parentDurationMs: number | null,
  childDurationMs: number | null,
): number | null {
  if (
    childDurationMs == null ||
    parentDurationMs == null ||
    !Number.isFinite(childDurationMs) ||
    !Number.isFinite(parentDurationMs) ||
    parentDurationMs <= 1e-12
  ) {
    return null;
  }
  return Math.max(0, Math.min(1, childDurationMs / parentDurationMs));
}

function childProfilingHeatScore(
  siblingCount: number,
  parentScore: number | null,
  parentDurationMs: number | null,
  childDurationMs: number | null,
): number | null {
  if (siblingCount <= 1) {
    return parentScore;
  }
  return profilingShareHeatScore(parentDurationMs, childDurationMs) ?? parentScore;
}

function profilingHeatColorCode(score: number | null): number | null {
  if (score == null) {
    return null;
  }
  if (score < 0.15) {
    return 32;
  }
  if (score < 0.65) {
    return 33;
  }
  return 31;
}

function colorizeSolveProfilingTime(value: string, heatScore: number | null): string {
  const colorCode = profilingHeatColorCode(heatScore);
  if (colorCode == null) {
    return value;
  }
  return `\u001b[${colorCode}m${value}\u001b[39m`;
}

function formatDurationFromMs(milliseconds: number | null | undefined): string {
  if (milliseconds == null || !Number.isFinite(milliseconds)) {
    return "--";
  }
  return formatDuration(milliseconds / 1000);
}

interface ProfilingLeaf {
  label: string;
  value: string;
  count: number;
  durationMs: number | null;
}

interface ProfilingSection {
  label: string;
  value: string;
  durationMs: number | null;
  items: ProfilingLeaf[];
  children: ProfilingSection[];
}

interface ProfilingRenderedLine {
  label: string;
  value: string;
  metadata: string | null;
  heatScore: number | null;
}

function profilingLeaf(detail: SolverPhaseDetail): ProfilingLeaf {
  return {
    label: detail.label,
    value: detail.value,
    count: detail.count,
    durationMs: parseDurationLabelToMs(detail.value),
  };
}

function timedProfilingLeaves(details: SolverPhaseDetail[]): ProfilingLeaf[] {
  return details.map(profilingLeaf).filter((detail) => detail.durationMs != null);
}

function takeProfilingLeaves(pool: ProfilingLeaf[], labels: readonly string[]): ProfilingLeaf[] {
  const wanted = new Set(labels);
  const taken: ProfilingLeaf[] = [];
  for (let index = 0; index < pool.length; index += 1) {
    const detail = pool[index];
    if (!wanted.has(detail.label)) {
      continue;
    }
    taken.push(detail);
    pool.splice(index, 1);
    index -= 1;
  }
  return taken;
}

function takeFirstProfilingLeaf(pool: ProfilingLeaf[], label: string): ProfilingLeaf | null {
  const index = pool.findIndex((detail) => detail.label === label);
  if (index < 0) {
    return null;
  }
  const [detail] = pool.splice(index, 1);
  return detail ?? null;
}

function sumProfilingDurations(items: readonly { durationMs: number | null }[]): number | null {
  let total = 0;
  let found = false;
  for (const item of items) {
    if (item.durationMs == null || !Number.isFinite(item.durationMs)) {
      continue;
    }
    total += item.durationMs;
    found = true;
  }
  return found ? total : null;
}

function createProfilingSection(
  label: string,
  explicitDurationMs: number | null,
  items: ProfilingLeaf[] = [],
  children: ProfilingSection[] = [],
): ProfilingSection | null {
  const filteredChildren = children.filter(
    (child) => child.durationMs != null || child.items.length > 0 || child.children.length > 0,
  );
  const computedDurationMs =
    explicitDurationMs ??
    sumProfilingDurations([
      ...items,
      ...filteredChildren.map((child) => ({ durationMs: child.durationMs })),
    ]);
  if (computedDurationMs == null && items.length === 0 && filteredChildren.length === 0) {
    return null;
  }
  return {
    label,
    value: formatDurationFromMs(computedDurationMs),
    durationMs: computedDurationMs,
    items,
    children: filteredChildren,
  };
}

function profilingTreePrefix(ancestorHasNext: readonly boolean[], isLast: boolean): string {
  const stem = ancestorHasNext.map((hasNext) => (hasNext ? "|  " : "   ")).join("");
  return `${stem}${isLast ? "`- " : "|- "}`;
}

function formatProfilingShare(
  durationMs: number | null,
  parentDurationMs: number | null,
): string | null {
  if (
    durationMs == null ||
    parentDurationMs == null ||
    !Number.isFinite(durationMs) ||
    !Number.isFinite(parentDurationMs) ||
    parentDurationMs <= 1e-12
  ) {
    return null;
  }
  const sharePercent = Math.max(0, Math.min(100, (durationMs / parentDurationMs) * 100));
  if (sharePercent >= 99.95) {
    return "100%";
  }
  if (sharePercent >= 10) {
    return `${sharePercent.toFixed(0)}%`;
  }
  if (sharePercent >= 1) {
    return `${sharePercent.toFixed(1)}%`;
  }
  return `${sharePercent.toFixed(2)}%`;
}

function formatProfilingLeafMetadata(item: ProfilingLeaf): string | null {
  if (item.count <= 0) {
    return null;
  }
  if (item.durationMs == null || !Number.isFinite(item.durationMs)) {
    return `${item.count}x`;
  }
  return `${item.count}x @ ${formatDurationFromMs(item.durationMs / item.count)}`;
}

function flattenProfilingSection(
  section: ProfilingSection,
  ancestorHasNext: boolean[],
  isLast: boolean,
  sectionHeatScore: number | null,
  parentDurationMs: number | null,
  root = false,
): ProfilingRenderedLine[] {
  const lines: ProfilingRenderedLine[] = [];
  const prefix = root ? "" : profilingTreePrefix(ancestorHasNext, isLast);
  lines.push({
    label: `${prefix}${section.label}`,
    value: section.value,
    metadata: root ? null : (() => {
      const share = formatProfilingShare(section.durationMs, parentDurationMs);
      return share == null ? null : `(${share})`;
    })(),
    heatScore: sectionHeatScore,
  });
  const nextAncestors = root ? [] : [...ancestorHasNext, !isLast];
  const children = [
    ...section.items.map((item) => ({ kind: "item" as const, item })),
    ...section.children.map((child) => ({ kind: "section" as const, section: child })),
  ];
  for (const [index, child] of children.entries()) {
    const childIsLast = index === children.length - 1;
    if (child.kind === "section") {
      const childHeatScore = childProfilingHeatScore(
        children.length,
        sectionHeatScore,
        section.durationMs,
        child.section.durationMs,
      );
      lines.push(
        ...flattenProfilingSection(
          child.section,
          nextAncestors,
          childIsLast,
          childHeatScore,
          section.durationMs,
        ),
      );
      continue;
    }
    const item = child.item;
    lines.push({
      label: `${profilingTreePrefix(nextAncestors, childIsLast)}${item.label}`,
      value: item.value,
      metadata: formatProfilingLeafMetadata(item),
      heatScore: childProfilingHeatScore(
        children.length,
        sectionHeatScore,
        section.durationMs,
        item.durationMs,
      ),
    });
  }
  return lines;
}

function chooseExplicitProfilingDuration(
  explicitDurationMs: number | null,
  fallbackDurationMs: number | null,
): number | null {
  if (
    explicitDurationMs == null ||
    !Number.isFinite(explicitDurationMs) ||
    explicitDurationMs <= 1e-12
  ) {
    return fallbackDurationMs;
  }
  return explicitDurationMs;
}

function buildSolveProfilingSections(solver: SolverReport): ProfilingSection[] {
  const setupSymbolicItems = timedProfilingLeaves(solver.phase_details.symbolic_setup);
  const setupJitItems = timedProfilingLeaves(solver.phase_details.jit);
  const solvePool = timedProfilingLeaves(solver.phase_details.solve);

  const initializationItems = takeProfilingLeaves(solvePool, [
    "Initial Guess Construction",
    "Runtime Bounds Construction",
  ]);
  const evaluateTotal = takeFirstProfilingLeaf(solvePool, "Evaluate Functions");
  const evaluateItems = takeProfilingLeaves(solvePool, [
    "Objective",
    "Gradient",
    "Equality Values",
    "Inequality Values",
    "Constraints",
    "Equality Jacobian",
    "Inequality Jacobian",
    "Jacobian",
    "Hessian",
    "Evaluate Other",
  ]);
  const adapterTotal = takeFirstProfilingLeaf(solvePool, "Adapter / Runtime Plumbing");
  const adapterItems = takeProfilingLeaves(solvePool, [
    "Adapter Callback",
    "Adapter IO",
    "Layout Projection",
  ]);
  const preprocessTotal = takeFirstProfilingLeaf(solvePool, "Preprocess");
  const preprocessItems = takeProfilingLeaves(solvePool, [
    "Jacobian Build",
    "Hessian Build",
    "Regularization",
    "Subproblem Assembly",
    "Preprocess Other",
  ]);
  const subproblemTotal = takeFirstProfilingLeaf(solvePool, "Subproblem Solve");
  const subproblemItems = takeProfilingLeaves(solvePool, [
    "QP Setup",
    "QP Solve",
    "Multiplier Estimation",
    "KKT Assembly",
    "Subproblem Solve Other",
  ]);
  const linearSolveTotal = takeFirstProfilingLeaf(solvePool, "Linear Solve");
  const linearSolveItems = takeProfilingLeaves(solvePool, [
    "Linear RHS Assembly",
    "Linear KKT Values",
    "Linear Symbolic Analysis",
    "Linear Numeric Factorization",
    "Linear Numeric Refactorization",
    "Linear Backsolve / Refinement",
    "Linear Solve Other",
  ]);
  const linearSolveSection = createProfilingSection(
    "Linear Solve",
    linearSolveTotal?.durationMs ?? null,
    linearSolveItems,
  );
  const subproblemSections = [
    linearSolveSection,
  ].filter((section): section is ProfilingSection => section != null);
  const lineSearchTotal = takeFirstProfilingLeaf(solvePool, "Line Search");
  const lineSearchItems = takeProfilingLeaves(solvePool, [
    "Line Search Eval",
    "Line Search Check",
    "Line Search Other",
  ]);
  const convergenceTotal = takeFirstProfilingLeaf(solvePool, "Convergence");
  const convergenceItems = takeProfilingLeaves(solvePool, [
    "Convergence Check",
    "Convergence Other",
  ]);
  const accountingItems = takeProfilingLeaves(solvePool, ["Unaccounted"]);
  const solveTotal = takeFirstProfilingLeaf(solvePool, "Total");
  const otherSolveItems = solvePool;

  const setupSections = [
    createProfilingSection(
      "Symbolics",
      solver.symbolic_setup_s == null ? null : solver.symbolic_setup_s * 1000,
      setupSymbolicItems,
    ),
    createProfilingSection("JIT", solver.jit_s == null ? null : solver.jit_s * 1000, setupJitItems),
    createProfilingSection("Initialization", null, initializationItems),
  ].filter((section): section is ProfilingSection => section != null);

  const solveSections = [
    createProfilingSection("Evaluate Functions", evaluateTotal?.durationMs ?? null, evaluateItems),
    createProfilingSection("Adapter / Runtime Plumbing", adapterTotal?.durationMs ?? null, adapterItems),
    createProfilingSection(
      "Preprocess / Model Assembly",
      preprocessTotal?.durationMs ?? null,
      preprocessItems,
    ),
    createProfilingSection(
      "Subproblem Solve",
      subproblemTotal?.durationMs ?? null,
      subproblemItems,
      subproblemSections,
    ),
    createProfilingSection("Line Search", lineSearchTotal?.durationMs ?? null, lineSearchItems),
    createProfilingSection("Convergence", convergenceTotal?.durationMs ?? null, convergenceItems),
    createProfilingSection("Accounting", null, accountingItems),
    createProfilingSection("Other Solve Timing", null, otherSolveItems),
  ].filter((section): section is ProfilingSection => section != null);

  return [
    createProfilingSection("Setup", null, [], setupSections),
    createProfilingSection(
      "Solve",
      chooseExplicitProfilingDuration(
        solver.solve_s == null ? null : solver.solve_s * 1000,
        solveTotal?.durationMs ?? null,
      ),
      [],
      solveSections,
    ),
  ].filter((section): section is ProfilingSection => section != null);
}

function appendSolveProfilingLog(solver: SolverReport | null | undefined): void {
  if (!solver) {
    return;
  }
  const sections = buildSolveProfilingSections(solver);
  if (sections.length === 0) {
    return;
  }
  const rootTotalMs = sumProfilingDurations(sections);
  const renderedBlocks = sections.map((section, index) => {
    const sectionHeatScore = profilingShareHeatScore(rootTotalMs, section.durationMs);
    return flattenProfilingSection(
      section,
      [],
      index === sections.length - 1,
      sectionHeatScore,
      null,
      true,
    );
  });
  const renderedLines = renderedBlocks.flat();
  const timeWidth = renderedLines.reduce((width, line) => Math.max(width, line.value.length), 0);
  const labelWidth = renderedLines.reduce((width, line) => Math.max(width, line.label.length), 0);
  const metadataWidth = renderedLines.reduce(
    (width, line) => Math.max(width, (line.metadata ?? "").length),
    0,
  );
  appendLogLine("solve profiling:", LOG_LEVEL.console);
  for (const [blockIndex, block] of renderedBlocks.entries()) {
    if (blockIndex > 0) {
      appendLogLine("", LOG_LEVEL.console);
    }
    for (const line of block) {
      const paddedLabel = line.label.padEnd(labelWidth, " ");
      const paddedTime = line.value.padStart(timeWidth, " ");
      const coloredTime = colorizeSolveProfilingTime(paddedTime, line.heatScore);
      const metadata =
        line.metadata == null ? "" : `  ${line.metadata.padStart(metadataWidth, " ")}`;
      appendLogLine(`  ${paddedLabel}  ${coloredTime}${metadata}`, LOG_LEVEL.console);
    }
  }
}

function solverPhaseDetailValue(
  solver: SolverReport | null | undefined,
  label: string,
): string | null {
  const detail = solver?.phase_details.solve.find((entry) => entry.label === label);
  return detail?.value ?? null;
}

function solveSummaryItems(
  progress: SolveProgress | null,
  solver: SolverReport | null,
): SolveSummaryItem[] {
  const items: SolveSummaryItem[] = [];
  if (progress) {
    const tfMetric = findMetric(state.artifact, METRIC_KEY.finalTime);
    items.push({ label: "Iteration", value: `${progress.iteration}` });
    items.push({ label: "Phase", value: phaseLabel(progress.phase) });
    items.push({ label: "Objective", value: progress.objective.toExponential(3) });
    items.push({ label: "T", value: tfMetric?.value ?? "--" });
    items.push({
      label: EQ_INF_LABEL,
      value: progress.eq_inf == null ? "--" : progress.eq_inf.toExponential(3),
    });
    items.push({
      label: INEQ_INF_LABEL,
      value: progress.ineq_inf == null ? "--" : progress.ineq_inf.toExponential(3),
    });
    items.push({ label: DUAL_INF_LABEL, value: progress.dual_inf.toExponential(3) });
    items.push({
      label: STEP_INF_LABEL,
      value: progress.step_inf == null ? "--" : progress.step_inf.toExponential(3),
    });
    if (progress.alpha_pr != null || progress.alpha_du != null) {
      items.push({
        label: "α_pr",
        value: progress.alpha_pr == null ? "--" : progress.alpha_pr.toExponential(3),
      });
      items.push({
        label: "α_du",
        value: progress.alpha_du == null ? "--" : progress.alpha_du.toExponential(3),
      });
    } else {
      items.push({ label: "α", value: progress.alpha == null ? "--" : progress.alpha.toExponential(3) });
    }
    return items;
  }

  if (solver?.iterations != null) {
    items.push({ label: "Iteration", value: `${solver.iterations}` });
    const tfMetric = findMetric(state.artifact, METRIC_KEY.finalTime);
    if (tfMetric) {
      items.push({ label: "T", value: tfMetric.value });
    }
  }

  if (solver?.status_kind === SOLVER_STATUS_KIND.error) {
    const detailPairs: Array<[string, string]> = [
      ["Grad", "Gradient"],
      ["Eq Jac", "Equality Jacobian"],
      ["Ineq Jac", "Inequality Jacobian"],
      ["Hess", "Hessian"],
      ["Linear", "Linear Solve"],
    ];
    for (const [chipLabel, detailLabel] of detailPairs) {
      const value = solverPhaseDetailValue(solver, detailLabel);
      if (value) {
        items.push({ label: chipLabel, value });
      }
    }
  }

  return items;
}

function createPhaseDetailsGrid(
  details: SolverPhaseDetail[],
  fallbackText: string,
): HTMLElement {
  if (details.length === 0) {
    const note = document.createElement("div");
    note.className = "solver-phase-note";
    note.textContent = fallbackText;
    return note;
  }

  const grid = document.createElement("div");
  grid.className = "solver-phase-detail-grid";
  for (const detail of details) {
    if (isLongSolverPhaseDetail(detail)) {
      grid.appendChild(createLongSolverPhaseDetail(detail));
      continue;
    }

    const item = document.createElement("div");
    item.className = "solver-phase-detail";

    const label = document.createElement("div");
    label.className = "solver-phase-detail-label";
    label.textContent = detail.label;
    item.appendChild(label);

    const value = document.createElement("div");
    value.className = "solver-phase-detail-value";
    value.textContent = detail.value;
    item.appendChild(value);

    grid.appendChild(item);
  }
  return grid;
}

function isLongSolverPhaseDetail(detail: SolverPhaseDetail): boolean {
  return detail.label === "Settings" || detail.value.length > 80 || detail.value.includes("\n");
}

function solverDetailPreview(value: string): string {
  const compact = value.replace(/\s+/g, " ").trim();
  if (compact.length <= 72) {
    return compact;
  }
  return `${compact.slice(0, 69).trimEnd()}...`;
}

function createLongSolverPhaseDetail(detail: SolverPhaseDetail): HTMLElement {
  const item = document.createElement("details");
  item.className = "solver-phase-detail solver-phase-detail-wide solver-phase-detail-collapsible";

  const summary = document.createElement("summary");
  summary.className = "solver-phase-detail-summary";

  const label = document.createElement("span");
  label.className = "solver-phase-detail-label";
  label.textContent = detail.label;
  summary.appendChild(label);

  const preview = document.createElement("span");
  preview.className = "solver-phase-detail-preview";
  preview.textContent = solverDetailPreview(detail.value);
  summary.appendChild(preview);

  item.appendChild(summary);

  const value = document.createElement("pre");
  value.className = "solver-phase-detail-code";
  value.textContent = detail.value;
  item.appendChild(value);

  return item;
}

function createSolverPhaseCard(options: {
  label: string;
  value: string;
  active: boolean;
  details: SolverPhaseDetail[];
  fallbackText: string;
}): HTMLElement {
  const card = document.createElement("article");
  card.className = `solver-phase-card ${options.active ? "solver-phase-card-active" : ""}`.trim();

  const head = document.createElement("div");
  head.className = "solver-phase-head";

  const label = document.createElement("div");
  label.className = "solver-phase-label";
  label.textContent = options.label;
  head.appendChild(label);

  const value = document.createElement("div");
  value.className = `solver-phase-time ${options.active ? "solver-phase-time-active" : ""}`.trim();
  if (options.active) {
    const spinner = document.createElement("span");
    spinner.className = "metric-spinner";
    spinner.setAttribute("aria-hidden", "true");
    value.appendChild(spinner);
  }
  value.append(document.createTextNode(options.value));
  head.appendChild(value);

  card.appendChild(head);
  card.appendChild(createPhaseDetailsGrid(options.details, options.fallbackText));
  return card;
}

function createSolverRunCard(
  progress: SolveProgress | null,
  solver: SolverReport,
): HTMLElement {
  const activeStage = currentSolveStage();
  const card = document.createElement("article");
  card.className = `solver-run-card ${activeStage === SOLVE_STAGE.solving ? "solver-phase-card-active" : ""}`.trim();

  const head = document.createElement("div");
  head.className = "solver-phase-head";

  const label = document.createElement("div");
  label.className = "solver-phase-label";
  label.textContent = "Solve";
  head.appendChild(label);

  const value = document.createElement("div");
  value.className = `solver-phase-time ${activeStage === SOLVE_STAGE.solving ? "solver-phase-time-active" : ""}`.trim();
  if (activeStage === SOLVE_STAGE.solving) {
    const spinner = document.createElement("span");
    spinner.className = "metric-spinner";
    spinner.setAttribute("aria-hidden", "true");
    value.appendChild(spinner);
  }
  value.append(document.createTextNode(formatDuration(solver.solve_s ?? null)));
  head.appendChild(value);
  card.appendChild(head);

  const items = solveSummaryItems(progress, solver);
  if (items.length === 0) {
    const note = document.createElement("div");
    note.className = "solver-phase-note";
    note.textContent = state.solving
      ? "Iteration diagnostics will appear once the nonlinear solve begins."
      : "No iteration diagnostics were produced for the last run.";
    card.appendChild(note);
    return card;
  }

  const grid = document.createElement("div");
  grid.className = "solver-summary-grid";
  for (const item of items) {
    grid.appendChild(createSolverSummaryChip(item));
  }
  card.appendChild(grid);
  return card;
}

function renderSolverPhaseSummary(solver: SolverReport): HTMLElement {
  const activeStage = currentSolveStage();
  const grid = document.createElement("div");
  grid.className = "solver-phase-grid";
  grid.append(
    createSolverPhaseCard({
      label: "Symbolic Setup",
      value: formatCompileDuration(solver.symbolic_setup_s ?? null),
      active: activeStage === SOLVE_STAGE.symbolicSetup,
      details: solver.phase_details.symbolic_setup,
      fallbackText: "Building symbolic model and derivatives.",
    }),
    createSolverPhaseCard({
      label: "JIT",
      value: formatJitDurationWithOutcome(solver.jit_s ?? null, cacheOutcomeForSolver(solver)),
      active: activeStage === SOLVE_STAGE.jitCompilation,
      details: solver.phase_details.jit,
      fallbackText: "Compiling numeric evaluation kernels.",
    }),
    createSolverPhaseCard({
      label: "Solve",
      value: formatDuration(solver.solve_s ?? null),
      active: activeStage === SOLVE_STAGE.solving,
      details: solver.phase_details.solve,
      fallbackText: "Running optimization iterations.",
    }),
  );
  return grid;
}

function renderSolverSummary(): void {
  const progress = state.latestProgress;
  const solver = currentSolverReport();
  if (!progress && !solver) {
    solverSummaryEl.innerHTML = `<div class="placeholder">Solve a problem to populate solver diagnostics.</div>`;
    return;
  }

  solverSummaryEl.innerHTML = "";
  if (solver) {
    solverSummaryEl.appendChild(renderSolverPhaseSummary(solver));
    solverSummaryEl.appendChild(createSolverRunCard(progress, solver));
  }
}

function formatConstraintValue(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) {
    return "--";
  }
  const abs = Math.abs(value);
  if (abs >= 1e3 || (abs > 0 && abs < 1e-2)) {
    return Number(value).toExponential(3);
  }
  return fmt(value, 3);
}

function formatConstraintSummaryValue(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) {
    return "--";
  }
  const abs = Math.abs(value);
  if (abs >= 1e3 || (abs > 0 && abs < 1e-2)) {
    return Number(value).toExponential(1);
  }
  return fmt(value, 2);
}

function worstConstraintViolation(entries: ConstraintPanelEntry[]): number | null {
  if (entries.length === 0) {
    return null;
  }
  let worst = 0;
  for (const entry of entries) {
    const magnitude = Math.abs(entry.worst_violation);
    if (magnitude > worst) {
      worst = magnitude;
    }
  }
  return worst;
}

function activeConstraintThreshold(tolerance: number | null | undefined): number {
  const numericTolerance = tolerance ?? Number.NaN;
  if (Number.isFinite(numericTolerance) && numericTolerance > 0) {
    return Math.max(100 * numericTolerance, 1e-8);
  }
  return 1e-8;
}

function isEqualityStyleBoundEntry(entry: ConstraintPanelEntry): boolean {
  const lower = entry.lower_bound;
  const upper = entry.upper_bound;
  if (lower == null || upper == null || !Number.isFinite(lower) || !Number.isFinite(upper)) {
    return false;
  }
  const scale = Math.max(Math.abs(lower), Math.abs(upper), 1);
  return Math.abs(lower - upper) <= 1e-12 * scale;
}

function constraintBoundSideLabel(side: ConstraintPanelBoundSideCode | null | undefined): string {
  switch (side) {
    case CONSTRAINT_PANEL_BOUND_SIDE.lower:
      return "lower";
    case CONSTRAINT_PANEL_BOUND_SIDE.upper:
      return "upper";
    case CONSTRAINT_PANEL_BOUND_SIDE.both:
      return "lower + upper";
    default:
      return "none";
  }
}

function renderBoundToken(
  value: number | null | undefined,
  severity: ConstraintPanelSeverityCode | null | undefined,
  fallback: string,
): string {
  const className = `constraint-bound constraint-bound-${boundSeverityClass(severity)}`;
  const label = value == null ? fallback : formatConstraintValue(value);
  return `<span class="${className}">${label}</span>`;
}

function renderConstraintPanel(
  target: HTMLElement,
  allEntries: ConstraintPanelEntry[],
  entries: ConstraintPanelEntry[],
  pendingText: string,
  kind: ConstraintPanelKind,
): void {
  if (!entries || entries.length === 0) {
    const worstViolation = worstConstraintViolation(allEntries);
    if (worstViolation == null) {
      target.innerHTML = `<div class="placeholder">${pendingText}</div>`;
      return;
    }
    if (kind === "active") {
      target.innerHTML = `
        <article class="constraint-entry constraint-entry-success constraint-entry-summary">
          <div class="constraint-entry-inline">
            <span class="constraint-inline-label">Active</span>
            <span class="constraint-inline-value">0</span>
          </div>
        </article>
      `;
      return;
    }
    if (kind === "ineq") {
      target.innerHTML = `
        <article class="constraint-entry constraint-entry-success constraint-entry-summary">
          <div class="constraint-entry-inline">
            <span class="constraint-inline-label">Violations</span>
            <span class="constraint-inline-value">0</span>
          </div>
        </article>
      `;
      return;
    }
    target.innerHTML = `
      <article class="constraint-entry constraint-entry-success constraint-entry-summary">
        <div class="constraint-entry-inline">
          <span class="constraint-inline-label">Worst Violation</span>
          <span class="constraint-inline-value">${formatConstraintSummaryValue(worstViolation)}</span>
        </div>
      </article>
    `;
    return;
  }
  target.innerHTML = entries
    .map((entry) => {
      const isActivePanel = kind === "active";
      const severityKind = isActivePanel ? "warning" : constraintSeverityClass(entry.severity);
      const severityClass =
        severityKind === "success" ? "" : `constraint-entry-${severityKind}`;
      const countMarkup = isActivePanel
        ? `active ${entry.active_instances ?? 0}/${entry.total_instances}`
        : `viol ${entry.violating_instances}/${entry.total_instances}`;
      const primaryLabel = isActivePanel ? "Nearest" : "Worst";
      const primaryValue = isActivePanel
        ? formatConstraintSummaryValue(entry.min_active_margin)
        : entry.worst_violation.toExponential(3);
      const side = isActivePanel ? entry.active_bound_side : entry.bound_side;
      const sideText = constraintBoundSideLabel(side);
      const sideMarkup =
        kind === "eq" || sideText === "none"
          ? ""
          : `
          <div class="constraint-entry-inline">
            <span class="constraint-inline-label">${isActivePanel ? "Active Bound" : "Violated Bound"}</span>
            <span class="constraint-inline-value">${sideText}</span>
          </div>`;
      const boundsMarkup =
        kind !== "eq"
          ? `
          <div class="constraint-entry-inline">
            <span class="constraint-inline-label">Bounds</span>
            ${renderBoundToken(entry.lower_bound, entry.lower_severity, "−∞")}
            <span class="constraint-bound-sep">…</span>
            ${renderBoundToken(entry.upper_bound, entry.upper_severity, "+∞")}
          </div>`
          : "";
      return `
        <article class="constraint-entry ${severityClass}">
          <div class="constraint-entry-top">
            <div class="constraint-entry-label">${entry.label}</div>
            <div class="constraint-entry-count">${countMarkup}</div>
          </div>
          <div class="constraint-entry-inline">
            <span class="constraint-inline-label">${primaryLabel}</span>
            <span class="constraint-inline-value">${primaryValue}</span>
          </div>
          ${sideMarkup}
          ${boundsMarkup}
        </article>
      `;
    })
    .join("");
}

function renderConstraintPanels(): void {
  const panels = state.artifact?.constraint_panels ?? { equalities: [], inequalities: [] };
  const allEqualities = panels.equalities;
  const allInequalities = panels.inequalities;
  const activeEqualities = allEqualities.filter((entry) => entry.violating_instances > 0);
  const activeInequalities = allInequalities.filter((entry) => entry.violating_instances > 0);
  const activeBoundConstraints = allInequalities
    .filter((entry) => (entry.active_instances ?? 0) > 0 && !isEqualityStyleBoundEntry(entry))
    .sort((lhs, rhs) => (lhs.min_active_margin ?? Number.POSITIVE_INFINITY)
      - (rhs.min_active_margin ?? Number.POSITIVE_INFINITY));
  const toleranceValue = currentSharedControlValue(
    CONTROL_SEMANTIC.solverConstraintTolerance,
    Number.NaN,
  );
  const activeToleranceText = formatConstraintSummaryValue(activeConstraintThreshold(toleranceValue));
  const toleranceText = formatSharedControlValue(
    CONTROL_SEMANTIC.solverConstraintTolerance,
  );
  const pendingText = state.artifact == null && !state.solving
    ? `tol ${toleranceText}`
    : "pending";
  const activePendingText = state.artifact == null && !state.solving
    ? `active ≤ ${activeToleranceText}`
    : "pending";
  renderConstraintPanel(
    eqViolationsEl,
    allEqualities,
    activeEqualities,
    pendingText,
    "eq",
  );
  renderConstraintPanel(
    ineqViolationsEl,
    allInequalities,
    activeInequalities,
    pendingText,
    "ineq",
  );
  renderConstraintPanel(
    activeConstraintsEl,
    allInequalities,
    activeBoundConstraints,
    activePendingText,
    "active",
  );
}

function appendLogLine(line: string, level: LogLevelCode = LOG_LEVEL.console): void {
  const normalized = line.replaceAll("\r\n", "\n").replaceAll("\r", "\n");
  const parts = normalized.split("\n");
  if (parts.length > 1 && parts[parts.length - 1] === "") {
    parts.pop();
  }
  if (parts.length === 0) {
    parts.push("");
  }
  const entries = parts.map((textPart) => ({ text: textPart, level }));
  state.logLines.push(...entries);
  solverLogEl.appendChild(buildLogLineElements(entries));
  syncCopyConsoleButtonAvailability();
  if (state.followSolverLog) {
    scrollConsoleToBottom();
  }
}

function positiveFiniteOrNull(value: number | null | undefined): number | null {
  const numericValue = value ?? Number.NaN;
  return Number.isFinite(numericValue) && numericValue > 0 ? numericValue : null;
}

function hexToRgba(hex: string, alpha: number): string {
  const normalized = hex.trim();
  if (!/^#[0-9a-fA-F]{6}$/.test(normalized)) {
    return hex;
  }
  const red = Number.parseInt(normalized.slice(1, 3), 16);
  const green = Number.parseInt(normalized.slice(3, 5), 16);
  const blue = Number.parseInt(normalized.slice(5, 7), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function filterPathMarkerColors(length: number): string[] {
  if (length <= 0) {
    return [];
  }
  if (length === 1) {
    return [hexToRgba(PALETTE[0], 0.9)];
  }
  return Array.from({ length }, (_, index) => {
    const t = index / (length - 1);
    const alpha = 0.08 + 0.72 * t;
    return hexToRgba(PALETTE[0], alpha);
  });
}

function filterPlotRanges(points: readonly FilterEntry[]): {
  xRange: NumericRange;
  yRange: NumericRange;
} {
  const positiveViolations = points
    .map((point) => Math.max(point.violation, 1e-14))
    .filter((value) => Number.isFinite(value) && value > 0);
  const objectives = points
    .map((point) => point.objective)
    .filter((value) => Number.isFinite(value));

  const minViolation = positiveViolations.length > 0 ? Math.min(...positiveViolations) : 1e-14;
  const maxViolation = positiveViolations.length > 0 ? Math.max(...positiveViolations) : 1;
  const minLogViolation = Math.log10(minViolation);
  const maxLogViolation = Math.log10(maxViolation);
  const logSpan = Math.max(maxLogViolation - minLogViolation, 0.35);
  const logPad = 0.14 * logSpan + 0.08;

  const minObjective = objectives.length > 0 ? Math.min(...objectives) : -1;
  const maxObjective = objectives.length > 0 ? Math.max(...objectives) : 1;
  const objectiveSpan = Math.max(maxObjective - minObjective, 1e-6);
  const objectivePad = 0.12 * objectiveSpan + 1e-6;

  return {
    xRange: [minLogViolation - logPad, maxLogViolation + logPad],
    yRange: [minObjective - objectivePad, maxObjective + objectivePad],
  };
}

const PROGRESS_TRACE = Object.freeze({
  objective: 0,
  eqInf: 1,
  ineqInf: 2,
  dualInf: 3,
  eqTol: 4,
  ineqTol: 5,
  dualTol: 6,
});

const TRUST_REGION_TRACE = Object.freeze({
  radius: 0,
  boundaryOff: 1,
  boundaryOn: 2,
});

function toleranceSeverity(
  value: number | null | undefined,
  tolerance: number | null | undefined,
): ConstraintPanelSeverityCode {
  const numericValue = value ?? Number.NaN;
  const numericTolerance = tolerance ?? Number.NaN;
  if (!Number.isFinite(numericValue) || !Number.isFinite(numericTolerance) || numericTolerance <= 0) {
    return CONSTRAINT_PANEL_SEVERITY.fullAccuracy;
  }
  if (numericValue <= numericTolerance) {
    return CONSTRAINT_PANEL_SEVERITY.fullAccuracy;
  }
  if (numericValue <= 100 * numericTolerance) {
    return CONSTRAINT_PANEL_SEVERITY.reducedAccuracy;
  }
  return CONSTRAINT_PANEL_SEVERITY.violated;
}

function toleranceStatusLabel(severity: ConstraintPanelSeverityCode): string {
  switch (severity) {
    case CONSTRAINT_PANEL_SEVERITY.reducedAccuracy:
      return "Satisfied to reduced accuracy";
    case CONSTRAINT_PANEL_SEVERITY.violated:
      return "Not satisfied";
    default:
      return "Satisfied to full accuracy";
  }
}

function toleranceMarkerSymbol(severity: ConstraintPanelSeverityCode): string {
  switch (severity) {
    case CONSTRAINT_PANEL_SEVERITY.reducedAccuracy:
      return "diamond-open";
    case CONSTRAINT_PANEL_SEVERITY.violated:
      return "x-open";
    default:
      return "circle-open";
  }
}

function toleranceTraceOpacity(severity: ConstraintPanelSeverityCode): number {
  switch (severity) {
    case CONSTRAINT_PANEL_SEVERITY.reducedAccuracy:
      return 0.72;
    case CONSTRAINT_PANEL_SEVERITY.violated:
      return 1.0;
    default:
      return 0.35;
  }
}

function progressToleranceTrace(name: string, color: string, dash: string): PlotlyTrace {
  return {
    type: "scatter",
    mode: "lines+markers",
    name,
    x: [],
    y: [],
    showlegend: false,
    line: {
      color,
      width: PLOT_LINE_WIDTH.tolerance,
      dash,
    },
    marker: {
      size: [],
      symbol: [],
      color,
      line: {
        color,
        width: PLOT_LINE_WIDTH.tolerance,
      },
    },
    opacity: 0.35,
    hovertemplate: `${name}<br>tol=%{y:.3e}<br>status=%{customdata}<extra></extra>`,
  };
}

function updateProgressThresholdTrace(
  traceIndex: number,
  name: string,
  color: string,
  dash: string,
  tolerance: number | null,
  currentValue: number | null,
  maxX: number,
): void {
  if (!window.Plotly) {
    return;
  }
  if (tolerance == null) {
    window.Plotly.restyle(
      progressPlotEl,
      {
        x: [[]],
        y: [[]],
        "marker.size": [[]],
        "marker.symbol": [[]],
        customdata: [[]],
      },
      [traceIndex],
    );
    return;
  }
  const severity = toleranceSeverity(currentValue, tolerance);
  const statusText = toleranceStatusLabel(severity);
  window.Plotly.restyle(
    progressPlotEl,
    {
      x: [[0, maxX]],
      y: [[tolerance, tolerance]],
      opacity: [toleranceTraceOpacity(severity)],
      "line.color": [color],
      "line.width": [1],
      "line.dash": [dash],
      "marker.size": [[0, 9]],
      "marker.symbol": [["circle-open", toleranceMarkerSymbol(severity)]],
      "marker.color": [color],
      "marker.line.color": [color],
      "marker.line.width": [1],
      customdata: [[statusText, statusText]],
      hovertemplate: [`${name}<br>tol=%{y:.3e}<br>status=%{customdata}<extra></extra>`],
    },
    [traceIndex],
  );
}

function updateProgressThresholds(progress: SolveProgress | null | undefined): void {
  if (!window.Plotly) {
    return;
  }
  const maxX = Math.max(1, Number(progress?.iteration ?? 0));
  const constraintTol = positiveFiniteOrNull(
    currentSharedControlValue(CONTROL_SEMANTIC.solverConstraintTolerance, Number.NaN),
  );
  const dualTol = positiveFiniteOrNull(
    currentSharedControlValue(CONTROL_SEMANTIC.solverDualTolerance, Number.NaN),
  );
  updateProgressThresholdTrace(
    PROGRESS_TRACE.eqTol,
    `${EQ_INF_LABEL} threshold`,
    PALETTE[1],
    "dot",
    constraintTol,
    residualValue(progress?.eq_inf),
    maxX,
  );
  updateProgressThresholdTrace(
    PROGRESS_TRACE.ineqTol,
    `${INEQ_INF_LABEL} threshold`,
    PALETTE[2],
    "longdash",
    constraintTol,
    residualValue(progress?.ineq_inf),
    maxX,
  );
  updateProgressThresholdTrace(
    PROGRESS_TRACE.dualTol,
    `${DUAL_INF_LABEL} threshold`,
    PALETTE[3],
    "dashdot",
    dualTol,
    residualValue(progress?.dual_inf),
    maxX,
  );
}

function ensureProgressPlot(): void {
  if (state.progressPlotReady || !window.Plotly) {
    return;
  }
  progressPlotEl.innerHTML = "";
  const data = [
    {
      type: "scatter",
      mode: "lines+markers",
      name: "Objective",
      x: [],
      y: [],
      yaxis: "y2",
      line: { color: PALETTE[0], width: PLOT_LINE_WIDTH.primary },
      marker: { size: PLOT_MARKER_SIZE.primary },
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: EQ_INF_LABEL,
      x: [],
      y: [],
      line: { color: PALETTE[1], width: PLOT_LINE_WIDTH.primary },
      marker: { size: PLOT_MARKER_SIZE.primary },
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: INEQ_INF_LABEL,
      x: [],
      y: [],
      line: { color: PALETTE[2], width: PLOT_LINE_WIDTH.primary },
      marker: { size: PLOT_MARKER_SIZE.primary },
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: DUAL_INF_LABEL,
      x: [],
      y: [],
      line: { color: PALETTE[3], width: PLOT_LINE_WIDTH.primary },
      marker: { size: PLOT_MARKER_SIZE.primary },
    },
    progressToleranceTrace(`${EQ_INF_LABEL} threshold`, PALETTE[1], "dot"),
    progressToleranceTrace(`${INEQ_INF_LABEL} threshold`, PALETTE[2], "longdash"),
    progressToleranceTrace(`${DUAL_INF_LABEL} threshold`, PALETTE[3], "dashdot"),
  ];
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(4, 15, 22, 0.92)",
    font: {
      color: "#e5f1f4",
      family: '"Avenir Next", Futura, "Trebuchet MS", sans-serif',
      size: 12,
    },
    margin: { l: 74, r: 72, t: 18, b: 58 },
    legend: {
      orientation: "h",
      y: -0.28,
      x: 0,
      font: { color: "#94b6bd", size: 11 },
    },
    xaxis: {
      title: "Iteration (-)",
      gridcolor: "rgba(229, 241, 244, 0.08)",
      linecolor: "rgba(177, 214, 222, 0.18)",
      zeroline: false,
      ticks: "outside",
      titlefont: { color: "#94b6bd" },
    },
    yaxis: {
      title: "Residual / violation (∞-norm)",
      type: "log",
      gridcolor: "rgba(229, 241, 244, 0.08)",
      linecolor: "rgba(177, 214, 222, 0.18)",
      zeroline: false,
      ticks: "outside",
      titlefont: { color: "#94b6bd" },
    },
    yaxis2: {
      title: "Objective (-)",
      overlaying: "y",
      side: "right",
      showgrid: false,
      linecolor: "rgba(177, 214, 222, 0.18)",
      ticks: "outside",
      titlefont: { color: "#94b6bd" },
    },
  };
  const config = {
    responsive: true,
    displaylogo: false,
    displayModeBar: false,
  };
  window.Plotly.newPlot(progressPlotEl, data, layout, config);
  state.progressPlotReady = true;
}

function positiveLogValue(value: number | null | undefined): number | null {
  if (value == null || !Number.isFinite(value)) {
    return null;
  }
  return Math.max(value, 1.0e-14);
}

function residualValue(value: number | null | undefined): number | null {
  return positiveLogValue(value);
}

function updateProgressPlot(progress: SolveProgress): void {
  if (!window.Plotly) {
    return;
  }
  ensureProgressPlot();
  const iteration = progress.iteration;
  window.Plotly.extendTraces(
    progressPlotEl,
    {
      x: [[iteration], [iteration], [iteration], [iteration]],
      y: [
        [progress.objective],
        [residualValue(progress.eq_inf)],
        [residualValue(progress.ineq_inf)],
        [residualValue(progress.dual_inf)],
      ],
    },
    [0, 1, 2, 3],
  );
  updateProgressThresholds(progress);
}

function trustRegionHoverText(
  progress: SolveProgress,
  trustRegion: SolveTrustRegionInfo,
): string {
  const acceptedStepNorm = positiveLogValue(trustRegion.step_norm);
  const attemptedStepNorm = positiveLogValue(trustRegion.largest_attempted_step_norm);
  const stepInf = positiveLogValue(progress.step_inf);
  const acceptedRatio =
    acceptedStepNorm != null && trustRegion.radius > 0
      ? Math.min(Math.max(acceptedStepNorm / trustRegion.radius, 0), 1)
      : null;
  const lines = [
    TRUST_REGION_RADIUS_LABEL,
    `iter=${progress.iteration}`,
    `radius=${trustRegion.radius.toExponential(3)}`,
    `accepted step=${acceptedStepNorm == null ? "--" : acceptedStepNorm.toExponential(3)}`,
    `largest attempted=${attemptedStepNorm == null ? "--" : attemptedStepNorm.toExponential(3)}`,
    `accepted / radius=${acceptedRatio == null ? "--" : acceptedRatio.toFixed(2)}`,
    `attempted radius=${trustRegion.attempted_radius.toExponential(3)}`,
    `contractions=${trustRegion.contraction_count}`,
    `qp retries=${trustRegion.qp_failure_retries}`,
    `boundary=${trustRegion.boundary_active ? "yes" : "no"}`,
    `restoration=${trustRegion.restoration_attempted ? "yes" : "no"}`,
    `elastic recovery=${trustRegion.elastic_recovery_attempted ? "yes" : "no"}`,
  ];
  if (stepInf != null) {
    lines.push(`step inf=${stepInf.toExponential(3)}`);
  }
  return lines.join("<br>");
}

function trustRegionMarkerSymbol(trustRegion: SolveTrustRegionInfo): string {
  if (trustRegion.elastic_recovery_attempted) {
    return "x";
  }
  if (trustRegion.restoration_attempted) {
    return "diamond";
  }
  return "circle";
}

function trustRegionMarkerSize(trustRegion: SolveTrustRegionInfo): number {
  const acceptedStepNorm = positiveLogValue(trustRegion.step_norm);
  if (acceptedStepNorm == null || !Number.isFinite(trustRegion.radius) || trustRegion.radius <= 0) {
    return 7;
  }
  const ratio = Math.min(Math.max(acceptedStepNorm / trustRegion.radius, 0), 1);
  return 6 + 8 * ratio;
}

function ensureTrustRegionPlot(): void {
  if (state.trustRegionPlotReady || !window.Plotly) {
    return;
  }
  trustRegionPlotEl.innerHTML = "";
  const data = [
    {
      type: "scatter",
      mode: "lines",
      name: TRUST_REGION_RADIUS_LABEL,
      x: [],
      y: [],
      line: { color: PALETTE[0], width: PLOT_LINE_WIDTH.primary },
      hoverinfo: "skip",
    },
    {
      type: "scatter",
      mode: "markers",
      name: "Boundary off",
      x: [],
      y: [],
      text: [],
      marker: {
        size: [],
        symbol: [],
        color: PALETTE[1],
        line: { color: "rgba(229, 241, 244, 0.22)", width: PLOT_LINE_WIDTH.tolerance },
      },
      hovertemplate: "%{text}<extra></extra>",
    },
    {
      type: "scatter",
      mode: "markers",
      name: "Boundary on",
      x: [],
      y: [],
      text: [],
      marker: {
        size: [],
        symbol: [],
        color: PALETTE[3],
        line: { color: "rgba(229, 241, 244, 0.22)", width: PLOT_LINE_WIDTH.tolerance },
      },
      hovertemplate: "%{text}<extra></extra>",
    },
  ];
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(4, 15, 22, 0.92)",
    font: {
      color: "#e5f1f4",
      family: '"Avenir Next", Futura, "Trebuchet MS", sans-serif',
      size: 12,
    },
    margin: { l: 74, r: 24, t: 18, b: 58 },
    legend: {
      orientation: "h",
      y: -0.28,
      x: 0,
      font: { color: "#94b6bd", size: 11 },
    },
    xaxis: {
      title: "Iteration (-)",
      gridcolor: "rgba(229, 241, 244, 0.08)",
      linecolor: "rgba(177, 214, 222, 0.18)",
      zeroline: false,
      ticks: "outside",
      titlefont: { color: "#94b6bd" },
    },
    yaxis: {
      title: "Radius / step size (2-norm)",
      type: "log",
      gridcolor: "rgba(229, 241, 244, 0.08)",
      linecolor: "rgba(177, 214, 222, 0.18)",
      zeroline: false,
      ticks: "outside",
      titlefont: { color: "#94b6bd" },
    },
    annotations: [
      {
        text: "Radius line · color = boundary · size = accepted/radius · diamond/x = restoration/elastic",
        xref: "paper",
        yref: "paper",
        x: 0,
        y: 1.12,
        showarrow: false,
        font: { color: "#94b6bd", size: 12 },
      },
    ],
  };
  const config = {
    responsive: true,
    displaylogo: false,
    displayModeBar: false,
  };
  window.Plotly.newPlot(trustRegionPlotEl, data, layout, config);
  state.trustRegionPlotReady = true;
}

function updateTrustRegionPlot(progress: SolveProgress): void {
  if (!window.Plotly) {
    return;
  }
  const trustRegion = progress.trust_region;
  if (!trustRegion) {
    renderTrustRegionPlotVisibility();
    return;
  }
  trustRegionPlotEl.hidden = false;
  ensureTrustRegionPlot();
  const iteration = progress.iteration;
  const radius = positiveLogValue(trustRegion.radius);
  const boundaryTrace = trustRegion.boundary_active
    ? TRUST_REGION_TRACE.boundaryOn
    : TRUST_REGION_TRACE.boundaryOff;
  window.Plotly.extendTraces(trustRegionPlotEl, {
    x: [[iteration]],
    y: [[radius]],
  }, [TRUST_REGION_TRACE.radius]);
  window.Plotly.extendTraces(
    trustRegionPlotEl,
    {
      x: [[iteration]],
      y: [[radius]],
      text: [[trustRegionHoverText(progress, trustRegion)]],
      "marker.size": [[trustRegionMarkerSize(trustRegion)]],
      "marker.symbol": [[trustRegionMarkerSymbol(trustRegion)]],
    },
    [boundaryTrace],
  );
}

function filterPointKey(progress: SolveProgress, filter: FilterInfo): string {
  return [
    progress.iteration,
    progress.phase,
    filter.current.violation.toExponential(6),
    filter.current.objective.toExponential(6),
  ].join(":");
}

function ensureFilterPlot(): void {
  if (state.filterPlotReady || !window.Plotly) {
    return;
  }
  filterPlotEl.innerHTML = "";
  const data = [
    {
      type: "scatter",
      mode: "markers",
      name: "Accepted history",
      showlegend: false,
      x: [],
      y: [],
      marker: {
        size: 2,
        color: hexToRgba(PALETTE[0], 0.14),
      },
    },
    {
      type: "scatter",
      mode: "markers",
      name: "Accepted path",
      x: [],
      y: [],
      marker: {
        size: 2,
        color: [],
      },
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: "Filter frontier",
      x: [],
      y: [],
      line: { color: PALETTE[1], width: PLOT_LINE_WIDTH.secondary, dash: "solid" },
      marker: {
        size: 3,
        color: "#2f9f89",
        line: { width: 0 },
      },
    },
    {
      type: "scatter",
      mode: "markers",
      name: "Current iterate",
      x: [],
      y: [],
      marker: {
        size: 4,
        color: PALETTE[3],
        line: { width: 0 },
      },
    },
  ];
  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(4, 15, 22, 0.92)",
    font: {
      color: "#e5f1f4",
      family: '"Avenir Next", Futura, "Trebuchet MS", sans-serif',
      size: 12,
    },
    margin: { l: 74, r: 24, t: 18, b: 58 },
    legend: {
      orientation: "h",
      y: -0.28,
      x: 0,
      font: { color: "#94b6bd", size: 11 },
    },
    xaxis: {
      title: "Violation (∞-norm)",
      type: "log",
      gridcolor: "rgba(229, 241, 244, 0.08)",
      linecolor: "rgba(177, 214, 222, 0.18)",
      zeroline: false,
      ticks: "outside",
      titlefont: { color: "#94b6bd" },
    },
    yaxis: {
      title: "Filter objective (-)",
      gridcolor: "rgba(229, 241, 244, 0.08)",
      linecolor: "rgba(177, 214, 222, 0.18)",
      zeroline: false,
      ticks: "outside",
      titlefont: { color: "#94b6bd" },
    },
    annotations: [
      {
        text: "Filter frontier",
        xref: "paper",
        yref: "paper",
        x: 0,
        y: 1.12,
        showarrow: false,
        font: { color: "#94b6bd", size: 12 },
      },
    ],
  };
  const config = {
    responsive: true,
    displaylogo: false,
    displayModeBar: false,
  };
  window.Plotly.newPlot(filterPlotEl, data, layout, config);
  state.filterPlotReady = true;
}

function updateFilterPlot(progress: SolveProgress): void {
  if (!window.Plotly) {
    return;
  }
  const filter = progress.filter;
  if (!filter) {
    if (window.Plotly && state.filterPlotReady) {
      window.Plotly.purge(filterPlotEl);
    }
    filterPlotEl.innerHTML = `<div class="placeholder">Filter telemetry will appear here when the active solver is using filter globalization.</div>`;
    state.filterPlotReady = false;
    state.filterRecentPath = [];
    state.lastFilterPointKey = null;
    return;
  }
  ensureFilterPlot();
  const pointKey = filterPointKey(progress, filter);
  if (state.lastFilterPointKey !== pointKey) {
    const acceptedPoint = {
      violation: Math.max(filter.current.violation, 1e-14),
      objective: filter.current.objective,
    };
    state.filterRecentPath.push(acceptedPoint);
    if (state.filterRecentPath.length > FILTER_RECENT_POINT_LIMIT) {
      state.filterRecentPath.shift();
    }
    state.lastFilterPointKey = pointKey;
    window.Plotly.extendTraces(
      filterPlotEl,
      {
        x: [[acceptedPoint.violation]],
        y: [[acceptedPoint.objective]],
      },
      [FILTER_TRACE.history],
    );
  }
  const recentPathX = state.filterRecentPath.map((entry) => entry.violation);
  const recentPathY = state.filterRecentPath.map((entry) => entry.objective);
  const acceptedPathColors = filterPathMarkerColors(state.filterRecentPath.length);
  window.Plotly.restyle(
    filterPlotEl,
    {
      x: [recentPathX],
      y: [recentPathY],
      "marker.color": [acceptedPathColors],
    },
    [FILTER_TRACE.recent],
  );
  const frontier = [...filter.entries].sort((lhs, rhs) => lhs.violation - rhs.violation);
  const acceptanceLabel = filter.accepted_mode == null
    ? "current frontier"
    : filter.accepted_mode === FILTER_ACCEPTANCE_MODE.violationReduction
      ? "accepted via violation reduction"
      : "accepted via objective Armijo";
  window.Plotly.restyle(
    filterPlotEl,
    {
      x: [frontier.map((entry) => Math.max(entry.violation, 1e-14))],
      y: [frontier.map((entry) => entry.objective)],
    },
    [FILTER_TRACE.frontier],
  );
  window.Plotly.restyle(
    filterPlotEl,
    {
      x: [[Math.max(filter.current.violation, 1e-14)]],
      y: [[filter.current.objective]],
    },
    [FILTER_TRACE.current],
  );
  const ranges = filterPlotRanges([...frontier, filter.current]);
  window.Plotly.relayout(filterPlotEl, {
    "annotations[0].text": `${filter.title} · ${acceptanceLabel}`,
    "yaxis.title": `${filter.objective_label} (-)`,
    "xaxis.autorange": false,
    "xaxis.range": ranges.xRange,
    "yaxis.autorange": false,
    "yaxis.range": ranges.yRange,
  });
}

function scheduleIterationUpdate(): void {
  if (state.iterationFlushScheduled) {
    return;
  }
  state.iterationFlushScheduled = true;
  requestAnimationFrame(() => {
    state.iterationFlushScheduled = false;
    const event = state.pendingIterationEvent;
    state.pendingIterationEvent = null;
    if (!event) {
      return;
    }
    applyIterationEvent(event, true);
  });
}

function shouldRenderIterationArtifact(event: IterationSolveEvent, forceArtifactRender: boolean): boolean {
  return forceArtifactRender
    || state.artifact == null
    || event.progress.iteration % ITERATION_ARTIFACT_RENDER_STRIDE === 0;
}

function applyIterationEvent(
  event: IterationSolveEvent,
  updateRunningStatus: boolean,
  forceArtifactRender = false,
): void {
  state.latestProgress = event.progress;
  state.liveSolver = event.artifact.solver;
  renderSolverSummary();
  updateProgressPlot(event.progress);
  updateTrustRegionPlot(event.progress);
  updateFilterPlot(event.progress);
  if (shouldRenderIterationArtifact(event, forceArtifactRender)) {
    state.artifact = event.artifact;
    scheduleArtifactRender(forceArtifactRender);
  }
  if (updateRunningStatus && state.liveStatus?.stage === SOLVE_STAGE.solving) {
    setStatusDisplay(statusDisplayForSolveStatus(state.liveStatus, event.progress.iteration));
  }
}

function applySolveFailure(message: string): void {
  const reportedFailureSolver =
    state.liveSolver?.completed && state.liveSolver.status_kind === SOLVER_STATUS_KIND.error
      ? state.liveSolver
      : null;
  if (state.pendingIterationEvent) {
    const pendingEvent = state.pendingIterationEvent;
    state.pendingIterationEvent = null;
    applyIterationEvent(pendingEvent, false, true);
  }
  state.liveStatus = null;
  state.terminalSolver =
    reportedFailureSolver == null
      ? buildFailureSolverReport(message)
      : mergeSolverReport(
          {
            ...reportedFailureSolver,
            failure_message: message,
          },
          reportedFailureSolver,
        );
  state.liveSolver = null;
  renderSolverSummary();
  renderMetrics();
  renderCompileCacheStatus();
  void refreshCompileCacheStatus();
  appendSolveProfilingLog(state.terminalSolver);
  appendLogLine(`error: ${message}`, LOG_LEVEL.error);
  setStatusDisplay(statusDisplayForSolverReport(state.terminalSolver));
}

function applySolveStopped(): void {
  if (state.pendingIterationEvent) {
    const pendingEvent = state.pendingIterationEvent;
    state.pendingIterationEvent = null;
    applyIterationEvent(pendingEvent, false, true);
  }
  state.liveStatus = null;
  state.terminalSolver = buildStoppedSolverReport();
  state.liveSolver = null;
  renderSolverSummary();
  renderMetrics();
  renderCompileCacheStatus();
  void refreshCompileCacheStatus();
  appendLogLine("solve stopped by user", LOG_LEVEL.warning);
  setStatusDisplay(statusDisplayForSolverReport(state.terminalSolver));
}

function typesetMath(root: Element | null | undefined): void {
  if (!root) {
    return;
  }
  if (!window.MathJax?.typesetPromise) {
    pendingMathRoots.add(root);
    scheduleMathTypesetRetry();
    return;
  }
  pendingMathRoots.delete(root);
  if (mathTypesetRetryHandle !== null) {
    window.clearTimeout(mathTypesetRetryHandle);
    mathTypesetRetryHandle = null;
  }
  if (typeof window.MathJax.typesetClear === "function") {
    window.MathJax.typesetClear([root]);
  }
  window.MathJax.typesetPromise([root]).catch(() => {});
}

function scheduleMathTypesetRetry(): void {
  if (mathTypesetRetryHandle !== null || pendingMathRoots.size === 0) {
    return;
  }
  mathTypesetRetryHandle = window.setTimeout(() => {
    mathTypesetRetryHandle = null;
    if (!window.MathJax?.typesetPromise) {
      scheduleMathTypesetRetry();
      return;
    }
    const roots = Array.from(pendingMathRoots);
    pendingMathRoots.clear();
    for (const root of roots) {
      typesetMath(root);
    }
  }, 100);
}

function renderModel(spec: ProblemSpec | null | undefined): void {
  modelEl.innerHTML = "";
  const sections = spec?.math_sections ?? [];
  if (sections.length === 0) {
    modelEl.innerHTML = `<div class="placeholder">This problem does not define model equations yet.</div>`;
    return;
  }

  const container = document.createElement("div");
  container.className = "model-sections";
  for (const section of sections) {
    const block = document.createElement("section");
    block.className = "model-section";

    const title = document.createElement("h3");
    title.className = "model-title";
    title.textContent = section.title;
    block.appendChild(title);

    for (const entry of section.entries) {
      const math = document.createElement("div");
      math.className = "math-entry";
      math.textContent = `\\[${entry}\\]`;
      block.appendChild(math);
    }

    container.appendChild(block);
  }

  modelEl.appendChild(container);
  typesetMath(modelEl);
}

function collectSceneBounds(scene: Scene2D): SceneBounds {
  const points: NumericPoint[] = [];
  for (const path of scene.paths) {
    path.x.forEach((x, index) => {
      const y = path.y[index];
      if (y != null) {
        points.push([x, y]);
      }
    });
  }
  for (const circle of scene.circles) {
    points.push([circle.cx - circle.radius, circle.cy - circle.radius]);
    points.push([circle.cx + circle.radius, circle.cy + circle.radius]);
  }
  for (const arrow of scene.arrows) {
    points.push([arrow.x, arrow.y], [arrow.x + arrow.dx, arrow.y + arrow.dy]);
  }
  if (scene.animation) {
    for (const frame of scene.animation.frames) {
      Object.values(frame.points).forEach((point) => points.push(point));
      frame.segments.forEach((segment) => {
        points.push(segment[0], segment[1]);
      });
    }
  }
  if (points.length === 0) {
    return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  }
  let minX = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const [x, y] of points) {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const dx = Math.max(maxX - minX, 1e-6);
  const dy = Math.max(maxY - minY, 1e-6);
  const padX = 0.12 * dx + 1e-6;
  const padY = 0.12 * dy + 1e-6;
  return {
    minX: minX - padX,
    maxX: maxX + padX,
    minY: minY - padY,
    maxY: maxY + padY,
  };
}

function renderScene(): void {
  const visualization = primarySceneVisualization(state.artifact);
  if (visualization) {
    sceneSubtitleEl.textContent = visualization.title;
    if (!window.Plotly) {
      state.sceneView = null;
      sceneEl.innerHTML = `<div class="placeholder">Plotly is still loading.</div>`;
      return;
    }
    if (
      !state.sceneView
      || state.sceneView.kind !== "paths_3d"
      || state.sceneView.visualization.title !== visualization.title
    ) {
      state.sceneView = createScene3DView(visualization);
      sceneEl.replaceChildren(state.sceneView.shell);
    }
    const view = state.sceneView;
    if (view.kind !== "paths_3d") {
      return;
    }
    view.visualization = visualization;
    view.meta.textContent = visualizationSubtitle(visualization);
    updatePaths3DVisualization(view, visualization);
    return;
  }

  const scene = state.artifact?.scene;
  sceneSubtitleEl.textContent = scene?.title ?? "";
  if (!scene) {
    state.sceneView = null;
    sceneEl.innerHTML = `<div class="placeholder">Solve a problem to render the semantic scene view.</div>`;
    return;
  }
  if (!window.Plotly) {
    state.sceneView = null;
    sceneEl.innerHTML = `<div class="placeholder">Plotly is still loading.</div>`;
    return;
  }
  if (!state.sceneView || state.sceneView.kind !== "scene_2d" || state.sceneView.scene !== scene) {
    state.sceneView = createSceneView(scene);
    sceneEl.replaceChildren(state.sceneView.shell);
  }

  const view = state.sceneView;
  if (!view || view.kind !== "scene_2d") {
    return;
  }
  view.meta.textContent = `${scene.x_label} · ${scene.y_label}`;
  if (view.playButton) {
    view.playButton.textContent = state.playing ? "Pause" : "Play";
  }
  if (view.slider) {
    const frameCount = scene.animation?.frames.length ?? 0;
    const frameIndex = frameCount > 0 ? Math.min(state.animationIndex, frameCount - 1) : 0;
    view.slider.max = String(Math.max(frameCount - 1, 0));
    view.slider.value = String(frameIndex);
  }
  updateScenePlot(view);
}

function visualizationSubtitle(visualization: ArtifactVisualization): string {
  if (visualization.kind === "paths_3d") {
    return `${visualization.x_label} · ${visualization.y_label} · ${visualization.z_label}`;
  }
  return `${visualization.x_label} · ${visualization.y_label}`;
}

function isPaths3DVisualization(visualization: ArtifactVisualization): visualization is Paths3DVisualization {
  return visualization.kind === "paths_3d";
}

function primarySceneVisualization(artifact: SolveArtifact | null): Paths3DVisualization | null {
  return artifact?.visualizations.find(isPaths3DVisualization) ?? null;
}

function artifactChartPanels(artifact: SolveArtifact | null): ChartPanel[] {
  if (!artifact) {
    return [];
  }
  const sceneVisualization = primarySceneVisualization(artifact);
  const secondaryVisualizations = artifact.visualizations.filter(
    (visualization) => visualization !== sceneVisualization,
  );
  return [
    ...artifact.charts.map((chart, index): ChartPanel => ({
      kind: "chart",
      key: `chart:${index}:${chart.title}`,
      title: chart.title,
      subtitle: chart.y_label,
      chart,
    })),
    ...secondaryVisualizations.map((visualization, index): ChartPanel => ({
      kind: "visualization",
      key: `visualization:${index}:${visualization.kind}:${visualization.title}`,
      title: visualization.title,
      subtitle: visualizationSubtitle(visualization),
      visualization,
    })),
  ];
}

function chartLayoutKey(panels: ChartPanel[]): string {
  return panels.map((panel) => panel.key).join("::");
}

function ensureChartViews(panels: ChartPanel[]): void {
  const nextKey = chartLayoutKey(panels);
  if (state.chartLayoutKey === nextKey && state.chartViews.size === panels.length) {
    return;
  }

  resetChartViews();
  chartsEl.innerHTML = "";
  state.chartLayoutKey = nextKey;

  for (const panel of panels) {
    const isPaths3D = panel.kind === "visualization" && panel.visualization.kind === "paths_3d";
    const shell = document.createElement("section");
    shell.className = "chart-shell";
    if (isPaths3D) {
      shell.classList.add("chart-shell-3d");
    }
    const header = document.createElement("div");
    header.className = "chart-header";
    header.innerHTML = `<div>${panel.title}</div><div class="card-subtitle">${panel.subtitle}</div>`;
    const plotEl = createPlotlyHostElement(
      isPaths3D ? "plot-surface plot-surface-3d" : "plot-surface",
    );
    shell.append(header, plotEl);
    chartsEl.appendChild(shell);
    state.chartViews.set(panel.key, {
      plotEl,
      linkedRangeBound: false,
      linkXRange: panel.kind === "chart",
      chartTraceSignature: null,
    });
  }
}

function buildChartTraces(chart: Chart): ChartTraceBuild {
  const groupOrder = new Map<string, number>();
  const colorIndexFor = (group: string): number => {
    if (!groupOrder.has(group)) {
      groupOrder.set(group, groupOrder.size);
    }
    return groupOrder.get(group)!;
  };
  const signatureTraces: PlotlyValue[] = [];
  const data = chart.series.map((series) => {
    const colorGroup = series.legend_group ?? series.name;
    const paletteIndex = colorIndexFor(colorGroup);
    const role = series.role ?? TIME_SERIES_ROLE.data;
    const isBound = role === TIME_SERIES_ROLE.lowerBound || role === TIME_SERIES_ROLE.upperBound;
    const color = PALETTE[paletteIndex % PALETTE.length];
    const dash = role === TIME_SERIES_ROLE.lowerBound
      ? "dash"
      : role === TIME_SERIES_ROLE.upperBound
        ? "longdash"
        : "solid";
    const mode = series.mode ?? "lines";
    const showlegend = series.show_legend ?? true;
    signatureTraces.push([
      series.name,
      colorGroup,
      role,
      mode,
      showlegend,
      color,
      dash,
      isBound,
    ]);
    return {
      type: "scatter",
      mode,
      name: series.name,
      legendgroup: series.name,
      showlegend,
      x: series.x,
      y: series.y,
      line: {
        color,
        width: isBound
          ? PLOT_LINE_WIDTH.bound
          : paletteIndex === 0
            ? PLOT_LINE_WIDTH.primary
            : PLOT_LINE_WIDTH.secondary,
        shape: "linear",
        dash,
      },
      marker: {
        color,
        size: isBound ? 0 : PLOT_MARKER_SIZE.primary,
      },
    };
  });
  const signature = JSON.stringify({
    title: chart.title,
    xLabel: chart.x_label,
    yLabel: chart.y_label,
    traces: signatureTraces,
  });
  return { data, signature };
}

function restyleChartData(view: ChartView, chart: Chart, signature: string): void {
  const plotly = window.Plotly;
  if (!plotly) {
    return;
  }
  const traceIndices = chart.series.map((_, index) => index);
  void plotly.restyle(
    view.plotEl,
    {
      x: chart.series.map((series) => series.x),
      y: chart.series.map((series) => series.y),
    },
    traceIndices,
  ).then(() => {
    view.chartTraceSignature = signature;
  }).catch((error) => {
    console.warn("chart restyle failed; rebuilding chart", error);
    view.chartTraceSignature = null;
    if (view.plotEl.isConnected) {
      scheduleArtifactRender(true);
    }
  });
}

function updateChart(view: ChartView | undefined, chart: Chart): void {
  if (!window.Plotly || !view) {
    return;
  }
  const { data, signature } = buildChartTraces(chart);
  if (view.chartTraceSignature === signature) {
    restyleChartData(view, chart, signature);
    return;
  }
  const layout: PlotlyLayout & { xaxis: PlotlyObject & { range?: number[] } } = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(4, 15, 22, 0.92)",
    font: {
      color: "#e5f1f4",
      family: '"Avenir Next", Futura, "Trebuchet MS", sans-serif',
      size: 12,
    },
    margin: { l: 74, r: 24, t: 18, b: 62 },
    legend: {
      orientation: "h",
      y: -0.28,
      x: 0,
      font: { color: "#94b6bd", size: 11 },
    },
    xaxis: {
      title: "",
      autorange: state.linkedChartAutorange,
      gridcolor: "rgba(229, 241, 244, 0.08)",
      linecolor: "rgba(177, 214, 222, 0.18)",
      zeroline: false,
      ticks: "outside",
    },
    yaxis: {
      title: chart.y_label,
      gridcolor: "rgba(229, 241, 244, 0.08)",
      linecolor: "rgba(177, 214, 222, 0.18)",
      zeroline: false,
      ticks: "outside",
      titlefont: { color: "#94b6bd" },
    },
  };
  const config = {
    responsive: true,
    displaylogo: false,
    displayModeBar: false,
  };
  if (view.linkXRange && !state.linkedChartAutorange && state.linkedChartRange) {
    layout.xaxis.range = state.linkedChartRange.slice();
  }
  window.Plotly.react(view.plotEl, data, layout, config).then(() => {
    view.chartTraceSignature = signature;
    if (view.linkXRange && !view.linkedRangeBound && typeof view.plotEl.on === "function") {
      view.plotEl.on("plotly_relayout", (eventData) => {
        syncLinkedChartRange(view, eventData);
      });
      view.linkedRangeBound = true;
    }
  }).catch((error) => {
    view.chartTraceSignature = null;
    console.warn("chart rebuild failed", error);
  });
}

function circleShapes(circles: SceneCircle[]): PlotlyTrace[] {
  return circles.map((circle) => ({
    type: "circle",
    xref: "x",
    yref: "y",
    x0: circle.cx - circle.radius,
    x1: circle.cx + circle.radius,
    y0: circle.cy - circle.radius,
    y1: circle.cy + circle.radius,
    line: {
      color: "#f25f5c",
      width: PLOT_LINE_WIDTH.shape,
      dash: "dash",
    },
    fillcolor: "rgba(242, 95, 92, 0.13)",
  }));
}

function circleAnnotations(circles: SceneCircle[]): PlotlyTrace[] {
  return circles
    .filter((circle) => circle.label)
    .map((circle) => ({
      x: circle.cx,
      y: circle.cy,
      text: circle.label,
      showarrow: false,
      font: {
        color: "#f25f5c",
        size: 11,
      },
      bgcolor: "rgba(4, 15, 22, 0.75)",
      bordercolor: "rgba(242, 95, 92, 0.28)",
      borderwidth: 1,
      borderpad: 3,
    }));
}

function updateContourVisualization(
  view: ChartView | undefined,
  visualization: Contour2DVisualization,
): void {
  if (!window.Plotly || !view) {
    return;
  }
  const pathTraces = visualization.paths.map((path, index) => ({
    type: "scatter",
    mode: "lines+markers",
    name: path.name,
    x: path.x,
    y: path.y,
    line: {
      color: index === 0 ? "#5bd1b5" : PALETTE[index % PALETTE.length],
      width: index === 0 ? PLOT_LINE_WIDTH.primary : PLOT_LINE_WIDTH.secondary,
    },
    marker: {
      color: index === 0 ? "#5bd1b5" : PALETTE[index % PALETTE.length],
      size: PLOT_MARKER_SIZE.primary,
      line: { color: "#041016", width: PLOT_LINE_WIDTH.tolerance },
    },
    hovertemplate: `${visualization.x_label}: %{x:.4f}<br>${visualization.y_label}: %{y:.4f}<extra>${path.name}</extra>`,
  }));
  const data: PlotlyTrace[] = [
    {
      type: "contour",
      name: visualization.title,
      x: visualization.x,
      y: visualization.y,
      z: visualization.z,
      colorscale: "Viridis",
      contours: {
        coloring: "heatmap",
        showlines: true,
      },
      line: {
        color: "rgba(229, 241, 244, 0.28)",
        width: PLOT_LINE_WIDTH.tolerance,
      },
      colorbar: {
        title: "f",
        tickfont: { color: "#94b6bd" },
        titlefont: { color: "#94b6bd" },
      },
      opacity: 0.9,
      hovertemplate: `${visualization.x_label}: %{x:.4f}<br>${visualization.y_label}: %{y:.4f}<br>f: %{z:.4g}<extra></extra>`,
    },
    ...pathTraces,
  ];
  const layout: PlotlyLayout = {
    uirevision: visualization.title,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(4, 15, 22, 0.92)",
    font: {
      color: "#e5f1f4",
      family: '"Avenir Next", Futura, "Trebuchet MS", sans-serif',
      size: 12,
    },
    margin: { l: 74, r: 42, t: 18, b: 62 },
    legend: {
      orientation: "h",
      y: -0.26,
      x: 0,
      font: { color: "#94b6bd", size: 11 },
    },
    dragmode: "zoom",
    hovermode: "closest",
    shapes: circleShapes(visualization.circles),
    annotations: circleAnnotations(visualization.circles),
    xaxis: {
      title: visualization.x_label,
      gridcolor: "rgba(229, 241, 244, 0.08)",
      linecolor: "rgba(177, 214, 222, 0.18)",
      zeroline: false,
      ticks: "outside",
      titlefont: { color: "#94b6bd" },
    },
    yaxis: {
      title: visualization.y_label,
      gridcolor: "rgba(229, 241, 244, 0.08)",
      linecolor: "rgba(177, 214, 222, 0.18)",
      zeroline: false,
      ticks: "outside",
      titlefont: { color: "#94b6bd" },
      scaleanchor: "x",
      scaleratio: 1,
    },
  };
  const config: PlotlyConfig = {
    responsive: true,
    displaylogo: false,
    displayModeBar: "hover",
    scrollZoom: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toImage"],
  };
  window.Plotly.react(view.plotEl, data, layout, config);
}

interface Paths3DTraceStyle {
  label: string;
  group: string;
  color: string;
  width: number;
  opacity: number;
  mode: string;
  markerSize: number;
  visible: true | "legendonly";
}

interface Paths3DTraceBuild {
  data: PlotlyTrace[];
  signature: string;
}

function escapePlotlyHoverText(value: string): string {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function compactAxisLabel(label: string): string {
  const trimmed = label.trim();
  const unitStart = trimmed.indexOf(" (");
  if (unitStart > 0) {
    return trimmed.slice(0, unitStart);
  }
  return trimmed;
}

function paths3DHoverTemplate(visualization: Paths3DVisualization, label: string): string {
  const xLabel = escapePlotlyHoverText(compactAxisLabel(visualization.x_label));
  const yLabel = escapePlotlyHoverText(compactAxisLabel(visualization.y_label));
  const zLabel = escapePlotlyHoverText(compactAxisLabel(visualization.z_label));
  return `<b>${escapePlotlyHoverText(label)}</b><br>`
    + `${xLabel} %{x:.1f} · ${yLabel} %{y:.1f} · ${zLabel} %{z:.0f}<extra></extra>`;
}

const DEFAULT_PATHS_3D_CAMERA: PlotlyObject = {
  up: { x: 0, y: 0, z: 1 },
  eye: { x: 1.05, y: -2.2, z: 1.35 },
  center: { x: 0.04, y: 0.0, z: -0.08 },
};

function isPlotlyObject(value: PlotlyValue | undefined): value is PlotlyObject {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function clonePlotlyObject(value: PlotlyObject): PlotlyObject {
  return JSON.parse(JSON.stringify(value)) as PlotlyObject;
}

function plotlyObjectProperty(object: PlotlyObject | null, key: string): PlotlyObject | null {
  const value = object?.[key];
  return isPlotlyObject(value) ? value : null;
}

function currentPaths3DCamera(view: PlotlyView): PlotlyObject | null {
  const plotEl = view.plotEl as PlotlyHostElement & {
    layout?: PlotlyObject;
    _fullLayout?: PlotlyObject;
  };
  const layoutCamera = plotlyObjectProperty(
    plotlyObjectProperty(plotEl.layout ?? null, "scene"),
    "camera",
  );
  if (layoutCamera) {
    return clonePlotlyObject(layoutCamera);
  }
  const fullLayoutCamera = plotlyObjectProperty(
    plotlyObjectProperty(plotEl._fullLayout ?? null, "scene"),
    "camera",
  );
  return fullLayoutCamera ? clonePlotlyObject(fullLayoutCamera) : null;
}

function paths3DCameraFromRelayout(
  view: PlotlyView,
  eventData: PlotlyRelayoutPayload,
): PlotlyObject | null {
  const camera = eventData["scene.camera"];
  if (isPlotlyObject(camera)) {
    return clonePlotlyObject(camera);
  }

  let hasCameraField = false;
  const merged = currentPaths3DCamera(view) ?? clonePlotlyObject(DEFAULT_PATHS_3D_CAMERA);
  for (const [key, value] of Object.entries(eventData)) {
    const prefix = "scene.camera.";
    if (!key.startsWith(prefix)) {
      continue;
    }
    const parts = key.slice(prefix.length).split(".");
    if (parts.length !== 2) {
      continue;
    }
    const [section, component] = parts;
    if (!section || !component) {
      continue;
    }
    let sectionObject = merged[section];
    if (!isPlotlyObject(sectionObject)) {
      sectionObject = {};
      merged[section] = sectionObject;
    }
    sectionObject[component] = value;
    hasCameraField = true;
  }
  return hasCameraField ? merged : null;
}

function capturePaths3DCamera(view: PlotlyView, eventData: PlotlyRelayoutPayload): void {
  const camera = paths3DCameraFromRelayout(view, eventData) ?? currentPaths3DCamera(view);
  if (camera) {
    view.sceneCamera = camera;
  }
}

function bindPaths3DInteraction(view: PlotlyView): void {
  if (view.sceneInteractionBound) {
    return;
  }
  const cancelIdleFrame = (): void => {
    if (view.sceneIdleFrameHandle != null) {
      window.cancelAnimationFrame(view.sceneIdleFrameHandle);
      view.sceneIdleFrameHandle = null;
    }
  };
  const markActive = (): void => {
    cancelIdleFrame();
    view.scenePointerActive = true;
    view.sceneInteracting = true;
  };
  const markIdle = (): void => {
    view.scenePointerActive = false;
    cancelIdleFrame();
    view.sceneIdleFrameHandle = window.requestAnimationFrame(() => {
      view.sceneIdleFrameHandle = null;
      view.sceneCamera = currentPaths3DCamera(view) ?? view.sceneCamera ?? null;
      view.sceneInteracting = false;
      if (view.plotEl.isConnected) {
        scheduleArtifactRender();
      }
    });
  };
  if (typeof view.plotEl.on === "function") {
    view.plotEl.on("plotly_relayouting", (eventData) => {
      view.sceneInteracting = true;
      capturePaths3DCamera(view, eventData);
    });
    view.plotEl.on("plotly_relayout", (eventData) => {
      capturePaths3DCamera(view, eventData);
      if (!view.scenePointerActive) {
        view.sceneInteracting = false;
        scheduleArtifactRender();
      }
    });
  }
  view.plotEl.addEventListener("pointerdown", markActive, { capture: true });
  window.addEventListener("pointerup", markIdle);
  window.addEventListener("pointercancel", markIdle);
  window.addEventListener("blur", markIdle);
  view.sceneInteractionBound = true;
}

function paths3DTraceStyle(path: ScenePath3D, index: number): Paths3DTraceStyle {
  const name = path.name.toLowerCase();
  if (name === "trajectory" || name.startsWith("trajectory arc")) {
    return {
      label: "Trajectory",
      group: "trajectory",
      color: "#5bd1b5",
      width: 2.8,
      opacity: 1,
      mode: "lines",
      markerSize: 0,
      visible: true,
    };
  }
  if (name.startsWith("lift")) {
    return {
      label: "Lift vectors",
      group: "lift",
      color: "#b8f2e6",
      width: 1.15,
      opacity: 0.78,
      mode: "lines",
      markerSize: 0,
      visible: "legendonly",
    };
  }
  if (name.startsWith("drag")) {
    return {
      label: "Drag vectors",
      group: "drag",
      color: "#f25f5c",
      width: 1.25,
      opacity: 0.9,
      mode: "lines",
      markerSize: 0,
      visible: "legendonly",
    };
  }
  if (name.startsWith("aero accel")) {
    return {
      label: "Aero acceleration",
      group: "aero",
      color: "#f7b267",
      width: 1.2,
      opacity: 0.74,
      mode: "lines",
      markerSize: 0,
      visible: true,
    };
  }
  if (name.startsWith("wind shear")) {
    return {
      label: "Wind shear",
      group: "wind-shear",
      color: "#7cc6fe",
      width: name.startsWith("wind shear frame") ? 0.9 : 1.05,
      opacity: name.startsWith("wind shear frame") ? 0.16 : 0.3,
      mode: "lines",
      markerSize: 0,
      visible: "legendonly",
    };
  }
  if (name.startsWith("wind")) {
    return {
      label: "Wind vectors",
      group: "wind",
      color: "#7cc6fe",
      width: 1.1,
      opacity: 0.68,
      mode: "lines",
      markerSize: 0,
      visible: "legendonly",
    };
  }
  if (name.startsWith("air axis")) {
    return {
      label: "Air-relative axis",
      group: "air-axis",
      color: "#e5f1f4",
      width: 0.95,
      opacity: 0.56,
      mode: "lines",
      markerSize: 0,
      visible: "legendonly",
    };
  }
  if (name.startsWith("zero-bank frame")) {
    return {
      label: "Zero-bank frame",
      group: "zero-bank-frame",
      color: "#d7aefb",
      width: 0.95,
      opacity: 0.58,
      mode: "lines",
      markerSize: 0,
      visible: "legendonly",
    };
  }
  if (name.startsWith("side frame")) {
    return {
      label: "Bank side frame",
      group: "side-frame",
      color: "#b8f2e6",
      width: 0.95,
      opacity: 0.58,
      mode: "lines",
      markerSize: 0,
      visible: "legendonly",
    };
  }
  return {
    label: path.name,
    group: path.name,
    color: PALETTE[index % PALETTE.length],
    width: 1,
    opacity: 0.48,
    mode: "lines",
    markerSize: 0,
    visible: "legendonly",
  };
}

function buildPaths3DTraces(visualization: Paths3DVisualization): Paths3DTraceBuild {
  const visibleLegendGroups = new Set<string>();
  const signatureTraces: PlotlyValue[] = [];
  const data = visualization.paths.map((path, index) => {
    const style = paths3DTraceStyle(path, index);
    const showlegend = !visibleLegendGroups.has(style.group);
    visibleLegendGroups.add(style.group);
    signatureTraces.push([
      path.name,
      style.label,
      style.group,
      style.color,
      style.width,
      style.opacity,
      style.mode,
      style.markerSize,
      style.visible,
      showlegend,
    ]);
    return {
      type: "scatter3d",
      mode: style.mode,
      name: style.label,
      legendgroup: style.group,
      x: path.x,
      y: path.y,
      z: path.z,
      visible: style.visible,
      showlegend,
      opacity: style.opacity,
      line: {
        color: style.color,
        width: style.width,
      },
      marker: {
        color: style.color,
        size: style.markerSize,
      },
      hovertemplate: paths3DHoverTemplate(visualization, style.label),
      hoverlabel: {
        bgcolor: "rgba(4, 15, 22, 0.94)",
        bordercolor: style.color,
        font: {
          color: "#e5f1f4",
          size: 11,
        },
      },
    };
  });
  const signature = JSON.stringify({
    title: visualization.title,
    labels: [visualization.x_label, visualization.y_label, visualization.z_label],
    traces: signatureTraces,
  });
  return { data, signature };
}

function restylePaths3DVisualization(
  view: PlotlyView,
  visualization: Paths3DVisualization,
  signature: string,
): void {
  const traceIndices = visualization.paths.map((_, index) => index);
  void window.Plotly?.restyle(
    view.plotEl,
    {
      x: visualization.paths.map((path) => path.x),
      y: visualization.paths.map((path) => path.y),
      z: visualization.paths.map((path) => path.z),
    },
    traceIndices,
  ).then(() => {
    view.sceneTraceSignature = signature;
  }).catch((error) => {
    console.warn("3D scene restyle failed; rebuilding figure", error);
    view.sceneTraceSignature = null;
    if (view.plotEl.isConnected) {
      scheduleArtifactRender(true);
    }
  });
}

function updatePaths3DVisualization(
  view: PlotlyView | undefined,
  visualization: Paths3DVisualization,
): void {
  if (!window.Plotly || !view) {
    return;
  }
  const { data, signature } = buildPaths3DTraces(visualization);
  const canRestyle = view.sceneTraceSignature === signature;
  if (canRestyle) {
    restylePaths3DVisualization(view, visualization, signature);
    return;
  }
  if (view.sceneInteracting) {
    return;
  }
  const layout: PlotlyLayout = {
    uirevision: visualization.title,
    dragmode: "turntable",
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(4, 15, 22, 0.92)",
    font: {
      color: "#e5f1f4",
      family: '"Avenir Next", Futura, "Trebuchet MS", sans-serif',
      size: 12,
    },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    legend: {
      orientation: "h",
      y: 0.02,
      x: 0.01,
      font: { color: "#94b6bd", size: 11 },
    },
    scene: {
      domain: { x: [0, 1], y: [0, 1] },
      bgcolor: "rgba(4, 15, 22, 0.92)",
      aspectmode: "manual",
      aspectratio: { x: 1.9, y: 1.05, z: 1.0 },
      xaxis: {
        title: visualization.x_label,
        showgrid: false,
        zeroline: false,
        titlefont: { color: "#94b6bd" },
      },
      yaxis: {
        title: visualization.y_label,
        showgrid: false,
        zeroline: false,
        titlefont: { color: "#94b6bd" },
      },
      zaxis: {
        title: visualization.z_label,
        showgrid: false,
        zeroline: false,
        titlefont: { color: "#94b6bd" },
      },
    },
  };
  const scene = layout.scene;
  if (isPlotlyObject(scene)) {
    const camera = view.sceneCamera
      ?? currentPaths3DCamera(view)
      ?? clonePlotlyObject(DEFAULT_PATHS_3D_CAMERA);
    view.sceneCamera = camera;
    scene.camera = clonePlotlyObject(camera);
  }
  const config: PlotlyConfig = {
    responsive: true,
    displaylogo: false,
    displayModeBar: "hover",
    scrollZoom: true,
    modeBarButtonsToRemove: ["toImage"],
  };
  window.Plotly.react(view.plotEl, data, layout, config).then(() => {
    view.sceneTraceSignature = signature;
    bindPaths3DInteraction(view);
  }).catch((error) => {
    view.sceneTraceSignature = null;
    console.warn("3D scene rebuild failed", error);
  });
}

function updateVisualization(
  view: ChartView | undefined,
  visualization: ArtifactVisualization,
): void {
  if (visualization.kind === "contour_2d") {
    updateContourVisualization(view, visualization);
  } else {
    updatePaths3DVisualization(view, visualization);
  }
}

function renderCharts(): void {
  const panels = artifactChartPanels(state.artifact);
  if (panels.length === 0) {
    resetChartViews();
    chartsEl.innerHTML = `<div class="placeholder">The solver will populate state, control, static, and constraint charts here.</div>`;
    return;
  }
  if (!window.Plotly) {
    chartsEl.innerHTML = `<div class="placeholder">Plotly is still loading.</div>`;
    return;
  }
  ensureChartViews(panels);
  for (const panel of panels) {
    const view = state.chartViews.get(panel.key);
    if (panel.kind === "chart") {
      updateChart(view, panel.chart);
    } else {
      updateVisualization(view, panel.visualization);
    }
  }
}

function stopAnimation(): void {
  if (state.playHandle !== null) {
    clearInterval(state.playHandle);
    state.playHandle = null;
  }
  state.playing = false;
}

function startAnimation(): void {
  const frames = state.artifact?.scene?.animation?.frames ?? [];
  if (frames.length === 0) {
    return;
  }
  stopAnimation();
  state.playing = true;
  state.playHandle = setInterval(() => {
    state.animationIndex = (state.animationIndex + 1) % frames.length;
    renderScene();
  }, 140);
}

function clearScheduledArtifactRender(): void {
  if (state.artifactRenderFrameHandle !== null) {
    window.cancelAnimationFrame(state.artifactRenderFrameHandle);
    state.artifactRenderFrameHandle = null;
  }
  state.renderScheduled = false;
}

function requestArtifactRenderFrame(): void {
  if (state.artifactRenderFrameHandle !== null) {
    return;
  }
  state.renderScheduled = true;
  state.artifactRenderFrameHandle = requestAnimationFrame(() => {
    state.artifactRenderFrameHandle = null;
    if (!state.renderScheduled) {
      return;
    }
    state.renderScheduled = false;
    renderMetrics();
    renderConstraintPanels();
    renderScene();
    renderCharts();
    renderNotes(state.artifact?.notes ?? currentSpec()?.notes ?? []);
  });
}

function scheduleArtifactRender(force = false): void {
  if (force) {
    clearScheduledArtifactRender();
  }
  requestArtifactRenderFrame();
}

function handleSolveEvent(event: SolveEvent): void {
  switch (event.kind) {
    case STREAM_EVENT_KIND.status:
      {
        const previousStage = state.liveStatus?.stage ?? null;
        state.liveStatus = event.status;
        state.liveSolver = buildStatusSolverReport(event.status);
        renderSolverSummary();
        renderCompileCacheStatus();
        if (previousStage !== SOLVE_STAGE.solving && event.status.stage === SOLVE_STAGE.solving) {
          void refreshCompileCacheStatus();
        }
        renderMetrics();
        setStatusDisplay(
          statusDisplayForSolveStatus(event.status, state.latestProgress?.iteration ?? null),
        );
        break;
      }
    case STREAM_EVENT_KIND.log:
      appendLogLine(event.line, event.level ?? LOG_LEVEL.console);
      break;
    case STREAM_EVENT_KIND.iteration:
      state.pendingIterationEvent = event;
      scheduleIterationUpdate();
      break;
    case STREAM_EVENT_KIND.final:
      state.pendingIterationEvent = null;
      state.liveStatus = null;
      state.terminalSolver = mergeSolverReport(event.artifact.solver, state.liveSolver);
      state.liveSolver = null;
      state.artifact = {
        ...event.artifact,
        solver: state.terminalSolver,
      };
      state.animationIndex = 0;
      renderSolverSummary();
      renderCompileCacheStatus();
      void refreshCompileCacheStatus();
      appendSolveProfilingLog(state.terminalSolver);
      scheduleArtifactRender(true);
      setStatusDisplay(statusDisplayForSolverReport(state.terminalSolver));
      break;
    case STREAM_EVENT_KIND.error:
    default:
      applySolveFailure(event.message);
      break;
  }
}

async function readNdjsonStream(response: Response, onEvent: (event: SolveEvent) => void): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Streaming response body is unavailable.");
  }
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      onEvent(normalizeSolveEvent(readWireSolveEvent(parseJsonValue(trimmed, "solve stream event"))));
    }
  }

  const tail = buffer.trim();
  if (tail) {
    onEvent(normalizeSolveEvent(readWireSolveEvent(parseJsonValue(tail, "solve stream event"))));
  }
}

async function solveCurrentProblem(event?: Event): Promise<void> {
  event?.preventDefault?.();
  const spec = currentSpec();
  if (!spec || state.solving) {
    return;
  }
  const abortController = new AbortController();

  try {
    state.solving = true;
    state.solveStopRequested = false;
    state.solveAbortController = abortController;
    syncCompileStatusPolling();
    clearScheduledPrewarm();
    solveButton.disabled = true;
    solveButton.setAttribute("aria-busy", "true");
    stopButton.disabled = false;
    stopAnimation();
    state.artifact = null;
    state.animationIndex = 0;
    state.sceneView = null;
    resetSolverPanel();
    renderMetrics();
    renderCompileCacheStatus();
    renderScene();
    renderCharts();
    setStatusDisplay(pendingSolveRequestStatusDisplay());

    const response = await fetch(`/api/solve_stream/${spec.wire_id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: abortController.signal,
      body: JSON.stringify({ values: state.values }),
    });

    if (!response.ok) {
      const payload = await readResponseJsonValue(response, `/api/solve_stream/${spec.wire_id}`);
      throw new Error(readOptionalErrorMessage(payload) ?? `Request failed with ${response.status}`);
    }

    await readNdjsonStream(response, handleSolveEvent);
  } catch (error) {
    if (state.solveStopRequested && error instanceof DOMException && error.name === "AbortError") {
      applySolveStopped();
      return;
    }
    applySolveFailure(
      describeThrownValue(
        error != null && typeof error === "object" ? error : String(error),
      ),
    );
  } finally {
    state.solving = false;
    state.solveAbortController = null;
    state.solveStopRequested = false;
    syncCompileStatusPolling();
    solveButton.disabled = false;
    solveButton.setAttribute("aria-busy", "false");
    stopButton.disabled = true;
    renderCompileCacheStatus();
  }
}

function stopCurrentSolve(event?: Event): void {
  event?.preventDefault?.();
  if (!state.solving || !state.solveAbortController) {
    return;
  }
  state.solveStopRequested = true;
  stopButton.disabled = true;
  appendLogLine("stop requested; closing solve stream", LOG_LEVEL.warning);
  setStatusDisplay({
    eyebrow: "Run Status",
    title: "Stopping solve",
    detail: "Waiting for the backend solver to observe the closed stream.",
    kind: "warning",
    active: true,
  });
  state.solveAbortController.abort();
}

async function init(): Promise<void> {
  try {
    const problemSpecs = await fetchJson("/api/problems", (value) => {
      const entries = readJsonArray(value, "/api/problems");
      return entries.map((entry, index) => readWireProblemSpec(entry, `/api/problems[${index}]`));
    });
    state.specs = problemSpecs.map(normalizeProblemSpec);
    if (state.specs.length === 0) {
      throw new Error("No problems are registered.");
    }
    controlsForm.addEventListener("submit", solveCurrentProblem);
    solveButton.addEventListener("click", solveCurrentProblem);
    stopButton.addEventListener("click", stopCurrentSolve);
    copyConsoleButton.addEventListener("click", () => {
      void copyConsoleTranscript();
    });
    consoleFollowCheckbox.addEventListener("change", () => {
      if (consoleFollowCheckbox.checked) {
        enableConsoleFollow();
      } else {
        setConsoleFollowState(false);
      }
    });
    solverLogEl.addEventListener("wheel", disableConsoleFollowForManualScroll, { passive: true });
    solverLogEl.addEventListener("touchstart", disableConsoleFollowForManualScroll, {
      passive: true,
    });
    clearJitCacheButton.addEventListener("click", () => {
      void clearJitCache();
    });
    setConsoleFollowState(state.followSolverLog);
    resetCopyConsoleButton();
    selectProblem(state.specs[0].id);
    void refreshCompileCacheStatus();
  } catch (error) {
    setStatus(error instanceof Error ? error.message : String(error), "error");
  }
}

init();

window.addEventListener("error", (event: ErrorEvent) => {
  reportFrontendError(
    event.error != null && typeof event.error === "object" ? event.error : event.message,
    "window error",
  );
});

window.addEventListener("unhandledrejection", (event: PromiseRejectionEvent) => {
  reportFrontendError(event.reason, "unhandled rejection");
});

window.addEventListener("mathjax-ready", () => {
  const pendingRoots = Array.from(pendingMathRoots);
  pendingMathRoots.clear();
  for (const root of pendingRoots) {
    typesetMath(root);
  }
  const spec = currentSpec();
  if (spec) {
    renderModel(spec);
  }
});
