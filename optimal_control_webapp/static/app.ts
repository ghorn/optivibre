const PALETTE = ["#f7b267", "#5bd1b5", "#7cc6fe", "#f25f5c", "#d7aefb", "#b8f2e6"] as const;
const EQ_INF_LABEL = "‖eq‖∞";
const INEQ_INF_LABEL = "‖ineq₊‖∞";
const DUAL_INF_LABEL = "‖∇L‖∞";
const STEP_INF_LABEL = "‖Δx‖∞";

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
const CONTROL_EDITOR = Object.freeze({
  slider: 0,
  select: 1,
  text: 2,
} as const);
const CONTROL_EDITOR_FROM_WIRE = Object.freeze({
  slider: CONTROL_EDITOR.slider,
  select: CONTROL_EDITOR.select,
  text: CONTROL_EDITOR.text,
} as const);
const CONTROL_SEMANTIC = Object.freeze({
  transcriptionMethod: 0,
  transcriptionIntervals: 1,
  collocationFamily: 2,
  collocationDegree: 3,
  solverMethod: 4,
  solverMaxIterations: 5,
  solverDualTolerance: 6,
  solverConstraintTolerance: 7,
  solverComplementarityTolerance: 8,
  problemParameter: 9,
} as const);
const CONTROL_SEMANTIC_FROM_WIRE = Object.freeze({
  transcription_method: CONTROL_SEMANTIC.transcriptionMethod,
  transcription_intervals: CONTROL_SEMANTIC.transcriptionIntervals,
  collocation_family: CONTROL_SEMANTIC.collocationFamily,
  collocation_degree: CONTROL_SEMANTIC.collocationDegree,
  solver_method: CONTROL_SEMANTIC.solverMethod,
  solver_max_iterations: CONTROL_SEMANTIC.solverMaxIterations,
  solver_dual_tolerance: CONTROL_SEMANTIC.solverDualTolerance,
  solver_constraint_tolerance: CONTROL_SEMANTIC.solverConstraintTolerance,
  solver_complementarity_tolerance: CONTROL_SEMANTIC.solverComplementarityTolerance,
  problem_parameter: CONTROL_SEMANTIC.problemParameter,
} as const);
const CONTROL_VISIBILITY = Object.freeze({
  always: 0,
  directCollocationOnly: 1,
} as const);
const CONTROL_VISIBILITY_FROM_WIRE = Object.freeze({
  always: CONTROL_VISIBILITY.always,
  direct_collocation_only: CONTROL_VISIBILITY.directCollocationOnly,
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
} as const);
const METRIC_KEY_FROM_WIRE = Object.freeze({
  custom: METRIC_KEY.custom,
  transcription_method: METRIC_KEY.transcriptionMethod,
  interval_count: METRIC_KEY.intervalCount,
  collocation_node_count: METRIC_KEY.collocationNodeCount,
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
const SOLVER_METHOD_FROM_WIRE = Object.freeze({
  sqp: SOLVER_METHOD.sqp,
  nlip: SOLVER_METHOD.nlip,
  ipopt: SOLVER_METHOD.ipopt,
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
const STREAM_EVENT_KIND_FROM_WIRE = Object.freeze({
  status: STREAM_EVENT_KIND.status,
  log: STREAM_EVENT_KIND.log,
  iteration: STREAM_EVENT_KIND.iteration,
  final: STREAM_EVENT_KIND.final,
  error: STREAM_EVENT_KIND.error,
} as const);
const PROBLEM_ID = Object.freeze({
  optimalDistanceGlider: 0,
  linearSManeuver: 1,
  sailboatUpwind: 2,
  craneTransfer: 3,
} as const);
const PROBLEM_ID_FROM_WIRE = Object.freeze({
  optimal_distance_glider: PROBLEM_ID.optimalDistanceGlider,
  linear_s_maneuver: PROBLEM_ID.linearSManeuver,
  sailboat_upwind: PROBLEM_ID.sailboatUpwind,
  crane_transfer: PROBLEM_ID.craneTransfer,
} as const);
const COMPILE_CACHE_STATE = Object.freeze({
  cold: 0,
  ready: 1,
} as const);
const COMPILE_CACHE_STATE_FROM_WIRE = Object.freeze({
  cold: COMPILE_CACHE_STATE.cold,
  ready: COMPILE_CACHE_STATE.ready,
} as const);

type ControlSectionCode = EnumValue<typeof CONTROL_SECTION>;
type ControlEditorCode = EnumValue<typeof CONTROL_EDITOR>;
type ControlSemanticCode = EnumValue<typeof CONTROL_SEMANTIC>;
type ControlVisibilityCode = EnumValue<typeof CONTROL_VISIBILITY>;
type ControlValueDisplayCode = EnumValue<typeof CONTROL_VALUE_DISPLAY>;
type LogLevelCode = EnumValue<typeof LOG_LEVEL>;
type TimeSeriesRoleCode = EnumValue<typeof TIME_SERIES_ROLE>;
type SolverStatusKindCode = EnumValue<typeof SOLVER_STATUS_KIND>;
type ConstraintPanelSeverityCode = EnumValue<typeof CONSTRAINT_PANEL_SEVERITY>;
type ConstraintPanelCategoryCode = EnumValue<typeof CONSTRAINT_PANEL_CATEGORY>;
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
  editor?: string | number;
  visibility?: string | number;
  semantic?: string | number;
  value_display?: string | number;
  choices?: ControlChoice[];
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
  editor: ControlEditorCode;
  visibility: ControlVisibilityCode;
  semantic: ControlSemanticCode;
  value_display: ControlValueDisplayCode;
  choices: ControlChoice[];
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
  symbolic_setup_s?: number | null;
  jit_s?: number | null;
}

interface CompileCacheStatus {
  wire_problem_id: string;
  problem_id: ProblemIdCode;
  problem_name: string;
  variant_id: string;
  variant_label: string;
  state: CompileCacheStateCode;
  symbolic_setup_s: number | null;
  jit_s: number | null;
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
}

interface WireConstraintPanelEntry extends Omit<
  ConstraintPanelEntry,
  "category" | "severity" | "lower_severity" | "upper_severity"
> {
  category: string | number;
  severity: string | number;
  lower_severity?: string | number | null;
  upper_severity?: string | number | null;
}

interface ConstraintPanels {
  equalities: ConstraintPanelEntry[];
  inequalities: ConstraintPanelEntry[];
}

interface WireConstraintPanels {
  equalities?: WireConstraintPanelEntry[];
  inequalities?: WireConstraintPanelEntry[];
}

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
  line_search_iterations?: number | null;
}

interface WireSolveProgress extends Omit<SolveProgress, "phase"> {
  phase: string | number;
}

interface SolverPhaseDetail {
  label: string;
  value: string;
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
  symbolic_setup_s?: number | null;
  jit_s?: number | null;
  solve_s?: number | null;
  compile_cached: boolean;
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
  constraint_panels: ConstraintPanels;
  charts: Chart[];
  scene: Scene2D;
  notes: string[];
}

interface WireSolveArtifact extends Omit<
  SolveArtifact,
  "summary" | "solver" | "constraint_panels" | "charts"
> {
  summary?: WireMetric[];
  solver: WireSolverReport;
  constraint_panels?: WireConstraintPanels | null;
  charts?: WireChart[];
}

interface SolveStatus {
  stage: SolveStageCode;
  solver_method?: SolverMethodCode | null;
  solver: SolverReport;
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
}

interface SceneView {
  scene: Scene2D;
  shell: HTMLDivElement;
  meta: HTMLDivElement;
  playButton: HTMLButtonElement | null;
  slider: HTMLInputElement | null;
  plotEl: PlotlyHostElement;
}

interface ControlSectionView {
  key: ControlSectionCode;
  title: string;
  subtitle: string;
  controls: ControlSpec[];
}

type ControlSectionCollapseState = Record<ControlSectionCode, boolean>;

interface FrontendState {
  specs: ProblemSpec[];
  selectedId: ProblemIdCode | null;
  values: Record<string, number>;
  compileCacheStatuses: CompileCacheStatus[];
  collapsedControlSections: ControlSectionCollapseState;
  artifact: SolveArtifact | null;
  animationIndex: number;
  playing: boolean;
  playHandle: number | null;
  solving: boolean;
  renderScheduled: boolean;
  chartViews: Map<string, ChartView>;
  chartLayoutKey: string;
  progressPlotReady: boolean;
  logLines: LogLine[];
  latestProgress: SolveProgress | null;
  liveStatus: SolveStatus | null;
  liveSolver: SolverReport | null;
  solveStartedAtMs: number | null;
  terminalSolver: SolverReport | null;
  pendingIterationEvent: IterationSolveEvent | null;
  iterationFlushScheduled: boolean;
  sceneView: SceneView | null;
  linkedChartRange: NumericRange | null;
  linkedChartAutorange: boolean;
  linkingChartRange: boolean;
  prewarmTimer: number | null;
  lastPrewarmSignature: string | null;
  prewarmInFlightSignature: string | null;
}

type PlotlyTrace = PlotlyObject;
type PlotlyLayout = PlotlyObject;
type PlotlyConfig = PlotlyObject;
type ConstraintPanelKind = "eq" | "ineq";
type AnsiColor = "red" | "green" | "yellow" | "cyan";

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
const PREWARM_DELAY_MS = 200;

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

function readWireControlSpec(value: JsonValue | undefined, context: string): WireControlSpec {
  const object = readJsonObject(value, context);
  const choicesValue = readOptionalJsonArray(readJsonValueAt(object, "choices"), `${context}.choices`);
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
  return {
    problem_id: readJsonStringOrNumber(readJsonValueAt(object, "problem_id"), `${context}.problem_id`),
    problem_name: readJsonString(readJsonValueAt(object, "problem_name"), `${context}.problem_name`),
    variant_id: readJsonString(readJsonValueAt(object, "variant_id"), `${context}.variant_id`),
    variant_label: readJsonString(readJsonValueAt(object, "variant_label"), `${context}.variant_label`),
    state: readJsonStringOrNumber(readJsonValueAt(object, "state"), `${context}.state`),
    symbolic_setup_s:
      readOptionalJsonNumber(readJsonValueAt(object, "symbolic_setup_s"), `${context}.symbolic_setup_s`) ??
      null,
    jit_s: readOptionalJsonNumber(readJsonValueAt(object, "jit_s"), `${context}.jit_s`) ?? null,
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
    line_search_iterations:
      readOptionalJsonNumber(
        readJsonValueAt(object, "line_search_iterations"),
        `${context}.line_search_iterations`,
      ) ?? null,
  };
}

function readSolverPhaseDetail(value: JsonValue | undefined, context: string): SolverPhaseDetail {
  const object = readJsonObject(value, context);
  return {
    label: readJsonString(readJsonValueAt(object, "label"), `${context}.label`),
    value: readJsonString(readJsonValueAt(object, "value"), `${context}.value`),
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
    symbolic_setup_s:
      readOptionalJsonNumber(readJsonValueAt(object, "symbolic_setup_s"), `${context}.symbolic_setup_s`) ??
      null,
    jit_s: readOptionalJsonNumber(readJsonValueAt(object, "jit_s"), `${context}.jit_s`) ?? null,
    solve_s: readOptionalJsonNumber(readJsonValueAt(object, "solve_s"), `${context}.solve_s`) ?? null,
    compile_cached:
      readOptionalJsonBoolean(readJsonValueAt(object, "compile_cached"), `${context}.compile_cached`) ??
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
  const charts = readOptionalJsonArray(readJsonValueAt(object, "charts"), `${context}.charts`);
  const notes = readOptionalJsonArray(readJsonValueAt(object, "notes"), `${context}.notes`);
  const constraintPanels = readJsonValueAt(object, "constraint_panels");
  return {
    title: readJsonString(readJsonValueAt(object, "title"), `${context}.title`),
    summary: summary?.map((metric, index) => readWireMetric(metric, `${context}.summary[${index}]`)),
    solver: readWireSolverReport(readJsonValueAt(object, "solver"), `${context}.solver`),
    constraint_panels:
      constraintPanels == null
        ? null
        : readWireConstraintPanels(constraintPanels, `${context}.constraint_panels`),
    charts: charts?.map((chart, index) => readWireChart(chart, `${context}.charts[${index}]`)),
    scene: readScene2D(readJsonValueAt(object, "scene"), `${context}.scene`),
    notes: notes?.map((note, index) => readJsonString(note, `${context}.notes[${index}]`)) ?? [],
  };
}

function readWireSolveStatus(value: JsonValue | undefined, context: string): WireSolveStatus {
  const object = readJsonObject(value, context);
  return {
    stage: readJsonStringOrNumber(readJsonValueAt(object, "stage"), `${context}.stage`),
    solver_method:
      readOptionalJsonStringOrNumber(
        readJsonValueAt(object, "solver_method"),
        `${context}.solver_method`,
      ) ?? null,
    solver: readWireSolverReport(readJsonValueAt(object, "solver"), `${context}.solver`),
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
  artifact: null,
  animationIndex: 0,
  playing: false,
  playHandle: null,
  solving: false,
  renderScheduled: false,
  chartViews: new Map<string, ChartView>(),
  chartLayoutKey: "",
  progressPlotReady: false,
  logLines: [],
  latestProgress: null,
  liveStatus: null,
  liveSolver: null,
  solveStartedAtMs: null,
  terminalSolver: null,
  pendingIterationEvent: null,
  iterationFlushScheduled: false,
  sceneView: null,
  linkedChartRange: null,
  linkedChartAutorange: true,
  linkingChartRange: false,
  prewarmTimer: null,
  lastPrewarmSignature: null,
  prewarmInFlightSignature: null,
};

const problemList = requiredElement<HTMLDivElement>("#problem-list");
const controls = requiredElement<HTMLDivElement>("#controls");
const controlsForm = requiredElement<HTMLFormElement>("#controls-form");
const solveButton = requiredElement<HTMLButtonElement>("#solve-button");
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
const solverLogEl = requiredElement<HTMLPreElement>("#solver-log");
const prewarmStatusEl = requiredElement<HTMLDivElement>("#prewarm-status");
const eqViolationsEl = requiredElement<HTMLDivElement>("#eq-violations");
const ineqViolationsEl = requiredElement<HTMLDivElement>("#ineq-violations");

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
const PHASE_LABEL = new Map<SolvePhaseCode, string>([
  [SOLVE_PHASE.initial, "initial"],
  [SOLVE_PHASE.acceptedStep, "accepted step"],
  [SOLVE_PHASE.postConvergence, "post convergence"],
  [SOLVE_PHASE.converged, "converged"],
  [SOLVE_PHASE.regular, "regular"],
  [SOLVE_PHASE.restoration, "restoration"],
]);
const DIRECT_COLLOCATION_VALUE = 1;

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

function formatCompileDuration(seconds: number | null | undefined, compileCached: boolean): string {
  const formatted = formatDuration(seconds);
  if (!compileCached || formatted === "--") {
    return formatted;
  }
  return `${formatted} (cached)`;
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

function solverWithLiveElapsed(solver: SolverReport | null | undefined): SolverReport | null {
  if (!solver) {
    return null;
  }
  if (solver.completed || state.solveStartedAtMs == null) {
    return solver;
  }
  const elapsedSeconds = Math.max(0, (performance.now() - state.solveStartedAtMs) / 1000);
  return {
    ...solver,
    solve_s: Math.max(solver.solve_s ?? 0, elapsedSeconds),
  };
}

function currentSolverReport(): SolverReport | null {
  if (state.terminalSolver) {
    return state.terminalSolver;
  }
  return solverWithLiveElapsed(state.liveSolver);
}

function currentSolveStage(): SolveStageCode | null {
  if (state.terminalSolver?.completed) {
    return null;
  }
  if (state.liveStatus) {
    return state.liveStatus.stage;
  }
  if (state.solving) {
    return SOLVE_STAGE.symbolicSetup;
  }
  return null;
}

function buildStatusSolverReport(status: SolveStatus): SolverReport {
  const liveSolver = solverWithLiveElapsed(state.liveSolver);
  const nextSolver = status.solver;
  return {
    ...nextSolver,
    symbolic_setup_s: nextSolver.symbolic_setup_s ?? liveSolver?.symbolic_setup_s ?? null,
    jit_s: nextSolver.jit_s ?? liveSolver?.jit_s ?? null,
    solve_s: nextSolver.solve_s ?? liveSolver?.solve_s ?? null,
    compile_cached: nextSolver.compile_cached || liveSolver?.compile_cached === true,
    phase_details: mergeSolverPhaseDetails(nextSolver.phase_details, liveSolver?.phase_details),
  };
}

function buildFailureSolverReport(message: string): SolverReport {
  const liveSolver = solverWithLiveElapsed(state.liveSolver);
  return {
    completed: true,
    status_label: "Failed",
    status_kind: SOLVER_STATUS_KIND.error,
    iterations: state.latestProgress?.iteration ?? null,
    symbolic_setup_s: liveSolver?.symbolic_setup_s ?? null,
    jit_s: liveSolver?.jit_s ?? null,
    solve_s: liveSolver?.solve_s ?? null,
    compile_cached: liveSolver?.compile_cached ?? false,
    phase_details: normalizeSolverPhaseDetails(liveSolver?.phase_details),
    failure_message: message,
  };
}

function mergeSolverReport(
  next: SolverReport,
  fallback: SolverReport | null | undefined,
): SolverReport {
  return {
    ...next,
    iterations: next.iterations ?? fallback?.iterations ?? null,
    symbolic_setup_s: next.symbolic_setup_s ?? fallback?.symbolic_setup_s ?? null,
    jit_s: next.jit_s ?? fallback?.jit_s ?? null,
    solve_s: next.solve_s ?? fallback?.solve_s ?? null,
    compile_cached: next.compile_cached || fallback?.compile_cached === true,
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

function renderLog(): void {
  solverLogEl.innerHTML = state.logLines
    .map((entry) => {
      const levelClass = logLevelClass(entry.level);
      return `<span class="log-line ${levelClass}">${ansiToHtml(entry.text) || "&nbsp;"}</span>`;
    })
    .join("");
  solverLogEl.scrollTop = solverLogEl.scrollHeight;
}

function renderCompileCacheStatus(): void {
  const rows = [...state.compileCacheStatuses].sort((left, right) => {
    const problemOrder = left.problem_name.localeCompare(right.problem_name);
    if (problemOrder !== 0) {
      return problemOrder;
    }
    return left.variant_label.localeCompare(right.variant_label);
  });

  if (rows.length === 0) {
    prewarmStatusEl.innerHTML = `<div class="placeholder">Compile cache status will appear here.</div>`;
    return;
  }

  const table = document.createElement("div");
  table.className = "compile-cache-table";

  const header = document.createElement("div");
  header.className = "compile-cache-row compile-cache-row-header";
  header.innerHTML = `
    <div>Problem</div>
    <div>Variant</div>
    <div>Symbolic</div>
    <div>JIT</div>
    <div>Status</div>
  `;
  table.appendChild(header);

  for (const row of rows) {
    const warming = isCompileWarmInProgress(row);
    const statusLabel = warming
      ? "warming"
      : row.state === COMPILE_CACHE_STATE.ready
        ? "ready"
        : "cold";
    const rowEl = document.createElement("div");
    rowEl.className = [
      "compile-cache-row",
      isCompileTarget(row) ? "compile-cache-row-current" : "",
    ].filter(Boolean).join(" ");
    rowEl.innerHTML = `
      <div class="compile-cache-problem">${escapeHtml(row.problem_name)}</div>
      <div class="compile-cache-variant">${escapeHtml(row.variant_label)}</div>
      <div>${escapeHtml(formatDuration(row.symbolic_setup_s))}</div>
      <div>${escapeHtml(formatDuration(row.jit_s))}</div>
      <div><span class="compile-cache-badge compile-cache-badge-${statusLabel}">${statusLabel}</span></div>
    `;
    table.appendChild(rowEl);
  }

  prewarmStatusEl.replaceChildren(table);
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
    state: decodeWireEnum(COMPILE_CACHE_STATE_FROM_WIRE, status.state, COMPILE_CACHE_STATE.cold),
    symbolic_setup_s: status.symbolic_setup_s ?? null,
    jit_s: status.jit_s ?? null,
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
  };
}

function normalizeSolveStatus(status: WireSolveStatus): SolveStatus {
  return {
    stage: decodeWireEnum(SOLVE_STAGE_FROM_WIRE, status.stage, SOLVE_STAGE.symbolicSetup),
    solver_method:
      status.solver_method == null
        ? null
        : decodeWireEnum(SOLVER_METHOD_FROM_WIRE, status.solver_method, SOLVER_METHOD.sqp),
    solver: normalizeSolverReport(status.solver),
  };
}

function normalizeArtifact(artifact: WireSolveArtifact): SolveArtifact {
  return {
    ...artifact,
    solver: normalizeSolverReport(artifact.solver),
    summary: (artifact.summary ?? []).map(normalizeMetric),
    constraint_panels: normalizeConstraintPanels(artifact.constraint_panels),
    charts: (artifact.charts ?? []).map(normalizeChart),
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

function findControlBySemantic(
  spec: ProblemSpec | undefined,
  semantic: ControlSemanticCode,
): ControlSpec | null {
  return spec?.controls.find((control) => control.semantic === semantic) ?? null;
}

function currentSharedControlValue(semantic: ControlSemanticCode, fallback = 0): number {
  const control = findControlBySemantic(currentSpec(), semantic);
  if (!control) {
    return fallback;
  }
  return Number(state.values[control.id] ?? control.default ?? fallback);
}

function formatSharedControlValue(
  semantic: ControlSemanticCode,
  fallback = "--",
): string {
  const control = findControlBySemantic(currentSpec(), semantic);
  if (!control) {
    return fallback;
  }
  const numeric = Number(state.values[control.id] ?? control.default);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return formatControlValue(control, numeric);
}

function currentTranscriptionMethodValue(): number {
  return currentSharedControlValue(CONTROL_SEMANTIC.transcriptionMethod, 0);
}

function isStructuralControl(control: ControlSpec): boolean {
  switch (control.semantic) {
    case CONTROL_SEMANTIC.transcriptionMethod:
    case CONTROL_SEMANTIC.transcriptionIntervals:
    case CONTROL_SEMANTIC.collocationFamily:
    case CONTROL_SEMANTIC.collocationDegree:
      return true;
    default:
      return false;
  }
}

function currentPrewarmSignature(): string | null {
  const spec = currentSpec();
  if (!spec) {
    return null;
  }
  return JSON.stringify({
    problem: spec.wire_id,
    method: currentSharedControlValue(CONTROL_SEMANTIC.transcriptionMethod, 0),
    intervals: currentSharedControlValue(CONTROL_SEMANTIC.transcriptionIntervals, 0),
    family: currentSharedControlValue(CONTROL_SEMANTIC.collocationFamily, 0),
    degree: currentSharedControlValue(CONTROL_SEMANTIC.collocationDegree, 0),
  });
}

function currentCompileVariantId(): string | null {
  if (!currentSpec()) {
    return null;
  }
  if (currentTranscriptionMethodValue() !== DIRECT_COLLOCATION_VALUE) {
    return "multiple_shooting";
  }
  return currentSharedControlValue(CONTROL_SEMANTIC.collocationFamily, 0) === 1
    ? "direct_collocation_radau_iia"
    : "direct_collocation_legendre";
}

function isCompileTarget(status: CompileCacheStatus): boolean {
  const spec = currentSpec();
  const variantId = currentCompileVariantId();
  return spec != null
    && variantId != null
    && status.problem_id === spec.id
    && status.variant_id === variantId;
}

function isCompileWarmInProgress(status: CompileCacheStatus): boolean {
  if (!isCompileTarget(status) || status.state === COMPILE_CACHE_STATE.ready) {
    return false;
  }
  if (state.prewarmInFlightSignature !== null) {
    return true;
  }
  const stage = currentSolveStage();
  return stage === SOLVE_STAGE.symbolicSetup || stage === SOLVE_STAGE.jitCompilation;
}

function clearScheduledPrewarm(): void {
  if (state.prewarmTimer !== null) {
    window.clearTimeout(state.prewarmTimer);
    state.prewarmTimer = null;
  }
}

async function refreshCompileCacheStatus(): Promise<void> {
  try {
    state.compileCacheStatuses = await fetchJson("/api/prewarm_status", (value) => {
      const entries = readJsonArray(value, "/api/prewarm_status");
      return entries.map((entry, index) =>
        normalizeCompileCacheStatus(
          readWireCompileCacheStatus(entry, `/api/prewarm_status[${index}]`),
        ));
    });
    renderCompileCacheStatus();
  } catch (error) {
    console.warn("compile cache status refresh failed", error);
  }
}

async function runPrewarm(expectedSignature: string): Promise<void> {
  const spec = currentSpec();
  if (!spec || currentPrewarmSignature() !== expectedSignature || state.solving) {
    return;
  }
  state.prewarmInFlightSignature = expectedSignature;
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
    if (currentPrewarmSignature() === expectedSignature) {
      state.lastPrewarmSignature = expectedSignature;
    }
  } catch (error) {
    console.warn("prewarm failed", error);
  } finally {
    if (state.prewarmInFlightSignature === expectedSignature) {
      state.prewarmInFlightSignature = null;
    }
    void refreshCompileCacheStatus();
    renderCompileCacheStatus();
    const latestSignature = currentPrewarmSignature();
    if (
      latestSignature !== null
      && latestSignature !== state.lastPrewarmSignature
      && latestSignature !== state.prewarmInFlightSignature
    ) {
      schedulePrewarm();
    }
  }
}

function schedulePrewarm(): void {
  if (state.solving) {
    return;
  }
  const signature = currentPrewarmSignature();
  if (
    signature === null
    || signature === state.lastPrewarmSignature
    || signature === state.prewarmInFlightSignature
  ) {
    return;
  }
  clearScheduledPrewarm();
  state.prewarmTimer = window.setTimeout(() => {
    state.prewarmTimer = null;
    void runPrewarm(signature);
  }, PREWARM_DELAY_MS);
}

function handleControlUpdate(control: ControlSpec): void {
  if (control.semantic === CONTROL_SEMANTIC.transcriptionMethod) {
    renderControls();
  }
  if (isStructuralControl(control)) {
    renderCompileCacheStatus();
    schedulePrewarm();
  }
}

function isControlVisible(control: ControlSpec): boolean {
  switch (control.visibility) {
    case CONTROL_VISIBILITY.directCollocationOnly:
      return currentTranscriptionMethodValue() === DIRECT_COLLOCATION_VALUE;
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

function isControlSectionCollapsed(section: ControlSectionCode): boolean {
  return state.collapsedControlSections[section];
}

function toggleControlSection(section: ControlSectionCode): void {
  state.collapsedControlSections[section] = !state.collapsedControlSections[section];
  renderControls();
}

function phaseLabel(phase: SolvePhaseCode): string {
  return PHASE_LABEL.get(phase) ?? "--";
}

function appendControl(wrapperParent: HTMLElement, control: ControlSpec): void {
  const wrapper = document.createElement("section");
  wrapper.className = "control-group";
  const value = state.values[control.id];
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
          `<option value="${choice.value}"${Number(choice.value) === Number(value) ? " selected" : ""}>${choice.label}</option>`,
      )
      .join("");
    wrapper.innerHTML = `
      <div class="control-header">
        <div>
          <div class="control-label">${control.label}</div>
          <div class="control-help">${control.help}</div>
        </div>
        <div class="value-pill">${formatValue(Number(value))}</div>
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
    wrapper.innerHTML = `
      <div class="control-header">
        <div>
          <div class="control-label">${control.label}</div>
          <div class="control-help">${control.help}</div>
        </div>
        <div class="value-pill">${formatValue(Number(value))}</div>
      </div>
      <div class="control-inputs control-inputs-select">
        <input type="text" value="${formatValue(Number(value))}" placeholder="${formatValue(control.default)}" spellcheck="false" />
      </div>
    `;
    const textInput = requiredChild<HTMLInputElement>(wrapper, "input");
    const pill = requiredChild<HTMLDivElement>(wrapper, ".value-pill");
    const sync = (raw: string): void => {
      const numeric = Number(raw);
      if (!Number.isFinite(numeric)) {
        return;
      }
      if (Number.isFinite(control.min) && numeric < control.min) {
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
      textInput.value = formatValue(Number(state.values[control.id]));
    });
    wrapperParent.appendChild(wrapper);
    return;
  }

  wrapper.innerHTML = `
    <div class="control-header">
      <div>
        <div class="control-label">${control.label}</div>
        <div class="control-help">${control.help}</div>
      </div>
      <div class="value-pill">${formatValue(Number(value))}</div>
    </div>
    <div class="control-inputs">
      <input type="range" min="${control.min}" max="${control.max}" step="${control.step}" value="${value}" />
      <input type="number" min="${control.min}" max="${control.max}" step="${control.step}" value="${value}" />
    </div>
  `;
  const rangeInput = requiredChild<HTMLInputElement>(wrapper, 'input[type="range"]');
  const numberInput = requiredChild<HTMLInputElement>(wrapper, 'input[type="number"]');
  const pill = requiredChild<HTMLDivElement>(wrapper, ".value-pill");
  const sync = (raw: string): void => {
    const numeric = Number(raw);
    state.values[control.id] = numeric;
    rangeInput.value = String(numeric);
    numberInput.value = String(numeric);
    pill.textContent = formatValue(numeric);
    handleControlUpdate(control);
  };
  rangeInput.addEventListener("input", (event) => {
    const target = readCurrentInputTarget(event, `${control.id} range input`);
    sync(target.value);
  });
  numberInput.addEventListener("input", (event) => {
    const target = readCurrentInputTarget(event, `${control.id} number input`);
    sync(target.value);
  });
  wrapperParent.appendChild(wrapper);
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
    if (view === sourceView) {
      continue;
    }
    tasks.push(window.Plotly.relayout(view.plotEl, payload));
  }
  Promise.allSettled(tasks).finally(() => {
    state.linkingChartRange = false;
  });
}

function createSceneView(scene: Scene2D): SceneView {
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
    scene,
    shell,
    meta,
    playButton,
    slider,
    plotEl,
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
      width: index === 0 ? 4 : 2.5,
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
        width: 3,
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
        size: 8,
        line: {
          color: "#e5f1f4",
          width: 1,
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
      width: 2,
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
      arrowwidth: 2.5,
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
    margin: { l: 74, r: 24, t: 18, b: 62 },
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
    scrollZoom: true,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toImage"],
  };
  window.Plotly.react(view.plotEl, data, layout, config);
}

function resetSolverPanel(): void {
  state.latestProgress = null;
  state.liveStatus = null;
  state.liveSolver = null;
  state.solveStartedAtMs = null;
  state.terminalSolver = null;
  state.pendingIterationEvent = null;
  state.iterationFlushScheduled = false;
  state.logLines = [];
  renderSolverSummary();
  renderConstraintPanels();
  renderLog();
  if (window.Plotly && state.progressPlotReady) {
    window.Plotly.purge(progressPlotEl);
  }
  progressPlotEl.innerHTML = `<div class="placeholder">Solve a problem to populate the live convergence history.</div>`;
  state.progressPlotReady = false;
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
  state.values = Object.fromEntries(spec.controls.map((control) => [control.id, control.default]));
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
      for (const control of section.controls) {
        appendControl(body, control);
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
      value: formatCompileDuration(solver?.symbolic_setup_s ?? null, solver?.compile_cached ?? false),
      active: activeStage === SOLVE_STAGE.symbolicSetup,
    },
    {
      label: "JIT",
      value: formatCompileDuration(solver?.jit_s ?? null, solver?.compile_cached ?? false),
      active: activeStage === SOLVE_STAGE.jitCompilation,
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
    items.push({ label: "α", value: progress.alpha == null ? "--" : progress.alpha.toExponential(3) });
    return items;
  }

  if (solver?.iterations != null) {
    items.push({ label: "Iteration", value: `${solver.iterations}` });
    const tfMetric = findMetric(state.artifact, METRIC_KEY.finalTime);
    if (tfMetric) {
      items.push({ label: "T", value: tfMetric.value });
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
      value: formatCompileDuration(solver.symbolic_setup_s ?? null, solver.compile_cached),
      active: activeStage === SOLVE_STAGE.symbolicSetup,
      details: solver.phase_details.symbolic_setup,
      fallbackText: "Building symbolic model and derivatives.",
    }),
    createSolverPhaseCard({
      label: "JIT",
      value: formatCompileDuration(solver.jit_s ?? null, solver.compile_cached),
      active: activeStage === SOLVE_STAGE.jitCompilation,
      details: solver.phase_details.jit,
      fallbackText: "Compiling numeric evaluation kernels.",
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
      const severityKind = constraintSeverityClass(entry.severity);
      const severityClass =
        severityKind === "success" ? "" : `constraint-entry-${severityKind}`;
      const boundsMarkup =
        kind === "ineq"
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
            <div class="constraint-entry-count">viol ${entry.violating_instances}/${entry.total_instances}</div>
          </div>
          <div class="constraint-entry-inline">
            <span class="constraint-inline-label">Worst</span>
            <span class="constraint-inline-value">${entry.worst_violation.toExponential(3)}</span>
          </div>
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
  const toleranceText = formatSharedControlValue(
    CONTROL_SEMANTIC.solverConstraintTolerance,
  );
  const pendingText = state.artifact == null && !state.solving
    ? `tol ${toleranceText}`
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
  state.logLines.push(...parts.map((textPart) => ({ text: textPart, level })));
  if (state.logLines.length > 240) {
    state.logLines.splice(0, state.logLines.length - 240);
  }
  renderLog();
}

function positiveFiniteOrNull(value: number | null | undefined): number | null {
  const numericValue = value ?? Number.NaN;
  return Number.isFinite(numericValue) && numericValue > 0 ? numericValue : null;
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
      width: 1,
      dash,
    },
    marker: {
      size: [],
      symbol: [],
      color,
      line: {
        color,
        width: 1,
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
      line: { color: PALETTE[0], width: 2.5 },
      marker: { size: 5 },
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: EQ_INF_LABEL,
      x: [],
      y: [],
      line: { color: PALETTE[1], width: 2.5 },
      marker: { size: 5 },
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: INEQ_INF_LABEL,
      x: [],
      y: [],
      line: { color: PALETTE[2], width: 2.5 },
      marker: { size: 5 },
    },
    {
      type: "scatter",
      mode: "lines+markers",
      name: DUAL_INF_LABEL,
      x: [],
      y: [],
      line: { color: PALETTE[3], width: 2.5 },
      marker: { size: 5 },
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

function residualValue(value: number | null | undefined): number | null {
  if (value == null || !Number.isFinite(value)) {
    return null;
  }
  return Math.max(value, 1.0e-14);
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
    300,
  );
  updateProgressThresholds(progress);
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

function applyIterationEvent(event: IterationSolveEvent, updateRunningStatus: boolean): void {
  state.latestProgress = event.progress;
  state.liveSolver = event.artifact.solver;
  state.artifact = event.artifact;
  renderSolverSummary();
  updateProgressPlot(event.progress);
  scheduleArtifactRender();
  if (updateRunningStatus && state.liveStatus?.stage === SOLVE_STAGE.solving) {
    setStatusDisplay(statusDisplayForSolveStatus(state.liveStatus, event.progress.iteration));
  }
}

function applySolveFailure(message: string): void {
  if (state.pendingIterationEvent) {
    const pendingEvent = state.pendingIterationEvent;
    state.pendingIterationEvent = null;
    applyIterationEvent(pendingEvent, false);
  }
  state.liveStatus = null;
  state.terminalSolver = buildFailureSolverReport(message);
  state.liveSolver = null;
  state.solveStartedAtMs = null;
  renderSolverSummary();
  renderMetrics();
  renderCompileCacheStatus();
  void refreshCompileCacheStatus();
  appendLogLine(`error: ${message}`, LOG_LEVEL.error);
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
  if (!state.sceneView || state.sceneView.scene !== scene) {
    state.sceneView = createSceneView(scene);
    sceneEl.replaceChildren(state.sceneView.shell);
  }

  const view = state.sceneView;
  if (!view) {
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

function chartLayoutKey(charts: Chart[]): string {
  return charts.map((chart) => chart.title).join("::");
}

function ensureChartViews(charts: Chart[]): void {
  const nextKey = chartLayoutKey(charts);
  if (state.chartLayoutKey === nextKey && state.chartViews.size === charts.length) {
    return;
  }

  resetChartViews();
  chartsEl.innerHTML = "";
  state.chartLayoutKey = nextKey;

  for (const chart of charts) {
    const shell = document.createElement("section");
    shell.className = "chart-shell";
    const header = document.createElement("div");
    header.className = "chart-header";
    header.innerHTML = `<div>${chart.title}</div><div class="card-subtitle">${chart.y_label}</div>`;
    const plotEl = createPlotlyHostElement("plot-surface");
    shell.append(header, plotEl);
    chartsEl.appendChild(shell);
    state.chartViews.set(chart.title, { plotEl, linkedRangeBound: false });
  }
}

function updateChart(view: ChartView | undefined, chart: Chart): void {
  if (!window.Plotly || !view) {
    return;
  }
  const groupOrder = new Map<string, number>();
  const colorIndexFor = (group: string): number => {
    if (!groupOrder.has(group)) {
      groupOrder.set(group, groupOrder.size);
    }
    return groupOrder.get(group)!;
  };
  const data = chart.series.map((series) => {
    const group = series.legend_group ?? series.name;
    const paletteIndex = colorIndexFor(group);
    const role = series.role ?? TIME_SERIES_ROLE.data;
    const isBound = role === TIME_SERIES_ROLE.lowerBound || role === TIME_SERIES_ROLE.upperBound;
    const color = isBound ? "#f25f5c" : PALETTE[paletteIndex % PALETTE.length];
    const dash = role === TIME_SERIES_ROLE.lowerBound
      ? "dash"
      : role === TIME_SERIES_ROLE.upperBound
        ? "longdash"
        : "solid";
    return {
      type: "scatter",
      mode: series.mode ?? "lines",
      name: series.name,
      legendgroup: group,
      showlegend: series.show_legend ?? true,
      x: series.x,
      y: series.y,
      line: {
        color,
        width: isBound ? 1.8 : paletteIndex === 0 ? 3.5 : 2.4,
        shape: "linear",
        dash,
      },
      marker: {
        color,
        size: isBound ? 0 : 6,
      },
    };
  });
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
      title: chart.x_label,
      autorange: state.linkedChartAutorange,
      gridcolor: "rgba(229, 241, 244, 0.08)",
      linecolor: "rgba(177, 214, 222, 0.18)",
      zeroline: false,
      ticks: "outside",
      titlefont: { color: "#94b6bd" },
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
  if (!state.linkedChartAutorange && state.linkedChartRange) {
    layout.xaxis.range = state.linkedChartRange.slice();
  }
  window.Plotly.react(view.plotEl, data, layout, config).then(() => {
    if (!view.linkedRangeBound && typeof view.plotEl.on === "function") {
      view.plotEl.on("plotly_relayout", (eventData) => {
        syncLinkedChartRange(view, eventData);
      });
      view.linkedRangeBound = true;
    }
  });
}

function renderCharts(): void {
  const charts = state.artifact?.charts ?? [];
  if (charts.length === 0) {
    resetChartViews();
    chartsEl.innerHTML = `<div class="placeholder">The solver will populate state, control, and constraint charts here.</div>`;
    return;
  }
  if (!window.Plotly) {
    chartsEl.innerHTML = `<div class="placeholder">Plotly is still loading.</div>`;
    return;
  }
  ensureChartViews(charts);
  charts.forEach((chart) => updateChart(state.chartViews.get(chart.title), chart));
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

function scheduleArtifactRender(): void {
  if (state.renderScheduled) {
    return;
  }
  state.renderScheduled = true;
  requestAnimationFrame(() => {
    state.renderScheduled = false;
    renderMetrics();
    renderConstraintPanels();
    renderScene();
    renderCharts();
    renderNotes(state.artifact?.notes ?? currentSpec()?.notes ?? []);
  });
}

function handleSolveEvent(event: SolveEvent): void {
  switch (event.kind) {
    case STREAM_EVENT_KIND.status:
      {
        const previousStage = state.liveStatus?.stage ?? null;
      state.liveStatus = event.status;
      if (event.status.stage === SOLVE_STAGE.solving && state.solveStartedAtMs == null) {
        state.solveStartedAtMs = performance.now();
      }
      if (event.status.stage !== SOLVE_STAGE.solving) {
        state.solveStartedAtMs = null;
      }
      state.liveSolver = buildStatusSolverReport(event.status);
      renderSolverSummary();
      renderCompileCacheStatus();
      if (previousStage !== SOLVE_STAGE.solving && event.status.stage === SOLVE_STAGE.solving) {
        void refreshCompileCacheStatus();
      }
      renderMetrics();
      setStatusDisplay(statusDisplayForSolveStatus(event.status, state.latestProgress?.iteration ?? null));
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
      state.solveStartedAtMs = null;
      state.artifact = {
        ...event.artifact,
        solver: state.terminalSolver,
      };
      state.animationIndex = 0;
      renderSolverSummary();
      renderCompileCacheStatus();
      void refreshCompileCacheStatus();
      scheduleArtifactRender();
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

  try {
    state.solving = true;
    clearScheduledPrewarm();
    solveButton.disabled = true;
    solveButton.setAttribute("aria-busy", "true");
    stopAnimation();
    state.artifact = null;
    state.animationIndex = 0;
    state.sceneView = null;
    resetSolverPanel();
    state.liveStatus = {
      stage: SOLVE_STAGE.symbolicSetup,
      solver_method: null,
      solver: {
        completed: false,
        status_label: "Setting up symbolic model...",
        status_kind: SOLVER_STATUS_KIND.info,
        iterations: null,
        symbolic_setup_s: null,
        jit_s: null,
        solve_s: null,
        compile_cached: false,
        phase_details: emptySolverPhaseDetails(),
      },
    };
    state.liveSolver = state.liveStatus.solver;
    state.solveStartedAtMs = null;
    renderMetrics();
    renderCompileCacheStatus();
    renderScene();
    renderCharts();
    setStatusDisplay(statusDisplayForSolveStatus(state.liveStatus));

    const response = await fetch(`/api/solve_stream/${spec.wire_id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values: state.values }),
    });

    if (!response.ok) {
      const payload = await readResponseJsonValue(response, `/api/solve_stream/${spec.wire_id}`);
      throw new Error(readOptionalErrorMessage(payload) ?? `Request failed with ${response.status}`);
    }

    await readNdjsonStream(response, handleSolveEvent);
  } catch (error) {
    applySolveFailure(
      describeThrownValue(
        error != null && typeof error === "object" ? error : String(error),
      ),
    );
  } finally {
    state.solving = false;
    solveButton.disabled = false;
    solveButton.setAttribute("aria-busy", "false");
    renderCompileCacheStatus();
  }
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
