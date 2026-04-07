const PALETTE = ["#f7b267", "#5bd1b5", "#7cc6fe", "#f25f5c", "#d7aefb", "#b8f2e6"];
const EQ_INF_LABEL = "‖eq‖∞";
const INEQ_INF_LABEL = "‖ineq₊‖∞";
const DUAL_INF_LABEL = "‖∇L‖∞";
const STEP_INF_LABEL = "‖Δx‖∞";

const state = {
  specs: [],
  selectedId: null,
  values: {},
  artifact: null,
  animationIndex: 0,
  playing: false,
  playHandle: null,
  solving: false,
  renderScheduled: false,
  chartViews: new Map(),
  chartLayoutKey: "",
  progressPlotReady: false,
  logLines: [],
  latestProgress: null,
  terminalSolver: null,
  pendingIterationEvent: null,
  iterationFlushScheduled: false,
  sceneView: null,
  linkedChartRange: null,
  linkedChartAutorange: true,
  linkingChartRange: false,
};

const problemList = document.querySelector("#problem-list");
const controls = document.querySelector("#controls");
const controlsForm = document.querySelector("#controls-form");
const solveButton = document.querySelector("#solve-button");
const statusEl = document.querySelector("#status");
const problemNameEl = document.querySelector("#problem-name");
const problemDescriptionEl = document.querySelector("#problem-description");
const metricsEl = document.querySelector("#metrics");
const sceneEl = document.querySelector("#scene");
const sceneSubtitleEl = document.querySelector("#scene-subtitle");
const chartsEl = document.querySelector("#charts");
const modelEl = document.querySelector("#model");
const notesEl = document.querySelector("#notes");
const solverSummaryEl = document.querySelector("#solver-summary");
const progressPlotEl = document.querySelector("#progress-plot");
const solverLogEl = document.querySelector("#solver-log");
const eqViolationsEl = document.querySelector("#eq-violations");
const ineqViolationsEl = document.querySelector("#ineq-violations");

const CONTROL_SECTION = Object.freeze({
  transcription: 0,
  solver: 1,
  problem: 2,
});
const CONTROL_SECTION_FROM_WIRE = Object.freeze({
  transcription: CONTROL_SECTION.transcription,
  solver: CONTROL_SECTION.solver,
  problem: CONTROL_SECTION.problem,
});
const CONTROL_EDITOR = Object.freeze({
  slider: 0,
  select: 1,
  text: 2,
});
const CONTROL_EDITOR_FROM_WIRE = Object.freeze({
  slider: CONTROL_EDITOR.slider,
  select: CONTROL_EDITOR.select,
  text: CONTROL_EDITOR.text,
});
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
});
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
});
const CONTROL_VISIBILITY = Object.freeze({
  always: 0,
  directCollocationOnly: 1,
});
const CONTROL_VISIBILITY_FROM_WIRE = Object.freeze({
  always: CONTROL_VISIBILITY.always,
  direct_collocation_only: CONTROL_VISIBILITY.directCollocationOnly,
});
const CONTROL_VALUE_DISPLAY = Object.freeze({
  scalar: 0,
  integer: 1,
  scientific: 2,
});
const CONTROL_VALUE_DISPLAY_FROM_WIRE = Object.freeze({
  scalar: CONTROL_VALUE_DISPLAY.scalar,
  integer: CONTROL_VALUE_DISPLAY.integer,
  scientific: CONTROL_VALUE_DISPLAY.scientific,
});
const LOG_LEVEL = Object.freeze({
  console: 0,
  info: 1,
  warning: 2,
  error: 3,
});
const LOG_LEVEL_FROM_WIRE = Object.freeze({
  console: LOG_LEVEL.console,
  info: LOG_LEVEL.info,
  warning: LOG_LEVEL.warning,
  error: LOG_LEVEL.error,
});
const TIME_SERIES_ROLE = Object.freeze({
  data: 0,
  lowerBound: 1,
  upperBound: 2,
});
const TIME_SERIES_ROLE_FROM_WIRE = Object.freeze({
  data: TIME_SERIES_ROLE.data,
  lower_bound: TIME_SERIES_ROLE.lowerBound,
  upper_bound: TIME_SERIES_ROLE.upperBound,
});
const SOLVER_STATUS_KIND = Object.freeze({
  success: 0,
  warning: 1,
  error: 2,
  info: 3,
});
const SOLVER_STATUS_KIND_FROM_WIRE = Object.freeze({
  success: SOLVER_STATUS_KIND.success,
  warning: SOLVER_STATUS_KIND.warning,
  error: SOLVER_STATUS_KIND.error,
  info: SOLVER_STATUS_KIND.info,
});
const CONSTRAINT_PANEL_SEVERITY = Object.freeze({
  fullAccuracy: 0,
  reducedAccuracy: 1,
  violated: 2,
});
const CONSTRAINT_PANEL_SEVERITY_FROM_WIRE = Object.freeze({
  full_accuracy: CONSTRAINT_PANEL_SEVERITY.fullAccuracy,
  reduced_accuracy: CONSTRAINT_PANEL_SEVERITY.reducedAccuracy,
  violated: CONSTRAINT_PANEL_SEVERITY.violated,
});
const CONSTRAINT_PANEL_CATEGORY = Object.freeze({
  boundaryEquality: 0,
  boundaryInequality: 1,
  path: 2,
  continuityState: 3,
  continuityControl: 4,
  collocationState: 5,
  collocationControl: 6,
  finalTime: 7,
});
const CONSTRAINT_PANEL_CATEGORY_FROM_WIRE = Object.freeze({
  boundary_equality: CONSTRAINT_PANEL_CATEGORY.boundaryEquality,
  boundary_inequality: CONSTRAINT_PANEL_CATEGORY.boundaryInequality,
  path: CONSTRAINT_PANEL_CATEGORY.path,
  continuity_state: CONSTRAINT_PANEL_CATEGORY.continuityState,
  continuity_control: CONSTRAINT_PANEL_CATEGORY.continuityControl,
  collocation_state: CONSTRAINT_PANEL_CATEGORY.collocationState,
  collocation_control: CONSTRAINT_PANEL_CATEGORY.collocationControl,
  final_time: CONSTRAINT_PANEL_CATEGORY.finalTime,
});
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
});
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
});
const SOLVE_PHASE = Object.freeze({
  initial: 0,
  acceptedStep: 1,
  postConvergence: 2,
  converged: 3,
  regular: 4,
  restoration: 5,
});
const SOLVE_PHASE_FROM_WIRE = Object.freeze({
  initial: SOLVE_PHASE.initial,
  accepted_step: SOLVE_PHASE.acceptedStep,
  post_convergence: SOLVE_PHASE.postConvergence,
  converged: SOLVE_PHASE.converged,
  regular: SOLVE_PHASE.regular,
  restoration: SOLVE_PHASE.restoration,
});
const STREAM_EVENT_KIND = Object.freeze({
  status: 0,
  log: 1,
  iteration: 2,
  final: 3,
  error: 4,
});
const STREAM_EVENT_KIND_FROM_WIRE = Object.freeze({
  status: STREAM_EVENT_KIND.status,
  log: STREAM_EVENT_KIND.log,
  iteration: STREAM_EVENT_KIND.iteration,
  final: STREAM_EVENT_KIND.final,
  error: STREAM_EVENT_KIND.error,
});
const PROBLEM_ID = Object.freeze({
  optimalDistanceGlider: 0,
  linearSManeuver: 1,
  sailboatUpwind: 2,
  craneTransfer: 3,
});
const PROBLEM_ID_FROM_WIRE = Object.freeze({
  optimal_distance_glider: PROBLEM_ID.optimalDistanceGlider,
  linear_s_maneuver: PROBLEM_ID.linearSManeuver,
  sailboat_upwind: PROBLEM_ID.sailboatUpwind,
  crane_transfer: PROBLEM_ID.craneTransfer,
});
const SECTION_META = [
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
const PHASE_LABEL = new Map([
  [SOLVE_PHASE.initial, "initial"],
  [SOLVE_PHASE.acceptedStep, "accepted step"],
  [SOLVE_PHASE.postConvergence, "post convergence"],
  [SOLVE_PHASE.converged, "converged"],
  [SOLVE_PHASE.regular, "regular"],
  [SOLVE_PHASE.restoration, "restoration"],
]);
const DIRECT_COLLOCATION_VALUE = 1;

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`.trim();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.error ?? `Request failed with ${response.status}`);
  }
  return payload;
}

function fmt(value, digits = 2) {
  return Number(value).toFixed(digits).replace(/\.00$/, "");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll(" ", "&nbsp;");
}

function isTextEntryControl(control) {
  return control.editor === CONTROL_EDITOR.text;
}

function formatControlValue(control, numeric) {
  switch (control.value_display) {
    case CONTROL_VALUE_DISPLAY.integer:
      return String(Math.round(numeric));
    case CONTROL_VALUE_DISPLAY.scientific:
      return Number(numeric).toExponential(2);
    default:
      return `${fmt(numeric, 3)} ${control.unit}`.trim();
  }
}

function formatDuration(seconds) {
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

function statusClass(kind) {
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

function constraintSeverityClass(severity) {
  switch (severity) {
    case CONSTRAINT_PANEL_SEVERITY.fullAccuracy:
      return "success";
    case CONSTRAINT_PANEL_SEVERITY.reducedAccuracy:
      return "warning";
    default:
      return "error";
  }
}

function boundSeverityClass(severity) {
  switch (severity) {
    case CONSTRAINT_PANEL_SEVERITY.violated:
      return "error";
    case CONSTRAINT_PANEL_SEVERITY.reducedAccuracy:
      return "warning";
    default:
      return "neutral";
  }
}

function logLevelClass(level) {
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

function solverStatus(artifact) {
  return artifact?.solver?.status_label ?? null;
}

function solverStatusKind(artifact) {
  return artifact?.solver?.status_kind ?? SOLVER_STATUS_KIND.info;
}

function hasSolverReport(solver) {
  return solver != null && solver.completed === true;
}

function currentSolverReport() {
  if (state.terminalSolver && hasSolverReport(state.terminalSolver)) {
    return state.terminalSolver;
  }
  return hasSolverReport(state.artifact?.solver) ? state.artifact.solver : null;
}

function buildFailureSolverReport(message) {
  return {
    completed: true,
    status_label: "Failed",
    status_kind: SOLVER_STATUS_KIND.error,
    iterations: state.latestProgress?.iteration ?? null,
    symbolic_setup_s: state.artifact?.solver?.symbolic_setup_s ?? null,
    jit_s: state.artifact?.solver?.jit_s ?? null,
    solve_s: state.artifact?.solver?.solve_s ?? null,
    failure_message: message,
  };
}

const ANSI_SGR_RE = /\[([0-9;]*)m/g;

function ansiClassName(state) {
  const classes = [];
  if (state.bold) {
    classes.push("ansi-bold");
  }
  if (state.color) {
    classes.push(`ansi-${state.color}`);
  }
  return classes.join(" ");
}

function pushAnsiSegment(parts, text, state) {
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

function applyAnsiCodes(state, codesText) {
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

function ansiToHtml(raw) {
  const input = String(raw ?? "");
  const parts = [];
  const state = { bold: false, color: null };
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

function renderLog() {
  solverLogEl.innerHTML = state.logLines
    .map((entry) => {
      const levelClass = logLevelClass(entry.level);
      return `<span class="log-line ${levelClass}">${ansiToHtml(entry.text) || "&nbsp;"}</span>`;
    })
    .join("");
  solverLogEl.scrollTop = solverLogEl.scrollHeight;
}

function decodeWireEnum(map, wireValue, fallback) {
  return Object.prototype.hasOwnProperty.call(map, wireValue) ? map[wireValue] : fallback;
}

function normalizeControl(control) {
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
  };
}

function normalizeProblemSpec(spec) {
  return {
    ...spec,
    wire_id: spec.id,
    id: decodeWireEnum(PROBLEM_ID_FROM_WIRE, spec.id, PROBLEM_ID.optimalDistanceGlider),
    controls: (spec.controls ?? []).map(normalizeControl),
  };
}

function normalizeSolverReport(solver) {
  if (!solver) {
    return solver;
  }
  return {
    ...solver,
    status_kind: decodeWireEnum(
      SOLVER_STATUS_KIND_FROM_WIRE,
      solver.status_kind,
      SOLVER_STATUS_KIND.info,
    ),
  };
}

function normalizeTimeSeries(series) {
  return {
    ...series,
    role: decodeWireEnum(TIME_SERIES_ROLE_FROM_WIRE, series.role, TIME_SERIES_ROLE.data),
  };
}

function normalizeChart(chart) {
  return {
    ...chart,
    series: (chart.series ?? []).map(normalizeTimeSeries),
  };
}

function normalizeMetric(metric) {
  return {
    ...metric,
    key: decodeWireEnum(METRIC_KEY_FROM_WIRE, metric.key, METRIC_KEY.custom),
  };
}

function normalizeConstraintPanelEntry(entry) {
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

function normalizeConstraintPanels(panels) {
  if (!panels) {
    return { equalities: [], inequalities: [] };
  }
  return {
    equalities: (panels.equalities ?? []).map(normalizeConstraintPanelEntry),
    inequalities: (panels.inequalities ?? []).map(normalizeConstraintPanelEntry),
  };
}

function normalizeProgress(progress) {
  if (!progress) {
    return progress;
  }
  return {
    ...progress,
    phase: decodeWireEnum(SOLVE_PHASE_FROM_WIRE, progress.phase, SOLVE_PHASE.initial),
  };
}

function normalizeArtifact(artifact) {
  if (!artifact) {
    return artifact;
  }
  return {
    ...artifact,
    solver: normalizeSolverReport(artifact.solver),
    summary: (artifact.summary ?? []).map(normalizeMetric),
    constraint_panels: normalizeConstraintPanels(artifact.constraint_panels),
    charts: (artifact.charts ?? []).map(normalizeChart),
  };
}

function findMetric(artifact, key) {
  return artifact?.summary?.find((metric) => metric.key === key) ?? null;
}

function normalizeSolveEvent(event) {
  const kind = decodeWireEnum(STREAM_EVENT_KIND_FROM_WIRE, event.kind, STREAM_EVENT_KIND.error);
  switch (kind) {
    case STREAM_EVENT_KIND.status:
      return { ...event, kind };
    case STREAM_EVENT_KIND.log:
      return {
        ...event,
        kind,
        level: decodeWireEnum(LOG_LEVEL_FROM_WIRE, event.level, LOG_LEVEL.console),
      };
    case STREAM_EVENT_KIND.iteration:
      return {
        ...event,
        kind,
        progress: normalizeProgress(event.progress),
        artifact: normalizeArtifact(event.artifact),
      };
    case STREAM_EVENT_KIND.final:
      return {
        ...event,
        kind,
        artifact: normalizeArtifact(event.artifact),
      };
    case STREAM_EVENT_KIND.error:
    default:
      return { ...event, kind: STREAM_EVENT_KIND.error };
  }
}

function currentSpec() {
  return state.specs.find((spec) => spec.id === state.selectedId);
}

function findControlBySemantic(spec, semantic) {
  return spec?.controls.find((control) => control.semantic === semantic) ?? null;
}

function currentSharedControlValue(semantic, fallback = 0) {
  const control = findControlBySemantic(currentSpec(), semantic);
  if (!control) {
    return fallback;
  }
  return Number(state.values[control.id] ?? control.default ?? fallback);
}

function currentTranscriptionMethodValue() {
  return currentSharedControlValue(CONTROL_SEMANTIC.transcriptionMethod, 0);
}

function isControlVisible(control) {
  switch (control.visibility) {
    case CONTROL_VISIBILITY.directCollocationOnly:
      return currentTranscriptionMethodValue() === DIRECT_COLLOCATION_VALUE;
    default:
      return true;
  }
}

function controlSections(spec) {
  const sections = SECTION_META.map((meta) => ({
    key: meta.key,
    title: meta.title,
    subtitle: meta.subtitle,
    controls: [],
  }));
  const byKey = new Map(sections.map((section) => [section.key, section]));
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

function phaseLabel(phase) {
  return PHASE_LABEL.get(phase) ?? "--";
}

function appendControl(wrapperParent, control) {
  const wrapper = document.createElement("section");
  wrapper.className = "control-group";
  const value = state.values[control.id];
  const choiceMap = new Map((control.choices ?? []).map((choice) => [Number(choice.value), choice.label]));
  const formatValue = (numeric) => {
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
    const selectInput = wrapper.querySelector("select");
    const pill = wrapper.querySelector(".value-pill");
    selectInput.addEventListener("input", (event) => {
      const numeric = Number(event.target.value);
      state.values[control.id] = numeric;
      pill.textContent = formatValue(numeric);
      if (control.semantic === CONTROL_SEMANTIC.transcriptionMethod) {
        renderControls();
      }
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
    const textInput = wrapper.querySelector("input");
    const pill = wrapper.querySelector(".value-pill");
    const sync = (raw) => {
      const numeric = Number(raw);
      if (!Number.isFinite(numeric)) {
        return;
      }
      if (Number.isFinite(control.min) && numeric < control.min) {
        return;
      }
      state.values[control.id] = numeric;
      pill.textContent = formatValue(numeric);
    };
    textInput.addEventListener("input", (event) => sync(event.target.value));
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
  const [rangeInput, numberInput] = wrapper.querySelectorAll("input");
  const pill = wrapper.querySelector(".value-pill");
  const sync = (raw) => {
    const numeric = Number(raw);
    state.values[control.id] = numeric;
    rangeInput.value = String(numeric);
    numberInput.value = String(numeric);
    pill.textContent = formatValue(numeric);
  };
  rangeInput.addEventListener("input", (event) => sync(event.target.value));
  numberInput.addEventListener("input", (event) => sync(event.target.value));
  wrapperParent.appendChild(wrapper);
}

function resetChartViews() {
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

function linkedChartRelayoutPayload() {
  if (state.linkedChartAutorange || !state.linkedChartRange) {
    return { "xaxis.autorange": true };
  }
  return {
    "xaxis.autorange": false,
    "xaxis.range": state.linkedChartRange.slice(),
  };
}

function extractLinkedChartRange(eventData) {
  if (!eventData || typeof eventData !== "object") {
    return null;
  }
  if (eventData["xaxis.autorange"]) {
    return { autorange: true, range: null };
  }
  if (Array.isArray(eventData["xaxis.range"]) && eventData["xaxis.range"].length === 2) {
    return {
      autorange: false,
      range: [eventData["xaxis.range"][0], eventData["xaxis.range"][1]],
    };
  }
  if ("xaxis.range[0]" in eventData && "xaxis.range[1]" in eventData) {
    return {
      autorange: false,
      range: [eventData["xaxis.range[0]"], eventData["xaxis.range[1]"],],
    };
  }
  return null;
}

function syncLinkedChartRange(sourceView, eventData) {
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
  const tasks = [];
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

function createSceneView(scene) {
  const shell = document.createElement("div");
  shell.className = "scene-shell";

  const toolbar = document.createElement("div");
  toolbar.className = "scene-toolbar";
  const meta = document.createElement("div");
  meta.className = "scene-meta";
  meta.textContent = `${scene.x_label} · ${scene.y_label}`;
  toolbar.appendChild(meta);

  let playButton = null;
  let slider = null;
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
      state.animationIndex = Number(event.target.value);
      renderScene();
    });
    controlsEl.append(playButton, slider);
    toolbar.appendChild(controlsEl);
  }

  const plotEl = document.createElement("div");
  plotEl.className = "plot-surface scene-plot-surface";
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

function scenePlotBounds(scene) {
  return collectSceneBounds(scene);
}

function scenePathTraces(scene) {
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

function sceneFrameTraces(scene, frameIndex) {
  const frames = scene.animation?.frames ?? [];
  if (frames.length === 0) {
    return [];
  }
  const frame = frames[Math.min(frameIndex, frames.length - 1)];
  const traces = [];

  if ((frame.segments ?? []).length > 0) {
    const x = [];
    const y = [];
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

  const entries = Object.entries(frame.points ?? {});
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

function sceneShapes(scene) {
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

function sceneAnnotations(scene) {
  const annotations = [];
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

function updateScenePlot(view) {
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
    dragmode: "pan",
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
    displayModeBar: false,
  };
  window.Plotly.react(view.plotEl, data, layout, config);
}

function resetSolverPanel() {
  state.latestProgress = null;
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
}

function selectProblem(problemId) {
  const spec = state.specs.find((item) => item.id === problemId);
  if (!spec) {
    return;
  }
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
}

function renderProblemList() {
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

function renderOverview() {
  const spec = currentSpec();
  problemNameEl.textContent = spec?.name ?? "";
  problemDescriptionEl.textContent = spec?.description ?? "";
}

function renderControls() {
  const spec = currentSpec();
  controls.innerHTML = "";
  if (!spec) {
    return;
  }

  const sections = controlSections(spec);
  for (const section of sections) {
    const shell = document.createElement("section");
    shell.className = "control-section";
    shell.innerHTML = `
      <div class="control-section-header">
        <div class="control-section-title">${section.title}</div>
        <div class="control-section-help">${section.subtitle}</div>
      </div>
      <div class="control-section-body"></div>
    `;
    const body = shell.querySelector(".control-section-body");
    for (const control of section.controls) {
      appendControl(body, control);
    }
    controls.appendChild(shell);
  }
}

function renderMetrics() {
  metricsEl.innerHTML = "";
  const solver = currentSolverReport();
  if (!hasSolverReport(solver)) {
    metricsEl.className = "metrics empty";
    return;
  }

  const cards = [
    { label: "Solve Status", value: solver.status_label, kind: solver.status_kind },
    { label: "Symbolic Setup", value: formatDuration(solver.symbolic_setup_s) },
    { label: "JIT Time", value: formatDuration(solver.jit_s) },
    { label: "Solve Time", value: formatDuration(solver.solve_s) },
    { label: "Iterations", value: solver.iterations == null ? "--" : String(solver.iterations) },
  ];
  metricsEl.className = "metrics";
  for (const metric of cards) {
    const card = document.createElement("article");
    const kindClass = metric.kind == null ? "" : `metric-card-${statusClass(metric.kind)}`;
    card.className = `metric-card ${kindClass}`.trim();
    card.innerHTML = `
      <div class="metric-label">${metric.label}</div>
      <div class="metric-value">${metric.value}</div>
    `;
    metricsEl.appendChild(card);
  }
}

function renderNotes(notes) {
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

function renderSolverSummary() {
  const progress = state.latestProgress;
  const solver = currentSolverReport();
  if (!progress && !solver) {
    solverSummaryEl.innerHTML = `<div class="placeholder">Solve a problem to populate the latest iteration metrics.</div>`;
    return;
  }

  const items = [];
  if (solver) {
    items.push(["Status", solver.status_label, solver.status_kind]);
  }
  if (progress) {
    const tfMetric = findMetric(state.artifact, METRIC_KEY.finalTime);
    items.push(
      ["Iteration", `${progress.iteration}`],
      ["Phase", phaseLabel(progress.phase)],
      ["Objective", progress.objective.toExponential(3)],
      ["T", tfMetric?.value ?? "--"],
      [EQ_INF_LABEL, progress.eq_inf == null ? "--" : progress.eq_inf.toExponential(3)],
      [INEQ_INF_LABEL, progress.ineq_inf == null ? "--" : progress.ineq_inf.toExponential(3)],
      [DUAL_INF_LABEL, progress.dual_inf.toExponential(3)],
      [STEP_INF_LABEL, progress.step_inf == null ? "--" : progress.step_inf.toExponential(3)],
      ["α", progress.alpha == null ? "--" : progress.alpha.toExponential(3)],
    );
  } else if (solver?.iterations != null) {
    items.push(["Iteration", `${solver.iterations}`]);
    const tfMetric = findMetric(state.artifact, METRIC_KEY.finalTime);
    if (tfMetric) {
      items.push(["T", tfMetric.value]);
    }
  }

  solverSummaryEl.innerHTML = items
    .map(([label, value, kind]) => {
      const kindClass = kind == null ? "" : `solver-chip-${statusClass(kind)}`;
      return `
        <article class="solver-chip ${kindClass}">
          <div class="solver-chip-label">${label}</div>
          <div class="solver-chip-value">${value}</div>
        </article>
      `;
    })
    .join("");
}

function formatConstraintValue(value) {
  if (value == null || !Number.isFinite(value)) {
    return "--";
  }
  const abs = Math.abs(value);
  if (abs >= 1e3 || (abs > 0 && abs < 1e-2)) {
    return Number(value).toExponential(3);
  }
  return fmt(value, 3);
}

function renderBoundToken(value, severity, fallback) {
  const className = `constraint-bound constraint-bound-${boundSeverityClass(severity)}`;
  const label = value == null ? fallback : formatConstraintValue(value);
  return `<span class="${className}">${label}</span>`;
}

function renderConstraintPanel(target, entries, emptyText, kind) {
  if (!target) {
    return;
  }
  if (!entries || entries.length === 0) {
    target.innerHTML = `<div class="placeholder">${emptyText}</div>`;
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

function renderConstraintPanels() {
  const panels = state.artifact?.constraint_panels ?? { equalities: [], inequalities: [] };
  renderConstraintPanel(
    eqViolationsEl,
    panels.equalities,
    "The largest grouped equality residuals will appear here while the solve is running.",
    "eq",
  );
  renderConstraintPanel(
    ineqViolationsEl,
    panels.inequalities,
    "The worst grouped inequality bounds will appear here while the solve is running.",
    "ineq",
  );
}

function appendLogLine(line, level = LOG_LEVEL.console) {
  const normalized = String(line ?? "").replaceAll("\r\n", "\n").replaceAll("\r", "\n");
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

function artifactTermination(artifact) {
  return solverStatus(artifact);
}

function positiveFiniteOrNull(value) {
  return Number.isFinite(value) && value > 0 ? value : null;
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

function toleranceSeverity(value, tolerance) {
  if (!Number.isFinite(value) || !Number.isFinite(tolerance) || tolerance <= 0) {
    return CONSTRAINT_PANEL_SEVERITY.fullAccuracy;
  }
  if (value <= tolerance) {
    return CONSTRAINT_PANEL_SEVERITY.fullAccuracy;
  }
  if (value <= 100 * tolerance) {
    return CONSTRAINT_PANEL_SEVERITY.reducedAccuracy;
  }
  return CONSTRAINT_PANEL_SEVERITY.violated;
}

function toleranceStatusLabel(severity) {
  switch (severity) {
    case CONSTRAINT_PANEL_SEVERITY.reducedAccuracy:
      return "Satisfied to reduced accuracy";
    case CONSTRAINT_PANEL_SEVERITY.violated:
      return "Not satisfied";
    default:
      return "Satisfied to full accuracy";
  }
}

function toleranceMarkerSymbol(severity) {
  switch (severity) {
    case CONSTRAINT_PANEL_SEVERITY.reducedAccuracy:
      return "diamond-open";
    case CONSTRAINT_PANEL_SEVERITY.violated:
      return "x-open";
    default:
      return "circle-open";
  }
}

function toleranceTraceOpacity(severity) {
  switch (severity) {
    case CONSTRAINT_PANEL_SEVERITY.reducedAccuracy:
      return 0.72;
    case CONSTRAINT_PANEL_SEVERITY.violated:
      return 1.0;
    default:
      return 0.35;
  }
}

function progressToleranceTrace(name, color, dash) {
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

function updateProgressThresholdTrace(traceIndex, name, color, dash, tolerance, currentValue, maxX) {
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

function updateProgressThresholds(progress) {
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

function ensureProgressPlot() {
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

function residualValue(value) {
  if (value == null || !Number.isFinite(value)) {
    return null;
  }
  return Math.max(value, 1.0e-14);
}

function updateProgressPlot(progress) {
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

function scheduleIterationUpdate() {
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

function applyIterationEvent(event, updateRunningStatus) {
  state.latestProgress = event.progress;
  state.artifact = event.artifact;
  renderSolverSummary();
  updateProgressPlot(event.progress);
  scheduleArtifactRender();
  if (updateRunningStatus) {
    setStatus(`Running iteration ${event.progress.iteration}...`, "");
  }
}

function typesetMath(root) {
  if (!root || !window.MathJax?.typesetPromise) {
    return;
  }
  if (typeof window.MathJax.typesetClear === "function") {
    window.MathJax.typesetClear([root]);
  }
  window.MathJax.typesetPromise([root]).catch(() => {});
}

function renderModel(spec) {
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

function collectSceneBounds(scene) {
  const points = [];
  for (const path of scene.paths) {
    path.x.forEach((x, index) => points.push([x, path.y[index]]));
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

function renderScene() {
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

function chartLayoutKey(charts) {
  return charts.map((chart) => chart.title).join("::");
}

function ensureChartViews(charts) {
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
    const plotEl = document.createElement("div");
    plotEl.className = "plot-surface";
    shell.append(header, plotEl);
    chartsEl.appendChild(shell);
    state.chartViews.set(chart.title, { plotEl, linkedRangeBound: false });
  }
}

function updateChart(view, chart) {
  if (!window.Plotly) {
    return;
  }
  const groupOrder = new Map();
  const colorIndexFor = (group) => {
    if (!groupOrder.has(group)) {
      groupOrder.set(group, groupOrder.size);
    }
    return groupOrder.get(group);
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
  const layout = {
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

function renderCharts() {
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

function stopAnimation() {
  if (state.playHandle !== null) {
    clearInterval(state.playHandle);
    state.playHandle = null;
  }
  state.playing = false;
}

function startAnimation() {
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

function scheduleArtifactRender() {
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

function handleSolveEvent(event) {
  switch (event.kind) {
    case STREAM_EVENT_KIND.status:
      setStatus(event.message, "");
      break;
    case STREAM_EVENT_KIND.log:
      appendLogLine(event.line, event.level ?? LOG_LEVEL.console);
      break;
    case STREAM_EVENT_KIND.iteration:
      state.pendingIterationEvent = event;
      scheduleIterationUpdate();
      break;
    case STREAM_EVENT_KIND.final:
      state.pendingIterationEvent = null;
      state.terminalSolver = hasSolverReport(event.artifact?.solver) ? event.artifact.solver : null;
      state.artifact = event.artifact;
      state.animationIndex = 0;
      renderSolverSummary();
      scheduleArtifactRender();
      setStatus(
        artifactTermination(event.artifact) ?? "Solve finished.",
        statusClass(solverStatusKind(event.artifact)),
      );
      break;
    case STREAM_EVENT_KIND.error:
    default:
      if (state.pendingIterationEvent) {
        const pendingEvent = state.pendingIterationEvent;
        state.pendingIterationEvent = null;
        applyIterationEvent(pendingEvent, false);
      }
      state.terminalSolver = buildFailureSolverReport(event.message);
      renderSolverSummary();
      renderMetrics();
      appendLogLine(`error: ${event.message}`, LOG_LEVEL.error);
      setStatus(event.message, "error");
      break;
  }
}

async function readNdjsonStream(response, onEvent) {
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
      onEvent(normalizeSolveEvent(JSON.parse(trimmed)));
    }
  }

  const tail = buffer.trim();
  if (tail) {
    onEvent(normalizeSolveEvent(JSON.parse(tail)));
  }
}

async function solveCurrentProblem(event) {
  event?.preventDefault?.();
  const spec = currentSpec();
  if (!spec || state.solving) {
    return;
  }

  state.solving = true;
  solveButton.disabled = true;
  solveButton.textContent = "Solving...";
  stopAnimation();
  state.artifact = null;
  state.animationIndex = 0;
  state.sceneView = null;
  renderMetrics();
  renderScene();
  renderCharts();
  resetSolverPanel();
  setStatus("Setting up symbolic model...", "");

  try {
    const response = await fetch(`/api/solve_stream/${spec.wire_id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values: state.values }),
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      throw new Error(payload?.error ?? `Request failed with ${response.status}`);
    }

    await readNdjsonStream(response, handleSolveEvent);
  } catch (error) {
    setStatus(error.message, "error");
    appendLogLine(`error: ${error.message}`, LOG_LEVEL.error);
  } finally {
    state.solving = false;
    solveButton.disabled = false;
    solveButton.textContent = "Solve";
  }
}

async function init() {
  try {
    state.specs = (await fetchJson("/api/problems")).map(normalizeProblemSpec);
    if (state.specs.length === 0) {
      throw new Error("No problems are registered.");
    }
    controlsForm.addEventListener("submit", solveCurrentProblem);
    solveButton.addEventListener("click", solveCurrentProblem);
    selectProblem(state.specs[0].id);
  } catch (error) {
    setStatus(error.message, "error");
  }
}

init();

window.addEventListener("mathjax-ready", () => {
  const spec = currentSpec();
  if (spec) {
    renderModel(spec);
  }
});
