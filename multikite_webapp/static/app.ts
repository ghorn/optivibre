import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

type PhaseMode = "adaptive" | "open_loop";
type Preset =
  | "free_flight1"
  | "star1"
  | "y2"
  | "y2_reference"
  | "star3"
  | "star4"
  | "simple_tether";
type TimeDilationPreset = "fast" | "1" | "0.5" | "0.1";
type CameraFollowTarget = "manual" | "disk_center" | `kite:${number}`;
type RuntimeTab = "console" | "plots";

type PlotlyDatum = Record<string, unknown>;

interface PlotlyLike {
  newPlot(
    element: HTMLElement,
    data: PlotlyDatum[],
    layout: PlotlyDatum,
    config?: PlotlyDatum
  ): Promise<unknown>;
  extendTraces(
    element: HTMLElement,
    update: { x: number[][]; y: number[][] },
    indices: number[],
    maxPoints?: number
  ): Promise<unknown>;
  restyle(
    element: HTMLElement,
    update: PlotlyDatum,
    indices: number[]
  ): Promise<unknown>;
  relayout(element: HTMLElement, update: PlotlyDatum): Promise<unknown>;
  purge(element: HTMLElement): void;
  Plots?: {
    resize(element: HTMLElement): void;
  };
}

declare const Plotly: PlotlyLike;

interface PlotlyRelayoutEvent {
  [key: string]: unknown;
}

interface PlotElement extends HTMLElement {
  on(event: "plotly_relayout", handler: (event: PlotlyRelayoutEvent) => void): void;
}

interface MathJaxLike {
  typesetPromise?(elements: Element[]): Promise<unknown>;
  typesetClear?(elements: Element[]): void;
}

interface MermaidLike {
  initialize(config: Record<string, unknown>): void;
  run(config?: { nodes?: Element[] }): Promise<unknown> | void;
}

declare global {
  interface Window {
    MathJax?: MathJaxLike;
    mermaid?: MermaidLike;
  }
}

interface PresetInfo {
  preset: Preset;
  name: string;
  description: string;
  kites: number;
  common_nodes: number;
  upper_nodes: number;
}

interface RunSummary {
  duration: number;
  accepted_steps: number;
  rejected_steps: number;
  max_phase_error: number;
  final_total_work: number;
  final_total_dissipated_work: number;
  final_total_kinetic_energy: number;
  final_total_potential_energy: number;
  final_total_tether_strain_energy: number;
  final_total_mechanical_energy: number;
  final_energy_conservation_residual: number;
  failure?: SimulationFailure | null;
}

interface SimulationFailure {
  time: number;
  kite_index: number;
  quantity: string;
  value_deg: number;
  lower_limit_deg: number;
  upper_limit_deg: number;
  alpha_deg: number;
  beta_deg: number;
  message: string;
}

interface SimulationProgress {
  iteration: number;
  time: number;
  duration: number;
  interval_dt: number;
  sample_count: number;
  accepted_steps_total: number;
  rejected_steps_total: number;
  accepted_steps_interval: number;
  rejected_steps_interval: number;
  substeps_interval: number;
  substep_budget: number;
}

interface ApiFrame {
  time: number;
  payload_position_n: [number, number, number];
  splitter_position_n: [number, number, number];
  clean_wind_n: [number, number, number];
  kite_gust_n: [number, number, number][];
  kite_airflow_n: [number, number, number][];
  kite_ref_span: number[];
  kite_ref_chord: number[];
  kite_ref_area: number[];
  kite_cad_offset_b: [number, number, number][];
  kite_bridle_pivot_b: [number, number, number][];
  kite_bridle_radius: number[];
  control_ring_center_n: [number, number, number];
  control_ring_radius: number;
  common_tether: [number, number, number][];
  common_tether_tensions: number[];
  upper_tethers: [number, number, number][][];
  upper_tether_tensions: number[][];
  kite_positions_n: [number, number, number][];
  kite_quaternions_n2b: [number, number, number, number][];
  kite_attitudes_rpy_deg: [number, number, number][];
  rabbit_targets_n: [number, number, number][];
  phase_error: number[];
  speed_target: number[];
  altitude: number[];
  altitude_ref: number[];
  kinetic_energy_specific: number[];
  kinetic_energy_ref_specific: number[];
  kinetic_energy_error_specific: number[];
  potential_energy_specific: number[];
  potential_energy_ref_specific: number[];
  potential_energy_error_specific: number[];
  total_energy_error_specific: number[];
  energy_balance_error_specific: number[];
  thrust_energy_integrator: number[];
  pitch_energy_integrator: number[];
  inertial_speed: number[];
  airspeed: number[];
  alpha_deg: number[];
  beta_deg: number[];
  body_omega_b: [number, number, number][];
  orbit_radius: number[];
  rabbit_radius: number[];
  curvature_y_b: number[];
  curvature_y_ref: number[];
  curvature_y_est: number[];
  omega_world_z_ref: number[];
  omega_world_z: number[];
  beta_ref_deg: number[];
  roll_ref_deg: number[];
  roll_ff_deg: number[];
  pitch_ref_deg: number[];
  curvature_z_b: number[];
  curvature_z_ref: number[];
  top_tension: number[];
  total_force_b: [number, number, number][];
  aero_force_b: [number, number, number][];
  aero_force_drag_b: [number, number, number][];
  aero_force_side_b: [number, number, number][];
  aero_force_lift_b: [number, number, number][];
  tether_force_b: [number, number, number][];
  gravity_force_b: [number, number, number][];
  motor_force_b: [number, number, number][];
  total_moment_b: [number, number, number][];
  aero_moment_b: [number, number, number][];
  tether_moment_b: [number, number, number][];
  motor_moment_b: [number, number, number][];
  cl_total: number[];
  cl_0_term: number[];
  cl_alpha_term: number[];
  cl_elevator_term: number[];
  cl_flap_term: number[];
  cd_total: number[];
  cd_0_term: number[];
  cd_induced_term: number[];
  cd_surface_term: number[];
  cy_total: number[];
  cy_beta_term: number[];
  cy_rudder_term: number[];
  roll_coeff_total: number[];
  roll_beta_term: number[];
  roll_p_term: number[];
  roll_r_term: number[];
  roll_aileron_term: number[];
  pitch_coeff_total: number[];
  pitch_0_term: number[];
  pitch_alpha_term: number[];
  pitch_q_term: number[];
  pitch_elevator_term: number[];
  pitch_flap_term: number[];
  yaw_coeff_total: number[];
  yaw_beta_term: number[];
  yaw_p_term: number[];
  yaw_r_term: number[];
  yaw_rudder_term: number[];
  aileron_cmd_deg: number[];
  flap_cmd_deg: number[];
  winglet_cmd_deg: number[];
  elevator_cmd_deg: number[];
  rudder_cmd_deg: number[];
  motor_torque: number[];
  total_work: number;
  total_dissipated_work: number;
  total_kinetic_energy: number;
  total_potential_energy: number;
  total_tether_strain_energy: number;
  total_mechanical_energy: number;
  energy_conservation_residual: number;
  work_minus_potential: number;
}

interface RunResponse {
  summary: RunSummary;
  frames: ApiFrame[];
}

interface KiteVisualDimensions {
  wingSpan: number;
  wingChord: number;
  wingThickness: number;
  fuselageLength: number;
  fuselageRadius: number;
  noseLength: number;
  tailConeLength: number;
  wingX: number;
  tailX: number;
  horizontalTailSpan: number;
  horizontalTailChord: number;
  verticalTailHeight: number;
  verticalTailChord: number;
  tailThickness: number;
  bridlePivotB: [number, number, number];
  bridleRadius: number;
}

type StreamEvent =
  | { kind: "log"; message: string }
  | { kind: "progress"; progress: SimulationProgress }
  | { kind: "frame"; frame: ApiFrame }
  | { kind: "plots"; frames: ApiFrame[] }
  | { kind: "summary"; summary: RunSummary };

const presetSelect = document.querySelector<HTMLSelectElement>("#preset")!;
const durationInput = document.querySelector<HTMLInputElement>("#duration")!;
const phaseModeSelect = document.querySelector<HTMLSelectElement>("#phase-mode")!;
const payloadInput = document.querySelector<HTMLInputElement>("#payload-mass")!;
const windInput = document.querySelector<HTMLInputElement>("#wind-speed")!;
const bridleEnabledInput = document.querySelector<HTMLInputElement>("#bridle-enabled")!;
const simNoiseInput = document.querySelector<HTMLInputElement>("#sim-noise")!;
const timeDilationSelect = document.querySelector<HTMLSelectElement>("#time-dilation")!;
const cameraFollowTargetSelect = document.querySelector<HTMLSelectElement>("#camera-follow-target")!;
const cameraFollowYawInput = document.querySelector<HTMLInputElement>("#camera-follow-yaw")!;
const cameraFollowYawLabel = cameraFollowYawInput.closest<HTMLLabelElement>(".checkbox-label")!;
const summaryNode = document.querySelector<HTMLElement>("#summary")!;
const failureNode = document.querySelector<HTMLElement>("#failure-pill")!;
const runtimeConsoleTab = document.querySelector<HTMLButtonElement>("#runtime-tab-console")!;
const runtimePlotsTab = document.querySelector<HTMLButtonElement>("#runtime-tab-plots")!;
const runtimeConsoleView = document.querySelector<HTMLElement>("#runtime-console-view")!;
const runtimePlotsView = document.querySelector<HTMLElement>("#runtime-plots-view")!;
const plotsNode = document.querySelector<HTMLElement>("#plots")!;
const viewport = document.querySelector<HTMLElement>("#viewport")!;
const runForm = document.querySelector<HTMLFormElement>("#run-form")!;
const runButton = document.querySelector<HTMLButtonElement>("#run-button")!;
const consoleNode = document.querySelector<HTMLElement>("#console")!;
const controllerDocsNode = document.querySelector<HTMLElement>("#controller-docs")!;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(viewport.clientWidth, viewport.clientHeight);
viewport.append(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color("#071019");
scene.fog = new THREE.Fog("#071019", 240, 1800);
const camera = new THREE.PerspectiveCamera(48, viewport.clientWidth / viewport.clientHeight, 0.1, 5000);
camera.up.set(0, 0, 1);
camera.position.set(240, -280, 290);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 220);
controls.screenSpacePanning = false;
controls.minPolarAngle = 0.05;
controls.maxPolarAngle = Math.PI - 0.05;
controls.update();

scene.add(new THREE.AmbientLight(0x9bc3ff, 0.9));
const sun = new THREE.DirectionalLight(0xb9f5da, 1.45);
sun.position.set(180, -240, 300);
scene.add(sun);

const rim = new THREE.DirectionalLight(0x5e87ff, 0.45);
rim.position.set(-220, 160, 180);
scene.add(rim);

const grid = new THREE.GridHelper(600, 30, 0x1fa874, 0x274355);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
const GRID_SIZE = 600;
const GRID_HALF_EXTENT = GRID_SIZE / 2;

const payloadMesh = new THREE.Mesh(
  new THREE.SphereGeometry(7, 24, 24),
  new THREE.MeshStandardMaterial({ color: 0xff7b72, roughness: 0.35, metalness: 0.08 })
);
scene.add(payloadMesh);

const splitterMesh = new THREE.Mesh(
  new THREE.SphereGeometry(4, 16, 16),
  new THREE.MeshStandardMaterial({ color: 0x45d7a7, roughness: 0.24, metalness: 0.15 })
);
scene.add(splitterMesh);

const orbitTargetMarker = new THREE.Group();
const orbitTargetCore = new THREE.Mesh(
  new THREE.SphereGeometry(1.0, 18, 18),
  new THREE.MeshStandardMaterial({
    color: 0xffd36b,
    emissive: 0xffd36b,
    emissiveIntensity: 0.55,
    transparent: true,
    opacity: 0.9,
    roughness: 0.2,
    metalness: 0.08
  })
);
const orbitTargetHalo = new THREE.Mesh(
  new THREE.SphereGeometry(2.2, 18, 18),
  new THREE.MeshBasicMaterial({
    color: 0xfff0b0,
    wireframe: true,
    transparent: true,
    opacity: 0.28
  })
);
orbitTargetMarker.add(orbitTargetCore);
orbitTargetMarker.add(orbitTargetHalo);
orbitTargetMarker.visible = false;
scene.add(orbitTargetMarker);

const ORBIT_TARGET_CORE_RADIUS_WORLD = 1.0;
const ORBIT_TARGET_HALO_RADIUS_WORLD = 2.2;
const ORBIT_TARGET_CORE_PIXELS = 14;
const ORBIT_TARGET_HALO_PIXELS = 34;
const ORBIT_TARGET_CORE_RADIUS_MIN = 0.2;
const ORBIT_TARGET_CORE_RADIUS_MAX = 2.4;
const ORBIT_TARGET_HALO_RADIUS_MIN = 0.55;
const ORBIT_TARGET_HALO_RADIUS_MAX = 6.0;

const controlRingLine = new THREE.LineLoop(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({
    color: 0x36d5c1,
    transparent: true,
    opacity: 0.42
  })
);
controlRingLine.visible = false;
scene.add(controlRingLine);

const controlAxisLine = new THREE.Line(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({
    color: 0xffd36b,
    transparent: true,
    opacity: 0.3
  })
);
controlAxisLine.visible = false;
scene.add(controlAxisLine);

const controlCenterMarker = new THREE.Mesh(
  new THREE.SphereGeometry(2.1, 16, 16),
  new THREE.MeshStandardMaterial({
    color: 0x36d5c1,
    emissive: 0x36d5c1,
    emissiveIntensity: 0.35,
    transparent: true,
    opacity: 0.92,
    roughness: 0.25,
    metalness: 0.06
  })
);
controlCenterMarker.visible = false;
scene.add(controlCenterMarker);

const kiteMeshes: THREE.Group[] = [];
const rabbitMeshes: THREE.Mesh[] = [];
const projectedPhaseMeshes: THREE.Mesh[] = [];
const guidanceLines: THREE.Line[] = [];
const phaseSlotMeshes: THREE.Mesh[] = [];
const commonSegmentMeshes: THREE.Mesh[] = [];
const commonNodeMeshes: THREE.Mesh[] = [];
const upperSegmentMeshes: THREE.Mesh[][] = [];
const upperNodeMeshes: THREE.Mesh[][] = [];
let kiteVisualGeometryKey: string | null = null;
const AIRFLOW_AMBIENT_PARTICLE_COUNT = 260;
const AIRFLOW_GUST_PARTICLES_PER_KITE = 48;
const AIRFLOW_MAX_KITES = 4;
const AIRFLOW_GUST_PARTICLE_COUNT = AIRFLOW_MAX_KITES * AIRFLOW_GUST_PARTICLES_PER_KITE;
const WINGTIP_TRAIL_LIFETIME_S = 5.0;
const WINGTIP_TRAIL_PARTICLE_COUNT = 4096;
const WINGTIP_TRAIL_LEFT_COLOR = new THREE.Color("#7cecff");
const WINGTIP_TRAIL_RIGHT_COLOR = new THREE.Color("#ffb170");

interface AirParticleState {
  position: THREE.Vector3;
  age: number;
  life: number;
  drift: number;
}

interface WingtipTrailParticleState {
  position: THREE.Vector3;
  age: number;
  life: number;
  active: boolean;
}

const ambientParticlePositions = new Float32Array(AIRFLOW_AMBIENT_PARTICLE_COUNT * 3);
const ambientParticleColors = new Float32Array(AIRFLOW_AMBIENT_PARTICLE_COUNT * 3);
const gustParticlePositions = new Float32Array(AIRFLOW_GUST_PARTICLE_COUNT * 3);
const gustParticleColors = new Float32Array(AIRFLOW_GUST_PARTICLE_COUNT * 3);
const wingtipTrailPositions = new Float32Array(WINGTIP_TRAIL_PARTICLE_COUNT * 3);
const wingtipTrailColors = new Float32Array(WINGTIP_TRAIL_PARTICLE_COUNT * 3);
const wingtipTrailAlpha = new Float32Array(WINGTIP_TRAIL_PARTICLE_COUNT);
const ambientParticleStates: AirParticleState[] = Array.from(
  { length: AIRFLOW_AMBIENT_PARTICLE_COUNT },
  () => ({
    position: new THREE.Vector3(),
    age: 0,
    life: 0,
    drift: Math.random()
  })
);
const gustParticleStates: AirParticleState[] = Array.from(
  { length: AIRFLOW_GUST_PARTICLE_COUNT },
  () => ({
    position: new THREE.Vector3(),
    age: 0,
    life: 0,
    drift: Math.random()
  })
);
const wingtipTrailStates: WingtipTrailParticleState[] = Array.from(
  { length: WINGTIP_TRAIL_PARTICLE_COUNT },
  () => ({
    position: new THREE.Vector3(),
    age: 0,
    life: 0,
    active: false
  })
);
let nextWingtipTrailParticleIndex = 0;

function makeSoftParticleMaterial(
  pointSize: number,
  opacity: number
): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    vertexColors: true,
    blending: THREE.NormalBlending,
    uniforms: {
      uPointSize: { value: pointSize },
      uOpacity: { value: opacity }
    },
    vertexShader: `
      varying vec3 vColor;
      uniform float uPointSize;

      void main() {
        vColor = color;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        float attenuation = clamp(180.0 / max(1.0, -mvPosition.z), 0.85, 2.1);
        gl_PointSize = uPointSize * attenuation;
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      uniform float uOpacity;

      void main() {
        float distance_from_center = length(gl_PointCoord - vec2(0.5));
        if (distance_from_center > 0.5) {
          discard;
        }
        float soft_edge = 1.0 - smoothstep(0.08, 0.5, distance_from_center);
        soft_edge = pow(soft_edge, 1.35);
        gl_FragColor = vec4(vColor, uOpacity * soft_edge);
      }
    `
  });
}

const ambientParticleGeometry = new THREE.BufferGeometry();
ambientParticleGeometry.setAttribute(
  "position",
  new THREE.BufferAttribute(ambientParticlePositions, 3).setUsage(THREE.DynamicDrawUsage)
);
ambientParticleGeometry.setAttribute(
  "color",
  new THREE.BufferAttribute(ambientParticleColors, 3).setUsage(THREE.DynamicDrawUsage)
);
const ambientParticleCloud = new THREE.Points(
  ambientParticleGeometry,
  makeSoftParticleMaterial(2.6, 0.26)
);
ambientParticleCloud.visible = false;
scene.add(ambientParticleCloud);

const gustParticleGeometry = new THREE.BufferGeometry();
gustParticleGeometry.setAttribute(
  "position",
  new THREE.BufferAttribute(gustParticlePositions, 3).setUsage(THREE.DynamicDrawUsage)
);
gustParticleGeometry.setAttribute(
  "color",
  new THREE.BufferAttribute(gustParticleColors, 3).setUsage(THREE.DynamicDrawUsage)
);
const gustParticleCloud = new THREE.Points(
  gustParticleGeometry,
  makeSoftParticleMaterial(3.15, 0.34)
);
gustParticleCloud.visible = false;
scene.add(gustParticleCloud);

const wingtipTrailGeometry = new THREE.BufferGeometry();
wingtipTrailGeometry.setAttribute(
  "position",
  new THREE.BufferAttribute(wingtipTrailPositions, 3).setUsage(THREE.DynamicDrawUsage)
);
wingtipTrailGeometry.setAttribute(
  "color",
  new THREE.BufferAttribute(wingtipTrailColors, 3).setUsage(THREE.DynamicDrawUsage)
);
wingtipTrailGeometry.setAttribute(
  "alpha",
  new THREE.BufferAttribute(wingtipTrailAlpha, 1).setUsage(THREE.DynamicDrawUsage)
);
const wingtipTrailMaterial = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  vertexColors: true,
  blending: THREE.NormalBlending,
  uniforms: {
    uPointSize: { value: 1.75 }
  },
  vertexShader: `
    attribute float alpha;
    varying vec3 vColor;
    varying float vAlpha;
    uniform float uPointSize;

    void main() {
      vColor = color;
      vAlpha = alpha;
      vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = uPointSize;
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  fragmentShader: `
    varying vec3 vColor;
    varying float vAlpha;

    void main() {
      float distance_from_center = length(gl_PointCoord - vec2(0.5));
      if (distance_from_center > 0.5) {
        discard;
      }
      float soft_edge = 1.0 - smoothstep(0.14, 0.5, distance_from_center);
      soft_edge = pow(soft_edge, 1.8);
      gl_FragColor = vec4(vColor, 0.72 * vAlpha * soft_edge);
    }
  `
});
const wingtipTrailCloud = new THREE.Points(wingtipTrailGeometry, wingtipTrailMaterial);
wingtipTrailCloud.visible = false;
scene.add(wingtipTrailCloud);

const tetherSegmentGeometry = new THREE.CylinderGeometry(1, 1, 1, 12, 1, true);
const tetherNodeGeometry = new THREE.SphereGeometry(1, 16, 16);
const tetherAxis = new THREE.Vector3(0, 1, 0);
const tensionColorLow = new THREE.Color("#5ba8ff");
const tensionColorMid = new THREE.Color("#ffbe6b");
const tensionColorHigh = new THREE.Color("#ff4d4d");
const TETHER_TENSION_MIN_N = 0;
const TETHER_TENSION_MAX_N = 2500;
const CONTROL_RING_SEGMENTS = 96;
const CONTROL_AXIS_HALF_LENGTH = 10;
const PHASE_ERROR_MAX_RAD = 0.5;
const phaseColorLow = new THREE.Color("#45d7a7");
const phaseColorMid = new THREE.Color("#ffbe6b");
const phaseColorHigh = new THREE.Color("#ff5c74");
const COMMON_SEGMENT_RADIUS = 0.46;
const UPPER_SEGMENT_RADIUS = 0.34;
const COMMON_NODE_RADIUS = 1.0;
const UPPER_NODE_RADIUS = 0.78;
const ambientAirColor = new THREE.Color("#6ee7ff");
const ambientAirColorHigh = new THREE.Color("#b8f7ff");
const gustAirColorLow = new THREE.Color("#55c5ff");
const gustAirColorMid = new THREE.Color("#ffbf72");
const gustAirColorHigh = new THREE.Color("#ff5b78");
const SUMMARY_REFRESH_MIN_INTERVAL_MS = 125;
let consoleLines: string[] = [];
let framesReceived = 0;
let framesRendered = 0;
let pendingPlaybackFrames: ApiFrame[] = [];
let pendingSummary: RunSummary | null = null;
let latestProgressState: SimulationProgress | null = null;
let activeSummaryRequest: {
  preset: string;
  phase_mode: PhaseMode;
  sim_noise_enabled: boolean;
  bridle_enabled: boolean;
} | null = null;
let currentPlaybackRate: number | null = null;
let currentPlaybackLabel = "Fast as possible";
let playbackStartWallTimeMs: number | null = null;
let playbackStartSimTime = 0;
let shouldSnapOrbitTargetToFrame = true;
let lastRenderedFrame: ApiFrame | null = null;
let lastCameraFollowHeadingRad: number | null = null;
let lastCameraFollowTarget: CameraFollowTarget | null = null;
let lastAirflowFrameTime: number | null = null;
let airflowUpdatesEnabled = true;
let lastSummaryRefreshWallTimeMs = 0;
let summaryRefreshPending = false;
let lastSummaryHtml = "";
const pendingMathRoots = new Set<Element>();
let mathTypesetRetryHandle: number | null = null;
const pendingMermaidRoots = new Set<Element>();
let mermaidRenderRetryHandle: number | null = null;
let mermaidInitialized = false;

type PlotDash = "solid" | "dash" | "dot" | "dashdot" | "longdash";

interface PlotTraceDefinition {
  name: string;
  color: string;
  signalKey?: string;
  legendName?: string;
  kiteIndex?: number;
  dash?: PlotDash;
  width?: number;
  value: (frame: ApiFrame) => number;
}

interface PlotGroupDefinition {
  title: string;
  yTitle: string;
  traces: PlotTraceDefinition[];
}

interface PlotSectionDefinition {
  title: string;
  description: string;
  groups: PlotGroupDefinition[];
  maxColumns?: number;
}

interface ActivePlotSection {
  host: HTMLElement;
  plot: HTMLElement;
  traces: PlotTraceDefinition[];
  traceIndices: number[];
}

interface SummaryMetric {
  label: string;
  value: string;
}

const KITE_COLORS = ["#45d7a7", "#66b8ff", "#ffbe6b", "#ff7b72"];
const REF_ALPHA = 0.6;
const LIMIT_COLOR = "rgba(255, 94, 94, 0.72)";
const ZERO_REF_COLOR = "rgba(211, 228, 245, 0.38)";
const MAX_PLOT_COLUMNS = 3;
const PLOT_GROUP_HEIGHT_PX = 330;
let activePlotSections: ActivePlotSection[] = [];
let plotKiteVisibility: boolean[] = [];
let plotSignalVisibility = new Map<string, boolean>();
let collapsedPlotSections = new Set<string>();
let syncingPlotXAxes = false;

interface KiteBreakdownTraceDefinition {
  name: string;
  value: (frame: ApiFrame, kiteIndex: number) => number;
  dash?: PlotDash;
  width?: number;
  alpha?: number;
}

function appendConsole(message: string): void {
  const stamp = new Date().toLocaleTimeString();
  consoleLines.push(`[${stamp}] ${message}`);
  if (consoleLines.length > 240) {
    consoleLines = consoleLines.slice(consoleLines.length - 240);
  }
  consoleNode.textContent = consoleLines.join("\n");
  consoleNode.scrollTop = consoleNode.scrollHeight;
}

function clearConsole(): void {
  consoleLines = [];
  consoleNode.textContent = "";
}

function showRuntimeTab(tab: RuntimeTab): void {
  const showConsole = tab === "console";
  runtimeConsoleTab.classList.toggle("active", showConsole);
  runtimePlotsTab.classList.toggle("active", !showConsole);
  runtimeConsoleView.classList.toggle("active", showConsole);
  runtimePlotsView.classList.toggle("active", !showConsole);
  runtimeConsoleTab.setAttribute("aria-selected", String(showConsole));
  runtimePlotsTab.setAttribute("aria-selected", String(!showConsole));
  if (!showConsole) {
    requestAnimationFrame(() => {
      activePlotSections.forEach((section) => {
        Plotly.Plots?.resize(section.plot);
      });
    });
  }
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function summaryRowsHtml(rows: SummaryMetric[]): string {
  return rows
    .map(
      (row) => `
        <div class="summary-row">
          <div class="summary-label">${escapeHtml(row.label)}</div>
          <div class="summary-value">${escapeHtml(row.value)}</div>
        </div>`
    )
    .join("");
}

function renderSummaryCard(
  statusLabel: string,
  statusValue: string,
  rows: SummaryMetric[],
  statusSubvalue?: string
): string {
  const bannerClass =
    statusLabel === "Running"
      ? " running"
      : statusLabel === "Terminated Early"
        ? " terminated"
        : "";
  return `
    <div class="status-banner${bannerClass}">
      <div class="status-label">${escapeHtml(statusLabel)}</div>
      <div class="status-value">${escapeHtml(statusValue)}</div>
      ${statusSubvalue ? `<div class="status-subvalue">${escapeHtml(statusSubvalue)}</div>` : ""}
    </div>
    <div class="summary-grid">
      ${summaryRowsHtml(rows)}
    </div>`;
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
    roots.forEach((root) => {
      typesetMath(root);
    });
  }, 100);
}

function ensureMermaidInitialized(): boolean {
  if (!window.mermaid) {
    return false;
  }
  if (!mermaidInitialized) {
    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "loose",
      theme: "dark",
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        nodeSpacing: 34,
        rankSpacing: 46,
        padding: 8,
        curve: "basis"
      },
      themeVariables: {
        background: "#081018",
        primaryColor: "#112231",
        primaryTextColor: "#edf6ff",
        primaryBorderColor: "#3ecf9b",
        lineColor: "#7ba6c4",
        secondaryColor: "#0b1722",
        tertiaryColor: "#081018",
        clusterBkg: "#0b1722",
        clusterBorder: "#345266",
        edgeLabelBackground: "#081018",
        fontFamily: "IBM Plex Sans, Helvetica Neue, sans-serif",
        fontSize: "16px"
      }
    });
    mermaidInitialized = true;
  }
  return true;
}

function renderMermaid(root: Element | null | undefined): void {
  if (!root) {
    return;
  }
  if (!ensureMermaidInitialized()) {
    pendingMermaidRoots.add(root);
    scheduleMermaidRenderRetry();
    return;
  }
  pendingMermaidRoots.delete(root);
  if (mermaidRenderRetryHandle !== null) {
    window.clearTimeout(mermaidRenderRetryHandle);
    mermaidRenderRetryHandle = null;
  }
  const nodes = Array.from(root.querySelectorAll(".mermaid"));
  nodes.forEach((node) => {
    node.removeAttribute("data-processed");
  });
  void window.mermaid?.run({ nodes });
}

function scheduleMermaidRenderRetry(): void {
  if (mermaidRenderRetryHandle !== null || pendingMermaidRoots.size === 0) {
    return;
  }
  mermaidRenderRetryHandle = window.setTimeout(() => {
    mermaidRenderRetryHandle = null;
    if (!ensureMermaidInitialized()) {
      scheduleMermaidRenderRetry();
      return;
    }
    const roots = Array.from(pendingMermaidRoots);
    pendingMermaidRoots.clear();
    roots.forEach((root) => {
      renderMermaid(root);
    });
  }, 100);
}

function controllerDocsHtml(phaseMode: PhaseMode): string {
  const modeLabel = phaseMode === "adaptive" ? "Adaptive" : "Open Loop";
  const phaseDiagram = String.raw`flowchart LR
    A["Measured phase<br/>φ_i"] --> B["Phase coordination<br/>phase error e_i"]
    M["Phase mode<br/>adaptive or open-loop"] --> B
    B --> C["Radius scheduler<br/>r_i^r"]
    B --> D["Airspeed scheduler<br/>V_i^*"]
    C --> E["Rabbit geometry<br/>p_i^r"]
    E --> H["Altitude reference<br/>h_i^*"]
    D --> F["TECS<br/>specific energy errors"]
    H --> F
    F --> G["Motor torque<br/>τ_i"]
    F --> I["Pitch reference<br/>θ_i^*"]
    classDef block fill:#112231,stroke:#3ecf9b,color:#edf6ff;`;
  const innerLoopDiagram = String.raw`flowchart LR
    A["Rabbit target<br/>p_i^r"] --> B["Guidance block<br/>body-frame curvature references κ_y^*, κ_z^*"]
    S["Measured state<br/>position, curvature, body rates"] --> B
    B --> C["Roll-reference loop<br/>κ_y^* - κ̂_y → φ^*"]
    C --> D["Roll inner loop<br/>φ^* - φ, p → δ_a"]
    T["TECS pitch reference<br/>θ^*"] --> E["Pitch inner loop<br/>θ^* - θ, q → δ_e"]
    B --> F["Rudder coordination loop<br/>β and r damping"]
    P["AOA backoff"] --> E
    D --> G["Aileron<br/>δ_a"]
    F --> H["Rudder<br/>δ_r"]
    E --> I["Elevator<br/>δ_e"]
    J["Flap<br/>trim only δ_f=δ_f0"]
    W["Winglet"] --> K["Trim only<br/>δ_w = δ_{w,0}"]
    classDef block fill:#112231,stroke:#66b8ff,color:#edf6ff;`;

  return `
    <section class="docs-card">
      <div class="docs-card-head">
        <div class="docs-card-title">Implementation Scope</div>
        <div class="docs-card-note">These equations are derived directly from the current Rust controller in <code>multikite_sim/src/controller.rs</code>. The main equations below show the nominal feedback laws; bounds, trims, and other implementation-specific saturations are called out separately.</div>
      </div>
      <div class="docs-card-body">
        <div class="docs-grid">
          <div class="docs-prose">
            <div class="docs-phase-pill">Active UI phase mode <strong>${modeLabel}</strong></div>
            <p>The propulsion loop now regulates <strong>airspeed-derived specific kinetic energy</strong>.</p>
            <p>The controller is naturally read as a cascade: phase scheduling and rabbit geometry, body-frame lateral curvature guidance, a total-energy layer, then roll/pitch inner loops and actuator commands.</p>
            <p><strong>The lateral aileron path introduces a desired roll angle.</strong> Rudder is used as a beta/yaw-rate coordination loop.</p>
            <p><strong>The vertical path is now TECS-style.</strong> Desired airspeed and altitude become kinetic and potential energy references. Motor torque closes kinetic-energy error, and pitch trades potential against kinetic energy.</p>
            <p>The flap and winglet commands are fixed at trim.</p>
          </div>
          <div class="docs-kv">
            <div class="docs-kv-row">
              <div class="docs-kv-label">Nominal Reading</div>
              <div class="docs-kv-value">Paper-style equations below omit clamps and write each channel with explicit proportional, derivative, and integral gains.</div>
            </div>
            <div class="docs-kv-row">
              <div class="docs-kv-label">Scheduling Variables</div>
              <div class="docs-kv-value">\(e_i\) drives both the rabbit radius \(r_i^r\) and the scheduled airspeed \(V_i^\star\).</div>
            </div>
            <div class="docs-kv-row">
              <div class="docs-kv-label">Inner-Loop Structure</div>
              <div class="docs-kv-value">Lateral curvature feeds desired roll; airspeed and altitude feed TECS, which commands motor torque and desired pitch.</div>
            </div>
            <div class="docs-kv-row">
              <div class="docs-kv-label">Implementation Bounds</div>
              <div class="docs-kv-value">Bounds are still active in code; they are summarized later instead of being embedded into every displayed equation.</div>
            </div>
          </div>
        </div>
        <div class="docs-equation">
          <div class="docs-equation-caption">Notation</div>
          \\[
          \\operatorname{wrap}(\\theta)\\in[-\\pi,\\pi],
          \\qquad
          q_i^b = R_{n\\to b} q_i^n,
          \\qquad
          \\tilde\\kappa_{\\bullet,i}=\\kappa_{\\bullet,i}-\\kappa^\\star_{\\bullet,i}.
          \\]
        </div>
      </div>
    </section>
    <section class="docs-card">
      <div class="docs-card-head">
        <div class="docs-card-title">Historical Haskell Comparison</div>
        <div class="docs-card-note">Compared against the flown 2015/early-2016 Haskell controller in <code>kittybutt/control/src/Kitty/Control/AircraftControl.hs</code> at commit <code>e18990d54</code>.</div>
      </div>
      <div class="docs-card-body">
        <div class="docs-note-list">
          <div class="docs-note-item"><strong>The outer geometry still matches closely.</strong> Both controllers use phase scheduling, rabbit-point geometry, and body-frame lateral curvature references.</div>
          <div class="docs-note-item"><strong>The lateral channel is now the deliberate divergence.</strong> The flown Haskell controller drove both aileron and rudder directly from lateral-curvature terms, whereas the current Rust controller maps lateral-curvature error into a desired roll angle for aileron and uses rudder as a beta/yaw-damper coordination loop.</div>
          <div class="docs-note-item"><strong>The energy controller is a deliberate modernization.</strong> The flown Haskell controller used inertial-speed PI to motor torque and curvature-to-surface vertical control; current Rust uses airspeed/altitude specific-energy PI loops for motor torque and pitch reference.</div>
          <div class="docs-note-item"><strong>Main implementation differences are wrapper-level.</strong> Rust computes phase error internally instead of consuming an external <code>phaseLag</code> signal, uses hard clamps where Haskell used <code>smoothSaturate</code>, and omits the Haskell RC/enable mixing path. Rust also exposes an open-loop phase mode that was not part of the flown Haskell path.</div>
          <div class="docs-note-item"><strong>There was no hidden roll-reference layer in the flown Haskell controller.</strong> The old Haskell code had only a commented roll-angle idea. So this new roll cascade is an intentional modernization, not a recovery of a missing historical layer.</div>
        </div>
      </div>
    </section>
    <section class="docs-card">
      <div class="docs-card-head">
        <div class="docs-card-title">Loop Topology</div>
        <div class="docs-card-note">This is the structural answer to the “what sits between error and actuator?” question. Lateral curvature becomes a roll reference; vertical path/airspeed control is now a TECS-style energy layer.</div>
      </div>
      <div class="docs-card-body">
        <div class="docs-equation">
          <div class="docs-equation-caption">Implemented signal path</div>
          \\[
          \\begin{aligned}
          p_i^r
          &\\rightarrow q_i^b
          \\rightarrow \\left(\\kappa_{y,i}^{\\star}, \\kappa_{z,i}^{\\star}\\right),\\\\[0.35em]
          \\left(\\kappa_{y,i}^{\\star}, \\hat\\kappa_{y,i}\\right)
          &\\rightarrow \\phi_i^{\\star},\\\\[0.35em]
          \\left(\\phi_i^{\\star}, \\phi_i, \\omega_{x,i}\\right)
          &\\rightarrow \\delta_{a,i},\\\\[0.35em]
          \\left(\\beta_i, \\omega_{z,i}\\right)
          &\\rightarrow \\delta_{r,i},\\\\[0.35em]
          \\left(V_i^\\star,h_i^\\star,V_i,h_i\\right)
          &\\rightarrow \\left(\\tau_i, \\theta_i^\\star\\right),\\\\[0.35em]
          \\left(\\theta_i^\\star,\\theta_i,\\omega_{y,i}\\right)
          &\\rightarrow \\delta_{e,i}.
          \\end{aligned}
          \\]
        </div>
        <div class="docs-note-list">
          <div class="docs-note-item">The aileron channel now has an explicit commanded roll angle <span class="docs-inline-math">\\(\\phi_i^{\\star}\\)</span>, generated from lateral-curvature error.</div>
          <div class="docs-note-item">The vertical channel now has an explicit commanded pitch angle <span class="docs-inline-math">\\(\\theta_i^{\\star}\\)</span> from the energy-balance PI loop.</div>
          <div class="docs-note-item">The most relevant “desired versus actual” lateral quantities are now <span class="docs-inline-math">\\(\\kappa_y^{\\star}\\)</span> versus <span class="docs-inline-math">\\(\\hat\\kappa_y\\)</span>, and <span class="docs-inline-math">\\(\\phi^{\\star}\\)</span> versus <span class="docs-inline-math">\\(\\phi\\)</span>.</div>
        </div>
      </div>
    </section>
    <section class="docs-card">
      <div class="docs-card-head">
        <div class="docs-card-title">Phase Coordination And Rabbit Geometry</div>
        <div class="docs-card-note">Each kite first receives a scalar phase error. That single error then schedules both orbit radius and speed command.</div>
      </div>
      <div class="docs-card-body">
        <div class="docs-equation">
          <div class="docs-equation-caption">Phase error</div>
          \\[
          \\begin{aligned}
          \\text{adaptive mode:}\\qquad
          \\varepsilon_i &= \\operatorname{wrap}\\!\\left(\\phi_i - \\frac{2\\pi i}{N_K}\\right),\\\\
          \\bar\\varepsilon &= \\operatorname{circmean}(\\{\\varepsilon_j\\}),\\\\
          e_i &= \\operatorname{wrap}(\\varepsilon_i - \\bar\\varepsilon),\\\\[0.35em]
          \\text{open-loop mode:}\\qquad
          \\omega_{\\mathrm{ref}} &= \\frac{v_{\\mathrm{ref}}}{r_d},\\\\
          \\phi_i^{\\mathrm{ref}}(t) &= \\phi_{i,0} + \\omega_{\\mathrm{ref}} t,\\\\
          e_i &= \\operatorname{wrap}\\!\\left(\\phi_i - \\phi_i^{\\mathrm{ref}}(t)\\right).
          \\end{aligned}
          \\]
        </div>
        <div class="docs-equation">
          <div class="docs-equation-caption">Rabbit geometry and speed scheduling</div>
          \\[
          \\begin{aligned}
          \\psi_i &= \\phi_i + \\frac{d_r}{r_d},\\\\
          r_i^{r} &= r_d\\left(1 + k_{\\phi r}\\frac{e_i}{\\pi}\\right),\\\\
          p_i^{r} &=
          \\begin{bmatrix}
            c_x + r_i^{r}\\cos\\psi_i \\\\
            c_y + r_i^{r}\\sin\\psi_i \\\\
            c_z - k_{\\dot z r} v_{i,z}^{\\mathrm{cad}}
          \\end{bmatrix},\\\\
          v_i^{\\star} &= v_0 - k_{v\\phi} e_i.
          \\end{aligned}
          \\]
        </div>
      </div>
    </section>
    <section class="docs-card">
      <div class="docs-card-head">
        <div class="docs-card-title">Filtered Rabbit, Guidance Curvature, And Integral States</div>
        <div class="docs-card-note">The rabbit point is filtered, transformed into body coordinates, and converted into curvature references for the inner loops.</div>
      </div>
      <div class="docs-card-body">
        <div class="docs-equation">
          <div class="docs-equation-caption">Discrete rabbit filters</div>
          \\[
          \\begin{aligned}
          \\ell_i^{(1)}[k{+}1] &= (1-\\alpha_1)\\,\\ell_i^{(1)}[k] + \\alpha_1 p_i^r[k],\\\\
          \\ell_i^{(2)}[k{+}1] &= (1-\\alpha_2)\\,\\ell_i^{(2)}[k] + \\alpha_2 p_i^r[k],
          \\end{aligned}
          \\qquad
          \\alpha_1 = \\frac{\\Delta t}{\\tau_1},\\;
          \\alpha_2 = \\frac{\\Delta t}{\\tau_2}.
          \\]
        </div>
        <div class="docs-equation">
          <div class="docs-equation-caption">Body-frame guidance curvature</div>
          \\[
          \\begin{aligned}
          q_i^n &= p_i^r - p_i^{\\mathrm{cad}},\\\\
          q_i^b &= R_{n\\to b} q_i^n,\\\\
          x_i &= \\max\\!\\left(|q_{i,x}^b|, 1\\right),\\\\
          \\kappa_{y,i}^{\\star} &= \\frac{2 q_{i,y}^b}{x_i^2},\\\\
          \\kappa_{z,i}^{\\star} &= \\frac{2 q_{i,z}^b}{x_i^2}.
          \\end{aligned}
          \\]
        </div>
        <div class="docs-equation">
          <div class="docs-equation-caption">Filtered-rabbit shaping and integral states</div>
          \\[
          \\begin{aligned}
          L_i &= \\lVert p_i^r - \\ell_i^{(2)} \\rVert + \\varepsilon,\\\\
          m_i &= \\tfrac12\\left(p_i^r + \\ell_i^{(2)}\\right) - \\ell_i^{(1)},\\\\
          k_i^n &= m_i\\,\\frac{\\lVert m_i \\rVert}{L_i},\\\\
          k_i^b &= R_{n\\to b} k_i^n,\\\\[0.35em]
          \\hat\\kappa_{y,i} &= \\frac{\\omega_{n,z,i}}{\\lVert v_i^{\\mathrm{cad}} \\rVert},\\\\
          I_{\\kappa\\phi,i}^{+} &= I_{\\kappa\\phi,i} + \\Delta t\\left(\\kappa_{y,i}^{\\star} - \\hat\\kappa_{y,i}\\right),\\\\
          \\phi_i^{\\star} &= k_{\\phi\\kappa,p}\\left(\\kappa_{y,i}^{\\star} - \\hat\\kappa_{y,i}\\right) + k_{\\phi\\kappa,i} I_{\\kappa\\phi,i},\\\\
          h_i &= -z_i,\\\\
          e_{h,i} &= \\operatorname{sat}(h_i^\\star-h_i),\\\\
          E_{k,i} &= \\tfrac12 V_i^2,\\qquad E_{k,i}^\\star=\\tfrac12(V_i^\\star)^2,\\\\
          E_{p,i} &= g h_i,\\qquad E_{p,i}^\\star=g(h_i+e_{h,i}).
          \\end{aligned}
          \\]
        </div>
      </div>
    </section>
    <section class="docs-card">
      <div class="docs-card-head">
        <div class="docs-card-title">Actuator And Propulsion Laws</div>
        <div class="docs-card-note">These are the nominal commanded laws. Saturation, anti-windup style bounds, and one-channel implementation quirks are summarized in the notes below rather than embedded into the equations themselves.</div>
      </div>
      <div class="docs-card-body">
        <div class="docs-equation">
          <div class="docs-equation-caption">Nominal TECS, surface, and torque commands</div>
          \\[
          \\begin{aligned}
          e_{k,i} &= E_{k,i}^{\\star}-E_{k,i},\\\\
          e_{p,i} &= E_{p,i}^{\\star}-E_{p,i},\\\\
          e_{b,i} &= e_{p,i}-e_{k,i},\\\\
          I_{\\tau,i}^{+} &= \\operatorname{aw}\\!\\left(I_{\\tau,i}+k_{\\tau,i}e_{k,i}\\Delta t\\right),\\\\
          I_{\\theta,i}^{+} &= \\operatorname{aw}\\!\\left(I_{\\theta,i}-k_{\\theta,i}e_{b,i}\\Delta t\\right),\\\\
          \\tau_i &= \\tau_0 + k_{\\tau,p}e_{k,i}+I_{\\tau,i},\\\\
          \\theta_i^\\star &= -k_{\\theta,p}e_{b,i}+I_{\\theta,i},\\\\[0.35em]
          \\delta_{a,i} &= \\delta_{a,0} + k_{a,\\phi}\\left(\\phi_i^{\\star} - \\phi_i\\right) - k_{a,p}\\,\\omega_{x,i},\\\\
          \\delta_{f,i} &= \\delta_{f,0},\\\\
          \\delta_{w,i} &= \\delta_{w,0},\\\\
          \\delta_{e,i} &= \\delta_{e,0} - k_{e,\\theta}\\left(\\theta_i^\\star-\\theta_i\\right) + k_{e,q}\\,\\omega_{y,i} + k_{e,\\alpha}\\alpha_i^{\\mathrm{prot}},\\\\
          \\delta_{r,i} &= \\delta_{r,0} - k_{r,\\beta}\\,\\beta_i + k_{r,\\Omega}\\left(\\Omega_{z,i}-\\Omega_{z,i}^\\star\\right).
          \\end{aligned}
          \\]
        </div>
        <div class="docs-gain-table">
          <div class="docs-gain-row docs-gain-head">
            <div>Symbol</div>
            <div>Current implementation value</div>
            <div>Meaning</div>
          </div>
          <div class="docs-gain-row">
            <div><span class="docs-inline-math">\(v_0,\;k_{v\phi}\)</span></div>
            <div><code>28</code>, <code>100</code></div>
            <div>Base scheduled speed and phase-to-speed gain.</div>
          </div>
          <div class="docs-gain-row">
            <div><span class="docs-inline-math">\(d_r,\;k_{\phi r},\;k_{\dot z r}\)</span></div>
            <div><code>rabbit_distance</code>, <code>phase_lag_to_radius</code>, <code>vert_vel_to_rabbit_height</code></div>
            <div>Rabbit lead distance, phase-to-radius gain, and vertical-velocity height shaping.</div>
          </div>
          <div class="docs-gain-row">
            <div><span class="docs-inline-math">\(k_{\phi\kappa,p},\;k_{\phi\kappa,i}\)</span></div>
            <div><code>8.0</code>, <code>2.0</code></div>
            <div>Outer-loop gains that map lateral-curvature tracking error into desired roll angle.</div>
          </div>
          <div class="docs-gain-row">
            <div><span class="docs-inline-math">\(k_{a,\phi},\;k_{a,p}\)</span></div>
            <div><code>0.7</code>, <code>0.22</code></div>
            <div>Inner roll loop gains from roll-angle error and roll rate to aileron command.</div>
          </div>
          <div class="docs-gain-row">
            <div><span class="docs-inline-math">\(k_{\tau,p},\;k_{\tau,i}\)</span></div>
            <div><code>0.04</code>, <code>0.008</code></div>
            <div>Specific kinetic-energy PI gains from airspeed error to motor torque.</div>
          </div>
          <div class="docs-gain-row">
            <div><span class="docs-inline-math">\(k_{\theta,p},\;k_{\theta,i}\)</span></div>
            <div><code>0.0012</code>, <code>0.00035</code></div>
            <div>Specific energy-balance PI gains from potential-minus-kinetic error to pitch reference.</div>
          </div>
          <div class="docs-gain-row">
            <div><span class="docs-inline-math">\(e_h^{\\max},\;I_{\\tau}^{\\max},\;I_{\\theta}^{\\max}\)</span></div>
            <div><code>25 m</code>, <code>8 N m</code>, <code>7 deg</code></div>
            <div>Altitude-error saturation and anti-windup bounds for the TECS integrators.</div>
          </div>
          <div class="docs-gain-row">
            <div><span class="docs-inline-math">\(k_{r,\beta},\;k_{r,\omega}\)</span></div>
            <div><code>0.7</code>, <code>0.2</code></div>
            <div>Rudder beta feedback and yaw-rate damping gains for coordinated-turn control.</div>
          </div>
          <div class="docs-gain-row">
            <div><span class="docs-inline-math">\(k_{e,\theta},\;k_{e,q},\;k_{e,\alpha}\)</span></div>
            <div><code>0.6</code>, <code>0.18</code>, <code>2.0</code></div>
            <div>Pitch inner-loop proportional gain, pitch-rate damping, and angle-of-attack protection.</div>
          </div>
        </div>
        <div class="docs-note-list">
          <div class="docs-note-item">The implementation applies bounds after the nominal laws are evaluated: scheduled speed, integral states, surface deflections, and motor torque are all clamped in code.</div>
          <div class="docs-note-item">The aileron channel uses a saturated lateral curvature reference <span class="docs-inline-math">\(\\bar\\kappa_{y,i}^{\\star}\)</span>, i.e. the current code limits <span class="docs-inline-math">\(\\kappa_{y,i}^{\\star}\)</span> before the proportional term.</div>
          <div class="docs-note-item">The altitude reference shown in the TECS plots is the saturated effective reference used by the energy controller.</div>
          <div class="docs-note-item">Angle-of-attack protection still biases the elevator command after the nominal pitch loop.</div>
          <div class="docs-note-item">Current output limits are: roll reference <span class="docs-inline-math">\(\\pm 35^{\\circ}\)</span>, pitch reference <span class="docs-inline-math">\(\\pm 14^{\\circ}\)</span>, aileron/flap/rudder <span class="docs-inline-math">\(\\pm 15^{\\circ}\)</span>, elevator <span class="docs-inline-math">\(\\pm 20^{\\circ}\)</span>, and motor torque <span class="docs-inline-math">\(0\\le\\tau\\le 16\\,\\mathrm{N\\,m}\)</span>.</div>
        </div>
      </div>
    </section>
    <section class="docs-card">
      <div class="docs-card-head">
        <div class="docs-card-title">Loop Diagrams</div>
        <div class="docs-card-note">These are abstract control block diagrams: first the outer scheduling and energy logic, then the roll, pitch, and coordination loops. There is still no commanded moment or angular-acceleration block.</div>
      </div>
      <div class="docs-card-body">
        <div class="docs-diagram-grid">
          <div class="docs-diagram">
            <div class="mermaid">${phaseDiagram}</div>
            <div class="docs-diagram-caption">Figure 1. Phase mode selection, rabbit scheduling, TECS references, and motor/pitch commands.</div>
          </div>
          <div class="docs-diagram">
            <div class="mermaid">${innerLoopDiagram}</div>
            <div class="docs-diagram-caption">Figure 2. Body-frame guidance, lateral roll cascade, TECS pitch cascade, and direct actuator channels.</div>
          </div>
        </div>
      </div>
    </section>`;
}

function renderControllerDocs(): void {
  controllerDocsNode.innerHTML = controllerDocsHtml(phaseModeSelect.value as PhaseMode);
  typesetMath(controllerDocsNode);
  renderMermaid(controllerDocsNode);
}

function syncOrbitTargetMarker(): void {
  orbitTargetMarker.position.copy(controls.target);
  const distance = camera.position.distanceTo(controls.target);
  const viewportHeight = Math.max(1, viewport.clientHeight);
  const worldPerPixel =
    (2 * distance * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) / viewportHeight;
  const coreRadius = THREE.MathUtils.clamp(
    worldPerPixel * ORBIT_TARGET_CORE_PIXELS,
    ORBIT_TARGET_CORE_RADIUS_MIN,
    ORBIT_TARGET_CORE_RADIUS_MAX
  );
  const haloRadius = THREE.MathUtils.clamp(
    worldPerPixel * ORBIT_TARGET_HALO_PIXELS,
    ORBIT_TARGET_HALO_RADIUS_MIN,
    ORBIT_TARGET_HALO_RADIUS_MAX
  );
  orbitTargetCore.scale.setScalar(coreRadius / ORBIT_TARGET_CORE_RADIUS_WORLD);
  orbitTargetHalo.scale.setScalar(haloRadius / ORBIT_TARGET_HALO_RADIUS_WORLD);
}

function setOrbitTargetMarkerVisible(visible: boolean): void {
  orbitTargetMarker.visible = visible;
  if (visible) {
    syncOrbitTargetMarker();
  }
}

function handleViewportPointerStart(event: PointerEvent): void {
  if (event.pointerType === "mouse" && event.button !== 0 && event.button !== 2) {
    return;
  }
  setOrbitTargetMarkerVisible(true);
}

function handleViewportPointerEnd(): void {
  setOrbitTargetMarkerVisible(false);
}

controls.addEventListener("change", syncOrbitTargetMarker);
renderer.domElement.addEventListener("pointerdown", handleViewportPointerStart);
renderer.domElement.addEventListener("pointerup", handleViewportPointerEnd);
renderer.domElement.addEventListener("pointercancel", handleViewportPointerEnd);
renderer.domElement.addEventListener("lostpointercapture", handleViewportPointerEnd);
window.addEventListener("pointerup", handleViewportPointerEnd);
window.addEventListener("pointercancel", handleViewportPointerEnd);
syncOrbitTargetMarker();

function setFailure(failure: SimulationFailure | null): void {
  if (!failure) {
    failureNode.innerHTML = "";
    failureNode.classList.remove("visible");
    return;
  }
  failureNode.innerHTML = `
    <div class="failure-head">
      <div class="failure-kicker">Simulation Terminated</div>
      <div class="failure-time">t = ${failure.time.toFixed(2)} s</div>
    </div>
    <div class="failure-title">
      Kite ${failure.kite_index + 1} ${escapeHtml(failure.quantity)} = ${failure.value_deg.toFixed(2)} deg
    </div>
    <div class="failure-detail">
      Allowed range: ${failure.lower_limit_deg.toFixed(1)} to ${failure.upper_limit_deg.toFixed(1)} deg
    </div>
    <div class="failure-chip-row">
      <div class="failure-chip">AOA ${failure.alpha_deg.toFixed(2)} deg</div>
      <div class="failure-chip">AOS ${failure.beta_deg.toFixed(2)} deg</div>
    </div>`;
  failureNode.classList.add("visible");
}

function presetKiteCount(preset: Preset): number {
  switch (preset) {
    case "free_flight1":
      return 1;
    case "star1":
      return 1;
    case "y2":
    case "y2_reference":
      return 2;
    case "star3":
      return 3;
    case "star4":
      return 4;
    default:
      return 0;
  }
}

function hexToRgba(hex: string, alpha: number): string {
  const normalized = hex.replace("#", "");
  const value = normalized.length === 3
    ? normalized.split("").map((char) => char + char).join("")
    : normalized;
  const red = Number.parseInt(value.slice(0, 2), 16);
  const green = Number.parseInt(value.slice(2, 4), 16);
  const blue = Number.parseInt(value.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function kiteColor(index: number): string {
  return KITE_COLORS[index % KITE_COLORS.length];
}

function timeDilationRate(preset: TimeDilationPreset): number | null {
  switch (preset) {
    case "1":
      return 1.0;
    case "0.5":
      return 0.5;
    case "0.1":
      return 0.1;
    default:
      return null;
  }
}

function timeDilationLabel(preset: TimeDilationPreset): string {
  switch (preset) {
    case "1":
      return "1x";
    case "0.5":
      return "0.5x";
    case "0.1":
      return "0.1x";
    default:
      return "Fast as possible";
  }
}

function bufferedFrameCount(): number {
  return pendingPlaybackFrames.length;
}

function resetPlaybackState(label: string, rate: number | null): void {
  framesReceived = 0;
  framesRendered = 0;
  pendingPlaybackFrames = [];
  pendingSummary = null;
  latestProgressState = null;
  currentPlaybackLabel = label;
  currentPlaybackRate = rate;
  playbackStartWallTimeMs = null;
  playbackStartSimTime = 0;
  shouldSnapOrbitTargetToFrame = true;
  lastRenderedFrame = null;
  resetCameraFollowState();
  lastAirflowFrameTime = null;
  airflowUpdatesEnabled = true;
  lastSummaryRefreshWallTimeMs = 0;
  summaryRefreshPending = false;
  lastSummaryHtml = "";
  ambientParticleCloud.visible = false;
  gustParticleCloud.visible = false;
  wingtipTrailCloud.visible = false;
  nextWingtipTrailParticleIndex = 0;
  ambientParticleStates.forEach((state) => {
    state.age = 0;
    state.life = 0;
  });
  gustParticleStates.forEach((state) => {
    state.age = 0;
    state.life = 0;
  });
  wingtipTrailStates.forEach((state, index) => {
    state.age = 0;
    state.life = 0;
    state.active = false;
    wingtipTrailAlpha[index] = 0;
  });
}

function plotColumnCount(): number {
  const width = plotsNode.clientWidth || window.innerWidth;
  if (width < 760) {
    return 1;
  }
  if (width < 1240) {
    return 2;
  }
  return MAX_PLOT_COLUMNS;
}

function ensurePlotKiteVisibility(kiteCount: number): void {
  plotKiteVisibility = Array.from(
    { length: kiteCount },
    (_, index) => plotKiteVisibility[index] ?? true
  );
}

function plotSignalKey(trace: PlotTraceDefinition): string {
  return trace.signalKey ?? trace.legendName ?? trace.name;
}

function plotSignalVisible(trace: PlotTraceDefinition): boolean {
  return plotSignalVisibility.get(plotSignalKey(trace)) ?? true;
}

function plotKiteTraceVisible(trace: PlotTraceDefinition): boolean {
  return trace.kiteIndex === undefined || (plotKiteVisibility[trace.kiteIndex] ?? true);
}

function plotTraceVisible(trace: PlotTraceDefinition): boolean {
  return plotSignalVisible(trace) && plotKiteTraceVisible(trace);
}

function plotGroupTraces(groups: PlotGroupDefinition[]): PlotTraceDefinition[] {
  return groups.flatMap((group) => group.traces);
}

function plotTraceVisibility(traces: PlotTraceDefinition[]): boolean[] {
  return traces.map((trace) => plotTraceVisible(trace));
}

function syncPlotKiteControlUi(): void {
  document.querySelectorAll<HTMLLabelElement>(".plot-kite-toggle").forEach((label) => {
    const kiteIndex = Number(label.dataset.kiteIndex);
    if (!Number.isInteger(kiteIndex)) {
      return;
    }
    const visible = plotKiteVisibility[kiteIndex] ?? true;
    const input = label.querySelector<HTMLInputElement>("input");
    const state = label.querySelector<HTMLElement>(".plot-kite-state");
    if (input) {
      input.checked = visible;
    }
    if (state) {
      state.textContent = visible ? "shown" : "hidden";
    }
    label.classList.toggle("muted", !visible);
  });
}

function applyPlotKiteVisibility(): void {
  if (activePlotSections.length === 0) {
    return;
  }
  syncPlotKiteControlUi();
  syncPlotSignalLegendUi();

  const updates = activePlotSections.map((section) =>
    Plotly.restyle(
      section.plot,
      { visible: plotTraceVisibility(section.traces) },
      section.traceIndices
    )
  );

  void Promise.all(updates).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    appendConsole(`plot visibility update failed: ${message}`);
  });
}

function renderPlotKiteControls(container: HTMLElement, kiteCount: number): void {
  ensurePlotKiteVisibility(kiteCount);
  container.innerHTML = "";
  container.classList.toggle("empty", kiteCount === 0);
  if (kiteCount === 0) {
    return;
  }

  const title = document.createElement("div");
  title.className = "plot-kite-controls-title";
  title.textContent = "Visible";
  container.append(title);

  const toggleGroup = document.createElement("div");
  toggleGroup.className = "plot-kite-toggle-group";
  container.append(toggleGroup);

  for (let kiteIndex = 0; kiteIndex < kiteCount; kiteIndex += 1) {
    const label = document.createElement("label");
    label.className = "plot-kite-toggle";
    label.dataset.kiteIndex = String(kiteIndex);
    label.style.setProperty("--kite-color", kiteColor(kiteIndex));

    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = plotKiteVisibility[kiteIndex] ?? true;

    const swatch = document.createElement("span");
    swatch.className = "plot-kite-swatch";

    const text = document.createElement("span");
    text.className = "plot-kite-label-text";
    text.textContent = `Kite ${kiteIndex + 1}`;

    const state = document.createElement("span");
    state.className = "plot-kite-state";
    state.textContent = input.checked ? "shown" : "hidden";

    label.classList.toggle("muted", !input.checked);
    input.addEventListener("change", () => {
      plotKiteVisibility[kiteIndex] = input.checked;
      applyPlotKiteVisibility();
    });
    label.append(input, swatch, text, state);
    toggleGroup.append(label);
  }
}

interface PlotSignalLegendItem {
  key: string;
  label: string;
  color: string;
  dash?: PlotDash;
}

function plotSignalLegendItems(traces: PlotTraceDefinition[]): PlotSignalLegendItem[] {
  const items: PlotSignalLegendItem[] = [];
  const seen = new Set<string>();
  for (const trace of traces) {
    const key = plotSignalKey(trace);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    items.push({
      key,
      label: trace.legendName ?? trace.name,
      color: trace.color,
      dash: trace.dash
    });
  }
  return items;
}

function syncPlotSignalLegendUi(): void {
  document.querySelectorAll<HTMLButtonElement>(".plot-signal-toggle").forEach((button) => {
    const signalKey = button.dataset.signalKey;
    if (!signalKey) {
      return;
    }
    const visible = plotSignalVisibility.get(signalKey) ?? true;
    const state = button.querySelector<HTMLElement>(".plot-signal-state");
    button.classList.toggle("muted", !visible);
    button.setAttribute("aria-pressed", String(visible));
    if (state) {
      state.textContent = visible ? "shown" : "hidden";
    }
  });
}

function renderPlotSignalLegend(container: HTMLElement, traces: PlotTraceDefinition[]): void {
  container.innerHTML = "";
  const items = plotSignalLegendItems(traces);
  container.classList.toggle("empty", items.length === 0);
  if (items.length === 0) {
    return;
  }

  const title = document.createElement("div");
  title.className = "plot-signal-legend-title";
  title.textContent = "Signals";
  container.append(title);

  const group = document.createElement("div");
  group.className = "plot-signal-toggle-group";
  container.append(group);

  for (const item of items) {
    const button = document.createElement("button");
    button.className = "plot-signal-toggle";
    button.type = "button";
    button.dataset.signalKey = item.key;
    button.title = item.label;
    button.style.setProperty("--signal-color", item.color);

    const swatch = document.createElement("span");
    swatch.className = `plot-signal-swatch ${item.dash ? `dash-${item.dash}` : ""}`;

    const text = document.createElement("span");
    text.className = "plot-signal-label-text";
    text.textContent = item.label;

    const state = document.createElement("span");
    state.className = "plot-signal-state";
    state.textContent = "shown";

    button.addEventListener("click", () => {
      const nextVisible = !(plotSignalVisibility.get(item.key) ?? true);
      plotSignalVisibility.set(item.key, nextVisible);
      applyPlotKiteVisibility();
    });

    button.append(swatch, text, state);
    group.append(button);
  }
  syncPlotSignalLegendUi();
}

function plotLegendBaseTitle(title: string): string {
  const strippedUnits = title.replace(/\s*\([^)]*\)\s*$/, "");
  const strippedComparison = strippedUnits.replace(
    /\s+Desired vs (Actual|Estimated)$/i,
    ""
  );
  const strippedSuffix = strippedComparison.replace(/\s+(Breakdown|Terms)$/i, "");
  return strippedSuffix.trim() || title;
}

function buildPerKiteGroup(
  kiteCount: number,
  title: string,
  yTitle: string,
  actualValue: (frame: ApiFrame, kiteIndex: number) => number,
  referenceValue?: (frame: ApiFrame, kiteIndex: number) => number,
  extraTraces: PlotTraceDefinition[] = []
): PlotGroupDefinition {
  const traces: PlotTraceDefinition[] = [];
  const actualSignalKey = `${title}:actual`;
  const referenceSignalKey = `${title}:reference`;
  const legendBase = plotLegendBaseTitle(title);
  for (let kiteIndex = 0; kiteIndex < kiteCount; kiteIndex += 1) {
    const color = kiteColor(kiteIndex);
    traces.push({
      name: `Kite ${kiteIndex + 1}`,
      color,
      signalKey: actualSignalKey,
      legendName: `${legendBase} actual`,
      kiteIndex,
      value: (frame) => actualValue(frame, kiteIndex)
    });
    if (referenceValue) {
      traces.push({
        name: `Kite ${kiteIndex + 1} Ref`,
        color: hexToRgba(color, REF_ALPHA),
        signalKey: referenceSignalKey,
        legendName: `${legendBase} ref`,
        kiteIndex,
        dash: "dash",
        value: (frame) => referenceValue(frame, kiteIndex)
      });
    }
  }
  return {
    title,
    yTitle,
    traces: [
      ...traces,
      ...extraTraces.map((trace) => ({
        ...trace,
        signalKey: trace.signalKey ?? `${title}:${trace.name}`,
        legendName: trace.legendName ?? `${legendBase} ${trace.name}`
      }))
    ]
  };
}

function buildPerKiteBreakdownGroup(
  kiteCount: number,
  title: string,
  yTitle: string,
  tracesByKite: KiteBreakdownTraceDefinition[]
): PlotGroupDefinition {
  const traces: PlotTraceDefinition[] = [];
  const legendBase = plotLegendBaseTitle(title);
  for (let kiteIndex = 0; kiteIndex < kiteCount; kiteIndex += 1) {
    const color = kiteColor(kiteIndex);
    for (const trace of tracesByKite) {
      traces.push({
        name: `Kite ${kiteIndex + 1} ${trace.name}`,
        color: hexToRgba(color, trace.alpha ?? 0.9),
        signalKey: `${title}:${trace.name}`,
        legendName: `${legendBase} ${trace.name}`,
        kiteIndex,
        dash: trace.dash,
        width: trace.width,
        value: (frame) => trace.value(frame, kiteIndex)
      });
    }
  }
  return { title, yTitle, traces };
}

function bodyComponent(
  values: [number, number, number][] | undefined,
  kiteIndex: number,
  axis: 0 | 1 | 2
): number {
  return values?.[kiteIndex]?.[axis] ?? 0;
}

function buildEnergyGroups(): PlotGroupDefinition[] {
  return [
    {
      title: "Energy Conservation Check (J)",
      yTitle: "J",
      traces: [
        {
          name: "Mechanical Energy",
          color: "#87d37c",
          value: (frame) => frame.total_mechanical_energy
        },
        {
          name: "Motor Work",
          color: "#66b8ff",
          value: (frame) => frame.total_work
        },
        {
          name: "Dissipated Work",
          color: "#ffbe6b",
          value: (frame) => frame.total_dissipated_work
        },
        {
          name: "E - W + D",
          color: "#c28dff",
          value: (frame) => frame.energy_conservation_residual
        }
      ]
    },
    {
      title: "Mechanical Energy Breakdown (J)",
      yTitle: "J",
      traces: [
        {
          name: "Kinetic Energy",
          color: "#45d7a7",
          value: (frame) => frame.total_kinetic_energy
        },
        {
          name: "Potential Energy",
          color: "#66b8ff",
          value: (frame) => frame.total_potential_energy
        },
        {
          name: "Tether Strain Energy",
          color: "#ffbe6b",
          value: (frame) => frame.total_tether_strain_energy
        }
      ]
    }
  ];
}

function buildAirspeedCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Airspeed Desired vs Actual (m/s)",
    "m/s",
    (frame, kiteIndex) => frame.airspeed[kiteIndex] ?? 0,
    (frame, kiteIndex) => frame.speed_target[kiteIndex] ?? 0
  );
}

function buildAltitudeCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Altitude Desired vs Actual (m)",
    "m",
    (frame, kiteIndex) => frame.altitude[kiteIndex] ?? 0,
    (frame, kiteIndex) => frame.altitude_ref[kiteIndex] ?? 0
  );
}

function buildRollCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Desired Roll vs Actual (deg)",
    "deg",
    (frame, kiteIndex) => frame.kite_attitudes_rpy_deg[kiteIndex]?.[0] ?? 0,
    (frame, kiteIndex) => frame.roll_ref_deg[kiteIndex] ?? 0
  );
}

function buildPitchCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Desired Pitch vs Actual (deg)",
    "deg",
    (frame, kiteIndex) => frame.kite_attitudes_rpy_deg[kiteIndex]?.[1] ?? 0,
    (frame, kiteIndex) => frame.pitch_ref_deg[kiteIndex] ?? 0
  );
}

function buildOrbitRadiusGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Orbit Radius (m)",
    "m",
    (frame, kiteIndex) => frame.orbit_radius[kiteIndex] ?? 0,
    (frame, kiteIndex) => frame.rabbit_radius[kiteIndex] ?? 0
  );
}

function buildPhaseErrorGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Phase Error (rad)",
    "rad",
    (frame, kiteIndex) => frame.phase_error[kiteIndex] ?? 0,
    undefined,
    [
      {
        name: "Zero Ref",
        color: ZERO_REF_COLOR,
        dash: "dash",
        value: () => 0
      }
    ]
  );
}

function buildAileronCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Aileron Command (deg)",
    "deg",
    (frame, kiteIndex) => frame.aileron_cmd_deg[kiteIndex] ?? 0
  );
}

function buildRudderCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Rudder Command (deg)",
    "deg",
    (frame, kiteIndex) => frame.rudder_cmd_deg[kiteIndex] ?? 0
  );
}

function buildElevatorCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Elevator Command (deg)",
    "deg",
    (frame, kiteIndex) => frame.elevator_cmd_deg[kiteIndex] ?? 0
  );
}

function buildMotorTorqueCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Motor Torque Command (N m)",
    "N m",
    (frame, kiteIndex) => frame.motor_torque[kiteIndex] ?? 0
  );
}

function buildTecsPitchCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "TECS Pitch Command (deg)",
    "deg",
    (frame, kiteIndex) => frame.pitch_ref_deg[kiteIndex] ?? 0
  );
}

function buildPlotSections(kiteCount: number): PlotSectionDefinition[] {
  if (kiteCount === 0) {
    return [
      {
        title: "Physics / Load State",
        description: "Passive tether-load states and energy consistency checks.",
        groups: buildEnergyGroups()
      }
    ];
  }

  return [
    {
      title: "Controller / Summary",
      description:
        "Top-level commanded quantities and their plant responses. Use this first to see whether the main loops are tracking before drilling into the actuator and physics sections.",
      groups: [
        buildAirspeedCommandGroup(kiteCount),
        buildAltitudeCommandGroup(kiteCount),
        buildRollCommandGroup(kiteCount),
        buildPitchCommandGroup(kiteCount),
        buildOrbitRadiusGroup(kiteCount),
        buildPhaseErrorGroup(kiteCount)
      ],
      maxColumns: 2
    },
    {
      title: "Controller / 1. Lateral Inner Loop",
      description:
        "Innermost lateral channel first. Desired path curvature is turned into a coordinated-turn roll feedforward plus a smaller curvature-error PI correction; the aileron then closes roll with body-rate damping. Rudder now closes a coordinated-turn / sideslip loop using desired world-Z turn rate together with beta regulation.",
      groups: [
        buildRollCommandGroup(kiteCount),
        buildPerKiteGroup(
          kiteCount,
          "Body Rate p (rad/s)",
          "rad/s",
          (frame, kiteIndex) => frame.body_omega_b[kiteIndex]?.[0] ?? 0
        ),
        buildAileronCommandGroup(kiteCount),
        buildPerKiteGroup(
          kiteCount,
          "Sideslip Beta (deg)",
          "deg",
          (frame, kiteIndex) => frame.beta_deg[kiteIndex] ?? 0,
          (frame, kiteIndex) => frame.beta_ref_deg[kiteIndex] ?? 0
        ),
        buildPerKiteGroup(
          kiteCount,
          "Body Rate r (rad/s)",
          "rad/s",
          (frame, kiteIndex) => frame.body_omega_b[kiteIndex]?.[2] ?? 0
        ),
        buildRudderCommandGroup(kiteCount),
        buildPerKiteGroup(
          kiteCount,
          "Curvature Y Desired vs Estimated (1/m)",
          "1/m",
          (frame, kiteIndex) => frame.curvature_y_est[kiteIndex] ?? 0,
          (frame, kiteIndex) => frame.curvature_y_ref[kiteIndex] ?? 0
        ),
        buildPerKiteGroup(
          kiteCount,
          "Roll Feedforward (deg)",
          "deg",
          (frame, kiteIndex) => frame.roll_ff_deg[kiteIndex] ?? 0
        ),
        buildPerKiteGroup(
          kiteCount,
          "World Z Angular Velocity Desired vs Actual (rad/s)",
          "rad/s",
          (frame, kiteIndex) => frame.omega_world_z[kiteIndex] ?? 0,
          (frame, kiteIndex) => frame.omega_world_z_ref[kiteIndex] ?? 0
        )
      ]
    },
    {
      title: "Controller / 2. Lateral Outer Loop",
      description:
        "Outer lateral path loop. Phase coordination biases the rabbit radius and speed scheduling; the path-tracking output into the inner loop is the rabbit-radius command.",
      groups: [
        buildOrbitRadiusGroup(kiteCount),
        buildPhaseErrorGroup(kiteCount)
      ]
    },
    {
      title: "Controller / 3. Total Energy Controller",
      description:
        "Specific kinetic and potential energy loops. Airspeed error is mapped to kinetic-energy error and closed with motor torque; altitude error is saturated, converted to potential-energy error, and pitch trades potential energy against kinetic energy.",
      groups: [
        buildAirspeedCommandGroup(kiteCount),
        buildPerKiteGroup(
          kiteCount,
          "Specific Kinetic Energy Desired vs Actual (m²/s²)",
          "m²/s²",
          (frame, kiteIndex) => frame.kinetic_energy_specific[kiteIndex] ?? 0,
          (frame, kiteIndex) => frame.kinetic_energy_ref_specific[kiteIndex] ?? 0
        ),
        buildMotorTorqueCommandGroup(kiteCount),
        buildAltitudeCommandGroup(kiteCount),
        buildPerKiteGroup(
          kiteCount,
          "Specific Potential Energy Desired vs Actual (m²/s²)",
          "m²/s²",
          (frame, kiteIndex) => frame.potential_energy_specific[kiteIndex] ?? 0,
          (frame, kiteIndex) => frame.potential_energy_ref_specific[kiteIndex] ?? 0
        ),
        buildTecsPitchCommandGroup(kiteCount),
        buildPerKiteGroup(
          kiteCount,
          "Specific Total Energy Desired vs Actual (m²/s²)",
          "m²/s²",
          (frame, kiteIndex) =>
            (frame.kinetic_energy_specific[kiteIndex] ?? 0) +
            (frame.potential_energy_specific[kiteIndex] ?? 0),
          (frame, kiteIndex) =>
            (frame.kinetic_energy_ref_specific[kiteIndex] ?? 0) +
            (frame.potential_energy_ref_specific[kiteIndex] ?? 0)
        ),
        buildPerKiteBreakdownGroup(
          kiteCount,
          "Specific Energy Errors (m²/s²)",
          "m²/s²",
          [
            {
              name: "Kinetic",
              value: (frame, kiteIndex) => frame.kinetic_energy_error_specific[kiteIndex] ?? 0
            },
            {
              name: "Potential",
              dash: "dash",
              alpha: 0.76,
              value: (frame, kiteIndex) => frame.potential_energy_error_specific[kiteIndex] ?? 0
            },
            {
              name: "Total",
              dash: "dot",
              alpha: 0.62,
              value: (frame, kiteIndex) => frame.total_energy_error_specific[kiteIndex] ?? 0
            },
            {
              name: "Balance",
              dash: "dashdot",
              alpha: 0.62,
              value: (frame, kiteIndex) => frame.energy_balance_error_specific[kiteIndex] ?? 0
            }
          ]
        ),
        buildPerKiteBreakdownGroup(
          kiteCount,
          "TECS PI Integrator Outputs",
          "cmd units",
          [
            {
              name: "Thrust Int (N m)",
              value: (frame, kiteIndex) => frame.thrust_energy_integrator[kiteIndex] ?? 0
            },
            {
              name: "Pitch Int (deg)",
              dash: "dash",
              alpha: 0.76,
              value: (frame, kiteIndex) =>
                ((frame.pitch_energy_integrator[kiteIndex] ?? 0) * 180) / Math.PI
            }
          ]
        )
      ]
    },
    {
      title: "Controller / 4. Pitch Inner Loop",
      description:
        "The TECS energy-balance output is a desired pitch angle. The elevator closes pitch with q damping. Flap is held at trim for now so the elevator loop can be tuned in isolation.",
      groups: [
        buildPitchCommandGroup(kiteCount),
        buildPerKiteGroup(
          kiteCount,
          "Body Rate q (rad/s)",
          "rad/s",
          (frame, kiteIndex) => frame.body_omega_b[kiteIndex]?.[1] ?? 0
        ),
        buildElevatorCommandGroup(kiteCount),
        buildPerKiteGroup(
          kiteCount,
          "Flap Command (deg)",
          "deg",
          (frame, kiteIndex) => frame.flap_cmd_deg[kiteIndex] ?? 0
        )
      ]
    },
    {
      title: "Physics / Plant State",
      description:
        "Plant outputs that are not direct top-level controller tracking plots: aero angles, yaw response, tether load, and other state that help explain why a run diverged.",
      groups: [
        buildPerKiteGroup(
          kiteCount,
          "Yaw (deg)",
          "deg",
          (frame, kiteIndex) => frame.kite_attitudes_rpy_deg[kiteIndex]?.[2] ?? 0
        ),
        buildPerKiteGroup(
          kiteCount,
          "Angle of Attack (deg)",
          "deg",
          (frame, kiteIndex) => frame.alpha_deg[kiteIndex] ?? 0,
          undefined,
          [
            {
              name: "AOA Upper Limit",
              color: LIMIT_COLOR,
              dash: "dot",
              value: () => 20
            },
            {
              name: "AOA Lower Limit",
              color: LIMIT_COLOR,
              dash: "dot",
              value: () => -15
            }
          ]
        ),
        buildPerKiteGroup(
          kiteCount,
          "Angle of Sideslip (deg)",
          "deg",
          (frame, kiteIndex) => frame.beta_deg[kiteIndex] ?? 0,
          undefined,
          [
            {
              name: "AOS Upper Limit",
              color: LIMIT_COLOR,
              dash: "dot",
              value: () => 30
            },
            {
              name: "AOS Lower Limit",
              color: LIMIT_COLOR,
              dash: "dot",
              value: () => -30
            }
          ]
        ),
        buildPerKiteGroup(
          kiteCount,
          "Top Tension (N)",
          "N",
          (frame, kiteIndex) => frame.top_tension[kiteIndex] ?? 0
        ),
        buildPerKiteGroup(
          kiteCount,
          "Vertical Curvature Desired vs Actual (1/m)",
          "1/m",
          (frame, kiteIndex) => frame.curvature_z_b[kiteIndex] ?? 0,
          (frame, kiteIndex) => frame.curvature_z_ref[kiteIndex] ?? 0
        )
      ]
    },
    {
      title: "Physics / Forces & Moments",
      description:
        "Body-axis loads generated by the simulated plant. These plots separate total force or moment from the source terms that create it, so you can see whether the tether, aerodynamics, gravity, or propulsion is driving the motion.",
      groups: [
        buildPerKiteBreakdownGroup(kiteCount, "Body X Force Breakdown (N)", "N", [
          {
            name: "Total",
            width: 3,
            alpha: 0.96,
            value: (frame, kiteIndex) => bodyComponent(frame.total_force_b, kiteIndex, 0)
          },
          {
            name: "Aero",
            dash: "dash",
            width: 2.1,
            alpha: 0.86,
            value: (frame, kiteIndex) => bodyComponent(frame.aero_force_b, kiteIndex, 0)
          },
          {
            name: "Tether",
            dash: "dot",
            width: 1.9,
            alpha: 0.84,
            value: (frame, kiteIndex) => bodyComponent(frame.tether_force_b, kiteIndex, 0)
          },
          {
            name: "Gravity",
            dash: "longdash",
            width: 1.7,
            alpha: 0.58,
            value: (frame, kiteIndex) => bodyComponent(frame.gravity_force_b, kiteIndex, 0)
          },
          {
            name: "Motor",
            dash: "dashdot",
            width: 1.8,
            alpha: 0.72,
            value: (frame, kiteIndex) => bodyComponent(frame.motor_force_b, kiteIndex, 0)
          }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "Body Y Force Breakdown (N)", "N", [
          {
            name: "Total",
            width: 3,
            alpha: 0.96,
            value: (frame, kiteIndex) => bodyComponent(frame.total_force_b, kiteIndex, 1)
          },
          {
            name: "Aero",
            dash: "dash",
            width: 2.1,
            alpha: 0.86,
            value: (frame, kiteIndex) => bodyComponent(frame.aero_force_b, kiteIndex, 1)
          },
          {
            name: "Tether",
            dash: "dot",
            width: 1.9,
            alpha: 0.84,
            value: (frame, kiteIndex) => bodyComponent(frame.tether_force_b, kiteIndex, 1)
          },
          {
            name: "Gravity",
            dash: "longdash",
            width: 1.7,
            alpha: 0.58,
            value: (frame, kiteIndex) => bodyComponent(frame.gravity_force_b, kiteIndex, 1)
          },
          {
            name: "Motor",
            dash: "dashdot",
            width: 1.8,
            alpha: 0.72,
            value: (frame, kiteIndex) => bodyComponent(frame.motor_force_b, kiteIndex, 1)
          }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "Body Z Force Breakdown (N)", "N", [
          {
            name: "Total",
            width: 3,
            alpha: 0.96,
            value: (frame, kiteIndex) => bodyComponent(frame.total_force_b, kiteIndex, 2)
          },
          {
            name: "Aero",
            dash: "dash",
            width: 2.1,
            alpha: 0.86,
            value: (frame, kiteIndex) => bodyComponent(frame.aero_force_b, kiteIndex, 2)
          },
          {
            name: "Tether",
            dash: "dot",
            width: 1.9,
            alpha: 0.84,
            value: (frame, kiteIndex) => bodyComponent(frame.tether_force_b, kiteIndex, 2)
          },
          {
            name: "Gravity",
            dash: "longdash",
            width: 1.7,
            alpha: 0.58,
            value: (frame, kiteIndex) => bodyComponent(frame.gravity_force_b, kiteIndex, 2)
          },
          {
            name: "Motor",
            dash: "dashdot",
            width: 1.8,
            alpha: 0.72,
            value: (frame, kiteIndex) => bodyComponent(frame.motor_force_b, kiteIndex, 2)
          }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "Body Roll Moment Breakdown (N m)", "N m", [
          {
            name: "Total",
            width: 3,
            alpha: 0.96,
            value: (frame, kiteIndex) => bodyComponent(frame.total_moment_b, kiteIndex, 0)
          },
          {
            name: "Aero",
            dash: "dash",
            width: 2.1,
            alpha: 0.86,
            value: (frame, kiteIndex) => bodyComponent(frame.aero_moment_b, kiteIndex, 0)
          },
          {
            name: "Tether",
            dash: "dot",
            width: 1.9,
            alpha: 0.84,
            value: (frame, kiteIndex) => bodyComponent(frame.tether_moment_b, kiteIndex, 0)
          },
          {
            name: "Motor",
            dash: "dashdot",
            width: 1.8,
            alpha: 0.72,
            value: (frame, kiteIndex) => bodyComponent(frame.motor_moment_b, kiteIndex, 0)
          }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "Body Pitch Moment Breakdown (N m)", "N m", [
          {
            name: "Total",
            width: 3,
            alpha: 0.96,
            value: (frame, kiteIndex) => bodyComponent(frame.total_moment_b, kiteIndex, 1)
          },
          {
            name: "Aero",
            dash: "dash",
            width: 2.1,
            alpha: 0.86,
            value: (frame, kiteIndex) => bodyComponent(frame.aero_moment_b, kiteIndex, 1)
          },
          {
            name: "Tether",
            dash: "dot",
            width: 1.9,
            alpha: 0.84,
            value: (frame, kiteIndex) => bodyComponent(frame.tether_moment_b, kiteIndex, 1)
          },
          {
            name: "Motor",
            dash: "dashdot",
            width: 1.8,
            alpha: 0.72,
            value: (frame, kiteIndex) => bodyComponent(frame.motor_moment_b, kiteIndex, 1)
          }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "Body Yaw Moment Breakdown (N m)", "N m", [
          {
            name: "Total",
            width: 3,
            alpha: 0.96,
            value: (frame, kiteIndex) => bodyComponent(frame.total_moment_b, kiteIndex, 2)
          },
          {
            name: "Aero",
            dash: "dash",
            width: 2.1,
            alpha: 0.86,
            value: (frame, kiteIndex) => bodyComponent(frame.aero_moment_b, kiteIndex, 2)
          },
          {
            name: "Tether",
            dash: "dot",
            width: 1.9,
            alpha: 0.84,
            value: (frame, kiteIndex) => bodyComponent(frame.tether_moment_b, kiteIndex, 2)
          },
          {
            name: "Motor",
            dash: "dashdot",
            width: 1.8,
            alpha: 0.72,
            value: (frame, kiteIndex) => bodyComponent(frame.motor_moment_b, kiteIndex, 2)
          }
        ])
      ],
      maxColumns: 2
    },
    {
      title: "Physics / Aero Model Terms",
      description:
        "Aero coefficient breakdowns from the actual model implementation. These show which modeled source terms are contributing to lift, drag, side force, and the aerodynamic moments at each instant.",
      groups: [
        buildPerKiteBreakdownGroup(kiteCount, "C_L Terms", "-", [
          { name: "Total", width: 3, alpha: 0.96, value: (frame, kiteIndex) => frame.cl_total[kiteIndex] ?? 0 },
          { name: "C_L0", dash: "dash", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.cl_0_term[kiteIndex] ?? 0 },
          { name: "C_Lα", dash: "dot", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.cl_alpha_term[kiteIndex] ?? 0 },
          { name: "C_Lδe", dash: "dashdot", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.cl_elevator_term[kiteIndex] ?? 0 },
          { name: "C_Lδf", dash: "longdash", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.cl_flap_term[kiteIndex] ?? 0 }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "C_D Terms", "-", [
          { name: "Total", width: 3, alpha: 0.96, value: (frame, kiteIndex) => frame.cd_total[kiteIndex] ?? 0 },
          { name: "C_D0", dash: "dash", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.cd_0_term[kiteIndex] ?? 0 },
          { name: "Induced", dash: "dot", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.cd_induced_term[kiteIndex] ?? 0 },
          { name: "Surface Abs", dash: "dashdot", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.cd_surface_term[kiteIndex] ?? 0 }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "C_Y Terms", "-", [
          { name: "Total", width: 3, alpha: 0.96, value: (frame, kiteIndex) => frame.cy_total[kiteIndex] ?? 0 },
          { name: "C_Yβ", dash: "dash", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.cy_beta_term[kiteIndex] ?? 0 },
          { name: "C_Yδr", dash: "dot", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.cy_rudder_term[kiteIndex] ?? 0 }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "C_l Terms", "-", [
          { name: "Total", width: 3, alpha: 0.96, value: (frame, kiteIndex) => frame.roll_coeff_total[kiteIndex] ?? 0 },
          { name: "C_lβ", dash: "dash", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.roll_beta_term[kiteIndex] ?? 0 },
          { name: "C_lp", dash: "dot", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.roll_p_term[kiteIndex] ?? 0 },
          { name: "C_lr", dash: "dashdot", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.roll_r_term[kiteIndex] ?? 0 },
          { name: "C_lδa", dash: "longdash", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.roll_aileron_term[kiteIndex] ?? 0 }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "C_m Terms", "-", [
          { name: "Total", width: 3, alpha: 0.96, value: (frame, kiteIndex) => frame.pitch_coeff_total[kiteIndex] ?? 0 },
          { name: "C_m0", dash: "dash", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.pitch_0_term[kiteIndex] ?? 0 },
          { name: "C_mα", dash: "dot", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.pitch_alpha_term[kiteIndex] ?? 0 },
          { name: "C_mq", dash: "dashdot", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.pitch_q_term[kiteIndex] ?? 0 },
          { name: "C_mδe", dash: "longdash", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.pitch_elevator_term[kiteIndex] ?? 0 },
          { name: "C_mδf", dash: "dash", width: 1.6, alpha: 0.58, value: (frame, kiteIndex) => frame.pitch_flap_term[kiteIndex] ?? 0 }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "C_n Terms", "-", [
          { name: "Total", width: 3, alpha: 0.96, value: (frame, kiteIndex) => frame.yaw_coeff_total[kiteIndex] ?? 0 },
          { name: "C_nβ", dash: "dash", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.yaw_beta_term[kiteIndex] ?? 0 },
          { name: "C_np", dash: "dot", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.yaw_p_term[kiteIndex] ?? 0 },
          { name: "C_nr", dash: "dashdot", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.yaw_r_term[kiteIndex] ?? 0 },
          { name: "C_nδr", dash: "longdash", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.yaw_rudder_term[kiteIndex] ?? 0 }
        ])
      ],
      maxColumns: 2
    },
    {
      title: "Physics / Energy & Consistency",
      description:
        "Whole-system energy diagnostics and work accounting used to sanity-check the integration.",
      groups: buildEnergyGroups()
    }
  ];
}

function formatProgressSummary(
  request: {
    preset: string;
    phase_mode: PhaseMode;
    sim_noise_enabled: boolean;
    bridle_enabled: boolean;
  },
  progress: SimulationProgress,
  receivedFrames: number,
  renderedFrames: number,
  bufferedFrames: number,
  playbackLabel: string
): string {
  const pct =
    progress.duration > 0 ? (100 * progress.time) / progress.duration : 0;
  return renderSummaryCard(
    "Running",
    `${progress.time.toFixed(2)} / ${progress.duration.toFixed(2)} s`,
    [
      { label: "Progress", value: `${pct.toFixed(1)}%` },
      { label: "Preset", value: request.preset },
      { label: "Phase Mode", value: request.phase_mode },
      { label: "Bridle", value: request.bridle_enabled ? "Enabled" : "CG attach" },
      { label: "Sim Noise", value: request.sim_noise_enabled ? "Dryden gusts" : "Off" },
      { label: "Time Dilation", value: playbackLabel },
      { label: "Iteration", value: String(progress.iteration) },
      { label: "Frames", value: `${receivedFrames} received / ${renderedFrames} rendered` },
      { label: "Buffered Frames", value: String(bufferedFrames) },
      {
        label: "Accepted / Rejected",
        value: `${progress.accepted_steps_total} / ${progress.rejected_steps_total}`
      },
      { label: "Last Interval dt", value: `${progress.interval_dt.toFixed(4)} s` },
      {
        label: "Last Interval Steps",
        value: `accepted ${progress.accepted_steps_interval}, rejected ${progress.rejected_steps_interval}`
      },
      {
        label: "Last Interval Substeps",
        value: `${progress.substeps_interval} / ${progress.substep_budget}`
      }
    ],
    "Live solver status"
  );
}

function formatRunSummary(
  summary: RunSummary,
  receivedFrames: number,
  renderedFrames: number,
  playbackLabel: string
): string {
  return renderSummaryCard(
    summary.failure ? "Terminated Early" : "Completed",
    `${summary.duration.toFixed(2)} s simulated`,
    [
      { label: "Time Dilation", value: playbackLabel },
      { label: "Frames", value: `${receivedFrames} received / ${renderedFrames} rendered` },
      {
        label: "Accepted / Rejected",
        value: `${summary.accepted_steps} / ${summary.rejected_steps}`
      },
      { label: "Max Phase Error", value: summary.max_phase_error.toFixed(4) },
      { label: "Final Motor Work", value: summary.final_total_work.toFixed(3) },
      {
        label: "Final Dissipated Work",
        value: summary.final_total_dissipated_work.toFixed(3)
      },
      {
        label: "Final Kinetic Energy",
        value: summary.final_total_kinetic_energy.toFixed(3)
      },
      {
        label: "Final Potential Energy",
        value: summary.final_total_potential_energy.toFixed(3)
      },
      {
        label: "Final Tether Strain Energy",
        value: summary.final_total_tether_strain_energy.toFixed(3)
      },
      {
        label: "Final Mechanical Energy",
        value: summary.final_total_mechanical_energy.toFixed(3)
      },
      {
        label: "Final E - W + D",
        value: summary.final_energy_conservation_residual.toFixed(3)
      }
    ],
    summary.failure ? "Ended early on a protection limit" : "Run finished normally"
  );
}

function applyPresetDefaults(): void {
  if (presetSelect.value === "simple_tether" && windInput.value === "5") {
    windInput.value = "0";
  }
  if (presetSelect.value === "free_flight1" && durationInput.value === "10") {
    durationInput.value = "60";
  }
}

function wrapAngleRad(angle: number): number {
  let wrapped = angle;
  while (wrapped > Math.PI) {
    wrapped -= 2 * Math.PI;
  }
  while (wrapped < -Math.PI) {
    wrapped += 2 * Math.PI;
  }
  return wrapped;
}

function currentCameraFollowTarget(): CameraFollowTarget {
  return cameraFollowTargetSelect.value as CameraFollowTarget;
}

function currentCameraFollowKiteIndex(): number | null {
  const selection = currentCameraFollowTarget();
  if (!selection.startsWith("kite:")) {
    return null;
  }
  const index = Number(selection.slice("kite:".length));
  return Number.isInteger(index) && index >= 0 ? index : null;
}

function updateCameraFollowUiState(): void {
  const followKite = currentCameraFollowKiteIndex() !== null;
  cameraFollowYawInput.disabled = !followKite;
  cameraFollowYawLabel.classList.toggle("disabled", !followKite);
}

function updateCameraFollowOptions(kiteCount: number): void {
  const previousValue = currentCameraFollowTarget();
  cameraFollowTargetSelect.innerHTML = "";

  const options: Array<{ value: CameraFollowTarget; label: string }> = [
    { value: "manual", label: "Manual" },
    { value: "disk_center", label: "Disk Center" }
  ];

  for (let index = 0; index < kiteCount; index += 1) {
    options.push({
      value: `kite:${index}`,
      label: `Kite ${index + 1}`
    });
  }

  options.forEach((optionDef) => {
    const option = document.createElement("option");
    option.value = optionDef.value;
    option.textContent = optionDef.label;
    cameraFollowTargetSelect.append(option);
  });

  const nextValue = options.some((optionDef) => optionDef.value === previousValue)
    ? previousValue
    : "disk_center";
  cameraFollowTargetSelect.value = nextValue;
  updateCameraFollowUiState();
}

function makeTetherMaterial(color = 0x66b8ff): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.22,
    roughness: 0.28,
    metalness: 0.08
  });
}

function toThree(point: [number, number, number]): THREE.Vector3 {
  return new THREE.Vector3(point[0], point[1], -point[2]);
}

function toThreeVector(vectorN: [number, number, number]): THREE.Vector3 {
  return new THREE.Vector3(vectorN[0], vectorN[1], -vectorN[2]);
}

function meanVector(points: [number, number, number][]): [number, number, number] {
  if (points.length === 0) {
    return [0, 0, 0];
  }
  const sum = points.reduce<[number, number, number]>(
    (acc, point) => [acc[0] + point[0], acc[1] + point[1], acc[2] + point[2]],
    [0, 0, 0]
  );
  return [sum[0] / points.length, sum[1] / points.length, sum[2] / points.length];
}

function flowBasis(direction: THREE.Vector3): {
  forward: THREE.Vector3;
  lateral: THREE.Vector3;
  vertical: THREE.Vector3;
} {
  const forward =
    direction.lengthSq() > 1.0e-9
      ? direction.clone().normalize()
      : new THREE.Vector3(1, 0, 0);
  const fallbackUp =
    Math.abs(forward.z) > 0.82
      ? new THREE.Vector3(0, 1, 0)
      : new THREE.Vector3(0, 0, 1);
  const lateral = new THREE.Vector3().crossVectors(forward, fallbackUp).normalize();
  const vertical = new THREE.Vector3().crossVectors(lateral, forward).normalize();
  return { forward, lateral, vertical };
}

function airflowCenter(frame: ApiFrame): THREE.Vector3 {
  if (frame.kite_positions_n.length > 0 && frame.control_ring_radius > 1.0e-6) {
    return toThree(frame.control_ring_center_n);
  }
  return toThree(frame.splitter_position_n);
}

function airflowLongitudinalSpan(frame: ApiFrame): number {
  const dx = frame.payload_position_n[0] - frame.splitter_position_n[0];
  const dy = frame.payload_position_n[1] - frame.splitter_position_n[1];
  const dz = frame.payload_position_n[2] - frame.splitter_position_n[2];
  const commonSpan = Math.hypot(dx, dy, dz);
  return Math.max(95, frame.control_ring_radius * 2.0, 1.15 * commonSpan);
}

function airflowCrossSpan(frame: ApiFrame): number {
  return Math.max(55, frame.control_ring_radius * 1.2);
}

function cameraFollowTargetPosition(frame: ApiFrame): THREE.Vector3 | null {
  const selection = currentCameraFollowTarget();
  if (selection === "manual") {
    return null;
  }
  if (selection === "disk_center") {
    if (frame.kite_positions_n.length > 0 && frame.control_ring_radius > 1.0e-6) {
      return toThree(frame.control_ring_center_n);
    }
    return toThree(frame.splitter_position_n);
  }
  const kiteIndex = currentCameraFollowKiteIndex();
  const kitePosition = kiteIndex === null ? null : frame.kite_positions_n[kiteIndex];
  return kitePosition ? toThree(kitePosition) : null;
}

function kiteHeadingRad(frame: ApiFrame, kiteIndex: number): number | null {
  const quaternionN2B = frame.kite_quaternions_n2b[kiteIndex];
  if (!quaternionN2B) {
    return null;
  }
  const orientation = kiteQuaternionToThree(quaternionN2B);
  const noseDirection = new THREE.Vector3(1, 0, 0).applyQuaternion(orientation);
  noseDirection.z = 0;
  if (noseDirection.lengthSq() < 1.0e-9) {
    return null;
  }
  noseDirection.normalize();
  return Math.atan2(noseDirection.y, noseDirection.x);
}

function resetCameraFollowState(): void {
  lastCameraFollowHeadingRad = null;
  lastCameraFollowTarget = null;
}

function snapCameraTargetToFrame(frame: ApiFrame): void {
  const target =
    cameraFollowTargetPosition(frame) ??
    (frame.kite_positions_n.length > 0 && frame.control_ring_radius > 1.0e-6
      ? toThree(frame.control_ring_center_n)
      : toThree(frame.splitter_position_n));
  const targetDelta = target.clone().sub(controls.target);
  camera.position.add(targetDelta);
  controls.target.copy(target);
  controls.update();
}

function applyCameraFollow(frame: ApiFrame): void {
  const selection = currentCameraFollowTarget();
  const target = cameraFollowTargetPosition(frame);
  if (!target) {
    resetCameraFollowState();
    return;
  }

  const currentOffset = camera.position.clone().sub(controls.target);
  let nextOffset = currentOffset;
  const kiteIndex = currentCameraFollowKiteIndex();
  const yawFollowEnabled = kiteIndex !== null && cameraFollowYawInput.checked;
  let currentHeading: number | null = null;

  if (kiteIndex !== null) {
    currentHeading = kiteHeadingRad(frame, kiteIndex);
    if (
      yawFollowEnabled &&
      currentHeading !== null &&
      lastCameraFollowTarget === selection &&
      lastCameraFollowHeadingRad !== null
    ) {
      const deltaYaw = wrapAngleRad(currentHeading - lastCameraFollowHeadingRad);
      nextOffset = currentOffset.clone().applyAxisAngle(new THREE.Vector3(0, 0, 1), deltaYaw);
    }
  }

  camera.position.copy(target.clone().add(nextOffset));
  controls.target.copy(target);
  controls.update();

  lastCameraFollowTarget = selection;
  lastCameraFollowHeadingRad = currentHeading;
}

function setParticleArrayEntry(
  positions: Float32Array,
  colors: Float32Array,
  index: number,
  position: THREE.Vector3,
  color: THREE.Color
): void {
  const base = index * 3;
  positions[base] = position.x;
  positions[base + 1] = position.y;
  positions[base + 2] = position.z;
  colors[base] = color.r;
  colors[base + 1] = color.g;
  colors[base + 2] = color.b;
}

function setWingtipTrailEntry(
  index: number,
  position: THREE.Vector3,
  color: THREE.Color,
  alpha: number
): void {
  const base = index * 3;
  wingtipTrailPositions[base] = position.x;
  wingtipTrailPositions[base + 1] = position.y;
  wingtipTrailPositions[base + 2] = position.z;
  wingtipTrailColors[base] = color.r;
  wingtipTrailColors[base + 1] = color.g;
  wingtipTrailColors[base + 2] = color.b;
  wingtipTrailAlpha[index] = alpha;
}

function randomCentered(span: number): number {
  return (Math.random() - 0.5) * 2.0 * span;
}

function randomBoxPosition(
  center: THREE.Vector3,
  forward: THREE.Vector3,
  lateral: THREE.Vector3,
  vertical: THREE.Vector3,
  longSpan: number,
  crossSpan: number,
  verticalSpan: number
): THREE.Vector3 {
  return center
    .clone()
    .addScaledVector(forward, randomCentered(longSpan))
    .addScaledVector(lateral, randomCentered(crossSpan))
    .addScaledVector(vertical, randomCentered(verticalSpan));
}

function randomGridVolumePosition(): THREE.Vector3 {
  return toThree([
    randomCentered(GRID_HALF_EXTENT),
    randomCentered(GRID_HALF_EXTENT),
    -Math.random() * GRID_HALF_EXTENT
  ]);
}

function isOutsideGridVolume(position: THREE.Vector3): boolean {
  return (
    Math.abs(position.x) > GRID_HALF_EXTENT ||
    Math.abs(position.y) > GRID_HALF_EXTENT ||
    position.z < 0 ||
    position.z > GRID_HALF_EXTENT
  );
}

function kiteQuaternionToThree(
  quaternionN2B: [number, number, number, number]
): THREE.Quaternion {
  const [w, i, j, k] = quaternionN2B;
  return new THREE.Quaternion(-i, -j, k, w).normalize();
}

function wingtipWorldPositions(
  frame: ApiFrame,
  kiteIndex: number
): { left: THREE.Vector3; right: THREE.Vector3 } | null {
  const dimensions = deriveKiteVisualDimensions(frame, kiteIndex);
  const position = frame.kite_positions_n[kiteIndex];
  const quaternionN2B = frame.kite_quaternions_n2b[kiteIndex];
  if (!dimensions || !position || !quaternionN2B) {
    return null;
  }

  const kitePosition = toThree(position);
  const orientation = kiteQuaternionToThree(quaternionN2B);
  const leftOffset = new THREE.Vector3(
    dimensions.wingX,
    0.5 * dimensions.wingSpan,
    0
  ).applyQuaternion(orientation);
  const rightOffset = new THREE.Vector3(
    dimensions.wingX,
    -0.5 * dimensions.wingSpan,
    0
  ).applyQuaternion(orientation);
  return {
    left: kitePosition.clone().add(leftOffset),
    right: kitePosition.clone().add(rightOffset)
  };
}

function airflowVelocity(frame: ApiFrame): THREE.Vector3 {
  const meanGustN = meanVector(frame.kite_gust_n);
  return toThreeVector([
    frame.clean_wind_n[0] + meanGustN[0],
    frame.clean_wind_n[1] + meanGustN[1],
    frame.clean_wind_n[2] + meanGustN[2]
  ]);
}

function gustMagnitude(frame: ApiFrame): number {
  const gustN = meanVector(frame.kite_gust_n);
  return Math.hypot(gustN[0], gustN[1], gustN[2]);
}

function initializeAmbientParticles(frame: ApiFrame, velocity: THREE.Vector3): void {
  const flowMagnitude = velocity.length();
  ambientParticleCloud.visible = flowMagnitude > 1.0e-4;
  if (!ambientParticleCloud.visible) {
    return;
  }

  const color = ambientAirColor.clone().lerp(
    ambientAirColorHigh,
    Math.min(1, flowMagnitude / 12.0)
  );

  for (let index = 0; index < AIRFLOW_AMBIENT_PARTICLE_COUNT; index += 1) {
    const state = ambientParticleStates[index];
    state.position.copy(randomGridVolumePosition());
    state.age = 0;
    state.life = (2.2 * GRID_SIZE) / Math.max(2.0, flowMagnitude) * (0.8 + 0.4 * Math.random());
    state.drift = 0.35 + 0.65 * Math.random();
    setParticleArrayEntry(ambientParticlePositions, ambientParticleColors, index, state.position, color);
  }

  (ambientParticleGeometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
  (ambientParticleGeometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
}

function resetAmbientParticle(index: number, frame: ApiFrame, velocity: THREE.Vector3): void {
  const flowMagnitude = Math.max(2.0, velocity.length());
  const state = ambientParticleStates[index];
  state.position.copy(randomGridVolumePosition());
  state.age = 0;
  state.life = (2.2 * GRID_SIZE) / flowMagnitude * (0.8 + 0.4 * Math.random());
  state.drift = 0.35 + 0.65 * Math.random();
}

function updateAmbientParticles(dtSimSeconds: number, frame: ApiFrame, velocity: THREE.Vector3): void {
  const flowMagnitude = velocity.length();
  ambientParticleCloud.visible = flowMagnitude > 1.0e-4;
  if (!ambientParticleCloud.visible) {
    return;
  }

  const color = ambientAirColor.clone().lerp(
    ambientAirColorHigh,
    Math.min(1, flowMagnitude / 12.0)
  );

  for (let index = 0; index < AIRFLOW_AMBIENT_PARTICLE_COUNT; index += 1) {
    const state = ambientParticleStates[index];
    if (state.life <= 0 || state.age >= state.life) {
      resetAmbientParticle(index, frame, velocity);
    }

    state.age += dtSimSeconds;
    state.position.addScaledVector(velocity, dtSimSeconds);

    if (isOutsideGridVolume(state.position)) {
      resetAmbientParticle(index, frame, velocity);
    }

    setParticleArrayEntry(
      ambientParticlePositions,
      ambientParticleColors,
      index,
      state.position,
      color
    );
  }

  (ambientParticleGeometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
  (ambientParticleGeometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
}

function gustParticleColor(gustMagnitude: number): THREE.Color {
  const normalized = Math.min(1, gustMagnitude / 6.0);
  if (normalized <= 0.5) {
    return gustAirColorLow.clone().lerp(gustAirColorMid, normalized / 0.5);
  }
  return gustAirColorMid.clone().lerp(gustAirColorHigh, (normalized - 0.5) / 0.5);
}

function resetGustParticle(index: number, frame: ApiFrame, velocity: THREE.Vector3): void {
  const state = gustParticleStates[index];
  state.position.copy(randomGridVolumePosition());
  state.age = 0;
  state.life = (1.6 * GRID_SIZE) / Math.max(1.0, velocity.length()) * (0.75 + 0.4 * Math.random());
  state.drift = 0.35 + 0.65 * Math.random();
}

function initializeGustParticles(frame: ApiFrame, velocity: THREE.Vector3): void {
  const gustStrength = gustMagnitude(frame);
  gustParticleCloud.visible = gustStrength > 1.0e-3;
  if (!gustParticleCloud.visible) {
    return;
  }

  const color = gustParticleColor(gustStrength);
  for (let index = 0; index < AIRFLOW_GUST_PARTICLE_COUNT; index += 1) {
    const state = gustParticleStates[index];
    state.position.copy(randomGridVolumePosition());
    state.age = 0;
    state.life = (2.0 * GRID_SIZE) / Math.max(1.0, velocity.length()) * (0.8 + 0.45 * Math.random());
    state.drift = 0.35 + 0.65 * Math.random();
    setParticleArrayEntry(gustParticlePositions, gustParticleColors, index, state.position, color);
  }

  (gustParticleGeometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
  (gustParticleGeometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
}

function updateGustParticles(dtSimSeconds: number, frame: ApiFrame, velocity: THREE.Vector3): void {
  const gustStrength = gustMagnitude(frame);
  gustParticleCloud.visible = gustStrength > 1.0e-3;
  if (!gustParticleCloud.visible) {
    return;
  }

  for (let index = 0; index < AIRFLOW_GUST_PARTICLE_COUNT; index += 1) {
    const state = gustParticleStates[index];
    if (state.life <= 0 || state.age >= state.life) {
      resetGustParticle(index, frame, velocity);
    }

    state.age += dtSimSeconds;
    state.position.addScaledVector(velocity, dtSimSeconds);

    if (isOutsideGridVolume(state.position)) {
      resetGustParticle(index, frame, velocity);
    }

    setParticleArrayEntry(
      gustParticlePositions,
      gustParticleColors,
      index,
      state.position,
      gustParticleColor(gustStrength)
    );
  }

  (gustParticleGeometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
  (gustParticleGeometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
}

function emitWingtipTrailParticle(
  position: THREE.Vector3,
  color: THREE.Color
): void {
  const index = nextWingtipTrailParticleIndex;
  nextWingtipTrailParticleIndex = (nextWingtipTrailParticleIndex + 1) % WINGTIP_TRAIL_PARTICLE_COUNT;
  const state = wingtipTrailStates[index];
  state.position.copy(position);
  state.age = 0;
  state.life = WINGTIP_TRAIL_LIFETIME_S;
  state.active = true;
  setWingtipTrailEntry(index, state.position, color, 1);
}

function updateWingtipTrailParticles(dtSimSeconds: number, frame: ApiFrame): void {
  const advectionVelocity = airflowVelocity(frame);
  let hasActiveParticles = false;

  for (let index = 0; index < WINGTIP_TRAIL_PARTICLE_COUNT; index += 1) {
    const state = wingtipTrailStates[index];
    if (!state.active) {
      continue;
    }

    hasActiveParticles = true;
    state.age += dtSimSeconds;
    if (state.age >= state.life) {
      state.active = false;
      wingtipTrailAlpha[index] = 0;
      continue;
    }

    state.position.addScaledVector(advectionVelocity, dtSimSeconds);
    const fade = Math.max(0, 1 - state.age / state.life);
    wingtipTrailPositions[index * 3] = state.position.x;
    wingtipTrailPositions[index * 3 + 1] = state.position.y;
    wingtipTrailPositions[index * 3 + 2] = state.position.z;
    wingtipTrailAlpha[index] = fade;
  }

  for (let kiteIndex = 0; kiteIndex < frame.kite_positions_n.length; kiteIndex += 1) {
    const wingtips = wingtipWorldPositions(frame, kiteIndex);
    if (!wingtips) {
      continue;
    }
    const baseColor = new THREE.Color(kiteColor(kiteIndex));
    emitWingtipTrailParticle(wingtips.left, baseColor.clone().lerp(WINGTIP_TRAIL_LEFT_COLOR, 0.72));
    emitWingtipTrailParticle(wingtips.right, baseColor.clone().lerp(WINGTIP_TRAIL_RIGHT_COLOR, 0.72));
    hasActiveParticles = true;
  }

  wingtipTrailCloud.visible = hasActiveParticles;
  (wingtipTrailGeometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
  (wingtipTrailGeometry.attributes.alpha as THREE.BufferAttribute).needsUpdate = true;
  (wingtipTrailGeometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
}

function advanceAirflowParticlesToFrame(frame: ApiFrame): void {
  const velocity = airflowVelocity(frame);
  if (lastAirflowFrameTime === null) {
    initializeAmbientParticles(frame, velocity);
    initializeGustParticles(frame, velocity);
    lastAirflowFrameTime = frame.time;
    return;
  }

  const dtSimSeconds = Math.max(0, frame.time - lastAirflowFrameTime);
  updateAmbientParticles(dtSimSeconds, frame, velocity);
  updateGustParticles(dtSimSeconds, frame, velocity);
  updateWingtipTrailParticles(dtSimSeconds, frame);
  lastAirflowFrameTime = frame.time;
}

function tensionToColor(tensionN: number): THREE.Color {
  const clamped = Math.max(TETHER_TENSION_MIN_N, tensionN);
  const normalized = Math.min(1, clamped / TETHER_TENSION_MAX_N);
  const shaped = Math.sqrt(normalized);
  if (shaped <= 0.55) {
    return tensionColorLow.clone().lerp(tensionColorMid, shaped / 0.55);
  }
  return tensionColorMid.clone().lerp(tensionColorHigh, (shaped - 0.55) / 0.45);
}

function phaseErrorColor(phaseErrorRad: number): THREE.Color {
  const normalized = Math.min(1, Math.abs(phaseErrorRad) / PHASE_ERROR_MAX_RAD);
  const shaped = Math.sqrt(normalized);
  if (shaped <= 0.55) {
    return phaseColorLow.clone().lerp(phaseColorMid, shaped / 0.55);
  }
  return phaseColorMid.clone().lerp(phaseColorHigh, (shaped - 0.55) / 0.45);
}

function setMaterialColor(
  material: THREE.Material | THREE.Material[],
  color: THREE.Color,
  emissiveIntensity = 0.22
): void {
  const materials = Array.isArray(material) ? material : [material];
  materials.forEach((entry) => {
    if (entry instanceof THREE.MeshStandardMaterial) {
      entry.color.copy(color);
      entry.emissive.copy(color);
      entry.emissiveIntensity = emissiveIntensity;
    }
  });
}

function ensureLineCount(target: number, bucket: THREE.Line[], factory: () => THREE.Line): void {
  while (bucket.length < target) {
    const line = factory();
    scene.add(line);
    bucket.push(line);
  }
  bucket.forEach((line, index) => {
    line.visible = index < target;
  });
}

function makeSceneLine(color: number, opacity: number): THREE.Line {
  return new THREE.Line(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity
    })
  );
}

function updateLine(
  line: THREE.Line,
  start: THREE.Vector3,
  end: THREE.Vector3,
  color: THREE.Color,
  opacity = 0.6
): void {
  if (start.distanceToSquared(end) < 1.0e-10) {
    line.visible = false;
    return;
  }
  line.visible = true;
  (line.geometry as THREE.BufferGeometry).setFromPoints([start, end]);
  const material = line.material as THREE.LineBasicMaterial;
  material.color.copy(color);
  material.opacity = opacity;
}

function ensureMeshCount(target: number, bucket: THREE.Mesh[], factory: () => THREE.Mesh): void {
  while (bucket.length < target) {
    const mesh = factory();
    scene.add(mesh);
    bucket.push(mesh);
  }
  bucket.forEach((mesh, index) => {
    mesh.visible = index < target;
  });
}

function makeTetherSegmentMesh(radius: number): THREE.Mesh {
  const mesh = new THREE.Mesh(tetherSegmentGeometry, makeTetherMaterial());
  mesh.userData.segmentRadius = radius;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  return mesh;
}

function makeTetherNodeMesh(radius: number): THREE.Mesh {
  const mesh = new THREE.Mesh(tetherNodeGeometry, makeTetherMaterial());
  mesh.scale.setScalar(radius);
  mesh.userData.nodeRadius = radius;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  return mesh;
}

function makeProjectedPhaseMesh(): THREE.Mesh {
  return new THREE.Mesh(
    new THREE.SphereGeometry(1.75, 14, 14),
    new THREE.MeshStandardMaterial({
      color: 0x66b8ff,
      emissive: 0x66b8ff,
      emissiveIntensity: 0.32,
      roughness: 0.2,
      metalness: 0.06
    })
  );
}

function makePhaseSlotMesh(): THREE.Mesh {
  return new THREE.Mesh(
    new THREE.SphereGeometry(1.25, 12, 12),
    new THREE.MeshStandardMaterial({
      color: 0xbfd8ea,
      emissive: 0xbfd8ea,
      emissiveIntensity: 0.15,
      transparent: true,
      opacity: 0.66,
      roughness: 0.34,
      metalness: 0.02
    })
  );
}

function updateSegmentMesh(
  mesh: THREE.Mesh,
  start: [number, number, number],
  end: [number, number, number],
  tensionN: number
): void {
  const startVec = toThree(start);
  const endVec = toThree(end);
  const delta = endVec.clone().sub(startVec);
  const length = delta.length();
  if (length < 1e-6) {
    mesh.visible = false;
    return;
  }
  mesh.visible = true;
  mesh.position.copy(startVec).add(endVec).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(tetherAxis, delta.normalize());
  mesh.scale.set(mesh.userData.segmentRadius as number, length, mesh.userData.segmentRadius as number);
  const color = tensionToColor(tensionN);
  const material = mesh.material as THREE.MeshStandardMaterial;
  material.color.copy(color);
  material.emissive.copy(color);
}

function updateNodeMesh(mesh: THREE.Mesh, point: [number, number, number], tensionN: number): void {
  mesh.visible = true;
  mesh.position.copy(toThree(point));
  const color = tensionToColor(tensionN);
  const material = mesh.material as THREE.MeshStandardMaterial;
  material.color.copy(color);
  material.emissive.copy(color);
}

function renderTether(
  points: [number, number, number][] | undefined,
  tensions: number[] | undefined,
  segmentMeshes: THREE.Mesh[],
  nodeMeshes: THREE.Mesh[],
  segmentRadius: number,
  nodeRadius: number
): void {
  const safePoints = points ?? [];
  const safeTensions = tensions ?? [];
  const segmentCount = Math.max(0, safePoints.length - 1);
  const nodeCount = Math.max(0, safePoints.length - 2);
  ensureMeshCount(segmentCount, segmentMeshes, () => makeTetherSegmentMesh(segmentRadius));
  ensureMeshCount(nodeCount, nodeMeshes, () => makeTetherNodeMesh(nodeRadius));

  for (let index = 0; index < segmentCount; index += 1) {
    updateSegmentMesh(
      segmentMeshes[index],
      safePoints[index],
      safePoints[index + 1],
      safeTensions[index] ?? 0
    );
  }

  for (let index = 0; index < nodeCount; index += 1) {
    const leftTension = safeTensions[index] ?? 0;
    const rightTension = safeTensions[index + 1] ?? leftTension;
    updateNodeMesh(nodeMeshes[index], safePoints[index + 1], 0.5 * (leftTension + rightTension));
  }
}

function updateControlRing(frame: ApiFrame): void {
  const kiteCount = frame.kite_positions_n.length;
  const ringVisible = kiteCount > 0 && frame.control_ring_radius > 1.0e-6;
  controlRingLine.visible = ringVisible;
  controlAxisLine.visible = ringVisible;
  controlCenterMarker.visible = ringVisible;

  if (!ringVisible) {
    phaseSlotMeshes.forEach((mesh) => {
      mesh.visible = false;
    });
    return;
  }

  const center = frame.control_ring_center_n;
  const radius = frame.control_ring_radius;
  const ringPoints = Array.from({ length: CONTROL_RING_SEGMENTS }, (_, index) => {
    const theta = (2 * Math.PI * index) / CONTROL_RING_SEGMENTS;
    return toThree([
      center[0] + radius * Math.cos(theta),
      center[1] + radius * Math.sin(theta),
      center[2]
    ]);
  });
  (controlRingLine.geometry as THREE.BufferGeometry).setFromPoints(ringPoints);
  controlCenterMarker.position.copy(toThree(center));

  const axisStart = toThree([center[0], center[1], center[2] - CONTROL_AXIS_HALF_LENGTH]);
  const axisEnd = toThree([center[0], center[1], center[2] + CONTROL_AXIS_HALF_LENGTH]);
  updateLine(controlAxisLine, axisStart, axisEnd, new THREE.Color("#ffd36b"), 0.28);

  const showAdaptiveSlots = activeSummaryRequest?.phase_mode === "adaptive";
  ensureMeshCount(showAdaptiveSlots ? kiteCount : 0, phaseSlotMeshes, makePhaseSlotMesh);
  if (showAdaptiveSlots) {
    for (let index = 0; index < kiteCount; index += 1) {
      const theta = (2 * Math.PI * index) / kiteCount;
      phaseSlotMeshes[index].visible = true;
      phaseSlotMeshes[index].position.copy(
        toThree([
          center[0] + radius * Math.cos(theta),
          center[1] + radius * Math.sin(theta),
          center[2]
        ])
      );
      setMaterialColor(
        phaseSlotMeshes[index].material,
        new THREE.Color(kiteColor(index)),
        0.12
      );
    }
  }
}

function deriveKiteVisualDimensions(frame: ApiFrame, index: number): KiteVisualDimensions | null {
  const wingSpan = frame.kite_ref_span[index];
  const wingChord = frame.kite_ref_chord[index];
  if (!Number.isFinite(wingSpan) || wingSpan <= 0 || !Number.isFinite(wingChord) || wingChord <= 0) {
    return null;
  }

  const wingArea = frame.kite_ref_area[index] ?? wingSpan * wingChord;
  const meanChord = Number.isFinite(wingArea) && wingArea > 1.0e-6 ? wingArea / wingSpan : wingChord;
  const effectiveChord = Math.max(meanChord, wingChord);
  const cadOffsetB = frame.kite_cad_offset_b[index] ?? [0, 0, 0];
  const bridlePivotB = frame.kite_bridle_pivot_b[index] ?? [0, 0, 0];
  const bridleRadius = Math.max(0.02, frame.kite_bridle_radius[index] ?? 0.05);
  const fuselageLength = Math.max(2.65 * effectiveChord, 0.44 * wingSpan);

  return {
    wingSpan,
    wingChord: effectiveChord,
    wingThickness: Math.max(0.045, 0.11 * effectiveChord),
    fuselageLength,
    fuselageRadius: Math.max(0.085, 0.18 * effectiveChord, 1.7 * bridleRadius),
    noseLength: 0.58 * effectiveChord,
    tailConeLength: 0.95 * effectiveChord,
    wingX: 0.06 * effectiveChord - cadOffsetB[0],
    tailX: -0.36 * fuselageLength,
    horizontalTailSpan: 0.36 * wingSpan,
    horizontalTailChord: 0.46 * effectiveChord,
    verticalTailHeight: 0.82 * effectiveChord,
    verticalTailChord: 0.52 * effectiveChord,
    tailThickness: Math.max(0.026, 0.09 * effectiveChord),
    bridlePivotB,
    bridleRadius
  };
}

function fallbackKiteVisualDimensions(frame: ApiFrame): KiteVisualDimensions | null {
  for (let index = 0; index < frame.kite_ref_span.length; index += 1) {
    const dimensions = deriveKiteVisualDimensions(frame, index);
    if (dimensions) {
      return dimensions;
    }
  }
  return null;
}

function kiteVisualKey(frame: ApiFrame): string {
  return JSON.stringify({
    span: frame.kite_ref_span,
    chord: frame.kite_ref_chord,
    area: frame.kite_ref_area,
    bridle: frame.kite_bridle_radius
  });
}

function cylinderAlongX(
  radiusStart: number,
  radiusEnd: number,
  length: number,
  material: THREE.Material
): THREE.Mesh {
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radiusEnd, radiusStart, Math.max(length, 0.02), 18, 1),
    material
  );
  mesh.rotation.z = Math.PI / 2;
  return mesh;
}

function makeKiteMesh(dimensions: KiteVisualDimensions): THREE.Group {
  const group = new THREE.Group();
  const fuselageMaterial = new THREE.MeshStandardMaterial({
    color: 0x7db7df,
    roughness: 0.36,
    metalness: 0.08
  });
  const wingMaterial = new THREE.MeshStandardMaterial({
    color: 0xdcecff,
    roughness: 0.28,
    metalness: 0.1
  });
  const tailMaterial = new THREE.MeshStandardMaterial({
    color: 0x9bcff3,
    roughness: 0.3,
    metalness: 0.08
  });
  const canopyMaterial = new THREE.MeshStandardMaterial({
    color: 0x9ae7ff,
    emissive: 0x36a9d9,
    emissiveIntensity: 0.1,
    roughness: 0.18,
    metalness: 0.2,
    transparent: true,
    opacity: 0.7
  });

  const aftX = -0.5 * dimensions.fuselageLength;
  const foreX = 0.5 * dimensions.fuselageLength;
  const tubeStart = aftX + dimensions.tailConeLength;
  const tubeEnd = foreX - dimensions.noseLength;
  const tubeLength = Math.max(0.18, tubeEnd - tubeStart);
  const tube = cylinderAlongX(
    dimensions.fuselageRadius * 0.92,
    dimensions.fuselageRadius,
    tubeLength,
    fuselageMaterial
  );
  tube.position.x = 0.5 * (tubeStart + tubeEnd);
  group.add(tube);

  const nose = new THREE.Mesh(
    new THREE.ConeGeometry(dimensions.fuselageRadius * 0.94, dimensions.noseLength, 18, 1),
    fuselageMaterial
  );
  nose.rotation.z = -Math.PI / 2;
  nose.position.x = foreX - 0.5 * dimensions.noseLength;
  group.add(nose);

  const tailCone = cylinderAlongX(
    dimensions.fuselageRadius * 0.76,
    dimensions.fuselageRadius * 0.28,
    dimensions.tailConeLength,
    fuselageMaterial
  );
  tailCone.position.x = aftX + 0.5 * dimensions.tailConeLength;
  group.add(tailCone);

  const canopy = new THREE.Mesh(
    new THREE.SphereGeometry(dimensions.fuselageRadius * 0.72, 18, 14),
    canopyMaterial
  );
  canopy.scale.set(1.25, 0.95, 0.72);
  canopy.position.set(0.18 * dimensions.fuselageLength, 0, dimensions.fuselageRadius * 0.55);
  group.add(canopy);

  const wing = new THREE.Mesh(
    new THREE.BoxGeometry(
      dimensions.wingChord,
      dimensions.wingSpan,
      dimensions.wingThickness
    ),
    wingMaterial
  );
  wing.position.set(dimensions.wingX, 0, 0);
  group.add(wing);

  const horizontalTail = new THREE.Mesh(
    new THREE.BoxGeometry(
      dimensions.horizontalTailChord,
      dimensions.horizontalTailSpan,
      dimensions.tailThickness
    ),
    tailMaterial
  );
  horizontalTail.position.set(dimensions.tailX, 0, 0.02);
  group.add(horizontalTail);

  const verticalTail = new THREE.Mesh(
    new THREE.BoxGeometry(
      dimensions.verticalTailChord,
      dimensions.tailThickness * 1.15,
      dimensions.verticalTailHeight
    ),
    tailMaterial
  );
  verticalTail.position.set(
    dimensions.tailX - 0.08 * dimensions.horizontalTailChord,
    0,
    0.5 * dimensions.verticalTailHeight
  );
  group.add(verticalTail);

  const bridlePickup = new THREE.Mesh(
    new THREE.SphereGeometry(Math.max(0.028, 0.55 * dimensions.bridleRadius), 12, 12),
    new THREE.MeshStandardMaterial({
      color: 0xffde9c,
      emissive: 0xffde9c,
      emissiveIntensity: 0.18,
      roughness: 0.32,
      metalness: 0.1
    })
  );
  bridlePickup.position.set(
    dimensions.bridlePivotB[0],
    dimensions.bridlePivotB[1],
    dimensions.bridlePivotB[2]
  );
  group.add(bridlePickup);

  return group;
}

function ensureKites(count: number, frame: ApiFrame): void {
  const geometryKey = kiteVisualKey(frame);
  const defaultDimensions = fallbackKiteVisualDimensions(frame);
  if (geometryKey !== kiteVisualGeometryKey && defaultDimensions) {
    for (let index = 0; index < kiteMeshes.length; index += 1) {
      const previousMesh = kiteMeshes[index];
      scene.remove(previousMesh);
      const dimensions = deriveKiteVisualDimensions(frame, index) ?? defaultDimensions;
      const nextMesh = makeKiteMesh(dimensions);
      kiteMeshes[index] = nextMesh;
      scene.add(nextMesh);
    }
    kiteVisualGeometryKey = geometryKey;
  }

  while (kiteMeshes.length < count) {
    const dimensions = deriveKiteVisualDimensions(frame, kiteMeshes.length) ?? defaultDimensions;
    if (!dimensions) {
      break;
    }
    const group = makeKiteMesh(dimensions);
    kiteMeshes.push(group);
    scene.add(group);

    const rabbit = new THREE.Mesh(
      new THREE.SphereGeometry(2.4, 12, 12),
      new THREE.MeshStandardMaterial({ color: 0xffbe6b, emissive: 0xffbe6b, emissiveIntensity: 0.32 })
    );
    rabbitMeshes.push(rabbit);
    scene.add(rabbit);

    const projectedPhase = makeProjectedPhaseMesh();
    projectedPhaseMeshes.push(projectedPhase);
    scene.add(projectedPhase);

    const guidanceLine = makeSceneLine(0x66d7c5, 0.58);
    guidanceLines.push(guidanceLine);
    scene.add(guidanceLine);

    upperSegmentMeshes.push([]);
    upperNodeMeshes.push([]);
  }

  kiteMeshes.forEach((mesh, index) => {
    const visible = index < count;
    mesh.visible = visible;
    rabbitMeshes[index].visible = visible;
    projectedPhaseMeshes[index].visible = visible;
    guidanceLines[index].visible = visible;
    upperSegmentMeshes[index].forEach((segment) => {
      segment.visible = visible;
    });
    upperNodeMeshes[index].forEach((node) => {
      node.visible = visible;
    });
  });
}

function renderFrame(frame: ApiFrame): void {
  lastRenderedFrame = frame;
  ensureKites(frame.kite_positions_n.length, frame);
  payloadMesh.position.copy(toThree(frame.payload_position_n));
  const splitterPosition = toThree(frame.splitter_position_n);
  splitterMesh.position.copy(splitterPosition);
  if (shouldSnapOrbitTargetToFrame) {
    snapCameraTargetToFrame(frame);
    shouldSnapOrbitTargetToFrame = false;
  }
  applyCameraFollow(frame);
  updateControlRing(frame);
  renderTether(
    frame.common_tether,
    frame.common_tether_tensions,
    commonSegmentMeshes,
    commonNodeMeshes,
    COMMON_SEGMENT_RADIUS,
    COMMON_NODE_RADIUS
  );

  frame.kite_positions_n.forEach((position, index) => {
    const mesh = kiteMeshes[index];
    const rabbitMesh = rabbitMeshes[index];
    const projectedPhaseMesh = projectedPhaseMeshes[index];
    const guidanceLine = guidanceLines[index];
    const upperSegments = upperSegmentMeshes[index];
    const upperNodes = upperNodeMeshes[index];
    const quatData = frame.kite_quaternions_n2b[index];
    if (
      !mesh ||
      !rabbitMesh ||
      !projectedPhaseMesh ||
      !guidanceLine ||
      !upperSegments ||
      !upperNodes ||
      !quatData
    ) {
      return;
    }
    const kitePosition = toThree(position);
    mesh.position.copy(kitePosition);
    const quat = kiteQuaternionToThree(quatData);
    mesh.setRotationFromQuaternion(quat);
    const rabbitTarget = frame.rabbit_targets_n[index] ?? position;
    const rabbitPosition = toThree(rabbitTarget);
    const phaseColor = phaseErrorColor(frame.phase_error[index] ?? 0);
    const projectedPhaseColor = new THREE.Color(kiteColor(index)).lerp(phaseColor, 0.4);

    rabbitMesh.position.copy(rabbitPosition);
    setMaterialColor(rabbitMesh.material, phaseColor, 0.3);

    const controlCenter = frame.control_ring_center_n;
    const dx = position[0] - controlCenter[0];
    const dy = position[1] - controlCenter[1];
    const phaseAngle = Math.atan2(dy, dx);
    const orbitRadius = frame.orbit_radius[index] ?? Math.hypot(dx, dy);
    projectedPhaseMesh.position.copy(
      toThree([
        controlCenter[0] + orbitRadius * Math.cos(phaseAngle),
        controlCenter[1] + orbitRadius * Math.sin(phaseAngle),
        controlCenter[2]
      ])
    );
    setMaterialColor(projectedPhaseMesh.material, projectedPhaseColor, 0.28);

    updateLine(guidanceLine, kitePosition, rabbitPosition, phaseColor, 0.68);

    renderTether(
      frame.upper_tethers[index],
      frame.upper_tether_tensions[index] ?? [],
      upperSegments,
      upperNodes,
      UPPER_SEGMENT_RADIUS,
      UPPER_NODE_RADIUS
    );
  });
}

function refreshProgressSummary(): void {
  if (!activeSummaryRequest || !latestProgressState) {
    return;
  }
  const now = performance.now();
  if (now - lastSummaryRefreshWallTimeMs < SUMMARY_REFRESH_MIN_INTERVAL_MS) {
    summaryRefreshPending = true;
    return;
  }
  const html = formatProgressSummary(
    activeSummaryRequest,
    latestProgressState,
    framesReceived,
    framesRendered,
    bufferedFrameCount(),
    currentPlaybackLabel
  );
  if (html !== lastSummaryHtml) {
    summaryNode.innerHTML = html;
    lastSummaryHtml = html;
  }
  lastSummaryRefreshWallTimeMs = now;
  summaryRefreshPending = false;
}

function renderFrameBatch(frames: ApiFrame[]): void {
  if (frames.length === 0) {
    return;
  }
  framesRendered += frames.length;
  const renderFrameForBatch = frames[frames.length - 1];
  if (airflowUpdatesEnabled) {
    try {
      if (currentPlaybackRate === null && frames.length > 1) {
        advanceAirflowParticlesToFrame(renderFrameForBatch);
      } else {
        frames.forEach((frame) => {
          advanceAirflowParticlesToFrame(frame);
        });
      }
    } catch (error) {
      airflowUpdatesEnabled = false;
      ambientParticleCloud.visible = false;
      gustParticleCloud.visible = false;
      wingtipTrailCloud.visible = false;
      const message = error instanceof Error ? error.message : String(error);
      appendConsole(`airflow rendering disabled after error: ${message}`);
    }
  }
  try {
    renderFrame(renderFrameForBatch);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    appendConsole(`3D render error: ${message}`);
    refreshProgressSummary();
    return;
  }
  refreshProgressSummary();
}

function drainPendingFrames(timestamp: number): void {
  if (pendingPlaybackFrames.length === 0) {
    return;
  }

  if (currentPlaybackRate === null) {
    const frames = pendingPlaybackFrames;
    pendingPlaybackFrames = [];
    renderFrameBatch(frames);
    return;
  }

  if (playbackStartWallTimeMs === null) {
    playbackStartWallTimeMs = timestamp;
    playbackStartSimTime = pendingPlaybackFrames[0].time;
  }

  const targetSimTime =
    playbackStartSimTime + ((timestamp - playbackStartWallTimeMs) / 1000) * currentPlaybackRate;

  let releaseCount = 0;
  while (
    releaseCount < pendingPlaybackFrames.length &&
    pendingPlaybackFrames[releaseCount].time <= targetSimTime + 1e-9
  ) {
    releaseCount += 1;
  }

  if (releaseCount > 0) {
    const frames = pendingPlaybackFrames.splice(0, releaseCount);
    renderFrameBatch(frames);
  }
}

function animate(timestamp: number): void {
  requestAnimationFrame(animate);
  drainPendingFrames(timestamp);
  if (summaryRefreshPending) {
    refreshProgressSummary();
  }
  syncOrbitTargetMarker();
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);

function sectionPlotColumns(groupCount: number, maxColumns?: number): number {
  const availableColumns = maxColumns
    ? Math.min(plotColumnCount(), Math.max(1, maxColumns))
    : plotColumnCount();
  if (groupCount <= 2) {
    return groupCount;
  }
  if (groupCount === 4) {
    return Math.min(2, availableColumns);
  }
  return Math.min(groupCount, availableColumns);
}

function plotGroupHeight(): number {
  return PLOT_GROUP_HEIGHT_PX;
}

function plotGroupData(group: PlotGroupDefinition, frames: ApiFrame[]): PlotlyDatum[] {
  const frameTimes = frames.map((frame) => frame.time);
  return group.traces.map((trace) => ({
    type: "scatter",
    mode: "lines",
    name: trace.name,
    x: frameTimes,
    y: frames.map((frame) => trace.value(frame)),
    line: {
      color: trace.color,
      width: trace.width ?? 2,
      dash: trace.dash ?? "solid"
    },
    visible: plotTraceVisible(trace),
    hovertemplate: `${trace.name}<br>t=%{x:.2f}s<br>%{y:.4f}<extra></extra>`
  })
  );
}

function plotGroupLayout(group: PlotGroupDefinition): PlotlyDatum {
  return {
    autosize: true,
    height: plotGroupHeight(),
    margin: { l: 54, r: 18, t: 18, b: 48 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#081018",
    font: {
      family: "IBM Plex Sans, Helvetica Neue, sans-serif",
      color: "#d9ecfb"
    },
    showlegend: false,
    xaxis: {
      title: { text: "Time (s)", standoff: 6 },
      gridcolor: "#243b4a",
      zerolinecolor: "#243b4a",
      linecolor: "#345266",
      mirror: true,
      automargin: true
    },
    yaxis: {
      title: { text: group.yTitle, standoff: 8 },
      gridcolor: "#243b4a",
      zerolinecolor: "#243b4a",
      linecolor: "#345266",
      mirror: true,
      automargin: true
    }
  };
}

function linkedXAxisUpdate(range: [number, number] | null): PlotlyDatum {
  const update: PlotlyDatum = {};
  if (range) {
    update["xaxis.range"] = range;
    update["xaxis.autorange"] = false;
  } else {
    update["xaxis.autorange"] = true;
  }
  return update;
}

function relayoutRange(event: PlotlyRelayoutEvent): [number, number] | null {
  for (const [key, value] of Object.entries(event)) {
    const match = key.match(/^xaxis\d*\.range$/);
    if (!match || !Array.isArray(value) || value.length !== 2) {
      continue;
    }
    const start = Number(value[0]);
    const end = Number(value[1]);
    if (Number.isFinite(start) && Number.isFinite(end)) {
      return [start, end];
    }
  }

  for (const key of Object.keys(event)) {
    const match = key.match(/^(xaxis\d*)\.range\[0\]$/);
    if (!match) {
      continue;
    }
    const axis = match[1];
    const start = Number(event[`${axis}.range[0]`]);
    const end = Number(event[`${axis}.range[1]`]);
    if (Number.isFinite(start) && Number.isFinite(end)) {
      return [start, end];
    }
  }

  return null;
}

function relayoutRequestsXAxisAutorange(event: PlotlyRelayoutEvent): boolean {
  return Object.entries(event).some(([key, value]) =>
    /^xaxis\d*\.autorange$/.test(key) && value === true
  );
}

function syncPlotXAxes(event: PlotlyRelayoutEvent): void {
  if (syncingPlotXAxes || activePlotSections.length === 0) {
    return;
  }

  const range = relayoutRange(event);
  const autorange = relayoutRequestsXAxisAutorange(event);
  if (!range && !autorange) {
    return;
  }

  syncingPlotXAxes = true;
  const updates = activePlotSections.map((section) =>
    Plotly.relayout(section.plot, linkedXAxisUpdate(range))
  );
  void Promise.all(updates)
    .catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      appendConsole(`plot x-axis sync failed: ${message}`);
    })
    .finally(() => {
      syncingPlotXAxes = false;
    });
}

function registerPlotXAxisSync(plot: HTMLElement): void {
  const plotElement = plot as PlotElement;
  if (typeof plotElement.on !== "function") {
    return;
  }
  plotElement.on("plotly_relayout", syncPlotXAxes);
}

function plotSectionKey(definition: PlotSectionDefinition): string {
  return definition.title;
}

function applyPlotSectionCollapsed(
  host: HTMLElement,
  body: HTMLElement,
  toggle: HTMLButtonElement,
  plots: HTMLElement[],
  collapsed: boolean
): void {
  host.classList.toggle("collapsed", collapsed);
  body.hidden = collapsed;
  toggle.setAttribute("aria-expanded", String(!collapsed));
  toggle.textContent = collapsed ? "Show plots" : "Hide plots";
  if (!collapsed) {
    plots.forEach((plot) => {
      if (plot.childElementCount > 0) {
        Plotly.Plots?.resize(plot);
      }
    });
  }
}

function clearPlots(message: string): void {
  activePlotSections.forEach((section) => {
    Plotly.purge(section.plot);
  });
  activePlotSections = [];
  plotsNode.innerHTML = "";
  const placeholder = document.createElement("div");
  placeholder.className = "plots-placeholder";
  placeholder.textContent = message;
  plotsNode.append(placeholder);
}

async function renderFinalPlots(frames: ApiFrame[], kiteCount: number): Promise<void> {
  activePlotSections.forEach((section) => {
    Plotly.purge(section.plot);
  });
  activePlotSections = [];
  plotSignalVisibility = new Map<string, boolean>();
  collapsedPlotSections = new Set<string>();
  plotsNode.innerHTML = "";
  ensurePlotKiteVisibility(kiteCount);

  const definitions = buildPlotSections(kiteCount);
  const sectionPromises = definitions.map(async (definition) => {
    const host = document.createElement("section");
    host.className = "plot-section";

    const header = document.createElement("div");
    header.className = "plot-section-head";
    const headerCopy = document.createElement("div");
    headerCopy.className = "plot-section-copy";
    const title = document.createElement("div");
    title.className = "plot-section-title";
    title.textContent = definition.title;
    const description = document.createElement("div");
    description.className = "plot-section-note";
    description.textContent = definition.description;
    const headerActions = document.createElement("div");
    headerActions.className = "plot-section-actions";
    const controls = document.createElement("div");
    controls.className = "plot-kite-controls";
    renderPlotKiteControls(controls, kiteCount);
    const collapseButton = document.createElement("button");
    collapseButton.className = "plot-section-collapse";
    collapseButton.type = "button";

    const body = document.createElement("div");
    body.className = "plot-section-body";
    const plotGrid = document.createElement("div");
    plotGrid.className = "plot-group-grid";
    plotGrid.style.setProperty(
      "--plot-columns",
      String(sectionPlotColumns(definition.groups.length, definition.maxColumns))
    );

    headerCopy.append(title, description);
    headerActions.append(controls, collapseButton);
    header.append(headerCopy, headerActions);
    body.append(plotGrid);
    host.append(header, body);
    plotsNode.append(host);

    const sectionPlots: HTMLElement[] = [];

    const sectionKey = plotSectionKey(definition);
    applyPlotSectionCollapsed(
      host,
      body,
      collapseButton,
      sectionPlots,
      collapsedPlotSections.has(sectionKey)
    );
    collapseButton.addEventListener("click", () => {
      const nextCollapsed = !collapsedPlotSections.has(sectionKey);
      if (nextCollapsed) {
        collapsedPlotSections.add(sectionKey);
      } else {
        collapsedPlotSections.delete(sectionKey);
      }
      applyPlotSectionCollapsed(host, body, collapseButton, sectionPlots, nextCollapsed);
    });

    const plotPromises = definition.groups.map(async (group) => {
      const groupCard = document.createElement("div");
      groupCard.className = "plot-group-card";
      const groupHead = document.createElement("div");
      groupHead.className = "plot-group-head";
      const groupTitle = document.createElement("div");
      groupTitle.className = "plot-group-title";
      groupTitle.textContent = group.title;
      const signalLegend = document.createElement("div");
      signalLegend.className = "plot-signal-legend";
      renderPlotSignalLegend(signalLegend, group.traces);
      const plot = document.createElement("div");
      plot.className = "plot-canvas";
      plot.style.height = `${plotGroupHeight()}px`;

      groupHead.append(groupTitle, signalLegend);
      groupCard.append(groupHead, plot);
      plotGrid.append(groupCard);
      sectionPlots.push(plot);

      const data = plotGroupData(group, frames);
      const traceIndices = data.map((_, index) => index);
      const activeSection: ActivePlotSection = {
        host,
        plot,
        traces: group.traces,
        traceIndices
      };

      await Plotly.newPlot(plot, data, plotGroupLayout(group), {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: [
          "select2d",
          "lasso2d",
          "autoScale2d",
          "toggleSpikelines"
        ]
      });
      activePlotSections.push(activeSection);
      registerPlotXAxisSync(plot);
    });
    await Promise.all(plotPromises);
  });

  await Promise.all(sectionPromises);
  applyPlotKiteVisibility();
}

function queueFrames(frames: ApiFrame[]): void {
  if (frames.length === 0) {
    return;
  }
  framesReceived += frames.length;
  if (
    framesRendered === 0 &&
    playbackStartWallTimeMs === null &&
    pendingPlaybackFrames.length === 0
  ) {
    const [firstFrame, ...remainingFrames] = frames;
    renderFrameBatch([firstFrame]);
    playbackStartWallTimeMs = performance.now();
    playbackStartSimTime = firstFrame.time;
    pendingPlaybackFrames.push(...remainingFrames);
  } else {
    pendingPlaybackFrames.push(...frames);
  }
  refreshProgressSummary();
}

function waitForPlaybackDrain(): Promise<void> {
  if (pendingPlaybackFrames.length === 0) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    const poll = () => {
      if (pendingPlaybackFrames.length === 0) {
        resolve();
        return;
      }
      requestAnimationFrame(poll);
    };
    requestAnimationFrame(poll);
  });
}

async function loadPresets(): Promise<void> {
  const response = await fetch("/api/presets");
  const presets = (await response.json()) as PresetInfo[];
  presetSelect.innerHTML = "";
  presets.forEach((preset) => {
    const option = document.createElement("option");
    option.value = preset.preset;
    option.textContent = `${preset.name} — ${preset.description}`;
    presetSelect.append(option);
  });
  applyPresetDefaults();
  updateCameraFollowOptions(presetKiteCount(presetSelect.value as Preset));
}

async function runSimulation(): Promise<void> {
  const selectedTimeDilation = timeDilationSelect.value as TimeDilationPreset;
  const playbackLabel = timeDilationLabel(selectedTimeDilation);
  const playbackRate = timeDilationRate(selectedTimeDilation);
  const durationSeconds = Number(durationInput.value);
  const request = {
    preset: presetSelect.value,
    duration: durationSeconds,
    phase_mode: phaseModeSelect.value as PhaseMode,
    payload_mass_kg: Number(payloadInput.value),
    wind_speed_mps: Number(windInput.value),
    bridle_enabled: bridleEnabledInput.checked,
    sim_noise_enabled: simNoiseInput.checked,
    sample_stride: 1
  };
  runButton.disabled = true;
  runButton.textContent = "Running...";
  activeSummaryRequest = {
    preset: request.preset,
    phase_mode: request.phase_mode,
    sim_noise_enabled: request.sim_noise_enabled,
    bridle_enabled: request.bridle_enabled
  };
  resetPlaybackState(playbackLabel, playbackRate);
  const kiteCount = presetKiteCount(request.preset as Preset);
  showRuntimeTab("console");
  clearConsole();
  clearPlots("Plots will be generated once after the simulation finishes.");
  setFailure(null);
  summaryNode.innerHTML = renderSummaryCard(
    "Queued",
    "Waiting for first solver update",
    [
      { label: "Preset", value: request.preset },
      { label: "Phase Mode", value: request.phase_mode },
      { label: "Bridle", value: request.bridle_enabled ? "Enabled" : "CG attach" },
      { label: "Sim Noise", value: request.sim_noise_enabled ? "Dryden gusts" : "Off" },
      { label: "Time Dilation", value: playbackLabel },
      { label: "Iteration", value: "0" },
      { label: "Frames", value: "0 received / 0 rendered" },
      { label: "Buffered Frames", value: "0" },
      { label: "Accepted / Rejected", value: "0 / 0" },
      { label: "Last Interval dt", value: "0.0000 s" },
      { label: "Last Interval Steps", value: "accepted 0, rejected 0" },
      { label: "Last Interval Substeps", value: "0 / 0" }
    ],
    "Run requested"
  );
  appendConsole(
    `run requested: preset=${request.preset}, duration=${request.duration}s, phase=${request.phase_mode}, bridle=${request.bridle_enabled ? "enabled" : "cg_attach"}, noise=${request.sim_noise_enabled ? "dryden" : "off"}, time_dilation=${playbackLabel}`
  );

  try {
    const response = await fetch("/api/run_stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request)
    });
    if (!response.ok) {
      throw new Error(`server returned ${response.status}`);
    }
    if (!response.body) {
      throw new Error("stream response body missing");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let summary: RunSummary | null = null;
    let finalPlotFrames: ApiFrame[] | null = null;
    let nextProgressLogIteration = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      const sceneFramesInChunk: ApiFrame[] = [];
      for (const line of lines) {
        if (!line.trim()) {
          continue;
        }
        const event = JSON.parse(line) as StreamEvent;
        if (event.kind === "log") {
          appendConsole(event.message);
          continue;
        }
        if (event.kind === "progress") {
          latestProgressState = event.progress;
          refreshProgressSummary();
          if (
            event.progress.iteration === 0 ||
            event.progress.iteration >= nextProgressLogIteration ||
            event.progress.time >= event.progress.duration - 1e-9
          ) {
            appendConsole(
              `iter=${event.progress.iteration} t=${event.progress.time.toFixed(2)}s ` +
                `substeps=${event.progress.substeps_interval}/${event.progress.substep_budget} ` +
                `accepted=${event.progress.accepted_steps_total} rejected=${event.progress.rejected_steps_total}`
            );
            nextProgressLogIteration = event.progress.iteration + 25;
          }
          continue;
        }
        if (event.kind === "frame") {
          sceneFramesInChunk.push(event.frame);
          continue;
        }
        if (event.kind === "plots") {
          finalPlotFrames = event.frames;
          appendConsole(`final plot buffer received: ${event.frames.length} samples`);
          continue;
        }
        summary = event.summary;
        pendingSummary = event.summary;
        setFailure(event.summary.failure ?? null);
      }
      queueFrames(sceneFramesInChunk);
    }

    if (buffer.trim()) {
      const event = JSON.parse(buffer) as StreamEvent;
      if (event.kind === "log") {
        appendConsole(event.message);
      } else if (event.kind === "progress") {
        latestProgressState = event.progress;
        refreshProgressSummary();
      } else if (event.kind === "frame") {
        queueFrames([event.frame]);
      } else if (event.kind === "plots") {
        finalPlotFrames = event.frames;
        appendConsole(`final plot buffer received: ${event.frames.length} samples`);
      } else {
        summary = event.summary;
        pendingSummary = event.summary;
        setFailure(event.summary.failure ?? null);
      }
    }

    await waitForPlaybackDrain();
    if (finalPlotFrames && finalPlotFrames.length > 0) {
      appendConsole(`rendering plots from ${finalPlotFrames.length} samples`);
      try {
        showRuntimeTab("plots");
        await renderFinalPlots(finalPlotFrames, kiteCount);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        clearPlots(`Plot rendering failed: ${message}`);
        appendConsole(`plot rendering failed: ${message}`);
      }
    } else {
      clearPlots("No plot samples were returned for this run.");
      appendConsole("no final plot buffer received");
    }
    if (pendingSummary) {
      summaryNode.innerHTML = formatRunSummary(
        pendingSummary,
        framesReceived,
        framesRendered,
        currentPlaybackLabel
      );
    }
    appendConsole(`received ${framesReceived} frames, rendered ${framesRendered}`);
    if (!summary) {
      appendConsole("run ended without a summary");
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    summaryNode.innerHTML = `<div class="summary-error">Run failed: ${escapeHtml(message)}</div>`;
    setFailure(null);
    appendConsole(`error: ${message}`);
  } finally {
    runButton.disabled = false;
    runButton.textContent = "Run";
  }
}

runForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void runSimulation();
});

runtimeConsoleTab.addEventListener("click", () => {
  showRuntimeTab("console");
});

runtimePlotsTab.addEventListener("click", () => {
  showRuntimeTab("plots");
});

presetSelect.addEventListener("change", () => {
  applyPresetDefaults();
  updateCameraFollowOptions(presetKiteCount(presetSelect.value as Preset));
  shouldSnapOrbitTargetToFrame = true;
  resetCameraFollowState();
});

phaseModeSelect.addEventListener("change", () => {
  renderControllerDocs();
});

cameraFollowTargetSelect.addEventListener("change", () => {
  updateCameraFollowUiState();
  shouldSnapOrbitTargetToFrame = true;
  resetCameraFollowState();
  if (lastRenderedFrame) {
    snapCameraTargetToFrame(lastRenderedFrame);
    applyCameraFollow(lastRenderedFrame);
  }
});

cameraFollowYawInput.addEventListener("change", () => {
  resetCameraFollowState();
});

window.addEventListener("mathjax-ready", () => {
  const roots = Array.from(pendingMathRoots);
  pendingMathRoots.clear();
  roots.forEach((root) => {
    typesetMath(root);
  });
  renderControllerDocs();
});

window.addEventListener("mermaid-ready", () => {
  ensureMermaidInitialized();
  const roots = Array.from(pendingMermaidRoots);
  pendingMermaidRoots.clear();
  roots.forEach((root) => {
    renderMermaid(root);
  });
  renderControllerDocs();
});

window.addEventListener("resize", () => {
  renderer.setSize(viewport.clientWidth, viewport.clientHeight);
  camera.aspect = viewport.clientWidth / viewport.clientHeight;
  camera.updateProjectionMatrix();
  syncOrbitTargetMarker();
  activePlotSections.forEach((section) => {
    Plotly.Plots?.resize(section.plot);
  });
});

void loadPresets().then(() => {
  applyPresetDefaults();
  renderControllerDocs();
  return runSimulation();
});
