import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

type PhaseMode = "adaptive" | "open_loop";
type LongitudinalMode = "total_energy" | "max_throttle_altitude_pitch";
type Preset = "swarm" | "free_flight1" | "simple_tether";
type TimeDilationPreset = "fast" | "10" | "5" | "2" | "1" | "0.5" | "0.1";
type CameraFollowTarget = "manual" | "disk_center" | "y_joint" | `kite:${number}`;
type RuntimeTab = "console" | "plots";
type TetherTensionScaleMode = "payload" | "run_peak" | "fixed";

type PlotlyDatum = Record<string, unknown>;

interface DrydenConfig {
  seed: number;
  intensity_scale: number;
  length_scale: number;
  altitude_intensity_enabled: boolean;
  altitude_length_scale_enabled: boolean;
}

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

type ControllerTuning = Record<string, number>;

interface SimulationDefaults {
  duration: number;
  dt_control: number;
  rk_abs_tol: number;
  rk_rel_tol: number;
  max_substeps: number;
  phase_mode: PhaseMode;
  sample_stride: number;
  sim_noise_enabled: boolean;
  dryden: DrydenConfig;
  bridle_enabled: boolean;
  longitudinal_mode: LongitudinalMode;
  controller_tuning: ControllerTuning;
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
  kite_control_roll_pitch_deg: [number, number][];
  rabbit_targets_n: [number, number, number][];
  phase_projected_n: [number, number, number][];
  closest_disk_n: [number, number, number][];
  disk_plane_projection_n: [number, number, number][];
  lookahead_on_disk_n: [number, number, number][];
  phase_slot_n: [number, number, number][];
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
  rotor_speed: number[];
  alpha_deg: number[];
  beta_deg: number[];
  body_omega_b: [number, number, number][];
  orbit_radius: number[];
  rabbit_radius: number[];
  rabbit_distance: number[];
  rabbit_target_distance: number[];
  rabbit_bearing_y_deg: number[];
  rabbit_vector_b: [number, number, number][];
  curvature_y_b: number[];
  curvature_y_ref: number[];
  curvature_y_est: number[];
  omega_world_z_ref: number[];
  omega_world_z: number[];
  beta_ref_deg: number[];
  roll_ref_deg: number[];
  roll_ff_deg: number[];
  roll_p_deg: number[];
  roll_i_deg: number[];
  pitch_ref_deg: number[];
  pitch_ref_p_deg: number[];
  pitch_ref_i_deg: number[];
  curvature_z_b: number[];
  curvature_z_ref: number[];
  aileron_trim_deg: number[];
  aileron_roll_p_deg: number[];
  aileron_roll_d_deg: number[];
  rudder_trim_deg: number[];
  rudder_beta_p_deg: number[];
  rudder_rate_d_deg: number[];
  rudder_world_z_p_deg: number[];
  elevator_trim_deg: number[];
  elevator_pitch_p_deg: number[];
  elevator_pitch_d_deg: number[];
  elevator_alpha_protection_deg: number[];
  motor_torque_trim: number[];
  motor_torque_p: number[];
  motor_torque_i: number[];
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
  rudder_force_b: [number, number, number][];
  rudder_moment_b: [number, number, number][];
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
  aileron_applied_deg: number[];
  flap_cmd_deg: number[];
  flap_applied_deg: number[];
  winglet_cmd_deg: number[];
  winglet_applied_deg: number[];
  elevator_cmd_deg: number[];
  elevator_applied_deg: number[];
  rudder_cmd_deg: number[];
  rudder_applied_deg: number[];
  motor_torque: number[];
  motor_torque_applied: number[];
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
  | { kind: "error"; message: string }
  | { kind: "progress"; progress: SimulationProgress }
  | { kind: "frame"; frame: ApiFrame }
  | { kind: "plots"; frames: ApiFrame[] }
  | { kind: "summary"; summary: RunSummary };

const presetSelect = document.querySelector<HTMLSelectElement>("#preset")!;
const swarmOptionsNode = document.querySelector<HTMLElement>("#swarm-options")!;
const swarmKitesSelect = document.querySelector<HTMLSelectElement>("#swarm-kites")!;
const swarmDiskAltitudeInput = document.querySelector<HTMLInputElement>(
  "#swarm-disk-altitude"
)!;
const swarmDiskRadiusInput = document.querySelector<HTMLInputElement>(
  "#swarm-disk-radius"
)!;
const swarmAircraftAltitudeInput = document.querySelector<HTMLInputElement>(
  "#swarm-aircraft-altitude"
)!;
const swarmUpperTetherLengthInput = document.querySelector<HTMLInputElement>(
  "#swarm-upper-tether-length"
)!;
const swarmCommonTetherLengthInput = document.querySelector<HTMLInputElement>(
  "#swarm-common-tether-length"
)!;
const durationInput = document.querySelector<HTMLInputElement>("#duration")!;
const dtControlInput = document.querySelector<HTMLInputElement>("#dt-control")!;
const phaseModeSelect = document.querySelector<HTMLSelectElement>("#phase-mode")!;
const payloadInput = document.querySelector<HTMLInputElement>("#payload-mass")!;
const windInput = document.querySelector<HTMLInputElement>("#wind-speed")!;
const bridleEnabledInput = document.querySelector<HTMLInputElement>("#bridle-enabled")!;
const simNoiseInput = document.querySelector<HTMLInputElement>("#sim-noise")!;
const drydenTuningFieldsNode = document.querySelector<HTMLElement>("#dryden-tuning-fields")!;
const drydenSeedInput = document.querySelector<HTMLInputElement>("#dryden-seed")!;
const drydenIntensityScaleInput = document.querySelector<HTMLInputElement>(
  "#dryden-intensity-scale"
)!;
const drydenLengthScaleInput = document.querySelector<HTMLInputElement>("#dryden-length-scale")!;
const drydenAltitudeIntensityInput = document.querySelector<HTMLInputElement>(
  "#dryden-altitude-intensity-enabled"
)!;
const drydenAltitudeLengthInput = document.querySelector<HTMLInputElement>(
  "#dryden-altitude-length-enabled"
)!;
const windShearEnabledInput = document.querySelector<HTMLInputElement>("#wind-shear-enabled")!;
const windShearNinetyHeightInput = document.querySelector<HTMLInputElement>(
  "#wind-shear-90-height"
)!;
const maxThrottleAltitudePitchInput = document.querySelector<HTMLInputElement>(
  "#max-throttle-altitude-pitch"
)!;
const rkAbsTolInput = document.querySelector<HTMLInputElement>("#rk-abs-tol")!;
const rkRelTolInput = document.querySelector<HTMLInputElement>("#rk-rel-tol")!;
const maxSubstepsInput = document.querySelector<HTMLInputElement>("#max-substeps")!;
const controllerTuningFieldsNode =
  document.querySelector<HTMLElement>("#controller-tuning-fields")!;
const timeDilationSelect = document.querySelector<HTMLSelectElement>("#time-dilation")!;
const cameraFollowTargetSelect = document.querySelector<HTMLSelectElement>("#camera-follow-target")!;
const cameraFollowYawInput = document.querySelector<HTMLInputElement>("#camera-follow-yaw")!;
const cameraFollowYawLabel = cameraFollowYawInput.closest<HTMLLabelElement>(".checkbox-label")!;
const trackpadNavigationInput = document.querySelector<HTMLInputElement>("#trackpad-navigation")!;
const controlLabelsEnabledInput = document.querySelector<HTMLInputElement>("#control-labels-enabled")!;
const controlDiskEnabledInput = document.querySelector<HTMLInputElement>("#control-disk-enabled")!;
const controlFeaturesEnabledInput = document.querySelector<HTMLInputElement>(
  "#control-features-enabled"
)!;
const controlFeatureLinesEnabledInput = document.querySelector<HTMLInputElement>(
  "#control-feature-lines-enabled"
)!;
const controlFeaturesAtTargetAltitudeInput = document.querySelector<HTMLInputElement>(
  "#control-features-at-target-altitude"
)!;
const controlFeaturesAtAircraftAltitudeInput = document.querySelector<HTMLInputElement>(
  "#control-features-at-aircraft-altitude"
)!;
const controlFeatureScaleInput = document.querySelector<HTMLInputElement>("#control-feature-scale")!;
const tetherNodesEnabledInput = document.querySelector<HTMLInputElement>("#tether-nodes-enabled")!;
const tetherNodeScaleInput = document.querySelector<HTMLInputElement>("#tether-node-scale")!;
const tetherTensionScaleModeSelect = document.querySelector<HTMLSelectElement>(
  "#tether-tension-scale-mode"
)!;
const tetherTensionPayloadMarginInput = document.querySelector<HTMLInputElement>(
  "#tether-tension-payload-margin"
)!;
const tetherTensionFixedMinInput = document.querySelector<HTMLInputElement>(
  "#tether-tension-fixed-min"
)!;
const tetherTensionFixedMaxInput = document.querySelector<HTMLInputElement>(
  "#tether-tension-fixed-max"
)!;
const fogEnabledInput = document.querySelector<HTMLInputElement>("#fog-enabled")!;
const airParticlesEnabledInput = document.querySelector<HTMLInputElement>("#air-particles-enabled")!;
const airParticleOpacityInput = document.querySelector<HTMLInputElement>("#air-particle-opacity")!;
const wingtipTrailsEnabledInput = document.querySelector<HTMLInputElement>("#wingtip-trails-enabled")!;
const wingtipConvectionEnabledInput = document.querySelector<HTMLInputElement>(
  "#wingtip-convection-enabled"
)!;
const summaryNode = document.querySelector<HTMLElement>("#summary")!;
const failureNode = document.querySelector<HTMLElement>("#failure-pill")!;
const sceneFailureNode = document.querySelector<HTMLElement>("#scene-failure-banner")!;
const runtimeConsoleTab = document.querySelector<HTMLButtonElement>("#runtime-tab-console")!;
const runtimePlotsTab = document.querySelector<HTMLButtonElement>("#runtime-tab-plots")!;
const runtimeConsoleView = document.querySelector<HTMLElement>("#runtime-console-view")!;
const runtimePlotsView = document.querySelector<HTMLElement>("#runtime-plots-view")!;
const plotsNode = document.querySelector<HTMLElement>("#plots")!;
const layoutNode = document.querySelector<HTMLElement>(".layout")!;
const viewport = document.querySelector<HTMLElement>("#viewport")!;
const sidebarResizeHandle = document.querySelector<HTMLElement>("#sidebar-resize-handle")!;
const sceneResizeHandle = document.querySelector<HTMLElement>("#scene-resize-handle")!;
const controlLabelLayer = document.querySelector<HTMLElement>("#control-label-layer")!;
const runForm = document.querySelector<HTMLFormElement>("#run-form")!;
const runButton = document.querySelector<HTMLButtonElement>("#run-button")!;
const restartButton = document.querySelector<HTMLButtonElement>("#restart-button")!;
const consoleNode = document.querySelector<HTMLElement>("#console")!;
const controllerDocsNode = document.querySelector<HTMLElement>("#controller-docs")!;

let presetInfoById = new Map<Preset, PresetInfo>();
let simulationDefaults: SimulationDefaults | null = null;
const DEFAULT_PRESET: Preset = "swarm";

type TuningMode = "total_energy" | "max_throttle_altitude_pitch";
type GuidanceMode = "rabbit" | "curvature" | "switch";

interface ControllerTuningField {
  key: string;
  label: string;
  group: string;
  step: string;
  kind?: "number" | "select";
  options?: { label: string; value: number }[];
  unit?: string;
  help?: string;
  min?: string;
  mode?: TuningMode;
  guidanceModes?: GuidanceMode[];
}

interface ControllerTuningSection {
  title: string;
  description: string;
  groups: string[];
}

const CONTROLLER_TUNING_FIELDS: ControllerTuningField[] = [
  { key: "speed_phase_gain", label: "Phase to speed gain", group: "Phase / speed scheduling", step: "0.1", unit: "m/s/rad" },
  { key: "speed_min_mps", label: "Minimum speed target", group: "Phase / speed scheduling", step: "0.1", unit: "m/s", min: "0" },
  { key: "speed_max_mps", label: "Maximum speed target", group: "Phase / speed scheduling", step: "0.1", unit: "m/s", min: "0" },
  { key: "rabbit_speed_to_distance_s", label: "Speed to rabbit distance", group: "Rabbit distance schedule", step: "0.1", unit: "s", min: "0", help: "Used as rabbit_distance = clamp(speed_target * this factor, min distance, max distance)." },
  { key: "rabbit_min_distance_m", label: "Minimum rabbit distance", group: "Rabbit distance schedule", step: "1", unit: "m", min: "0" },
  { key: "rabbit_max_distance_m", label: "Maximum rabbit distance", group: "Rabbit distance schedule", step: "1", unit: "m", min: "0" },
  { key: "roll_feedforward_gain", label: "Roll feedforward gain", group: "Lateral outer loop", step: "0.05", guidanceModes: ["curvature", "switch"] },
  { key: "rabbit_bearing_roll_p", label: "Rabbit bearing to roll P", group: "Lateral outer loop", step: "0.05", guidanceModes: ["rabbit", "switch"], help: "Direct rabbit mode only: body-frame bearing angle to the rabbit becomes roll reference without converting to curvature." },
  { key: "rabbit_bearing_roll_i", label: "Rabbit bearing to roll I", group: "Lateral outer loop", step: "0.01", guidanceModes: ["rabbit", "switch"], help: "Integrator on direct body-frame rabbit bearing angle." },
  { key: "roll_curvature_p", label: "Curvature to roll P", group: "Lateral outer loop", step: "0.1", guidanceModes: ["curvature", "switch"] },
  { key: "roll_curvature_i", label: "Curvature to roll I", group: "Lateral outer loop", step: "0.1", guidanceModes: ["curvature", "switch"] },
  { key: "roll_curvature_integrator_limit", label: "Curvature integrator limit", group: "Lateral outer loop", step: "0.005", min: "0", guidanceModes: ["curvature", "switch"] },
  { key: "roll_ref_limit_deg", label: "Roll reference limit", group: "Lateral outer loop", step: "1", unit: "deg", min: "0" },
  { key: "tethered_roll_ref_rate_limit_degps", label: "Tethered roll ref rate limit", group: "Lateral outer loop", step: "5", unit: "deg/s", min: "0" },
  {
    key: "guidance_mode",
    label: "Lateral guidance mode",
    group: "Lateral outer loop / guidance geometry",
    step: "1",
    kind: "select",
    options: [
      { label: "Always rabbit bearing", value: 0 },
      { label: "Always curvature conversion", value: 1 },
      { label: "Switch: rabbit ahead, curvature behind", value: 2 }
    ],
    help: "Always rabbit points at the body-frame rabbit bearing directly. Always curvature converts the rabbit vector to pure-pursuit curvature. Switch uses direct rabbit while the rabbit is ahead, then converts to curvature if it falls behind."
  },
  {
    key: "guidance_min_lookahead_fraction",
    label: "Minimum forward lookahead",
    group: "Lateral outer loop / guidance geometry",
    step: "0.01",
    unit: "fraction of rabbit_distance",
    min: "0",
    guidanceModes: ["curvature", "switch"],
    help: "Curvature-conversion only. Sets the minimum forward body-X lookahead: min_x = this value * rabbit_distance."
  },
  {
    key: "guidance_lateral_lookahead_ratio_limit",
    label: "Rabbit lateral clamp ratio",
    group: "Lateral outer loop / guidance geometry",
    step: "0.05",
    unit: "|y,z| / forward lookahead",
    min: "0",
    guidanceModes: ["curvature", "switch"],
    help: "Curvature-conversion only. It clamps sideways/up rabbit offset before converting the target vector to curvature; it is not the switch threshold."
  },
  { key: "guidance_curvature_limit", label: "Rabbit curvature clamp", group: "Lateral outer loop / guidance geometry", step: "0.005", unit: "1/m", min: "0", guidanceModes: ["curvature", "switch"], help: "Curvature-conversion only. Clamp applied to rabbit-derived curvature before it becomes roll-reference correction." },
  { key: "tethered_aileron_roll_p", label: "Aileron roll P", group: "Roll inner loop", step: "0.05", help: "Shared by tethered and free-flight presets: aileron response to roll angle error." },
  { key: "tethered_aileron_roll_d", label: "Aileron body-p D", group: "Roll inner loop", step: "0.01", help: "Shared roll-rate damping using body X angular velocity p." },
  { key: "tethered_rudder_beta_p", label: "Rudder sideslip P", group: "Sideslip / yaw damper", step: "0.05", help: "Shared by tethered and free-flight presets: rudder response to sideslip beta." },
  { key: "tethered_rudder_rate_d", label: "Rudder body-r D", group: "Sideslip / yaw damper", step: "0.01", help: "Shared yaw damping using body Z angular velocity r." },
  { key: "tethered_rudder_world_z_p", label: "Rudder world-Z rate P", group: "Sideslip / yaw damper", step: "0.05", help: "Optional correction on world-Z turn-rate tracking. Leave at 0 for a pure beta/body-r damper." },
  { key: "tethered_rudder_trim_offset_deg", label: "Rudder trim offset", group: "Sideslip / yaw damper", step: "0.5", unit: "deg" },
  { key: "tecs_altitude_error_limit_m", label: "Altitude error clamp", group: "Longitudinal shared", step: "1", unit: "m", min: "0" },
  { key: "free_pitch_ref_limit_deg", label: "Free-flight pitch ref limit", group: "Longitudinal shared", step: "1", unit: "deg", min: "0" },
  { key: "tethered_pitch_ref_limit_deg", label: "Tethered pitch ref limit", group: "Longitudinal shared", step: "1", unit: "deg", min: "0" },
  { key: "elevator_pitch_p", label: "Elevator pitch P", group: "Pitch inner loop", step: "0.05" },
  { key: "elevator_pitch_d", label: "Elevator pitch-rate D", group: "Pitch inner loop", step: "0.01" },
  { key: "altitude_pitch_p", label: "Altitude-to-pitch P", group: "Max-throttle altitude mode", step: "0.001", mode: "max_throttle_altitude_pitch" },
  { key: "altitude_pitch_i", label: "Altitude-to-pitch I", group: "Max-throttle altitude mode", step: "0.0001", mode: "max_throttle_altitude_pitch" },
  { key: "tecs_thrust_kinetic_p", label: "Kinetic energy to thrust P", group: "TECS mode", step: "0.01", mode: "total_energy" },
  { key: "tecs_thrust_kinetic_i", label: "Kinetic energy to thrust I", group: "TECS mode", step: "0.005", mode: "total_energy" },
  { key: "tecs_thrust_integrator_limit_nm", label: "Thrust integrator limit", group: "TECS mode", step: "0.5", unit: "N m", min: "0", mode: "total_energy" },
  { key: "tethered_thrust_positive_potential_blend", label: "Tethered potential-to-thrust blend", group: "TECS mode", step: "0.005", mode: "total_energy" },
  { key: "tethered_tecs_potential_error_limit", label: "Tethered PE error clamp", group: "TECS mode", step: "5", unit: "m²/s²", min: "0", mode: "total_energy" },
  { key: "tethered_tecs_potential_balance_weight", label: "Tethered PE balance weight", group: "TECS mode", step: "0.05", mode: "total_energy" },
  { key: "tethered_tecs_kinetic_deficit_balance_weight", label: "Tethered KE deficit balance weight", group: "TECS mode", step: "0.05", mode: "total_energy" },
  { key: "tethered_tecs_kinetic_surplus_balance_weight", label: "Tethered KE surplus balance weight", group: "TECS mode", step: "0.05", mode: "total_energy" },
  { key: "tecs_pitch_balance_p", label: "Energy-balance to pitch P", group: "TECS mode", step: "0.0001", mode: "total_energy" },
  { key: "tecs_pitch_balance_i", label: "Energy-balance to pitch I", group: "TECS mode", step: "0.00005", mode: "total_energy" },
  { key: "tecs_pitch_integrator_limit_deg", label: "Pitch integrator limit", group: "TECS mode", step: "1", unit: "deg", min: "0", mode: "total_energy" },
  { key: "alpha_protection_min_deg", label: "Alpha protection min", group: "Alpha protection", step: "1", unit: "deg" },
  { key: "alpha_protection_max_deg", label: "Alpha protection max", group: "Alpha protection", step: "1", unit: "deg" },
  { key: "alpha_to_elevator", label: "Alpha protection to elevator", group: "Alpha protection", step: "0.5" },
  { key: "surface_limit_aileron_deg", label: "Aileron limit", group: "Actuator limits", step: "1", unit: "deg", min: "0" },
  { key: "surface_limit_rudder_deg", label: "Rudder limit", group: "Actuator limits", step: "1", unit: "deg", min: "0" },
  { key: "surface_limit_elevator_deg", label: "Elevator limit", group: "Actuator limits", step: "1", unit: "deg", min: "0" },
  { key: "motor_torque_max_nm", label: "Motor torque max", group: "Actuator limits", step: "1", unit: "N m", min: "0" },
  { key: "actuator_surface_tau_s", label: "Surface lag time constant", group: "Actuator dynamics", step: "0.005", unit: "s", min: "0" },
  { key: "actuator_motor_tau_s", label: "Motor lag time constant", group: "Actuator dynamics", step: "0.005", unit: "s", min: "0" },
  { key: "rotor_speed_soft_limit_radps", label: "Rotor soft speed limit", group: "Actuator limits", step: "10", unit: "rad/s", min: "0" },
  { key: "rotor_speed_hard_limit_radps", label: "Rotor hard speed limit", group: "Actuator limits", step: "10", unit: "rad/s", min: "0" }
];

const CONTROLLER_TUNING_SECTIONS: ControllerTuningSection[] = [
  {
    title: "Formation Scheduling",
    description: "Phase error and target-speed scheduling before the individual kite controller.",
    groups: ["Phase / speed scheduling", "Rabbit distance schedule"]
  },
  {
    title: "Lateral Outer Loop",
    description: "Rabbit geometry, guidance mode, lookahead, and the commanded roll reference.",
    groups: ["Lateral outer loop", "Lateral outer loop / guidance geometry"]
  },
  {
    title: "Lateral Inner Loops",
    description: "Direct roll and sideslip/yaw-damper gains that command aileron and rudder.",
    groups: ["Roll inner loop", "Sideslip / yaw damper"]
  },
  {
    title: "Longitudinal",
    description: "Pitch, altitude, airspeed, and TECS gains. Mode-specific fields hide when inactive.",
    groups: ["Longitudinal shared", "Pitch inner loop", "Max-throttle altitude mode", "TECS mode"]
  },
  {
    title: "Protection & Actuators",
    description: "Alpha protection, actuator authority limits, rotor limits, and first-order actuator lag.",
    groups: ["Alpha protection", "Actuator limits", "Actuator dynamics"]
  }
];

const CONTROLLER_TUNING_GROUP_TO_SECTION = new Map<string, ControllerTuningSection>(
  CONTROLLER_TUNING_SECTIONS.flatMap((section) => section.groups.map((group) => [group, section] as const))
);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(viewport.clientWidth, viewport.clientHeight);
viewport.append(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color("#071019");
const sceneFog = new THREE.Fog("#071019", 240, 1800);
scene.fog = null;
const camera = new THREE.PerspectiveCamera(48, viewport.clientWidth / viewport.clientHeight, 0.1, 5000);
camera.up.set(0, 0, 1);
camera.position.set(240, -280, 290);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 220);
controls.screenSpacePanning = false;
controls.minPolarAngle = 0.05;
controls.maxPolarAngle = Math.PI - 0.05;

function applyPointerNavigationMode(): void {
  const trackpadMode = trackpadNavigationInput.checked;
  controls.mouseButtons = {
    LEFT: trackpadMode ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN,
    MIDDLE: null,
    RIGHT: trackpadMode ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE
  };
  controls.update();
}
applyPointerNavigationMode();

const WORLD_Z_AXIS = new THREE.Vector3(0, 0, 1);
const middlePanStart = new THREE.Vector2();
const middlePanEnd = new THREE.Vector2();
const middlePanVertical = new THREE.Vector3();
let middleZPanPointerId: number | null = null;
const SCENE_RESIZE_MIN_HEIGHT = 360;
const SCENE_RESIZE_MAX_HEIGHT = 1400;
const SIDEBAR_WIDTH_STORAGE_KEY = "multikite.sidebarWidthPx";
const SIDEBAR_RESIZE_MIN_WIDTH = 280;
const SIDEBAR_RESIZE_MAX_WIDTH = 760;
const RIGHT_WORKBENCH_MIN_WIDTH = 520;
let sidebarResizePointerId: number | null = null;
let sidebarResizeStartX = 0;
let sidebarResizeStartWidth = 0;
let sceneResizePointerId: number | null = null;
let sceneResizeStartY = 0;
let sceneResizeStartHeight = 0;

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
const AIR_PARTICLE_DISK_CLEARANCE_M = 100;

const payloadMesh = new THREE.Mesh(
  new THREE.SphereGeometry(7, 24, 24),
  new THREE.MeshStandardMaterial({ color: 0xff7b72, roughness: 0.35, metalness: 0.08 })
);
scene.add(payloadMesh);

const splitterMesh = new THREE.Mesh(
  new THREE.SphereGeometry(4, 16, 16),
  new THREE.MeshStandardMaterial({ color: 0x45d7a7, roughness: 0.24, metalness: 0.15 })
);

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
orbitTargetMarker.add(orbitTargetCore);
orbitTargetMarker.visible = false;
scene.add(orbitTargetMarker);

const ORBIT_TARGET_CORE_RADIUS_WORLD = 1.0;
const ORBIT_TARGET_CORE_PIXELS = 14;

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

const aircraftControlRingLine = new THREE.LineLoop(
  new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({
    color: 0x36d5c1,
    transparent: true,
    opacity: 0.26
  })
);
aircraftControlRingLine.visible = false;
scene.add(aircraftControlRingLine);

type ControlFeatureAltitudeLayer = "target" | "aircraft";

const CONTROL_VIS_COLORS = {
  lookahead: "#ff5c74",
  lookaheadOnDisk: "#ff7a66",
  projectedPhase: "#66b8ff",
  closestDisk: "#d6c3ff",
  phaseSlot: "#ffd36b",
  disk: "#36d5c1"
} as const;
const controlLookaheadColor = new THREE.Color(CONTROL_VIS_COLORS.lookahead);
const controlLookaheadOnDiskColor = new THREE.Color(CONTROL_VIS_COLORS.lookaheadOnDisk);
const controlProjectedPhaseColor = new THREE.Color(CONTROL_VIS_COLORS.projectedPhase);
const controlClosestDiskColor = new THREE.Color(CONTROL_VIS_COLORS.closestDisk);
const controlPhaseSlotColor = new THREE.Color(CONTROL_VIS_COLORS.phaseSlot);
const controlRabbitRelationshipLineColor = controlLookaheadOnDiskColor;

const kiteMeshes: THREE.Group[] = [];
const rabbitMeshes: THREE.Mesh[] = [];
const lookaheadOnDiskMeshes: THREE.Mesh[] = [];
const projectedPhaseMeshes: THREE.Mesh[] = [];
const guidanceLines: THREE.Line[] = [];
const lookaheadRadialOffsetLines: THREE.Line[] = [];
const projectedToDiskLines: THREE.Line[] = [];
const phaseSlotToClosestDiskLines: THREE.Line[] = [];
const phaseSlotMeshes: THREE.Mesh[] = [];
const aircraftRabbitMeshes: THREE.Mesh[] = [];
const aircraftLookaheadOnDiskMeshes: THREE.Mesh[] = [];
const aircraftProjectedPhaseMeshes: THREE.Mesh[] = [];
const aircraftGuidanceLines: THREE.Line[] = [];
const aircraftLookaheadRadialOffsetLines: THREE.Line[] = [];
const aircraftProjectedToDiskLines: THREE.Line[] = [];
const aircraftPhaseSlotToClosestDiskLines: THREE.Line[] = [];
const aircraftPhaseSlotMeshes: THREE.Mesh[] = [];

interface ControlFeatureLayerMeshes {
  mode: ControlFeatureAltitudeLayer;
  controlRingLine: THREE.LineLoop;
  rabbitMeshes: THREE.Mesh[];
  lookaheadOnDiskMeshes: THREE.Mesh[];
  projectedPhaseMeshes: THREE.Mesh[];
  guidanceLines: THREE.Line[];
  lookaheadRadialOffsetLines: THREE.Line[];
  projectedToDiskLines: THREE.Line[];
  phaseSlotToClosestDiskLines: THREE.Line[];
  phaseSlotMeshes: THREE.Mesh[];
}

const targetControlFeatureLayer: ControlFeatureLayerMeshes = {
  mode: "target",
  controlRingLine,
  rabbitMeshes,
  lookaheadOnDiskMeshes,
  projectedPhaseMeshes,
  guidanceLines,
  lookaheadRadialOffsetLines,
  projectedToDiskLines,
  phaseSlotToClosestDiskLines,
  phaseSlotMeshes
};

const aircraftControlFeatureLayer: ControlFeatureLayerMeshes = {
  mode: "aircraft",
  controlRingLine: aircraftControlRingLine,
  rabbitMeshes: aircraftRabbitMeshes,
  lookaheadOnDiskMeshes: aircraftLookaheadOnDiskMeshes,
  projectedPhaseMeshes: aircraftProjectedPhaseMeshes,
  guidanceLines: aircraftGuidanceLines,
  lookaheadRadialOffsetLines: aircraftLookaheadRadialOffsetLines,
  projectedToDiskLines: aircraftProjectedToDiskLines,
  phaseSlotToClosestDiskLines: aircraftPhaseSlotToClosestDiskLines,
  phaseSlotMeshes: aircraftPhaseSlotMeshes
};

const controlFeatureLayers: ControlFeatureLayerMeshes[] = [
  targetControlFeatureLayer,
  aircraftControlFeatureLayer
];
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

interface AirParticleState {
  position: THREE.Vector3;
  age: number;
  life: number;
  drift: number;
}

interface WingtipTrailParticleState {
  position: THREE.Vector3;
  velocity: THREE.Vector3;
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
    velocity: new THREE.Vector3(),
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
  makeSoftParticleMaterial(2.6, 0.45)
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
  makeSoftParticleMaterial(3.15, 0.63)
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
    uPointSize: { value: 3.5 }
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
const tetherNodeColor = new THREE.Color("#c8b894");
const tensionColorSlack = new THREE.Color("#46515e");
const tensionColorLow = new THREE.Color("#2f7dff");
const tensionColorMid = new THREE.Color("#f0d98a");
const tensionColorHigh = new THREE.Color("#ff2f2f");
const TETHER_TENSION_FALLBACK_MIN_N = 0;
const TETHER_TENSION_FALLBACK_MAX_N = 3000;
const TETHER_TENSION_PEAK_PADDING_FRACTION = 0.08;
const TETHER_SLACK_TENSION_N = 0.5;
const TETHER_SLACK_RANGE_FRACTION = 0.005;
const CONTROL_RING_SEGMENTS = 96;
const COMMON_SEGMENT_RADIUS = 0.46 / 3;
const UPPER_SEGMENT_RADIUS = 0.34 / 3;
const COMMON_NODE_RADIUS = 1.0 / 3;
const UPPER_NODE_RADIUS = 0.78 / 3;
const ambientAirColor = new THREE.Color("#6ee7ff");
const ambientAirColorHigh = new THREE.Color("#b8f7ff");
const gustAirColorLow = new THREE.Color("#55c5ff");
const gustAirColorMid = new THREE.Color("#ffbf72");
const gustAirColorHigh = new THREE.Color("#ff5b78");
const WINGTIP_LEFT_COLORS = ["#b83a42", "#c6533f", "#a63d55", "#c46a4c"];
const WINGTIP_RIGHT_COLORS = ["#3fa35b", "#5c9f45", "#2f9871", "#6aa85a"];
const SUMMARY_REFRESH_MIN_INTERVAL_MS = 125;
let consoleLines: string[] = [];
let framesReceived = 0;
let observedTetherTensionMin = Number.POSITIVE_INFINITY;
let observedTetherTensionMax = Number.NEGATIVE_INFINITY;
let framesRendered = 0;
let pendingPlaybackFrames: ApiFrame[] = [];
let pendingSummary: RunSummary | null = null;
let latestProgressState: SimulationProgress | null = null;
let activeSummaryRequest: {
  preset: string;
  swarm_kites: number;
  swarm_disk_altitude_m: number | null;
  swarm_disk_radius_m: number | null;
  swarm_aircraft_altitude_m: number | null;
  swarm_upper_tether_length_m: number | null;
  swarm_common_tether_length_m: number | null;
  phase_mode: PhaseMode;
  longitudinal_mode: LongitudinalMode;
  sim_noise_enabled: boolean;
  dryden?: DrydenConfig;
  bridle_enabled: boolean;
  dt_control: number;
  rk_abs_tol: number;
  rk_rel_tol: number;
  max_substeps: number;
  controller_tuning: ControllerTuning;
} | null = null;
let runInProgress = false;
let runStreamComplete = false;
let playbackPaused = false;
let playbackReleased = false;
let lastFailureConsoleKey: string | null = null;
let streamAbortController: AbortController | null = null;
let activeRunSequence = 0;
let controllerTuningChangedDuringRun = false;
let currentPlaybackRate: number | null = null;
let currentPlaybackLabel = "1x";
let playbackStartWallTimeMs: number | null = null;
let playbackStartSimTime = 0;
let shouldSnapOrbitTargetToFrame = true;
let lastRenderedFrame: ApiFrame | null = null;
let hasRenderedSimulationFrame = false;
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
const controlLabelNodes = new Map<string, HTMLDivElement>();

type PlotDash = "solid" | "dash" | "dot" | "dashdot" | "longdash";

interface PlotTraceDefinition {
  name: string;
  color: string;
  signalKey?: string;
  legendName?: string;
  kiteIndex?: number;
  alwaysVisible?: boolean;
  defaultVisible?: boolean;
  dash?: PlotDash;
  width?: number;
  shape?: "linear" | "hv";
  hoverTemplate?: string;
  hoverText?: (frame: ApiFrame) => string | null;
  value: (frame: ApiFrame) => number | null;
}

interface PlotGroupDefinition {
  title: string;
  yTitle: string;
  traces: PlotTraceDefinition[];
  height?: number;
  yTickVals?: number[];
  yTickText?: string[];
  yRange?: [number, number];
  showSignalLegend?: boolean;
}

interface PlotSectionDefinition {
  title: string;
  description: string;
  groups: PlotGroupDefinition[];
  maxColumns?: number;
  showKiteControls?: boolean;
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
const LIMITER_TIMELINE_HEIGHT_PX = 280;
const GRAVITY_MPS2 = 9.80665;
const RAD_TO_DEG = 180 / Math.PI;
let activePlotSections: ActivePlotSection[] = [];
let plotKiteVisibility: boolean[] = [];
let plotSignalVisibility = new Map<string, boolean>();
let collapsedPlotSections = new Set<string>();
let activePlotTabKey: string | null = null;
let syncingPlotXAxes = false;

interface KiteBreakdownTraceDefinition {
  name: string;
  value: (frame: ApiFrame, kiteIndex: number) => number;
  dash?: PlotDash;
  width?: number;
  alpha?: number;
  defaultVisible?: boolean;
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

interface DocsSymbolRow {
  symbol: string;
  source: string | string[];
  meaning: string;
}

function docsInlineMath(latex: string): string {
  return `<span class="docs-inline-math">\\(${latex}\\)</span>`;
}

function docsSourceList(source: string | string[]): string {
  const sources = Array.isArray(source) ? source : [source];
  return sources.map((item) => `<code>${escapeHtml(item)}</code>`).join(", ");
}

function docsSymbolLegend(rows: DocsSymbolRow[], title = "Symbol legend"): string {
  return `
    <div class="docs-gain-table docs-symbol-legend">
      <div class="docs-gain-row docs-gain-head">
        <div>${escapeHtml(title)}</div>
        <div>Implementation source</div>
        <div>Meaning</div>
      </div>
      ${rows
        .map(
          (row) => `
            <div class="docs-gain-row">
              <div>${docsInlineMath(row.symbol)}</div>
              <div>${docsSourceList(row.source)}</div>
              <div>${escapeHtml(row.meaning)}</div>
            </div>`
        )
        .join("")}
    </div>`;
}

function docsLegendStack(...legends: string[]): string {
  return `<div class="docs-legend-stack">${legends.join("")}</div>`;
}

function docsEquation(caption: string, latex: string, legendHtml: string, noteHtml = ""): string {
  return `
    <div class="docs-equation-row">
      <div class="docs-equation">
        <div class="docs-equation-caption">${escapeHtml(caption)}</div>
        ${latex}
        ${noteHtml}
      </div>
      <div class="docs-equation-legend">
        ${legendHtml}
      </div>
    </div>`;
}

function controllerDocsHtml(phaseMode: PhaseMode, longitudinalMode: LongitudinalMode): string {
  const modeLabel = phaseMode === "adaptive" ? "Adaptive" : "Open Loop";
  const longitudinalLabel = longitudinalModeLabel(longitudinalMode);
  const usesMaxThrottle = longitudinalMode === "max_throttle_altitude_pitch";
  const verticalBlock = usesMaxThrottle
    ? "Altitude PI<br/>pitch reference θ_i^*"
    : "TECS<br/>specific energy errors";
  const torqueBlock = usesMaxThrottle ? "Max motor torque<br/>τ_i=τ_max" : "Motor torque<br/>τ_i";
  const pitchBlock = usesMaxThrottle
    ? "Pitch reference<br/>from altitude error"
    : "Pitch reference<br/>θ_i^*";
  const longitudinalParagraph = usesMaxThrottle
    ? "<p><strong>The vertical path is in max-throttle experiment mode.</strong> Motor torque is pinned at the configured limit, and pitch is driven by a saturated altitude-error PI loop. Energy traces remain useful diagnostics, but they are not commanding throttle in this mode.</p>"
    : "<p><strong>The vertical path is TECS-style.</strong> Desired airspeed and altitude become kinetic and potential energy references. Motor torque closes kinetic-energy error, and pitch trades potential against kinetic energy.</p>";
  const propulsionParagraph = usesMaxThrottle
    ? "<p>The propulsion command is intentionally open-loop at maximum torque for lateral-loop tuning.</p>"
    : "<p>The propulsion loop regulates <strong>airspeed-derived specific kinetic energy</strong>.</p>";
  const longitudinalStructure = usesMaxThrottle
    ? "Altitude feeds a pitch PI loop; motor torque is held at the configured maximum."
    : "Selectable lateral guidance feeds desired roll; airspeed and altitude feed TECS, which commands motor torque and desired pitch.";
  const notationEquation = String.raw`
          \[
          \operatorname{wrap}(\theta)\in[-\pi,\pi],
          \qquad
          q_i^b = R_{n\to b} q_i^n,
          \qquad
          \tilde\kappa_{\bullet,i}=\kappa_{\bullet,i}-\kappa^\star_{\bullet,i}.
          \]`;
  const signalPathEquation = String.raw`
          \[
          \begin{aligned}
          p_i^r
          &\rightarrow q_i^b
          \rightarrow \theta_{r,i}\ \text{or}\ \left(\kappa_{y,i}^{\star}, \kappa_{z,i}^{\star}\right),\\[0.35em]
          \theta_{r,i}\ \text{or}\ \left(\kappa_{y,i}^{\star}, \hat\kappa_{y,i}\right)
          &\rightarrow \phi_i^{\star},\\[0.35em]
          \left(\phi_i^{\star}, \phi_i, \omega_{x,i}\right)
          &\rightarrow \delta_{a,i},\\[0.35em]
          \left(\beta_i, \omega_{z,i}\right)
          &\rightarrow \delta_{r,i},\\[0.35em]
          \left(V_i^\star,h_i^\star,V_i,h_i\right)
          &\rightarrow \left(\tau_i, \theta_i^\star\right),\\[0.35em]
          \left(\theta_i^\star,\theta_i,\omega_{y,i}\right)
          &\rightarrow \delta_{e,i}.
          \end{aligned}
          \]`;
  const phaseErrorEquation = String.raw`
          \[
          \begin{aligned}
          \text{adaptive mode:}\qquad
          \varepsilon_i &= \operatorname{wrap}\!\left(\phi_i - \frac{2\pi i}{N_K}\right),\\
          \bar\varepsilon &= \operatorname{circmean}(\{\varepsilon_j\}),\\
          e_i &= \operatorname{wrap}(\varepsilon_i - \bar\varepsilon),\\[0.35em]
          \text{open-loop mode:}\qquad
          \omega_{\mathrm{ref}} &= \frac{v_{\mathrm{ref}}}{r_d},\\
          \phi_i^{\mathrm{ref}}(t) &= \phi_{i,0} + \omega_{\mathrm{ref}} t,\\
          e_i &= \operatorname{wrap}\!\left(\phi_i - \phi_i^{\mathrm{ref}}(t)\right).
          \end{aligned}
          \]`;
  const rabbitScheduleEquation = String.raw`
          \[
          \begin{aligned}
          v_i^{\star} &= v_0 - k_{v\phi} e_i,\\
          d_{r,i} &= \operatorname{sat}_{[d_{r,\min},d_{r,\max}]}
            \left(k_{dv}V_i^\star\right),\\
          \psi_i &= \phi_i + \frac{d_{r,i}}{r_d},\\
          r_i^{r} &= r_d\left(1 + k_{\phi r}\frac{e_i}{\pi}\right).
          \end{aligned}
          \]`;
  const rabbitGeometryEquation = String.raw`
          \[
          \begin{aligned}
          p_i^{r} &=
          \begin{bmatrix}
            c_x + r_i^{r}\cos\psi_i \\
            c_y + r_i^{r}\sin\psi_i \\
            c_z
          \end{bmatrix}.
          \end{aligned}
          \]`;
  const lateralGuidanceEquation = String.raw`
          \[
          \begin{aligned}
          \bar p_i^r &=
          \begin{bmatrix}
            p_{i,x}^{r} \\
            p_{i,y}^{r} \\
            p_{i,z}^{\mathrm{cad}}
          \end{bmatrix},\\
          q_i^n &= \bar p_i^r - p_i^{\mathrm{cad}},\\
          q_i^\chi &= R_z(\chi_i)^\top q_i^n,\qquad \chi_i = \operatorname{yaw}(q_{n\to b,i}),\\
          \theta_{r,i} &= \operatorname{atan2}(q_{i,y}^\chi,q_{i,x}^\chi),\\
          \phi_{i,\mathrm{rabbit}}^\star &= k_{\phi\theta,p}\theta_{r,i}
            + k_{\phi\theta,i}\int\theta_{r,i}\,dt,\\
          x_i &= \max\!\left(q_{i,x}^\chi, f_x d_{r,i}, 1\right),\\
          \tilde q_{i,y}^\chi &= \operatorname{sat}_{\rho x_i}\!\left(q_{i,y}^\chi\right),
          \qquad
          \tilde q_{i,z}^\chi = 0,\\
          \kappa_{y,i}^{R} &= \operatorname{sat}_{\kappa_{\max}}\!\left(\frac{2\tilde q_{i,y}^\chi}{x_i^2}\right),
          \qquad
          \kappa_{z,i}^{R} = 0,\\
          (\phi_i^\star,\kappa_{y,i}^{\star},\kappa_{z,i}^{\star}) &=
          \begin{cases}
          (\phi_{i,\mathrm{rabbit}}^\star,0,0), & \text{rabbit mode},\\
          (\mathrm{curv2roll}(\kappa_{y,i}^{R}),\kappa_{y,i}^{R},\kappa_{z,i}^{R}), & \text{curvature mode},\\
          (\phi_{i,\mathrm{rabbit}}^\star,0,0), & \text{switch mode and } q_{i,x}^\chi \ge \max(f_xd_{r,i},1),\\
          (\mathrm{curv2roll}(\kappa_{y,i}^{R}),\kappa_{y,i}^{R},\kappa_{z,i}^{R}), & \text{switch mode otherwise}.
          \end{cases}
          \end{aligned}
          \]`;
  const lateralGuidanceNote = `<div class="docs-card-note">Here \\(k_{dv}\\) is the speed-to-distance factor, \\(f_x\\) is the minimum-forward-lookahead fraction, \\(\\rho\\) is the lateral clamp ratio, and \\(\\kappa_{\\max}\\) is the curvature clamp. The direct rabbit path does not convert the rabbit vector to curvature.</div>`;
  const curvatureTrackingEquation = String.raw`
          \[
          \begin{aligned}
          \hat\kappa_{y,i} &= \frac{\omega_{n,z,i}}{\lVert v_i^{\mathrm{cad}} \rVert},\\
          I_{\kappa\phi,i}^{+} &= I_{\kappa\phi,i} + \Delta t\left(\kappa_{y,i}^{\star} - \hat\kappa_{y,i}\right),\\
          \phi_i^{\star} &= k_{\phi\kappa,p}\left(\kappa_{y,i}^{\star} - \hat\kappa_{y,i}\right) + k_{\phi\kappa,i} I_{\kappa\phi,i}.
          \end{aligned}
          \]`;
  const energyStateEquation = String.raw`
          \[
          \begin{aligned}
          h_i &= -z_i-z_g,\\
          h_i^\star &= -c_z-z_g+k_{\dot z h}v_{i,z}^{\mathrm{cad}},\\
          e_{h,i} &= \operatorname{sat}(h_i^\star-h_i),\\
          E_{k,i} &= \tfrac12 V_i^2,\qquad E_{k,i}^\star=\tfrac12(V_i^\star)^2,\\
          E_{p,i} &= g h_i,\qquad E_{p,i}^\star=g(h_i+e_{h,i}).
          \end{aligned}
          \]`;
  const actuatorEquation = String.raw`
          \[
          \begin{aligned}
          e_{k,i} &= E_{k,i}^{\star}-E_{k,i},\\
          e_{p,i} &= E_{p,i}^{\star}-E_{p,i},\\
          e_{b,i} &= e_{p,i}-e_{k,i},\\
          I_{\tau,i}^{+} &= \operatorname{aw}\!\left(I_{\tau,i}+k_{\tau,i}e_{k,i}\Delta t\right),\\
          I_{\theta,i}^{+} &= \operatorname{aw}\!\left(I_{\theta,i}-k_{\theta,i}e_{b,i}\Delta t\right),\\
          \tau_i &= \tau_0 + k_{\tau,p}e_{k,i}+I_{\tau,i},\\
          \theta_i^\star &= -k_{\theta,p}e_{b,i}+I_{\theta,i},\\[0.35em]
          \delta_{a,i} &= \delta_{a,0} + k_{a,\phi}\left(\phi_i^{\star} - \phi_i\right) - k_{a,p}\,\omega_{x,i},\\
          \delta_{f,i} &= \delta_{f,0},\\
          \delta_{w,i} &= \delta_{w,0},\\
          \delta_{e,i} &= \delta_{e,0} - k_{e,\theta}\left(\theta_i^\star-\theta_i\right) + k_{e,q}\,\omega_{y,i} + k_{e,\alpha}\alpha_i^{\mathrm{prot}},\\
          \delta_{r,i} &= \delta_{r,0} - k_{r,\beta}\,\beta_i - k_{r,r}\,\omega_{z,i}.
          \end{aligned}
          \]`;
  const phaseDiagram = String.raw`flowchart LR
    A["Measured phase<br/>φ_i"] --> B["Phase coordination<br/>phase error e_i"]
    M["Phase mode<br/>adaptive or open-loop"] --> B
    B --> C["Radius scheduler<br/>r_i^r"]
    B --> D["Airspeed scheduler<br/>V_i^*"]
    D --> R["Rabbit distance<br/>d_r = clamp(k_dv V_i^*)"]
    C --> E["Lateral rabbit geometry<br/>p_i^r in disk plane"]
    R --> E
    E --> L["Lateral guidance<br/>bearing or curvature"]
    L --> J["Roll and sideslip loops<br/>δ_a, δ_r"]
    H["Disk-height altitude reference<br/>h_i^*"] --> F
    D --> F["${verticalBlock}"]
    F --> G["${torqueBlock}"]
    F --> I["${pitchBlock}"]
    classDef block fill:#112231,stroke:#3ecf9b,color:#edf6ff;`;
  const innerLoopDiagram = String.raw`flowchart LR
    A["Rabbit target<br/>p_i^r"] --> B["Guidance selector<br/>direct bearing or curvature"]
    S["Measured state<br/>position, curvature, body rates"] --> B
    B --> C["Roll-reference loop<br/>θ_r or κ_y^* - κ̂_y → φ^*"]
    C --> D["Roll inner loop<br/>φ^* - φ, p → δ_a"]
    T["${pitchBlock}"] --> E["Pitch inner loop<br/>θ^* - θ, q → δ_e"]
    B --> F["Rudder coordination loop<br/>β and r damping"]
    P["AOA backoff"] --> E
    D --> G["Aileron<br/>δ_a"]
    F --> H["Rudder<br/>δ_r"]
    E --> I["Elevator<br/>δ_e"]
    J["Flap<br/>trim only δ_f=δ_f0"]
    W["Winglet"] --> K["Trim only<br/>δ_w = δ_{w,0}"]
    classDef block fill:#112231,stroke:#66b8ff,color:#edf6ff;`;
  const notationLegend = docsSymbolLegend([
    {
      symbol: String.raw`\operatorname{wrap}(\cdot)`,
      source: "wrap_angle",
      meaning: "Angle wrapping into the [-pi, pi] interval."
    },
    {
      symbol: String.raw`R_{n\to b}`,
      source: "rotate_nav_to_body",
      meaning: "Rotation from navigation frame into the kite body frame."
    },
    {
      symbol: String.raw`i,\;N_K`,
      source: "const NK",
      meaning: "Kite index and number of kites in the const-generic specialization."
    }
  ]);
  const phaseLegend = docsSymbolLegend([
    {
      symbol: String.raw`\phi_i,\;\varepsilon_i,\;\bar\varepsilon,\;e_i`,
      source: ["phase_angle", "pairwise_phase_errors"],
      meaning: "Measured phase, slot-relative phase, circular mean, and scalar phase error."
    },
    {
      symbol: String.raw`r_d,\;c`,
      source: ["controller.disk_radius", "controller.disk_center_n"],
      meaning: "Nominal control-disk radius and center."
    },
    {
      symbol: String.raw`V_i^\star,\;v_0,\;k_{v\phi}`,
      source: ["controller.speed_ref", "speed_phase_gain"],
      meaning: "Scheduled airspeed, base speed, and phase-to-speed gain."
    },
    {
      symbol: String.raw`d_{r,i},\;k_{dv},\;d_{r,\min},\;d_{r,\max}`,
      source: ["rabbit_speed_to_distance_s", "rabbit_min_distance_m", "rabbit_max_distance_m"],
      meaning: "Rabbit lead distance and the speed-to-distance clamp schedule."
    },
    {
      symbol: String.raw`r_i^r,\;k_{\phi r}`,
      source: "phase_lag_to_radius",
      meaning: "Phase-biased rabbit radius and phase-to-radius gain."
    },
    {
      symbol: String.raw`p_i^r`,
      source: "rabbit_targets_n",
      meaning: "Lateral rabbit target point in the control-disk plane."
    }
  ]);
  const guidanceLegend = docsSymbolLegend([
    {
      symbol: String.raw`q_i^n,\;q_i^b`,
      source: ["rabbit_targets_n", "rotate_nav_to_body"],
      meaning: "Vector from kite CAD point to rabbit target, in navigation and body frames."
    },
    {
      symbol: String.raw`\theta_{r,i},\;\phi_{i,\mathrm{rabbit}}^\star`,
      source: ["direct_rabbit_bearing_y", "direct_rabbit_roll_reference"],
      meaning: "Body-frame rabbit bearing and the direct-rabbit desired roll angle."
    },
    {
      symbol: String.raw`k_{\phi\theta,p},\;k_{\phi\theta,i}`,
      source: ["rabbit_bearing_roll_p", "rabbit_bearing_roll_i"],
      meaning: "Direct-rabbit PI gains from bearing error to roll reference."
    },
    {
      symbol: String.raw`f_x,\;\rho,\;\kappa_{\max}`,
      source: [
        "guidance_min_lookahead_fraction",
        "guidance_lateral_lookahead_ratio_limit",
        "guidance_curvature_limit"
      ],
      meaning: "Curvature-conversion lookahead, lateral clamp ratio, and curvature clamp."
    },
    {
      symbol: String.raw`\kappa_{y,i}^R,\;\kappa_{z,i}^R`,
      source: "lateral_guidance_curvatures",
      meaning: "Pure-pursuit curvature references used only by curvature and switch fallback modes."
    },
    {
      symbol: String.raw`\hat\kappa_{y,i},\;I_{\kappa\phi,i},\;k_{\phi\kappa,p},\;k_{\phi\kappa,i}`,
      source: ["omega_world_z", "roll_curvature_p", "roll_curvature_i"],
      meaning: "Estimated lateral curvature, curvature-to-roll integrator, and curvature-to-roll gains."
    }
  ]);
  const energyLegend = docsSymbolLegend([
    {
      symbol: String.raw`h_i,\;h_i^\star,\;e_{h,i}`,
      source: ["altitude", "altitude_ref", "tecs_altitude_error_limit_m"],
      meaning: "Altitude, altitude reference, and saturated altitude error."
    },
    {
      symbol: String.raw`c_z,\;z_g,\;k_{\dot z h}`,
      source: ["controller.disk_center_n", "ground_altitude", "vert_vel_to_rabbit_height"],
      meaning: "Control-disk height, ground altitude, and vertical-velocity shaping for the altitude reference."
    },
    {
      symbol: String.raw`E_{k,i},\;E_{p,i},\;E_{k,i}^\star,\;E_{p,i}^\star`,
      source: "tecs_terms",
      meaning: "Specific kinetic and potential energy values and references."
    },
    {
      symbol: String.raw`e_{k,i},\;e_{p,i},\;e_{b,i}`,
      source: "tecs_terms",
      meaning: "Specific kinetic, potential, and balance-energy errors."
    },
    {
      symbol: String.raw`I_{\tau,i},\;I_{\theta,i}`,
      source: ["thrust_energy_integrator", "pitch_energy_integrator"],
      meaning: "Motor-torque and pitch-reference PI integrator states."
    },
    {
      symbol: String.raw`k_{\tau,p},\;k_{\tau,i},\;k_{\theta,p},\;k_{\theta,i}`,
      source: [
        "tecs_thrust_kinetic_p",
        "tecs_thrust_kinetic_i",
        "tecs_pitch_balance_p",
        "tecs_pitch_balance_i"
      ],
      meaning: "TECS thrust and pitch PI gains."
    }
  ]);
  const actuatorLegend = docsSymbolLegend([
    {
      symbol: String.raw`\delta_a,\;\delta_e,\;\delta_r,\;\delta_f,\;\delta_w,\;\tau`,
      source: "Controls",
      meaning: "Aileron, elevator, rudder, flap, winglet, and motor-torque commands."
    },
    {
      symbol: String.raw`k_{a,\phi},\;k_{a,p}`,
      source: ["tethered_aileron_roll_p", "tethered_aileron_roll_d"],
      meaning: "Roll-angle and body-x-rate feedback gains for aileron."
    },
    {
      symbol: String.raw`k_{e,\theta},\;k_{e,q},\;k_{e,\alpha}`,
      source: ["elevator_pitch_p", "elevator_pitch_d", "alpha_to_elevator"],
      meaning: "Pitch-angle, pitch-rate, and angle-of-attack-protection gains for elevator."
    },
    {
      symbol: String.raw`k_{r,\beta},\;k_{r,r}`,
      source: ["tethered_rudder_beta_p", "tethered_rudder_rate_d"],
      meaning: "Sideslip and body-z-rate feedback gains for rudder."
    },
    {
      symbol: String.raw`\alpha_i^{\mathrm{prot}}`,
      source: ["alpha_protection_min_deg", "alpha_protection_max_deg"],
      meaning: "Angle-of-attack protection term added after the nominal pitch loop."
    },
    {
      symbol: String.raw`\delta_{\bullet,0}`,
      source: "trim surfaces",
      meaning: "Trim value for each surface command."
    }
  ]);

  return `
    <section class="docs-card">
      <div class="docs-card-head">
        <div class="docs-card-title">Implementation Scope</div>
        <div class="docs-card-note">These equations are generated directly from the Rust controller module under <code>multikite_sim/src/controller/</code>. The equations show the nominal feedback laws; bounds, trims, and other implementation-specific saturations are called out separately.</div>
      </div>
      <div class="docs-card-body">
        <div class="docs-grid">
          <div class="docs-prose">
            <div class="docs-phase-pill">Active UI phase mode <strong>${modeLabel}</strong></div>
            <div class="docs-phase-pill">Active longitudinal mode <strong>${longitudinalLabel}</strong></div>
            ${propulsionParagraph}
            <p>The controller is naturally read as a cascade: phase scheduling, lateral rabbit geometry, selectable lateral guidance, an independent altitude/speed energy layer, then roll/pitch inner loops and actuator commands.</p>
            <p><strong>The lateral aileron path introduces a desired roll angle.</strong> Rudder is used as a beta/yaw-rate coordination loop.</p>
            ${longitudinalParagraph}
            <p>The flap and winglet commands are fixed at trim.</p>
          </div>
          <div class="docs-kv">
            <div class="docs-kv-row">
              <div class="docs-kv-label">Nominal Reading</div>
              <div class="docs-kv-value">Paper-style equations below omit clamps and write each channel with explicit proportional, derivative, and integral gains.</div>
            </div>
            <div class="docs-kv-row">
              <div class="docs-kv-label">Scheduling Variables</div>
              <div class="docs-kv-value">\\(e_i\\) drives both the rabbit radius \\(r_i^r\\) and the scheduled airspeed \\(V_i^\\star\\); \\(V_i^\\star\\) schedules rabbit distance. The altitude reference is set by disk height, not by the lateral rabbit point.</div>
            </div>
            <div class="docs-kv-row">
              <div class="docs-kv-label">Inner-Loop Structure</div>
              <div class="docs-kv-value">${longitudinalStructure}</div>
            </div>
            <div class="docs-kv-row">
              <div class="docs-kv-label">Implementation Bounds</div>
              <div class="docs-kv-value">Bounds are applied in code and summarized separately instead of being embedded into every displayed equation.</div>
            </div>
          </div>
        </div>
        ${docsEquation("Notation", notationEquation, notationLegend)}
      </div>
    </section>
    <section class="docs-card">
      <div class="docs-card-head">
        <div class="docs-card-title">Loop Topology</div>
        <div class="docs-card-note">This is the structural answer to the “what sits between error and actuator?” question. Direct rabbit bearing or converted lateral curvature becomes a roll reference; disk-height altitude and airspeed control form the TECS-style energy layer.</div>
      </div>
      <div class="docs-card-body">
        ${docsEquation("Implemented signal path", signalPathEquation, docsLegendStack(guidanceLegend, energyLegend, actuatorLegend))}
        <div class="docs-note-list">
          <div class="docs-note-item">The aileron channel has an explicit commanded roll angle <span class="docs-inline-math">\\(\\phi_i^{\\star}\\)</span>, generated either from direct rabbit bearing or from lateral-curvature error depending on guidance mode.</div>
          <div class="docs-note-item">The vertical channel has an explicit commanded pitch angle <span class="docs-inline-math">\\(\\theta_i^{\\star}\\)</span> from the energy-balance PI loop.</div>
          <div class="docs-note-item">The most relevant “desired versus actual” lateral quantity is always <span class="docs-inline-math">\\(\\phi^{\\star}\\)</span> versus <span class="docs-inline-math">\\(\\phi\\)</span>. Curvature plots are meaningful only in curvature-conversion modes.</div>
        </div>
      </div>
    </section>
    <section class="docs-card">
      <div class="docs-card-head">
        <div class="docs-card-title">Phase Coordination And Scheduling</div>
        <div class="docs-card-note">Each kite first receives a scalar phase error. That single error schedules orbit radius, airspeed, and rabbit lead distance.</div>
      </div>
      <div class="docs-card-body">
        ${docsEquation("Phase error", phaseErrorEquation, phaseLegend)}
        ${docsEquation("Radius, speed, and lookahead scheduling", rabbitScheduleEquation, phaseLegend)}
      </div>
    </section>
    <section class="docs-card">
      <div class="docs-card-head">
        <div class="docs-card-title">Lateral Guidance And Integral States</div>
        <div class="docs-card-note">Rabbit geometry is a lateral target in the control-disk plane. Lateral guidance uses that target's X/Y coordinates projected to the aircraft altitude and rotated only by body yaw, so roll, pitch, and altitude error stay out of the lateral bearing. Default guidance is direct rabbit bearing. Curvature mode converts the rabbit vector to pure-pursuit curvature; switch mode uses direct rabbit while the target is ahead and curvature conversion when it falls behind.</div>
      </div>
      <div class="docs-card-body">
        ${docsEquation("Lateral rabbit geometry", rabbitGeometryEquation, phaseLegend)}
        ${docsEquation("Selectable lateral guidance", lateralGuidanceEquation, guidanceLegend, lateralGuidanceNote)}
        ${docsEquation("Curvature-to-roll tracking", curvatureTrackingEquation, guidanceLegend)}
      </div>
    </section>
    <section class="docs-card">
      <div class="docs-card-head">
        <div class="docs-card-title">Longitudinal, Actuator, And Propulsion Laws</div>
        <div class="docs-card-note">These are the nominal commanded laws. Saturation, anti-windup style bounds, and one-channel implementation quirks are summarized in the notes below rather than embedded into the equations themselves.</div>
      </div>
      <div class="docs-card-body">
        ${docsEquation("Independent altitude reference and energy states", energyStateEquation, energyLegend)}
        ${docsEquation("Nominal TECS, surface, and torque commands", actuatorEquation, docsLegendStack(energyLegend, actuatorLegend))}
        <div class="docs-note-list">
          <div class="docs-note-item">The implementation applies bounds after the nominal laws are evaluated: scheduled speed, integral states, surface deflections, and motor torque are all clamped in code.</div>
          <div class="docs-note-item">In direct-rabbit mode the aileron channel closes a rabbit bearing angle directly. In curvature modes the implementation first clamps the converted curvature reference <span class="docs-inline-math">\\(\\kappa_{y,i}^{\\star}\\)</span> before the curvature-to-roll term.</div>
          <div class="docs-note-item">The altitude reference shown in the TECS plots is the saturated effective reference used by the energy controller.</div>
          <div class="docs-note-item">Angle-of-attack protection biases the elevator command after the nominal pitch loop.</div>
          <div class="docs-note-item">Output limits come from the exposed controller tuning fields: <span class="docs-inline-math">\\(\\phi^\\star_{\\max}\\)</span>, <span class="docs-inline-math">\\(\\theta^\\star_{\\max}\\)</span>, surface deflection limits, and <span class="docs-inline-math">\\(\\tau_{\\max}\\)</span>.</div>
        </div>
      </div>
    </section>
    <section class="docs-card">
      <div class="docs-card-head">
        <div class="docs-card-title">Loop Diagrams</div>
        <div class="docs-card-note">These abstract control block diagrams show the outer scheduling and energy logic, then the roll, pitch, and coordination loops. The implementation has no commanded moment or angular-acceleration block.</div>
      </div>
      <div class="docs-card-body">
        <div class="docs-diagram-grid">
          <div class="docs-diagram">
            <div class="mermaid">${phaseDiagram}</div>
            <div class="docs-diagram-caption">Figure 1. Phase mode selection and rabbit scheduling feed lateral guidance; disk-height altitude and scheduled speed feed TECS.</div>
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
  const longitudinalMode = (maxThrottleAltitudePitchInput.checked
    ? "max_throttle_altitude_pitch"
    : "total_energy") as LongitudinalMode;
  controllerDocsNode.innerHTML = controllerDocsHtml(
    phaseModeSelect.value as PhaseMode,
    longitudinalMode
  );
  typesetMath(controllerDocsNode);
  renderMermaid(controllerDocsNode);
}

function syncOrbitTargetMarker(): void {
  orbitTargetMarker.position.copy(controls.target);
  const distance = camera.position.distanceTo(controls.target);
  const viewportHeight = Math.max(1, viewport.clientHeight);
  const worldPerPixel =
    (2 * distance * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)) / viewportHeight;
  const coreRadius = worldPerPixel * ORBIT_TARGET_CORE_PIXELS;
  orbitTargetCore.scale.setScalar(coreRadius / ORBIT_TARGET_CORE_RADIUS_WORLD);
}

function setOrbitTargetMarkerVisible(visible: boolean): void {
  orbitTargetMarker.visible = visible;
  if (visible) {
    syncOrbitTargetMarker();
  }
}

function numericInputValue(input: HTMLInputElement, fallback: number): number {
  const value = input.valueAsNumber;
  return Number.isFinite(value) ? value : fallback;
}

function clampedInputValue(
  input: HTMLInputElement,
  fallback: number,
  minValue: number,
  maxValue: number
): number {
  return THREE.MathUtils.clamp(numericInputValue(input, fallback), minValue, maxValue);
}

function nonnegativeIntegerInputValue(input: HTMLInputElement, fallback: number): number {
  const value = Math.floor(input.valueAsNumber);
  return Number.isFinite(value) && value >= 0 ? value : fallback;
}

function defaultDrydenConfig(): DrydenConfig {
  return (
    simulationDefaults?.dryden ?? {
      seed: 42,
      intensity_scale: 1,
      length_scale: 1,
      altitude_intensity_enabled: true,
      altitude_length_scale_enabled: true
    }
  );
}

function drydenConfigFromInputs(): DrydenConfig {
  const defaults = defaultDrydenConfig();
  return {
    seed: nonnegativeIntegerInputValue(drydenSeedInput, defaults.seed),
    intensity_scale: clampedInputValue(drydenIntensityScaleInput, defaults.intensity_scale, 0, 100),
    length_scale: clampedInputValue(drydenLengthScaleInput, defaults.length_scale, 0.01, 100),
    altitude_intensity_enabled: drydenAltitudeIntensityInput.checked,
    altitude_length_scale_enabled: drydenAltitudeLengthInput.checked
  };
}

function syncDrydenTuningVisibility(): void {
  drydenTuningFieldsNode.hidden = !simNoiseInput.checked;
}

function controlFeaturesVisible(): boolean {
  return controlFeaturesEnabledInput.checked;
}

function controlFeatureLinesVisible(): boolean {
  return controlFeatureLinesEnabledInput.checked;
}

function controlFeaturesAtTargetAltitude(): boolean {
  return controlFeaturesAtTargetAltitudeInput.checked;
}

function controlFeaturesAtAircraftAltitude(): boolean {
  return controlFeaturesAtAircraftAltitudeInput.checked;
}

function controlFeatureLayerVisible(layer: ControlFeatureAltitudeLayer): boolean {
  return layer === "aircraft"
    ? controlFeaturesAtAircraftAltitude()
    : controlFeaturesAtTargetAltitude();
}

function controlFeatureScale(): number {
  return clampedInputValue(controlFeatureScaleInput, 0.5, 0.05, 8);
}

function tetherNodeScale(): number {
  return clampedInputValue(tetherNodeScaleInput, 1.5, 0, 8);
}

function tetherTensionScaleMode(): TetherTensionScaleMode {
  const value = tetherTensionScaleModeSelect.value;
  if (value === "run_peak" || value === "fixed") {
    return value;
  }
  return "payload";
}

function tetherTensionPayloadMargin(): number {
  return clampedInputValue(tetherTensionPayloadMarginInput, 3, 0.1, 20);
}

function fallbackTetherTensionRange(): { min: number; max: number } {
  return {
    min: TETHER_TENSION_FALLBACK_MIN_N,
    max: TETHER_TENSION_FALLBACK_MAX_N
  };
}

function payloadTetherTensionRange(): { min: number; max: number } {
  const payloadMassKg = Math.max(1, numericInputValue(payloadInput, 100));
  const max = payloadMassKg * GRAVITY_MPS2 * tetherTensionPayloadMargin();
  return {
    min: 0,
    max: Math.max(1, max)
  };
}

function observedTetherTensionRange(): { min: number; max: number } | null {
  if (
    !Number.isFinite(observedTetherTensionMin) ||
    !Number.isFinite(observedTetherTensionMax) ||
    observedTetherTensionMax <= observedTetherTensionMin
  ) {
    return null;
  }
  const span = observedTetherTensionMax - observedTetherTensionMin;
  const padding = Math.max(1, span * TETHER_TENSION_PEAK_PADDING_FRACTION);
  return {
    min: Math.max(0, observedTetherTensionMin - padding),
    max: observedTetherTensionMax + padding
  };
}

function fixedTetherTensionRange(): { min: number; max: number } | null {
  const min = numericInputValue(tetherTensionFixedMinInput, TETHER_TENSION_FALLBACK_MIN_N);
  const max = numericInputValue(tetherTensionFixedMaxInput, TETHER_TENSION_FALLBACK_MAX_N);
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) {
    return null;
  }
  return { min, max };
}

function tetherTensionColorRange(): { min: number; max: number } {
  switch (tetherTensionScaleMode()) {
    case "run_peak":
      return observedTetherTensionRange() ?? payloadTetherTensionRange();
    case "fixed":
      return fixedTetherTensionRange() ?? fallbackTetherTensionRange();
    case "payload":
    default:
      return payloadTetherTensionRange();
  }
}

function tetherSlackThresholdN(): number {
  const { max } = tetherTensionColorRange();
  return Math.max(TETHER_SLACK_TENSION_N, max * TETHER_SLACK_RANGE_FRACTION);
}

function syncTetherTensionScaleVisibility(): void {
  const mode = tetherTensionScaleMode();
  document.querySelectorAll<HTMLElement>("[data-tension-scale-mode]").forEach((node) => {
    node.hidden = node.dataset.tensionScaleMode !== mode;
  });
}

function rerenderTetherColors(): void {
  if (lastRenderedFrame) {
    renderFrame(lastRenderedFrame);
  }
}

function airParticlesVisible(): boolean {
  return airParticlesEnabledInput.checked;
}

function airParticleOpacity(): number {
  return clampedInputValue(airParticleOpacityInput, 0.45, 0, 1);
}

function tetherNodesVisible(): boolean {
  return tetherNodesEnabledInput.checked;
}

function windShearVisible(): boolean {
  return windShearEnabledInput.checked && windShearNinetyHeightMeters() !== null;
}

function windShearNinetyHeightMeters(): number | null {
  if (windShearNinetyHeightInput.value.trim() === "") {
    return null;
  }
  const value = windShearNinetyHeightInput.valueAsNumber;
  return Number.isFinite(value) && value > 0 ? value : null;
}

function windShearFactorAtHeight(heightMeters: number): number {
  if (!windShearVisible()) {
    return 1;
  }
  const height = Math.max(0, heightMeters);
  const ninetyHeight = windShearNinetyHeightMeters();
  if (ninetyHeight === null) {
    return 1;
  }
  const scaleHeight = ninetyHeight / Math.log(10);
  if (!Number.isFinite(scaleHeight) || scaleHeight <= 0) {
    return 1;
  }
  return THREE.MathUtils.clamp(1 - Math.exp(-height / scaleHeight), 0, 1);
}

function wingtipTrailsVisible(): boolean {
  return wingtipTrailsEnabledInput.checked;
}

function wingtipConvectionEnabled(): boolean {
  return wingtipConvectionEnabledInput.checked;
}

function applyFogVisibility(): void {
  scene.fog = fogEnabledInput.checked ? sceneFog : null;
}

function applyAirParticleOpacity(): void {
  const opacity = airParticleOpacity();
  const ambientMaterial = ambientParticleCloud.material as THREE.ShaderMaterial;
  const gustMaterial = gustParticleCloud.material as THREE.ShaderMaterial;
  ambientMaterial.uniforms.uOpacity.value = opacity;
  gustMaterial.uniforms.uOpacity.value = Math.min(1, opacity * 1.4);
}

function applyVisualizationScales(): void {
  const markerScale = controlFeatureScale();
  payloadMesh.scale.setScalar(markerScale);
  controlFeatureLayers.forEach((layer) => {
    layer.rabbitMeshes.forEach((mesh) => mesh.scale.setScalar(markerScale));
    layer.lookaheadOnDiskMeshes.forEach((mesh) => mesh.scale.setScalar(markerScale));
    layer.projectedPhaseMeshes.forEach((mesh) => mesh.scale.setScalar(markerScale));
    layer.phaseSlotMeshes.forEach((mesh) => mesh.scale.setScalar(markerScale));
  });

  const nodeScale = tetherNodeScale();
  commonNodeMeshes.forEach((mesh) => {
    mesh.scale.setScalar((mesh.userData.nodeRadius as number) * nodeScale);
  });
  upperNodeMeshes.forEach((nodes) => {
    nodes.forEach((mesh) => {
      mesh.scale.setScalar((mesh.userData.nodeRadius as number) * nodeScale);
    });
  });
}

function resizeSceneRenderer(): void {
  renderer.setSize(viewport.clientWidth, viewport.clientHeight);
  camera.aspect = viewport.clientWidth / Math.max(1, viewport.clientHeight);
  camera.updateProjectionMatrix();
  syncOrbitTargetMarker();
  updateControlLabels();
}

function applyVisualizationVisibility(): void {
  const showControlFeatures = controlFeaturesVisible();
  const showControlLines = controlFeatureLinesVisible();
  const showTetherNodes = tetherNodesVisible();
  const frame = lastRenderedFrame;
  const kiteCount = frame?.kite_positions_n.length ?? 0;

  payloadMesh.visible = true;
  splitterMesh.visible = false;

  commonNodeMeshes.forEach((mesh) => {
    mesh.visible = showTetherNodes;
  });
  upperNodeMeshes.forEach((nodes, kiteIndex) => {
    nodes.forEach((mesh) => {
      mesh.visible = showTetherNodes && (!frame || kiteIndex < kiteCount);
    });
  });

  controlFeatureLayers.forEach((layer) => {
    const layerVisible = controlFeatureLayerVisible(layer.mode);
    const markerVisible = showControlFeatures && layerVisible;
    const lineVisible = showControlLines && layerVisible;
    layer.rabbitMeshes.forEach((mesh, kiteIndex) => {
      mesh.visible = markerVisible && (!frame || kiteIndex < kiteCount);
    });
    layer.lookaheadOnDiskMeshes.forEach((mesh, kiteIndex) => {
      mesh.visible = markerVisible && (!frame || kiteIndex < kiteCount);
    });
    layer.projectedPhaseMeshes.forEach((mesh, kiteIndex) => {
      mesh.visible = markerVisible && (!frame || kiteIndex < kiteCount);
    });
    [
      layer.guidanceLines,
      layer.lookaheadRadialOffsetLines,
      layer.projectedToDiskLines,
      layer.phaseSlotToClosestDiskLines
    ].forEach((lines) => {
      lines.forEach((line, kiteIndex) => {
        line.visible = lineVisible && (!frame || kiteIndex < kiteCount);
      });
    });
    layer.phaseSlotMeshes.forEach((mesh, kiteIndex) => {
      mesh.visible = markerVisible && (!frame || kiteIndex < kiteCount);
    });
  });

  applyFogVisibility();
  applyAirParticleOpacity();
  applyVisualizationScales();
  if (frame) {
    renderFrame(frame);
  } else {
    controlRingLine.visible = false;
    aircraftControlRingLine.visible = false;
  }
  updateControlLabels();
}

function stopMiddleZPanEvent(event: PointerEvent | MouseEvent): void {
  event.preventDefault();
  event.stopPropagation();
}

function middleZPanScalePerPixel(): number {
  const targetDistance = camera.position.distanceTo(controls.target);
  const fovScale = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
  return (2 * targetDistance * fovScale * controls.panSpeed) / Math.max(1, viewport.clientHeight);
}

function applyMiddleZPan(deltaY: number): void {
  const scale = middleZPanScalePerPixel();
  middlePanVertical.copy(WORLD_Z_AXIS).multiplyScalar(deltaY * scale);
  camera.position.add(middlePanVertical);
  controls.target.add(middlePanVertical);
  controls.update();
  syncOrbitTargetMarker();
}

function handleMiddleZPanStart(event: PointerEvent): void {
  if (event.pointerType !== "mouse" || event.button !== 1) {
    return;
  }
  stopMiddleZPanEvent(event);
  middleZPanPointerId = event.pointerId;
  middlePanStart.set(event.clientX, event.clientY);
  renderer.domElement.setPointerCapture(event.pointerId);
  setOrbitTargetMarkerVisible(true);
}

function handleMiddleZPanMove(event: PointerEvent): void {
  if (middleZPanPointerId !== event.pointerId) {
    return;
  }
  stopMiddleZPanEvent(event);
  middlePanEnd.set(event.clientX, event.clientY);
  applyMiddleZPan(middlePanEnd.y - middlePanStart.y);
  middlePanStart.copy(middlePanEnd);
}

function endMiddleZPan(event?: PointerEvent): void {
  if (middleZPanPointerId === null) {
    return;
  }
  if (event && event.pointerId !== middleZPanPointerId) {
    return;
  }
  if (event) {
    stopMiddleZPanEvent(event);
    if (renderer.domElement.hasPointerCapture(event.pointerId)) {
      renderer.domElement.releasePointerCapture(event.pointerId);
    }
  }
  middleZPanPointerId = null;
  setOrbitTargetMarkerVisible(false);
}

function preventMiddleAuxClick(event: MouseEvent): void {
  if (event.button === 1) {
    event.preventDefault();
  }
}

function sidebarMaxWidth(): number {
  return Math.max(
    SIDEBAR_RESIZE_MIN_WIDTH,
    Math.min(SIDEBAR_RESIZE_MAX_WIDTH, layoutNode.getBoundingClientRect().width - RIGHT_WORKBENCH_MIN_WIDTH)
  );
}

function clampSidebarWidth(widthPx: number): number {
  return THREE.MathUtils.clamp(widthPx, SIDEBAR_RESIZE_MIN_WIDTH, sidebarMaxWidth());
}

function setSidebarWidth(widthPx: number, persist: boolean): void {
  const width = clampSidebarWidth(widthPx);
  layoutNode.style.setProperty("--controls-width", `${width.toFixed(0)}px`);
  if (persist) {
    localStorage.setItem(SIDEBAR_WIDTH_STORAGE_KEY, width.toFixed(0));
  }
  resizeSceneRenderer();
}

function restoreSidebarWidth(): void {
  const saved = Number(localStorage.getItem(SIDEBAR_WIDTH_STORAGE_KEY));
  if (Number.isFinite(saved) && saved > 0) {
    setSidebarWidth(saved, false);
  }
}

function sidebarResizeWidthFromEvent(event: PointerEvent): number {
  return sidebarResizeStartWidth + event.clientX - sidebarResizeStartX;
}

function handleSidebarResizeStart(event: PointerEvent): void {
  if (event.pointerType === "mouse" && event.button !== 0) {
    return;
  }
  event.preventDefault();
  sidebarResizePointerId = event.pointerId;
  sidebarResizeStartX = event.clientX;
  sidebarResizeStartWidth = document.querySelector<HTMLElement>(".controls")!.getBoundingClientRect().width;
  sidebarResizeHandle.setPointerCapture(event.pointerId);
  layoutNode.classList.add("sidebar-resizing");
}

function handleSidebarResizeMove(event: PointerEvent): void {
  if (event.pointerId !== sidebarResizePointerId) {
    return;
  }
  event.preventDefault();
  setSidebarWidth(sidebarResizeWidthFromEvent(event), true);
}

function endSidebarResize(event?: PointerEvent): void {
  if (sidebarResizePointerId === null) {
    return;
  }
  if (event && event.pointerId !== sidebarResizePointerId) {
    return;
  }
  if (event && sidebarResizeHandle.hasPointerCapture(event.pointerId)) {
    sidebarResizeHandle.releasePointerCapture(event.pointerId);
  }
  sidebarResizePointerId = null;
  layoutNode.classList.remove("sidebar-resizing");
  resizeSceneRenderer();
}

function handleSidebarResizeKeydown(event: KeyboardEvent): void {
  const step = event.shiftKey ? 80 : 24;
  const currentWidth = document.querySelector<HTMLElement>(".controls")!.getBoundingClientRect().width;
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    setSidebarWidth(currentWidth - step, true);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    setSidebarWidth(currentWidth + step, true);
  }
}

function sceneResizeHeightFromEvent(event: PointerEvent): number {
  return THREE.MathUtils.clamp(
    sceneResizeStartHeight + event.clientY - sceneResizeStartY,
    SCENE_RESIZE_MIN_HEIGHT,
    SCENE_RESIZE_MAX_HEIGHT
  );
}

function setSceneHeight(heightPx: number): void {
  layoutNode.style.setProperty("--top-workbench-height", `${heightPx.toFixed(0)}px`);
  resizeSceneRenderer();
}

function handleSceneResizeStart(event: PointerEvent): void {
  if (event.pointerType === "mouse" && event.button !== 0) {
    return;
  }
  event.preventDefault();
  sceneResizePointerId = event.pointerId;
  sceneResizeStartY = event.clientY;
  sceneResizeStartHeight = viewport.getBoundingClientRect().height;
  sceneResizeHandle.setPointerCapture(event.pointerId);
  layoutNode.classList.add("scene-resizing");
}

function handleSceneResizeMove(event: PointerEvent): void {
  if (event.pointerId !== sceneResizePointerId) {
    return;
  }
  event.preventDefault();
  setSceneHeight(sceneResizeHeightFromEvent(event));
}

function endSceneResize(event?: PointerEvent): void {
  if (sceneResizePointerId === null) {
    return;
  }
  if (event && event.pointerId !== sceneResizePointerId) {
    return;
  }
  if (event && sceneResizeHandle.hasPointerCapture(event.pointerId)) {
    sceneResizeHandle.releasePointerCapture(event.pointerId);
  }
  sceneResizePointerId = null;
  layoutNode.classList.remove("scene-resizing");
  resizeSceneRenderer();
}

function handleSceneResizeKeydown(event: KeyboardEvent): void {
  const step = event.shiftKey ? 80 : 24;
  if (event.key === "ArrowUp") {
    event.preventDefault();
    setSceneHeight(Math.max(SCENE_RESIZE_MIN_HEIGHT, viewport.getBoundingClientRect().height - step));
  } else if (event.key === "ArrowDown") {
    event.preventDefault();
    setSceneHeight(Math.min(SCENE_RESIZE_MAX_HEIGHT, viewport.getBoundingClientRect().height + step));
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
renderer.domElement.addEventListener("pointerdown", handleMiddleZPanStart, { capture: true });
renderer.domElement.addEventListener("pointermove", handleMiddleZPanMove, { capture: true });
renderer.domElement.addEventListener("pointerup", endMiddleZPan, { capture: true });
renderer.domElement.addEventListener("pointercancel", endMiddleZPan, { capture: true });
renderer.domElement.addEventListener("lostpointercapture", () => endMiddleZPan());
renderer.domElement.addEventListener("auxclick", preventMiddleAuxClick);
renderer.domElement.addEventListener("pointerdown", handleViewportPointerStart);
renderer.domElement.addEventListener("pointerup", handleViewportPointerEnd);
renderer.domElement.addEventListener("pointercancel", handleViewportPointerEnd);
renderer.domElement.addEventListener("lostpointercapture", handleViewportPointerEnd);
window.addEventListener("pointerup", (event) => {
  endMiddleZPan(event);
  endSidebarResize(event);
  endSceneResize(event);
  handleViewportPointerEnd();
});
window.addEventListener("pointercancel", (event) => {
  endMiddleZPan(event);
  endSidebarResize(event);
  endSceneResize(event);
  handleViewportPointerEnd();
});
sidebarResizeHandle.addEventListener("pointerdown", handleSidebarResizeStart);
sidebarResizeHandle.addEventListener("pointermove", handleSidebarResizeMove);
sidebarResizeHandle.addEventListener("pointerup", endSidebarResize);
sidebarResizeHandle.addEventListener("pointercancel", endSidebarResize);
sidebarResizeHandle.addEventListener("lostpointercapture", () => endSidebarResize());
sidebarResizeHandle.addEventListener("keydown", handleSidebarResizeKeydown);
sceneResizeHandle.addEventListener("pointerdown", handleSceneResizeStart);
sceneResizeHandle.addEventListener("pointermove", handleSceneResizeMove);
sceneResizeHandle.addEventListener("pointerup", endSceneResize);
sceneResizeHandle.addEventListener("pointercancel", endSceneResize);
sceneResizeHandle.addEventListener("lostpointercapture", () => endSceneResize());
sceneResizeHandle.addEventListener("keydown", handleSceneResizeKeydown);
restoreSidebarWidth();
syncOrbitTargetMarker();

function failureKey(failure: SimulationFailure): string {
  return [
    failure.kite_index,
    failure.quantity,
    failure.time.toFixed(6),
    failure.value_deg.toFixed(6),
    failure.alpha_deg.toFixed(6),
    failure.beta_deg.toFixed(6)
  ].join(":");
}

function failureTitle(failure: SimulationFailure): string {
  return `Kite ${failure.kite_index + 1} ${failure.quantity} = ${failure.value_deg.toFixed(2)} deg`;
}

function failureLimitText(failure: SimulationFailure): string {
  return `limit [${failure.lower_limit_deg.toFixed(1)}, ${failure.upper_limit_deg.toFixed(1)}] deg`;
}

function failureChipHtml(failure: SimulationFailure): string {
  return `
    <div class="failure-chip-row">
      <div class="failure-chip">AOA ${failure.alpha_deg.toFixed(2)} deg</div>
      <div class="failure-chip">AOS ${failure.beta_deg.toFixed(2)} deg</div>
    </div>`;
}

function logFailureIfNeeded(failure: SimulationFailure | null): void {
  if (!failure) {
    return;
  }
  const key = failureKey(failure);
  if (key === lastFailureConsoleKey) {
    return;
  }
  lastFailureConsoleKey = key;
  appendConsole(
    `TERMINATED: ${failureTitle(failure)} at t=${failure.time.toFixed(2)}s; ` +
      `${failureLimitText(failure)}; AOA=${failure.alpha_deg.toFixed(2)} deg, ` +
      `AOS=${failure.beta_deg.toFixed(2)} deg`
  );
}

function setFailure(failure: SimulationFailure | null): void {
  if (!failure) {
    failureNode.innerHTML = "";
    failureNode.classList.remove("visible");
    sceneFailureNode.innerHTML = "";
    sceneFailureNode.classList.remove("visible");
    return;
  }
  failureNode.innerHTML = `
    <div class="failure-head">
      <div class="failure-kicker">Simulation Terminated</div>
      <div class="failure-time">t = ${failure.time.toFixed(2)} s</div>
    </div>
    <div class="failure-title">${escapeHtml(failureTitle(failure))}</div>
    <div class="failure-detail">
      Allowed range: ${failure.lower_limit_deg.toFixed(1)} to ${failure.upper_limit_deg.toFixed(1)} deg
    </div>
    ${failureChipHtml(failure)}`;
  sceneFailureNode.innerHTML = `
    <div class="scene-failure-kicker">Protection Limit</div>
    <div class="scene-failure-title">${escapeHtml(failureTitle(failure))}</div>
    <div class="scene-failure-meta">
      t = ${failure.time.toFixed(2)} s · ${escapeHtml(failureLimitText(failure))}
    </div>
    ${failureChipHtml(failure)}`;
  failureNode.classList.add("visible");
  sceneFailureNode.classList.add("visible");
}

function setSolverFailure(message: string, time: number | null): void {
  const timeText = time === null ? "time unavailable" : `t = ${time.toFixed(2)} s`;
  failureNode.innerHTML = `
    <div class="failure-head">
      <div class="failure-kicker">Simulation Failed</div>
      <div class="failure-time">${escapeHtml(timeText)}</div>
    </div>
    <div class="failure-title">${escapeHtml(message)}</div>
    <div class="failure-detail">The integrator or runtime stopped before producing a completed summary.</div>`;
  sceneFailureNode.innerHTML = `
    <div class="scene-failure-kicker">Solver Failure</div>
    <div class="scene-failure-title">${escapeHtml(message)}</div>
    <div class="scene-failure-meta">${escapeHtml(timeText)}</div>`;
  failureNode.classList.add("visible");
  sceneFailureNode.classList.add("visible");
}

function presetKiteCount(preset: Preset): number {
  if (preset === "swarm") {
    return selectedSwarmKiteCount();
  }
  const presetInfo = presetInfoById.get(preset);
  if (presetInfo) {
    return presetInfo.kites;
  }
  const option = Array.from(presetSelect.options).find((item) => item.value === preset);
  return Number(option?.dataset.kites ?? 0);
}

function selectedSwarmKiteCount(): number {
  const value = Number(swarmKitesSelect.value);
  return Number.isInteger(value) ? Math.min(12, Math.max(1, value)) : 2;
}

function optionalInputValue(input: HTMLInputElement): number | null {
  if (input.value.trim() === "") {
    return null;
  }
  const value = input.valueAsNumber;
  return Number.isFinite(value) ? value : null;
}

function optionalMetersLabel(value: number | null): string {
  return value === null ? "auto" : `${compactNumberInputValue(value)} m`;
}

function syncSwarmOptionsVisibility(): void {
  swarmOptionsNode.hidden = presetSelect.value !== "swarm";
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
    case "10":
      return 10.0;
    case "5":
      return 5.0;
    case "2":
      return 2.0;
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
    case "10":
      return "10x";
    case "5":
      return "5x";
    case "2":
      return "2x";
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

function positiveInputValue(input: HTMLInputElement, fallback: number): number {
  const value = input.valueAsNumber;
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function nonnegativeInputValue(input: HTMLInputElement, fallback: number): number {
  const value = input.valueAsNumber;
  return Number.isFinite(value) && value >= 0 ? value : fallback;
}

function positiveIntegerInputValue(input: HTMLInputElement, fallback: number): number {
  const value = Math.floor(input.valueAsNumber);
  return Number.isFinite(value) && value >= 1 ? value : fallback;
}

function toleranceLabel(value: number): string {
  return Number.isFinite(value) && value > 0 ? value.toExponential(1) : "n/a";
}

function toleranceInputValue(value: number): string {
  return Number.isFinite(value) && value > 0 ? value.toExponential(0) : "";
}

function compactNumberInputValue(value: number): string {
  if (!Number.isFinite(value)) {
    return "";
  }
  const abs = Math.abs(value);
  if (abs > 0 && (abs < 1.0e-3 || abs >= 1.0e4)) {
    return value.toExponential(3).replace(/\.?0+e/, "e");
  }
  return String(Number(value.toPrecision(8)));
}

function renderControllerTuningControls(tuning: ControllerTuning): void {
  controllerTuningFieldsNode.innerHTML = "";
  const sections = new Map<string, HTMLElement>();
  const groups = new Map<string, HTMLElement>();

  const ensureSection = (section: ControllerTuningSection): HTMLElement => {
    const existing = sections.get(section.title);
    if (existing) {
      return existing;
    }
    const node = document.createElement("section");
    node.className = "tuning-section";
    node.dataset.section = section.title;
    node.innerHTML = `
      <div class="tuning-section-head">
        <div class="tuning-section-title">${escapeHtml(section.title)}</div>
        <div class="tuning-section-description">${escapeHtml(section.description)}</div>
      </div>
    `;
    sections.set(section.title, node);
    controllerTuningFieldsNode.append(node);
    return node;
  };

  CONTROLLER_TUNING_FIELDS.forEach((field) => {
    let group = groups.get(field.group);
    if (!group) {
      const section = CONTROLLER_TUNING_GROUP_TO_SECTION.get(field.group) ?? {
        title: "Other",
        description: "Less commonly adjusted controller parameters.",
        groups: [field.group]
      };
      const sectionNode = ensureSection(section);
      group = document.createElement("section");
      group.className = "tuning-group";
      group.dataset.group = field.group;
      group.innerHTML = `<div class="tuning-group-title">${escapeHtml(field.group)}</div>`;
      groups.set(field.group, group);
      sectionNode.append(group);
    }

    const row = document.createElement("label");
    row.className = field.kind === "select" ? "tuning-field tuning-field-wide" : "tuning-field";
    if (field.mode) {
      row.dataset.mode = field.mode;
    }
    if (field.guidanceModes) {
      row.dataset.guidanceModes = field.guidanceModes.join(",");
    }
    const value = tuning[field.key] ?? 0;
    const controlId = `controller-tuning-${field.key}`;
    const control =
      field.kind === "select"
        ? `
          <select
            id="${escapeHtml(controlId)}"
            data-tuning-key="${escapeHtml(field.key)}"
            data-tuning-step="${escapeHtml(field.step)}"
          >
            ${(field.options ?? [])
              .map((option) => {
                const selected = Math.round(value) === option.value ? " selected" : "";
                return `<option value="${option.value}"${selected}>${escapeHtml(option.label)}</option>`;
              })
              .join("")}
          </select>
        `
        : `
          <input
            id="${escapeHtml(controlId)}"
            data-tuning-key="${escapeHtml(field.key)}"
            data-tuning-step="${escapeHtml(field.step)}"
            type="number"
            step="any"
            ${field.min ? `min="${escapeHtml(field.min)}"` : ""}
            value="${escapeHtml(compactNumberInputValue(value))}"
          />
        `;
    row.innerHTML = `
      <span>
        <span class="tuning-label">${escapeHtml(field.label)}</span>
        ${field.unit ? `<span class="tuning-unit">${escapeHtml(field.unit)}</span>` : ""}
        ${field.help ? `<span class="tuning-help">${escapeHtml(field.help)}</span>` : ""}
      </span>
      ${control}
    `;
    group.append(row);
  });

  syncControllerTuningVisibility();
}

function controllerTuningFromInputs(): ControllerTuning {
  const defaults = simulationDefaults?.controller_tuning ?? {};
  const tuning: ControllerTuning = { ...defaults };
  CONTROLLER_TUNING_FIELDS.forEach((field) => {
    const input = controllerTuningFieldsNode.querySelector<HTMLInputElement | HTMLSelectElement>(
      `[data-tuning-key="${field.key}"]`
    );
    const value =
      input instanceof HTMLInputElement ? input.valueAsNumber : Number(input?.value);
    tuning[field.key] =
      value !== undefined && Number.isFinite(value) ? value : defaults[field.key] ?? 0;
  });
  return tuning;
}

function controllerTuningFingerprint(tuning: ControllerTuning): string {
  const canonical = CONTROLLER_TUNING_FIELDS.map((field) => {
    const value = tuning[field.key];
    return `${field.key}=${Number.isFinite(value) ? Number(value).toPrecision(12) : "nan"}`;
  }).join(";");
  let hash = 2166136261;
  for (let index = 0; index < canonical.length; index += 1) {
    hash ^= canonical.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function controllerTuningSnapshotLabel(tuning: ControllerTuning): string {
  const defaults = simulationDefaults?.controller_tuning ?? {};
  const changedFields = CONTROLLER_TUNING_FIELDS.filter((field) => {
    const value = tuning[field.key];
    const defaultValue = defaults[field.key];
    if (!Number.isFinite(value) || !Number.isFinite(defaultValue)) {
      return value !== defaultValue;
    }
    return Math.abs(value - defaultValue) > Math.max(1.0e-10, Math.abs(defaultValue) * 1.0e-9);
  });
  const fingerprint = controllerTuningFingerprint(tuning);
  if (changedFields.length === 0) {
    return `controller tuning captured: defaults, fingerprint=${fingerprint}`;
  }
  const visible = changedFields.slice(0, 6).map((field) => {
    const value = tuning[field.key];
    return `${field.label}=${compactNumberInputValue(value)}`;
  });
  const extra = changedFields.length > visible.length ? `, +${changedFields.length - visible.length} more` : "";
  return `controller tuning captured: ${visible.join(", ")}${extra}; fingerprint=${fingerprint}`;
}

function noteControllerTuningEditedDuringRun(): void {
  if (!runInProgress || controllerTuningChangedDuringRun) {
    return;
  }
  controllerTuningChangedDuringRun = true;
  appendConsole("controller tuning edited during active run; current run is unchanged, use Restart to apply");
}

function activeGuidanceMode(): GuidanceMode {
  const input = controllerTuningFieldsNode.querySelector<HTMLSelectElement>(
    `[data-tuning-key="guidance_mode"]`
  );
  switch (Math.round(Number(input?.value))) {
    case 1:
      return "curvature";
    case 2:
      return "switch";
    default:
      return "rabbit";
  }
}

function syncControllerTuningVisibility(): void {
  const activeMode: TuningMode = maxThrottleAltitudePitchInput.checked
    ? "max_throttle_altitude_pitch"
    : "total_energy";
  const guidanceMode = activeGuidanceMode();
  controllerTuningFieldsNode.querySelectorAll<HTMLElement>(".tuning-field").forEach((field) => {
    field.hidden = false;
  });
  controllerTuningFieldsNode.querySelectorAll<HTMLElement>(".tuning-field[data-mode]").forEach(
    (field) => {
      field.hidden = field.dataset.mode !== activeMode;
    }
  );
  controllerTuningFieldsNode
    .querySelectorAll<HTMLElement>(".tuning-field[data-guidance-modes]")
    .forEach((field) => {
      const allowed = (field.dataset.guidanceModes ?? "").split(",");
      field.hidden ||= !allowed.includes(guidanceMode);
    });
  controllerTuningFieldsNode.querySelectorAll<HTMLElement>(".tuning-group").forEach((group) => {
    const fields = Array.from(group.querySelectorAll<HTMLElement>(".tuning-field"));
    group.hidden = fields.length > 0 && fields.every((field) => field.hidden);
  });
  controllerTuningFieldsNode.querySelectorAll<HTMLElement>(".tuning-section").forEach((section) => {
    const groups = Array.from(section.querySelectorAll<HTMLElement>(".tuning-group"));
    section.hidden = groups.length > 0 && groups.every((group) => group.hidden);
  });
}

function playbackAnchorSimTime(): number {
  return lastRenderedFrame?.time ?? pendingPlaybackFrames[0]?.time ?? playbackStartSimTime;
}

function applyTimeDilationSelection(logChange: boolean): void {
  const selectedTimeDilation = timeDilationSelect.value as TimeDilationPreset;
  const nextLabel = timeDilationLabel(selectedTimeDilation);
  const nextRate = timeDilationRate(selectedTimeDilation);
  const changed = nextLabel !== currentPlaybackLabel || nextRate !== currentPlaybackRate;
  currentPlaybackLabel = nextLabel;
  currentPlaybackRate = nextRate;
  playbackStartSimTime = playbackAnchorSimTime();
  playbackStartWallTimeMs = nextRate === null ? null : performance.now();
  if (logChange && changed && runInProgress) {
    appendConsole(
      playbackPaused
        ? `time dilation changed to ${nextLabel}; applies on resume`
        : `time dilation changed to ${nextLabel}`
    );
  }
  refreshProgressSummary();
}

function setRunControls(): void {
  runButton.disabled = false;
  if (runInProgress) {
    runButton.textContent = runStreamComplete
      ? "Run"
      : !playbackReleased
        ? "Solving"
        : playbackPaused
          ? "Resume"
          : "Pause";
  } else {
    runButton.textContent = "Run";
  }
  restartButton.disabled = false;
}

function resumePlaybackClock(): void {
  if (currentPlaybackRate === null) {
    return;
  }
  playbackStartWallTimeMs = performance.now();
  playbackStartSimTime = playbackAnchorSimTime();
}

function togglePlaybackPause(): void {
  if (!runInProgress || !playbackReleased) {
    return;
  }
  playbackPaused = !playbackPaused;
  if (!playbackPaused) {
    resumePlaybackClock();
    appendConsole("playback resumed");
  } else {
    appendConsole("playback paused");
  }
  setRunControls();
  refreshProgressSummary();
}

function longitudinalModeLabel(mode: LongitudinalMode): string {
  switch (mode) {
    case "max_throttle_altitude_pitch":
      return "Max throttle + altitude pitch";
    default:
      return "Total energy";
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
  playbackPaused = false;
  playbackReleased = false;
  currentPlaybackLabel = label;
  currentPlaybackRate = rate;
  playbackStartWallTimeMs = null;
  playbackStartSimTime = 0;
  shouldSnapOrbitTargetToFrame = !hasRenderedSimulationFrame && currentCameraFollowTarget() === "manual";
  lastRenderedFrame = null;
  observedTetherTensionMin = Number.POSITIVE_INFINITY;
  observedTetherTensionMax = Number.NEGATIVE_INFINITY;
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
    state.velocity.set(0, 0, 0);
    wingtipTrailAlpha[index] = 0;
  });
}

function clearWingtipTrailParticles(): void {
  wingtipTrailCloud.visible = false;
  nextWingtipTrailParticleIndex = 0;
  wingtipTrailStates.forEach((state, index) => {
    state.age = 0;
    state.life = 0;
    state.active = false;
    state.velocity.set(0, 0, 0);
    wingtipTrailAlpha[index] = 0;
  });
  (wingtipTrailGeometry.attributes.alpha as THREE.BufferAttribute).needsUpdate = true;
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

function plotSignalDefaultVisible(signalKey: string): boolean {
  for (const section of activePlotSections) {
    for (const trace of section.traces) {
      if (plotSignalKey(trace) === signalKey) {
        return trace.defaultVisible ?? true;
      }
    }
  }
  return true;
}

function plotSignalVisible(trace: PlotTraceDefinition): boolean {
  return plotSignalVisibility.get(plotSignalKey(trace)) ?? trace.defaultVisible ?? true;
}

function plotKiteTraceVisible(trace: PlotTraceDefinition): boolean {
  if (trace.alwaysVisible) {
    return true;
  }
  return trace.kiteIndex === undefined || (plotKiteVisibility[trace.kiteIndex] ?? true);
}

function plotTraceVisible(trace: PlotTraceDefinition): boolean {
  if (trace.alwaysVisible) {
    return true;
  }
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
    const visible = plotSignalVisibility.get(signalKey) ?? plotSignalDefaultVisible(signalKey);
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
      const nextVisible = !(plotSignalVisibility.get(item.key) ?? plotSignalDefaultVisible(item.key));
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
        defaultVisible: trace.defaultVisible,
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
    (frame, kiteIndex) => frame.altitude_ref[kiteIndex] ?? 0,
    [
      {
        name: "Payload",
        color: "#ffbe6b",
        signalKey: "Altitude Desired vs Actual (m):payload",
        legendName: "Payload altitude",
        dash: "dot",
        width: 2,
        value: (frame) =>
          frame.common_tether.length > 2 ? -frame.payload_position_n[2] : Number.NaN
      }
    ]
  );
}

function buildRollCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Desired Roll vs Actual (deg)",
    "deg",
    (frame, kiteIndex) => frame.kite_control_roll_pitch_deg[kiteIndex]?.[0] ?? 0,
    (frame, kiteIndex) => frame.roll_ref_deg[kiteIndex] ?? 0
  );
}

function buildPitchCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Desired Pitch vs Actual (deg)",
    "deg",
    (frame, kiteIndex) => frame.kite_control_roll_pitch_deg[kiteIndex]?.[1] ?? 0,
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
    "Phase Error (deg)",
    "deg",
    (frame, kiteIndex) => (frame.phase_error[kiteIndex] ?? 0) * RAD_TO_DEG,
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

function buildRabbitBearingGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Rabbit Bearing Error (deg)",
    "deg",
    (frame, kiteIndex) => frame.rabbit_bearing_y_deg[kiteIndex] ?? 0,
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

function buildRabbitDistanceGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Rabbit Lead / Target Distance (m)",
    "m",
    (frame, kiteIndex) => frame.rabbit_target_distance[kiteIndex] ?? 0,
    (frame, kiteIndex) => frame.rabbit_distance[kiteIndex] ?? 0
  );
}

function buildRabbitVectorGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteBreakdownGroup(kiteCount, "Rabbit Vector in Body Frame (m)", "m", [
    {
      name: "Forward x",
      alpha: 0.95,
      value: (frame, kiteIndex) => bodyComponent(frame.rabbit_vector_b, kiteIndex, 0)
    },
    {
      name: "Lateral y",
      alpha: 0.78,
      dash: "dash",
      value: (frame, kiteIndex) => bodyComponent(frame.rabbit_vector_b, kiteIndex, 1)
    },
    {
      name: "Vertical z",
      alpha: 0.62,
      dash: "dot",
      defaultVisible: false,
      value: (frame, kiteIndex) => bodyComponent(frame.rabbit_vector_b, kiteIndex, 2)
    }
  ]);
}

function buildSpeedTargetGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Scheduled Speed Target (m/s)",
    "m/s",
    (frame, kiteIndex) => frame.speed_target[kiteIndex] ?? 0
  );
}

function buildAileronCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Aileron Command vs Applied (deg)",
    "deg",
    (frame, kiteIndex) => frame.aileron_applied_deg[kiteIndex] ?? 0,
    (frame, kiteIndex) => frame.aileron_cmd_deg[kiteIndex] ?? 0
  );
}

function buildRudderCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Rudder Command vs Applied (deg)",
    "deg",
    (frame, kiteIndex) => frame.rudder_applied_deg[kiteIndex] ?? 0,
    (frame, kiteIndex) => frame.rudder_cmd_deg[kiteIndex] ?? 0
  );
}

function buildElevatorCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Elevator Command vs Applied (deg)",
    "deg",
    (frame, kiteIndex) => frame.elevator_applied_deg[kiteIndex] ?? 0,
    (frame, kiteIndex) => frame.elevator_cmd_deg[kiteIndex] ?? 0
  );
}

function buildMotorTorqueCommandGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteGroup(
    kiteCount,
    "Motor Torque Command vs Applied (N m)",
    "N m",
    (frame, kiteIndex) => frame.motor_torque_applied[kiteIndex] ?? 0,
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

function buildRollReferenceBreakdownGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteBreakdownGroup(
    kiteCount,
    "Roll Reference Breakdown (deg)",
    "deg",
    [
      {
        name: "Total",
        width: 2.6,
        value: (frame, kiteIndex) => frame.roll_ref_deg[kiteIndex] ?? 0
      },
      {
        name: "Feedforward",
        dash: "dash",
        alpha: 0.7,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.roll_ff_deg[kiteIndex] ?? 0
      },
      {
        name: "P",
        dash: "dot",
        alpha: 0.7,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.roll_p_deg[kiteIndex] ?? 0
      },
      {
        name: "I",
        dash: "dashdot",
        alpha: 0.7,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.roll_i_deg[kiteIndex] ?? 0
      }
    ]
  );
}

function buildAileronBreakdownGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteBreakdownGroup(
    kiteCount,
    "Aileron Command Breakdown (deg)",
    "deg",
    [
      {
        name: "Total",
        width: 2.6,
        value: (frame, kiteIndex) => frame.aileron_cmd_deg[kiteIndex] ?? 0
      },
      {
        name: "Trim",
        dash: "dash",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.aileron_trim_deg[kiteIndex] ?? 0
      },
      {
        name: "Roll P",
        dash: "dot",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.aileron_roll_p_deg[kiteIndex] ?? 0
      },
      {
        name: "Roll-rate D",
        dash: "dashdot",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.aileron_roll_d_deg[kiteIndex] ?? 0
      }
    ]
  );
}

function buildRudderBreakdownGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteBreakdownGroup(
    kiteCount,
    "Rudder Command Breakdown (deg)",
    "deg",
    [
      {
        name: "Total",
        width: 2.6,
        value: (frame, kiteIndex) => frame.rudder_cmd_deg[kiteIndex] ?? 0
      },
      {
        name: "Trim/offset",
        dash: "dash",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.rudder_trim_deg[kiteIndex] ?? 0
      },
      {
        name: "Beta P",
        dash: "dot",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.rudder_beta_p_deg[kiteIndex] ?? 0
      },
      {
        name: "Body-rate D",
        dash: "dashdot",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.rudder_rate_d_deg[kiteIndex] ?? 0
      },
      {
        name: "Turn-rate P",
        dash: "longdash",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.rudder_world_z_p_deg[kiteIndex] ?? 0
      }
    ]
  );
}

function buildMotorTorqueBreakdownGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteBreakdownGroup(
    kiteCount,
    "Motor Torque Breakdown (N m)",
    "N m",
    [
      {
        name: "Total",
        width: 2.6,
        value: (frame, kiteIndex) => frame.motor_torque[kiteIndex] ?? 0
      },
      {
        name: "Trim",
        dash: "dash",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.motor_torque_trim[kiteIndex] ?? 0
      },
      {
        name: "Kinetic P",
        dash: "dot",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.motor_torque_p[kiteIndex] ?? 0
      },
      {
        name: "Kinetic I",
        dash: "dashdot",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.motor_torque_i[kiteIndex] ?? 0
      }
    ]
  );
}

function buildPitchReferenceBreakdownGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteBreakdownGroup(
    kiteCount,
    "Pitch Reference Breakdown (deg)",
    "deg",
    [
      {
        name: "Total",
        width: 2.6,
        value: (frame, kiteIndex) => frame.pitch_ref_deg[kiteIndex] ?? 0
      },
      {
        name: "Energy P",
        dash: "dot",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.pitch_ref_p_deg[kiteIndex] ?? 0
      },
      {
        name: "Energy I",
        dash: "dashdot",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.pitch_ref_i_deg[kiteIndex] ?? 0
      }
    ]
  );
}

function buildElevatorBreakdownGroup(kiteCount: number): PlotGroupDefinition {
  return buildPerKiteBreakdownGroup(
    kiteCount,
    "Elevator Command Breakdown (deg)",
    "deg",
    [
      {
        name: "Total",
        width: 2.6,
        value: (frame, kiteIndex) => frame.elevator_cmd_deg[kiteIndex] ?? 0
      },
      {
        name: "Trim",
        dash: "dash",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.elevator_trim_deg[kiteIndex] ?? 0
      },
      {
        name: "Pitch P",
        dash: "dot",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.elevator_pitch_p_deg[kiteIndex] ?? 0
      },
      {
        name: "Pitch-rate D",
        dash: "dashdot",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.elevator_pitch_d_deg[kiteIndex] ?? 0
      },
      {
        name: "AOA protection",
        dash: "longdash",
        alpha: 0.68,
        defaultVisible: false,
        value: (frame, kiteIndex) => frame.elevator_alpha_protection_deg[kiteIndex] ?? 0
      }
    ]
  );
}

interface LimiterLaneDefinition {
  label: string;
  color: string;
  modes?: LimiterLaneMode[];
  active: (frame: ApiFrame, kiteIndex: number, tuning: ControllerTuning) => LimiterActivation | null;
}

interface LimiterLaneMode {
  key: string;
  label: string;
  color: string;
}

interface LimiterActivation {
  mode: string;
  label: string;
  detail?: string;
}

function tuningNumber(tuning: ControllerTuning, key: string, fallback: number): number {
  const value = tuning[key] ?? simulationDefaults?.controller_tuning?.[key];
  return Number.isFinite(value) ? Number(value) : fallback;
}

function nearAbsLimit(value: number, limit: number, absoluteMargin = 1.0e-4): boolean {
  if (!Number.isFinite(value) || !Number.isFinite(limit) || limit <= 0) {
    return false;
  }
  return Math.abs(value) >= limit - Math.max(absoluteMargin, limit * 1.0e-4);
}

function nearUpperLimit(value: number, limit: number, absoluteMargin = 1.0e-4): boolean {
  if (!Number.isFinite(value) || !Number.isFinite(limit) || limit <= 0) {
    return false;
  }
  return value >= limit - Math.max(absoluteMargin, limit * 1.0e-4);
}

function nearLowerLimit(value: number, limit: number, absoluteMargin = 1.0e-4): boolean {
  if (!Number.isFinite(value) || !Number.isFinite(limit)) {
    return false;
  }
  return value <= limit + Math.max(absoluteMargin, Math.abs(limit) * 1.0e-4);
}

function limiterActivation(mode: string, label: string, detail?: string): LimiterActivation {
  return { mode, label, detail };
}

function simpleLimiterActivation(
  condition: boolean,
  label = "active",
  detail?: string
): LimiterActivation | null {
  return condition ? limiterActivation("active", label, detail) : null;
}

function signedLimiterActivation(
  value: number,
  limit: number,
  positiveLabel: string,
  negativeLabel: string,
  unit: string,
  margin = 0.02
): LimiterActivation | null {
  if (!nearAbsLimit(value, limit, margin)) {
    return null;
  }
  const positive = value >= 0;
  return limiterActivation(
    positive ? "positive" : "negative",
    positive ? positiveLabel : negativeLabel,
    `${value.toFixed(3)} ${unit} at ${positive ? "+" : "-"}${limit.toFixed(3)} ${unit} limit`
  );
}

function upperLimiterActivation(
  value: number,
  limit: number,
  label: string,
  unit: string,
  margin = 0.02
): LimiterActivation | null {
  return nearUpperLimit(value, limit, margin)
    ? limiterActivation("upper", label, `${value.toFixed(3)} ${unit} at ${limit.toFixed(3)} ${unit} upper limit`)
    : null;
}

function lowerLimiterActivation(
  value: number,
  limit: number,
  label: string,
  unit: string,
  margin = 0.02
): LimiterActivation | null {
  return nearLowerLimit(value, limit, margin)
    ? limiterActivation("lower", label, `${value.toFixed(3)} ${unit} at ${limit.toFixed(3)} ${unit} lower limit`)
    : null;
}

function positiveNegativeModes(colorPositive = "#ff7b72", colorNegative = "#66b8ff"): LimiterLaneMode[] {
  return [
    { key: "positive", label: "positive", color: colorPositive },
    { key: "negative", label: "negative", color: colorNegative }
  ];
}

function upperLowerModes(colorUpper = "#ff7b72", colorLower = "#66b8ff"): LimiterLaneMode[] {
  return [
    { key: "upper", label: "upper", color: colorUpper },
    { key: "lower", label: "lower", color: colorLower }
  ];
}

function runUsesTetheredPitchLimit(): boolean {
  const preset = activeSummaryRequest?.preset ?? presetSelect.value;
  return preset !== "free_flight1";
}

function pitchReferenceLimitDeg(tuning: ControllerTuning): number {
  return runUsesTetheredPitchLimit()
    ? tuningNumber(tuning, "tethered_pitch_ref_limit_deg", 22)
    : tuningNumber(tuning, "free_pitch_ref_limit_deg", 22);
}

function limiterLaneDefinitions(): LimiterLaneDefinition[] {
  return [
    {
      label: "Roll ref clamp",
      color: "#ffd166",
      modes: positiveNegativeModes("#ffd166", "#8dd7ff"),
      active: (frame, kiteIndex, tuning) =>
        signedLimiterActivation(
          frame.roll_ref_deg[kiteIndex] ?? 0,
          tuningNumber(tuning, "roll_ref_limit_deg", 35),
          "positive roll reference clamp",
          "negative roll reference clamp",
          "deg"
        )
    },
    {
      label: "Pitch ref clamp",
      color: "#ffd166",
      modes: positiveNegativeModes("#ffd166", "#8dd7ff"),
      active: (frame, kiteIndex, tuning) =>
        signedLimiterActivation(
          frame.pitch_ref_deg[kiteIndex] ?? 0,
          pitchReferenceLimitDeg(tuning),
          "pitch-up reference clamp",
          "pitch-down reference clamp",
          "deg"
        )
    },
    {
      label: "Aileron sat",
      color: "#ff7b72",
      modes: positiveNegativeModes(),
      active: (frame, kiteIndex, tuning) =>
        signedLimiterActivation(
          frame.aileron_cmd_deg[kiteIndex] ?? 0,
          tuningNumber(tuning, "surface_limit_aileron_deg", 28.65),
          "positive aileron saturation",
          "negative aileron saturation",
          "deg"
        )
    },
    {
      label: "Rudder sat",
      color: "#ff7b72",
      modes: positiveNegativeModes(),
      active: (frame, kiteIndex, tuning) =>
        signedLimiterActivation(
          frame.rudder_cmd_deg[kiteIndex] ?? 0,
          tuningNumber(tuning, "surface_limit_rudder_deg", 25),
          "positive rudder saturation",
          "negative rudder saturation",
          "deg"
        )
    },
    {
      label: "Elevator sat",
      color: "#ff7b72",
      modes: positiveNegativeModes(),
      active: (frame, kiteIndex, tuning) =>
        signedLimiterActivation(
          frame.elevator_cmd_deg[kiteIndex] ?? 0,
          tuningNumber(tuning, "surface_limit_elevator_deg", 17.19),
          "positive elevator saturation",
          "negative elevator saturation",
          "deg"
        )
    },
    {
      label: "Motor max",
      color: "#ff7b72",
      active: (frame, kiteIndex, tuning) =>
        upperLimiterActivation(
          frame.motor_torque[kiteIndex] ?? 0,
          tuningNumber(tuning, "motor_torque_max_nm", 45.6),
          "motor torque at upper limit",
          "N m"
        )
    },
    {
      label: "Motor min",
      color: "#ff7b72",
      active: (frame, kiteIndex) =>
        lowerLimiterActivation(
          frame.motor_torque[kiteIndex] ?? 0,
          0,
          "motor torque at lower limit",
          "N m"
        )
    },
    {
      label: "Thrust I limit",
      color: "#c28dff",
      modes: positiveNegativeModes("#c28dff", "#72d7ff"),
      active: (frame, kiteIndex, tuning) =>
        signedLimiterActivation(
          frame.thrust_energy_integrator[kiteIndex] ?? 0,
          tuningNumber(tuning, "tecs_thrust_integrator_limit_nm", 8),
          "positive thrust integrator limit",
          "negative thrust integrator limit",
          "N m"
        )
    },
    {
      label: "Pitch I limit",
      color: "#c28dff",
      modes: positiveNegativeModes("#c28dff", "#72d7ff"),
      active: (frame, kiteIndex, tuning) =>
        signedLimiterActivation(
          ((frame.pitch_energy_integrator[kiteIndex] ?? 0) * 180) / Math.PI,
          tuningNumber(tuning, "tecs_pitch_integrator_limit_deg", 22),
          "positive pitch integrator limit",
          "negative pitch integrator limit",
          "deg"
        )
    },
    {
      label: "Thrust antiwindup hold",
      color: "#c28dff",
      modes: upperLowerModes("#c28dff", "#72d7ff"),
      active: (frame, kiteIndex, tuning) => {
        const motor = frame.motor_torque[kiteIndex] ?? Number.NaN;
        const motorMax = tuningNumber(tuning, "motor_torque_max_nm", 45.6);
        const error = frame.kinetic_energy_error_specific[kiteIndex] ?? 0;
        if (nearUpperLimit(motor, motorMax, 0.02) && error > 0) {
          return limiterActivation("upper", "upper antiwindup hold", `motor saturated high while kinetic-energy error is ${error.toFixed(2)} m²/s²`);
        }
        if (nearLowerLimit(motor, 0, 0.02) && error < 0) {
          return limiterActivation("lower", "lower antiwindup hold", `motor saturated low while kinetic-energy error is ${error.toFixed(2)} m²/s²`);
        }
        return null;
      }
    },
    {
      label: "Pitch antiwindup hold",
      color: "#c28dff",
      modes: upperLowerModes("#c28dff", "#72d7ff"),
      active: (frame, kiteIndex, tuning) => {
        const pitchRef = frame.pitch_ref_deg[kiteIndex] ?? Number.NaN;
        const limit = pitchReferenceLimitDeg(tuning);
        const error = frame.energy_balance_error_specific[kiteIndex] ?? 0;
        if (pitchRef >= limit - 0.02 && error > 0) {
          return limiterActivation("upper", "upper pitch antiwindup hold", `pitch ref high while balance error is ${error.toFixed(2)} m²/s²`);
        }
        if (pitchRef <= -limit + 0.02 && error < 0) {
          return limiterActivation("lower", "lower pitch antiwindup hold", `pitch ref low while balance error is ${error.toFixed(2)} m²/s²`);
        }
        return null;
      }
    },
    {
      label: "Altitude error clamp",
      color: "#f4a261",
      modes: positiveNegativeModes("#f4a261", "#72d7ff"),
      active: (frame, kiteIndex, tuning) => {
        const error =
          (frame.potential_energy_ref_specific[kiteIndex] ?? 0) -
          (frame.potential_energy_specific[kiteIndex] ?? 0);
        return signedLimiterActivation(
          error,
          tuningNumber(tuning, "tecs_altitude_error_limit_m", 25) * GRAVITY_MPS2,
          "positive altitude-energy clamp",
          "negative altitude-energy clamp",
          "m²/s²",
          0.25
        );
      }
    },
    {
      label: "Tether PE clamp",
      color: "#f4a261",
      modes: positiveNegativeModes("#f4a261", "#72d7ff"),
      active: (frame, kiteIndex, tuning) =>
        signedLimiterActivation(
          frame.potential_energy_error_specific[kiteIndex] ?? 0,
          tuningNumber(tuning, "tethered_tecs_potential_error_limit", 245),
          "positive tethered potential-energy clamp",
          "negative tethered potential-energy clamp",
          "m²/s²",
          0.25
        )
    },
    {
      label: "Speed target clamp",
      color: "#f4a261",
      modes: upperLowerModes("#f4a261", "#72d7ff"),
      active: (frame, kiteIndex, tuning) => {
        const speedTarget = frame.speed_target[kiteIndex] ?? Number.NaN;
        return (
          upperLimiterActivation(speedTarget, tuningNumber(tuning, "speed_max_mps", 35), "speed target at max clamp", "m/s") ??
          lowerLimiterActivation(speedTarget, tuningNumber(tuning, "speed_min_mps", 30), "speed target at min clamp", "m/s")
        );
      }
    },
    {
      label: "Alpha protection",
      color: "#ff4d6d",
      modes: upperLowerModes("#ff4d6d", "#72d7ff"),
      active: (frame, kiteIndex, tuning) => {
        const alpha = frame.alpha_deg[kiteIndex] ?? Number.NaN;
        if (alpha >= tuningNumber(tuning, "alpha_protection_max_deg", 10)) {
          return limiterActivation("upper", "high-alpha protection", `alpha = ${alpha.toFixed(2)} deg`);
        }
        if (alpha <= tuningNumber(tuning, "alpha_protection_min_deg", -8)) {
          return limiterActivation("lower", "low-alpha protection", `alpha = ${alpha.toFixed(2)} deg`);
        }
        return null;
      }
    },
    {
      label: "Rotor soft limit",
      color: "#66b8ff",
      active: (frame, kiteIndex, tuning) =>
        upperLimiterActivation(
          frame.rotor_speed[kiteIndex] ?? 0,
          tuningNumber(tuning, "rotor_speed_soft_limit_radps", 800),
          "rotor speed at soft limit",
          "rad/s",
          0.5
        )
    },
    {
      label: "Rotor hard limit",
      color: "#ff4d6d",
      active: (frame, kiteIndex, tuning) =>
        upperLimiterActivation(
          frame.rotor_speed[kiteIndex] ?? 0,
          tuningNumber(tuning, "rotor_speed_hard_limit_radps", 900),
          "rotor speed at hard limit",
          "rad/s",
          0.5
        )
    },
    {
      label: "Full-throttle mode",
      color: "#89f0ff",
      active: () =>
        simpleLimiterActivation(
          activeSummaryRequest?.longitudinal_mode === "max_throttle_altitude_pitch",
          "full-throttle experiment mode"
        )
    }
  ];
}

function limiterLaneModes(lane: LimiterLaneDefinition): LimiterLaneMode[] {
  return lane.modes ?? [{ key: "active", label: "active", color: lane.color }];
}

function limiterLaneY(laneIndex: number, kiteIndex: number, kiteCount: number): number {
  return laneIndex * (kiteCount + 0.65) + kiteIndex;
}

function limiterHoverHtml(
  lane: LimiterLaneDefinition,
  activation: LimiterActivation,
  kiteIndex: number
): string {
  const detail = activation.detail
    ? `<br><span style="color:#9fb9cc">${escapeHtml(activation.detail)}</span>`
    : "";
  return [
    `<b>${escapeHtml(lane.label)}</b>`,
    `<br><span style="color:#cfe8f8">Kite ${kiteIndex + 1}</span>`,
    `<br><span style="color:#9ff0d0">${escapeHtml(activation.label)}</span>`,
    detail
  ].join("");
}

function buildLimiterTimelineGroup(kiteCount: number): PlotGroupDefinition {
  const tuning = activeSummaryRequest?.controller_tuning ?? simulationDefaults?.controller_tuning ?? {};
  const lanes = limiterLaneDefinitions();
  const traces: PlotTraceDefinition[] = [];
  for (let kiteIndex = 0; kiteIndex < kiteCount; kiteIndex += 1) {
    for (let laneIndex = 0; laneIndex < lanes.length; laneIndex += 1) {
      const lane = lanes[laneIndex];
      const laneY = limiterLaneY(laneIndex, kiteIndex, kiteCount);
      for (const mode of limiterLaneModes(lane)) {
        traces.push({
          name: `Kite ${kiteIndex + 1} ${lane.label} ${mode.label}`,
          color: hexToRgba(mode.color, 0.94),
          signalKey: `limiter:${lane.label}:${mode.key}:kite-${kiteIndex}`,
          legendName: `${lane.label} ${mode.label}`,
          kiteIndex,
          alwaysVisible: true,
          width: 7,
          shape: "hv",
          hoverText: (frame) => {
            const activation = lane.active(frame, kiteIndex, tuning);
            return activation?.mode === mode.key
              ? limiterHoverHtml(lane, activation, kiteIndex)
              : null;
          },
          value: (frame) => {
            const activation = lane.active(frame, kiteIndex, tuning);
            return activation?.mode === mode.key ? laneY : null;
          }
        });
      }
    }
  }
  const yTickVals: number[] = [];
  const yTickText: string[] = [];
  for (let laneIndex = 0; laneIndex < lanes.length; laneIndex += 1) {
    for (let kiteIndex = 0; kiteIndex < kiteCount; kiteIndex += 1) {
      yTickVals.push(limiterLaneY(laneIndex, kiteIndex, kiteCount));
      yTickText.push(`K${kiteIndex + 1} · ${lanes[laneIndex].label}`);
    }
  }
  const laneSpan = Math.max(1, lanes.length * (kiteCount + 0.65) - 0.65);
  return {
    title: "Limiter / Mode Timeline",
    yTitle: "",
    traces,
    height: Math.max(LIMITER_TIMELINE_HEIGHT_PX, Math.min(820, yTickVals.length * 20)),
    yTickVals,
    yTickText,
    yRange: [-0.7, laneSpan - 0.3],
    showSignalLegend: false
  };
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
      title: "Controller / Limiters & Mode Switches",
      description:
        "Compact timeline of controller regions that are hard clamps or mode changes: reference clamps, actuator saturation, integrator limits, energy clamps, alpha protection, rotor limits, and full-throttle experiment mode.",
      groups: [buildLimiterTimelineGroup(kiteCount)],
      maxColumns: 1,
      showKiteControls: false
    },
    {
      title: "Controller / 1. Lateral Inner Loop",
      description:
        "Innermost lateral channel first. In direct-rabbit mode, body-frame rabbit bearing commands roll directly. In curvature modes, converted path curvature commands roll. The aileron then closes roll with body-rate damping; rudder is a sideslip/yaw-damper loop using beta and body z-rate.",
      groups: [
        buildRollCommandGroup(kiteCount),
        buildRollReferenceBreakdownGroup(kiteCount),
        buildPerKiteGroup(
          kiteCount,
          "Body Rate p (rad/s)",
          "rad/s",
          (frame, kiteIndex) => frame.body_omega_b[kiteIndex]?.[0] ?? 0
        ),
        buildAileronCommandGroup(kiteCount),
        buildAileronBreakdownGroup(kiteCount),
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
        buildRudderBreakdownGroup(kiteCount),
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
        "Outer lateral path loop. Phase coordination biases rabbit radius and speed scheduling; desired speed schedules rabbit lead distance before the target point feeds the selected guidance mode.",
      groups: [
        buildRabbitBearingGroup(kiteCount),
        buildRabbitVectorGroup(kiteCount),
        buildRabbitDistanceGroup(kiteCount),
        buildOrbitRadiusGroup(kiteCount),
        buildPhaseErrorGroup(kiteCount),
        buildSpeedTargetGroup(kiteCount)
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
        buildMotorTorqueBreakdownGroup(kiteCount),
        buildAltitudeCommandGroup(kiteCount),
        buildPerKiteGroup(
          kiteCount,
          "Specific Potential Energy Desired vs Actual (m²/s²)",
          "m²/s²",
          (frame, kiteIndex) => frame.potential_energy_specific[kiteIndex] ?? 0,
          (frame, kiteIndex) => frame.potential_energy_ref_specific[kiteIndex] ?? 0
        ),
        buildTecsPitchCommandGroup(kiteCount),
        buildPitchReferenceBreakdownGroup(kiteCount),
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
        "The TECS energy-balance output is a desired pitch angle. The elevator closes pitch with q damping. Flap is held at trim so the elevator loop can be tuned in isolation.",
      groups: [
        buildPitchCommandGroup(kiteCount),
        buildPerKiteGroup(
          kiteCount,
          "Body Rate q (rad/s)",
          "rad/s",
          (frame, kiteIndex) => frame.body_omega_b[kiteIndex]?.[1] ?? 0
        ),
        buildElevatorCommandGroup(kiteCount),
        buildElevatorBreakdownGroup(kiteCount),
        buildPerKiteGroup(
          kiteCount,
          "Flap Command vs Applied (deg)",
          "deg",
          (frame, kiteIndex) => frame.flap_applied_deg[kiteIndex] ?? 0,
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
          "Rotor Speed (rad/s)",
          "rad/s",
          (frame, kiteIndex) => frame.rotor_speed[kiteIndex] ?? 0,
          undefined,
          [
            {
              name: "Rotor Fit Soft Limit",
              color: LIMIT_COLOR,
              dash: "dot",
              value: () => 500
            },
            {
              name: "Rotor Fit Hard Limit",
              color: LIMIT_COLOR,
              dash: "dash",
              value: () => 600
            }
          ]
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
            name: "Rudder Aero",
            dash: "dashdot",
            width: 2.0,
            alpha: 0.9,
            value: (frame, kiteIndex) => bodyComponent(frame.rudder_force_b, kiteIndex, 1)
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
            name: "Rudder Aero",
            dash: "dashdot",
            width: 2.1,
            alpha: 0.92,
            value: (frame, kiteIndex) => bodyComponent(frame.rudder_moment_b, kiteIndex, 2)
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
        "Aero coefficient breakdowns from the actual model implementation. Nominal/rate/surface labels are contribution buckets, not local linear stability derivatives.",
      groups: [
        buildPerKiteBreakdownGroup(kiteCount, "C_L Terms", "-", [
          { name: "Total", width: 3, alpha: 0.96, value: (frame, kiteIndex) => frame.cl_total[kiteIndex] ?? 0 },
          { name: "Nominal", dash: "dash", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.cl_0_term[kiteIndex] ?? 0 },
          { name: "Rate", dash: "dot", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.cl_alpha_term[kiteIndex] ?? 0 },
          { name: "C_Lδe", dash: "dashdot", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.cl_elevator_term[kiteIndex] ?? 0 },
          { name: "C_Lδf", dash: "longdash", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.cl_flap_term[kiteIndex] ?? 0 }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "C_D Terms", "-", [
          { name: "Total", width: 3, alpha: 0.96, value: (frame, kiteIndex) => frame.cd_total[kiteIndex] ?? 0 },
          { name: "Nominal", dash: "dash", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.cd_0_term[kiteIndex] ?? 0 },
          { name: "Rate", dash: "dot", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.cd_induced_term[kiteIndex] ?? 0 },
          { name: "Surface", dash: "dashdot", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.cd_surface_term[kiteIndex] ?? 0 }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "C_Yw Terms", "-", [
          { name: "Total", width: 3, alpha: 0.96, value: (frame, kiteIndex) => frame.cy_total[kiteIndex] ?? 0 },
          { name: "Nominal + Rate", dash: "dash", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.cy_beta_term[kiteIndex] ?? 0 },
          { name: "C_Ywδr", dash: "dot", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.cy_rudder_term[kiteIndex] ?? 0 }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "C_l Terms", "-", [
          { name: "Total", width: 3, alpha: 0.96, value: (frame, kiteIndex) => frame.roll_coeff_total[kiteIndex] ?? 0 },
          { name: "Nominal", dash: "dash", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.roll_beta_term[kiteIndex] ?? 0 },
          { name: "Rate", dash: "dot", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.roll_p_term[kiteIndex] ?? 0 },
          { name: "C_lδa", dash: "longdash", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.roll_aileron_term[kiteIndex] ?? 0 }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "C_m Terms", "-", [
          { name: "Total", width: 3, alpha: 0.96, value: (frame, kiteIndex) => frame.pitch_coeff_total[kiteIndex] ?? 0 },
          { name: "Nominal", dash: "dash", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.pitch_0_term[kiteIndex] ?? 0 },
          { name: "Rate", dash: "dashdot", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.pitch_q_term[kiteIndex] ?? 0 },
          { name: "C_mδe", dash: "longdash", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.pitch_elevator_term[kiteIndex] ?? 0 },
          { name: "C_mδf", dash: "dash", width: 1.6, alpha: 0.58, value: (frame, kiteIndex) => frame.pitch_flap_term[kiteIndex] ?? 0 }
        ]),
        buildPerKiteBreakdownGroup(kiteCount, "C_n Terms", "-", [
          { name: "Total", width: 3, alpha: 0.96, value: (frame, kiteIndex) => frame.yaw_coeff_total[kiteIndex] ?? 0 },
          { name: "Nominal", dash: "dash", width: 1.9, alpha: 0.82, value: (frame, kiteIndex) => frame.yaw_beta_term[kiteIndex] ?? 0 },
          { name: "Rate", dash: "dashdot", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.yaw_r_term[kiteIndex] ?? 0 },
          { name: "C_nδr", dash: "longdash", width: 1.8, alpha: 0.76, value: (frame, kiteIndex) => frame.yaw_rudder_term[kiteIndex] ?? 0 }
        ])
      ],
      maxColumns: 2
    },
    {
      title: "Physics / Energy & Consistency",
      description:
        "Whole-system energy diagnostics and work accounting for sanity-checking the integration.",
      groups: buildEnergyGroups()
    }
  ];
}

function formatProgressSummary(
  request: {
    preset: string;
    swarm_disk_altitude_m: number | null;
    swarm_disk_radius_m: number | null;
    swarm_aircraft_altitude_m: number | null;
    swarm_upper_tether_length_m: number | null;
    swarm_common_tether_length_m: number | null;
    phase_mode: PhaseMode;
    longitudinal_mode: LongitudinalMode;
    sim_noise_enabled: boolean;
    dryden?: DrydenConfig;
    bridle_enabled: boolean;
    dt_control: number;
    rk_abs_tol: number;
    rk_rel_tol: number;
    max_substeps: number;
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
      { label: "Disk Altitude", value: optionalMetersLabel(request.swarm_disk_altitude_m) },
      { label: "Disk Radius", value: optionalMetersLabel(request.swarm_disk_radius_m) },
      { label: "Aircraft Altitude", value: optionalMetersLabel(request.swarm_aircraft_altitude_m) },
      {
        label: "Tethers",
        value: `lower ${compactNumberInputValue(request.swarm_common_tether_length_m ?? 150)} m / upper ${compactNumberInputValue(request.swarm_upper_tether_length_m ?? 120)} m`
      },
      { label: "Phase Mode", value: request.phase_mode },
      { label: "Longitudinal", value: longitudinalModeLabel(request.longitudinal_mode) },
      { label: "Bridle", value: request.bridle_enabled ? "Enabled" : "CG attach" },
      { label: "Sim Noise", value: request.sim_noise_enabled ? "Dryden gusts" : "Off" },
      {
        label: "Dryden",
        value:
          request.sim_noise_enabled && request.dryden
            ? `intensity ${compactNumberInputValue(request.dryden.intensity_scale)}, length ${compactNumberInputValue(request.dryden.length_scale)}`
            : "disabled"
      },
      {
        label: "RK abs / rel tol",
        value: `${toleranceLabel(request.rk_abs_tol)} / ${toleranceLabel(request.rk_rel_tol)}`
      },
      {
        label: "Sample Period / Rate",
        value: `${compactNumberInputValue(request.dt_control)} s / ${(1 / request.dt_control).toFixed(1)} Hz`
      },
      { label: "Substep Budget", value: String(request.max_substeps) },
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
      {
        label: "Longitudinal",
        value: longitudinalModeLabel(activeSummaryRequest?.longitudinal_mode ?? "total_energy")
      },
      { label: "Frames", value: `${receivedFrames} received / ${renderedFrames} rendered` },
      {
        label: "Accepted / Rejected",
        value: `${summary.accepted_steps} / ${summary.rejected_steps}`
      },
      {
        label: "Max Phase Error",
        value: `${(summary.max_phase_error * RAD_TO_DEG).toFixed(2)} deg`
      },
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
    { value: "disk_center", label: "Disk Center" },
    { value: "y_joint", label: "Y Joint" }
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
    transparent: true,
    opacity: 1.0,
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
  if (selection === "y_joint") {
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

function airParticleVolumeTop(frame?: ApiFrame): number {
  const diskCenter = frame?.control_ring_center_n;
  if (!diskCenter) {
    return GRID_HALF_EXTENT;
  }
  const diskAltitude = -diskCenter[2];
  return Math.max(GRID_HALF_EXTENT, diskAltitude + AIR_PARTICLE_DISK_CLEARANCE_M);
}

function randomGridVolumePosition(frame?: ApiFrame): THREE.Vector3 {
  const top = airParticleVolumeTop(frame);
  return toThree([
    randomCentered(GRID_HALF_EXTENT),
    randomCentered(GRID_HALF_EXTENT),
    -Math.random() * top
  ]);
}

function isOutsideGridVolume(position: THREE.Vector3, frame?: ApiFrame): boolean {
  return (
    Math.abs(position.x) > GRID_HALF_EXTENT ||
    Math.abs(position.y) > GRID_HALF_EXTENT ||
    position.z < 0 ||
    position.z > airParticleVolumeTop(frame)
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

function airflowVelocity(frame: ApiFrame, positionThree?: THREE.Vector3): THREE.Vector3 {
  const meanGustN = meanVector(frame.kite_gust_n);
  const shearFactor = positionThree ? windShearFactorAtHeight(positionThree.z) : 1;
  return toThreeVector([
    frame.clean_wind_n[0] * shearFactor + meanGustN[0],
    frame.clean_wind_n[1] * shearFactor + meanGustN[1],
    frame.clean_wind_n[2] * shearFactor + meanGustN[2]
  ]);
}

function kiteWindVelocityAtPosition(
  frame: ApiFrame,
  kiteIndex: number,
  positionThree: THREE.Vector3
): THREE.Vector3 {
  const gustN = frame.kite_gust_n[kiteIndex] ?? [0, 0, 0];
  const shearFactor = windShearFactorAtHeight(positionThree.z);
  return toThreeVector([
    frame.clean_wind_n[0] * shearFactor + gustN[0],
    frame.clean_wind_n[1] * shearFactor + gustN[1],
    frame.clean_wind_n[2] * shearFactor + gustN[2]
  ]);
}

function wingtipTrailColor(kiteIndex: number, side: "left" | "right"): THREE.Color {
  const palette = side === "left" ? WINGTIP_LEFT_COLORS : WINGTIP_RIGHT_COLORS;
  return new THREE.Color(palette[kiteIndex % palette.length]);
}

function gustMagnitude(frame: ApiFrame): number {
  const gustN = meanVector(frame.kite_gust_n);
  return Math.hypot(gustN[0], gustN[1], gustN[2]);
}

function initializeAmbientParticles(frame: ApiFrame, velocity: THREE.Vector3): void {
  const flowMagnitude = velocity.length();
  ambientParticleCloud.visible = airParticlesVisible() && flowMagnitude > 1.0e-4;
  if (!ambientParticleCloud.visible) {
    return;
  }

  const color = ambientAirColor.clone().lerp(
    ambientAirColorHigh,
    Math.min(1, flowMagnitude / 12.0)
  );

  for (let index = 0; index < AIRFLOW_AMBIENT_PARTICLE_COUNT; index += 1) {
    const state = ambientParticleStates[index];
    state.position.copy(randomGridVolumePosition(frame));
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
  state.position.copy(randomGridVolumePosition(frame));
  state.age = 0;
  state.life = (2.2 * GRID_SIZE) / flowMagnitude * (0.8 + 0.4 * Math.random());
  state.drift = 0.35 + 0.65 * Math.random();
}

function updateAmbientParticles(dtSimSeconds: number, frame: ApiFrame, velocity: THREE.Vector3): void {
  const flowMagnitude = velocity.length();
  ambientParticleCloud.visible = airParticlesVisible() && flowMagnitude > 1.0e-4;
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

    if (isOutsideGridVolume(state.position, frame)) {
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
  state.position.copy(randomGridVolumePosition(frame));
  state.age = 0;
  state.life = (1.6 * GRID_SIZE) / Math.max(1.0, velocity.length()) * (0.75 + 0.4 * Math.random());
  state.drift = 0.35 + 0.65 * Math.random();
}

function initializeGustParticles(frame: ApiFrame, velocity: THREE.Vector3): void {
  const gustStrength = gustMagnitude(frame);
  gustParticleCloud.visible = airParticlesVisible() && gustStrength > 1.0e-3;
  if (!gustParticleCloud.visible) {
    return;
  }

  const color = gustParticleColor(gustStrength);
  for (let index = 0; index < AIRFLOW_GUST_PARTICLE_COUNT; index += 1) {
    const state = gustParticleStates[index];
    state.position.copy(randomGridVolumePosition(frame));
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
  gustParticleCloud.visible = airParticlesVisible() && gustStrength > 1.0e-3;
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

    if (isOutsideGridVolume(state.position, frame)) {
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
  color: THREE.Color,
  velocity: THREE.Vector3
): void {
  const index = nextWingtipTrailParticleIndex;
  nextWingtipTrailParticleIndex = (nextWingtipTrailParticleIndex + 1) % WINGTIP_TRAIL_PARTICLE_COUNT;
  const state = wingtipTrailStates[index];
  state.position.copy(position);
  state.velocity.copy(velocity);
  state.age = 0;
  state.life = WINGTIP_TRAIL_LIFETIME_S;
  state.active = true;
  setWingtipTrailEntry(index, state.position, color, 1);
}

function updateWingtipTrailParticles(dtSimSeconds: number, frame: ApiFrame): void {
  if (!wingtipTrailsVisible()) {
    wingtipTrailCloud.visible = false;
    return;
  }
  const currentAdvectionVelocity = airflowVelocity(frame, airflowCenter(frame));
  const useCapturedVelocity = wingtipConvectionEnabled();
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

    const velocity = useCapturedVelocity ? state.velocity : currentAdvectionVelocity;
    state.position.addScaledVector(velocity, dtSimSeconds);
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
    emitWingtipTrailParticle(
      wingtips.left,
      wingtipTrailColor(kiteIndex, "left"),
      kiteWindVelocityAtPosition(frame, kiteIndex, wingtips.left)
    );
    emitWingtipTrailParticle(
      wingtips.right,
      wingtipTrailColor(kiteIndex, "right"),
      kiteWindVelocityAtPosition(frame, kiteIndex, wingtips.right)
    );
    hasActiveParticles = true;
  }

  wingtipTrailCloud.visible = hasActiveParticles;
  (wingtipTrailGeometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;
  (wingtipTrailGeometry.attributes.alpha as THREE.BufferAttribute).needsUpdate = true;
  (wingtipTrailGeometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
}

function advanceAirflowParticlesToFrame(frame: ApiFrame): void {
  const velocity = airflowVelocity(frame, airflowCenter(frame));
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

function recordTetherTension(tensionN: number): void {
  if (!Number.isFinite(tensionN)) {
    return;
  }
  observedTetherTensionMin = Math.min(observedTetherTensionMin, tensionN);
  observedTetherTensionMax = Math.max(observedTetherTensionMax, tensionN);
}

function recordFrameTetherTensions(frame: ApiFrame): void {
  frame.common_tether_tensions.forEach(recordTetherTension);
  frame.upper_tether_tensions.forEach((kiteTensions) => {
    kiteTensions.forEach(recordTetherTension);
  });
}

function tensionToColor(tensionN: number): THREE.Color {
  if (!Number.isFinite(tensionN) || tensionN <= tetherSlackThresholdN()) {
    return tensionColorSlack.clone();
  }
  const { min, max } = tetherTensionColorRange();
  const span = Math.max(1.0e-9, max - min);
  const normalized = THREE.MathUtils.clamp((tensionN - min) / span, 0, 1);
  const shaped = Math.pow(normalized, 0.85);
  if (shaped <= 0.55) {
    return tensionColorLow.clone().lerp(tensionColorMid, shaped / 0.55);
  }
  return tensionColorMid.clone().lerp(tensionColorHigh, (shaped - 0.55) / 0.45);
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

function makeFadedSceneLine(color: number): THREE.Line {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uColor: { value: new THREE.Color(color) }
    },
    vertexShader: `
      attribute float lineAlpha;
      varying float vAlpha;

      void main() {
        vAlpha = lineAlpha;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      varying float vAlpha;

      void main() {
        gl_FragColor = vec4(uColor, vAlpha);
      }
    `,
    transparent: true,
    depthWrite: false
  });
  return new THREE.Line(new THREE.BufferGeometry(), material);
}

function updateLine(
  line: THREE.Line,
  start: THREE.Vector3,
  end: THREE.Vector3,
  color: THREE.Color,
  opacity = 0.6,
  enabled = true
): void {
  if (!enabled || start.distanceToSquared(end) < 1.0e-10) {
    line.visible = false;
    return;
  }
  line.visible = true;
  (line.geometry as THREE.BufferGeometry).setFromPoints([start, end]);
  const material = line.material as THREE.LineBasicMaterial;
  material.color.copy(color);
  material.opacity = opacity;
}

function updateFadedLine(
  line: THREE.Line,
  start: THREE.Vector3,
  end: THREE.Vector3,
  color: THREE.Color,
  startOpacity: number,
  endOpacity: number,
  enabled = true,
  segments = 18
): void {
  if (!enabled || start.distanceToSquared(end) < 1.0e-10) {
    line.visible = false;
    return;
  }

  line.visible = true;
  const pointCount = Math.max(2, segments + 1);
  const positions = new Float32Array(pointCount * 3);
  const alphas = new Float32Array(pointCount);
  for (let index = 0; index < pointCount; index += 1) {
    const t = index / (pointCount - 1);
    const point = start.clone().lerp(end, t);
    positions[index * 3] = point.x;
    positions[index * 3 + 1] = point.y;
    positions[index * 3 + 2] = point.z;
    alphas[index] = startOpacity + t * (endOpacity - startOpacity);
  }

  const geometry = line.geometry as THREE.BufferGeometry;
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("lineAlpha", new THREE.BufferAttribute(alphas, 1));
  geometry.computeBoundingSphere();
  const material = line.material as THREE.ShaderMaterial;
  (material.uniforms.uColor.value as THREE.Color).copy(color);
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
  mesh.scale.setScalar(radius * tetherNodeScale());
  mesh.userData.nodeRadius = radius;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  setMaterialColor(mesh.material, tetherNodeColor, 0.12);
  return mesh;
}

function makeProjectedPhaseMesh(): THREE.Mesh {
  return new THREE.Mesh(
    new THREE.SphereGeometry(1.75, 14, 14),
    new THREE.MeshStandardMaterial({
      color: controlProjectedPhaseColor,
      emissive: controlProjectedPhaseColor,
      emissiveIntensity: 0.32,
      roughness: 0.2,
      metalness: 0.06
    })
  );
}

function makeLookaheadOnDiskMesh(): THREE.Mesh {
  return new THREE.Mesh(
    new THREE.SphereGeometry(1.45, 14, 14),
    new THREE.MeshStandardMaterial({
      color: controlLookaheadOnDiskColor,
      emissive: controlLookaheadOnDiskColor,
      emissiveIntensity: 0.24,
      transparent: true,
      opacity: 0.82,
      roughness: 0.24,
      metalness: 0.04
    })
  );
}

function makeRabbitMesh(): THREE.Mesh {
  return new THREE.Mesh(
    new THREE.SphereGeometry(2.4, 12, 12),
    new THREE.MeshStandardMaterial({
      color: controlLookaheadColor,
      emissive: controlLookaheadColor,
      emissiveIntensity: 0.32
    })
  );
}

function makePhaseSlotMesh(): THREE.Mesh {
  return new THREE.Mesh(
    new THREE.SphereGeometry(1.25, 12, 12),
    new THREE.MeshStandardMaterial({
      color: controlPhaseSlotColor,
      emissive: controlPhaseSlotColor,
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
  const isSlack = !Number.isFinite(tensionN) || tensionN <= tetherSlackThresholdN();
  const color = tensionToColor(tensionN);
  const material = mesh.material as THREE.MeshStandardMaterial;
  material.color.copy(color);
  material.emissive.copy(color);
  material.emissiveIntensity = isSlack ? 0.04 : 0.22;
  material.opacity = isSlack ? 0.46 : 1.0;
}

function updateNodeMesh(mesh: THREE.Mesh, point: [number, number, number]): void {
  mesh.visible = tetherNodesVisible();
  mesh.position.copy(toThree(point));
  mesh.scale.setScalar((mesh.userData.nodeRadius as number) * tetherNodeScale());
  setMaterialColor(mesh.material, tetherNodeColor, 0.12);
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
    updateNodeMesh(nodeMeshes[index], safePoints[index + 1]);
  }
}

function updateControlRing(frame: ApiFrame, layer: ControlFeatureLayerMeshes): void {
  const kiteCount = frame.kite_positions_n.length;
  const ringVisible = kiteCount > 0 && frame.control_ring_radius > 1.0e-6;
  const layerVisible = controlFeatureLayerVisible(layer.mode);
  layer.controlRingLine.visible = ringVisible && controlDiskEnabledInput.checked && layerVisible;

  if (!ringVisible) {
    layer.phaseSlotMeshes.forEach((mesh) => {
      mesh.visible = false;
    });
    return;
  }

  const radius = frame.control_ring_radius;
  const ringDown = displayControlRingDown(frame, layer.mode);
  const ringPoints = Array.from({ length: CONTROL_RING_SEGMENTS }, (_, index) => {
    const theta = (2 * Math.PI * index) / CONTROL_RING_SEGMENTS;
    return controlRingPoint(frame, theta, radius, ringDown);
  });
  (layer.controlRingLine.geometry as THREE.BufferGeometry).setFromPoints(ringPoints);

  const showAdaptiveSlots = activeSummaryRequest?.phase_mode === "adaptive";
  ensureMeshCount(showAdaptiveSlots ? kiteCount : 0, layer.phaseSlotMeshes, makePhaseSlotMesh);
  if (showAdaptiveSlots) {
    for (let index = 0; index < kiteCount; index += 1) {
      layer.phaseSlotMeshes[index].visible = controlFeaturesVisible() && layerVisible;
      layer.phaseSlotMeshes[index].position.copy(phaseSlotPosition(frame, index, layer.mode));
      setMaterialColor(
        layer.phaseSlotMeshes[index].material,
        controlPhaseSlotColor,
        0.18
      );
    }
  }
}

function updateControlRings(frame: ApiFrame): void {
  controlFeatureLayers.forEach((layer) => updateControlRing(frame, layer));
}

function averageKiteDown(frame: ApiFrame): number {
  if (frame.kite_positions_n.length === 0) {
    return frame.control_ring_center_n[2];
  }
  return frame.kite_positions_n.reduce((sum, position) => sum + position[2], 0) /
    frame.kite_positions_n.length;
}

function displayControlRingDown(frame: ApiFrame, layer: ControlFeatureAltitudeLayer): number {
  return layer === "aircraft" ? averageKiteDown(frame) : frame.control_ring_center_n[2];
}

function displayFeatureDown(
  frame: ApiFrame,
  kiteIndex: number,
  layer: ControlFeatureAltitudeLayer
): number | null {
  if (layer !== "aircraft") {
    return null;
  }
  return frame.kite_positions_n[kiteIndex]?.[2] ?? frame.control_ring_center_n[2];
}

function toDisplayDiskFeature(
  frame: ApiFrame,
  kiteIndex: number,
  point: [number, number, number],
  layer: ControlFeatureAltitudeLayer
): THREE.Vector3 {
  const featureDown = displayFeatureDown(frame, kiteIndex, layer);
  return toThree([point[0], point[1], featureDown ?? point[2]]);
}

function controlRingPoint(
  frame: ApiFrame,
  theta: number,
  radius = frame.control_ring_radius,
  down = frame.control_ring_center_n[2]
): THREE.Vector3 {
  const center = frame.control_ring_center_n;
  return toThree([
    center[0] + radius * Math.cos(theta),
    center[1] + radius * Math.sin(theta),
    down
  ]);
}

function projectedPhasePosition(
  frame: ApiFrame,
  kiteIndex: number,
  layer: ControlFeatureAltitudeLayer
): THREE.Vector3 {
  return toDisplayDiskFeature(
    frame,
    kiteIndex,
    frame.phase_projected_n[kiteIndex] ?? frame.kite_positions_n[kiteIndex],
    layer
  );
}

function closestDiskPosition(
  frame: ApiFrame,
  kiteIndex: number,
  layer: ControlFeatureAltitudeLayer
): THREE.Vector3 {
  return toDisplayDiskFeature(
    frame,
    kiteIndex,
    frame.closest_disk_n[kiteIndex] ?? frame.kite_positions_n[kiteIndex],
    layer
  );
}

function diskPlaneProjectionPosition(
  frame: ApiFrame,
  kiteIndex: number,
  layer: ControlFeatureAltitudeLayer
): THREE.Vector3 {
  return toDisplayDiskFeature(
    frame,
    kiteIndex,
    frame.disk_plane_projection_n[kiteIndex] ?? frame.kite_positions_n[kiteIndex],
    layer
  );
}

function lookaheadOnDiskPosition(
  frame: ApiFrame,
  kiteIndex: number,
  layer: ControlFeatureAltitudeLayer
): THREE.Vector3 {
  return toDisplayDiskFeature(
    frame,
    kiteIndex,
    frame.lookahead_on_disk_n[kiteIndex] ??
      frame.rabbit_targets_n[kiteIndex] ??
      frame.kite_positions_n[kiteIndex],
    layer
  );
}

function rabbitTargetPosition(
  frame: ApiFrame,
  kiteIndex: number,
  layer: ControlFeatureAltitudeLayer
): THREE.Vector3 {
  return toDisplayDiskFeature(
    frame,
    kiteIndex,
    frame.rabbit_targets_n[kiteIndex] ?? frame.kite_positions_n[kiteIndex],
    layer
  );
}

function phaseSlotPosition(
  frame: ApiFrame,
  kiteIndex: number,
  layer: ControlFeatureAltitudeLayer
): THREE.Vector3 {
  return toDisplayDiskFeature(
    frame,
    kiteIndex,
    frame.phase_slot_n[kiteIndex] ?? frame.kite_positions_n[kiteIndex],
    layer
  );
}

interface ControlLabelSpec {
  key: string;
  text: string;
  color: string;
  position: THREE.Vector3;
}

function labelScreenPosition(worldPosition: THREE.Vector3): { x: number; y: number } | null {
  const projected = worldPosition.clone().project(camera);
  if (
    projected.z < -1 ||
    projected.z > 1 ||
    projected.x < -1.1 ||
    projected.x > 1.1 ||
    projected.y < -1.1 ||
    projected.y > 1.1
  ) {
    return null;
  }
  return {
    x: (projected.x * 0.5 + 0.5) * viewport.clientWidth,
    y: (-projected.y * 0.5 + 0.5) * viewport.clientHeight
  };
}

function setControlLabel(spec: ControlLabelSpec): void {
  let label = controlLabelNodes.get(spec.key);
  if (!label) {
    label = document.createElement("div");
    label.className = "control-label";
    controlLabelLayer.append(label);
    controlLabelNodes.set(spec.key, label);
  }
  label.textContent = spec.text;
  label.style.setProperty("--label-color", spec.color);
  const screen = labelScreenPosition(spec.position);
  if (!screen) {
    label.style.display = "none";
    return;
  }
  label.style.display = "block";
  label.style.transform = `translate3d(${screen.x}px, ${screen.y}px, 0) translate(-50%, calc(-100% - 8px))`;
}

function updateControlLabels(): void {
  const frame = lastRenderedFrame;
  const enabled =
    controlLabelsEnabledInput.checked &&
    controlFeaturesVisible() &&
    (controlFeaturesAtTargetAltitude() || controlFeaturesAtAircraftAltitude()) &&
    frame &&
    frame.kite_positions_n.length > 0;
  controlLabelLayer.classList.toggle("visible", Boolean(enabled));
  if (!enabled || !frame) {
    controlLabelNodes.forEach((label) => {
      label.style.display = "none";
    });
    return;
  }

  const specs: ControlLabelSpec[] = [];
  const showAdaptiveSlots = activeSummaryRequest?.phase_mode === "adaptive";
  const labelLayer: ControlFeatureAltitudeLayer =
    controlFeaturesAtTargetAltitude() ? "target" : "aircraft";

  frame.kite_positions_n.forEach((_position, kiteIndex) => {
    const kiteLabel = `K${kiteIndex + 1}`;
    specs.push({
      key: `lookahead-${kiteIndex}`,
      text: `${kiteLabel} lookahead`,
      color: CONTROL_VIS_COLORS.lookahead,
      position: rabbitTargetPosition(frame, kiteIndex, labelLayer)
    });
    specs.push({
      key: `lookahead-disk-${kiteIndex}`,
      text: `${kiteLabel} lookahead on disk`,
      color: CONTROL_VIS_COLORS.lookaheadOnDisk,
      position: lookaheadOnDiskPosition(frame, kiteIndex, labelLayer)
    });
    specs.push({
      key: `projected-${kiteIndex}`,
      text: `${kiteLabel} projected phase`,
      color: CONTROL_VIS_COLORS.projectedPhase,
      position: projectedPhasePosition(frame, kiteIndex, labelLayer)
    });
    if (showAdaptiveSlots) {
      specs.push({
        key: `slot-${kiteIndex}`,
        text: `${kiteLabel} phase slot`,
        color: CONTROL_VIS_COLORS.phaseSlot,
        position: phaseSlotPosition(frame, kiteIndex, labelLayer)
      });
    }
  });

  const activeKeys = new Set(specs.map((spec) => spec.key));
  controlLabelNodes.forEach((label, key) => {
    if (!activeKeys.has(key)) {
      label.style.display = "none";
    }
  });
  specs.forEach(setControlLabel);
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

    const rabbit = makeRabbitMesh();
    rabbitMeshes.push(rabbit);
    scene.add(rabbit);

    const lookaheadOnDisk = makeLookaheadOnDiskMesh();
    lookaheadOnDiskMeshes.push(lookaheadOnDisk);
    scene.add(lookaheadOnDisk);

    const projectedPhase = makeProjectedPhaseMesh();
    projectedPhaseMeshes.push(projectedPhase);
    scene.add(projectedPhase);

    const guidanceLine = makeFadedSceneLine(0xff5c74);
    guidanceLines.push(guidanceLine);
    scene.add(guidanceLine);

    const lookaheadRadialOffsetLine = makeSceneLine(0xff7a66, 0.58);
    lookaheadRadialOffsetLines.push(lookaheadRadialOffsetLine);
    scene.add(lookaheadRadialOffsetLine);

    const projectedToDiskLine = makeSceneLine(0xd6c3ff, 0.48);
    projectedToDiskLines.push(projectedToDiskLine);
    scene.add(projectedToDiskLine);

    const phaseSlotToClosestDiskLine = makeSceneLine(0xff7a66, 0.42);
    phaseSlotToClosestDiskLines.push(phaseSlotToClosestDiskLine);
    scene.add(phaseSlotToClosestDiskLine);

    const aircraftRabbit = makeRabbitMesh();
    aircraftRabbitMeshes.push(aircraftRabbit);
    scene.add(aircraftRabbit);

    const aircraftLookaheadOnDisk = makeLookaheadOnDiskMesh();
    aircraftLookaheadOnDiskMeshes.push(aircraftLookaheadOnDisk);
    scene.add(aircraftLookaheadOnDisk);

    const aircraftProjectedPhase = makeProjectedPhaseMesh();
    aircraftProjectedPhaseMeshes.push(aircraftProjectedPhase);
    scene.add(aircraftProjectedPhase);

    const aircraftGuidanceLine = makeFadedSceneLine(0xff5c74);
    aircraftGuidanceLines.push(aircraftGuidanceLine);
    scene.add(aircraftGuidanceLine);

    const aircraftLookaheadRadialOffsetLine = makeSceneLine(0xff7a66, 0.44);
    aircraftLookaheadRadialOffsetLines.push(aircraftLookaheadRadialOffsetLine);
    scene.add(aircraftLookaheadRadialOffsetLine);

    const aircraftProjectedToDiskLine = makeSceneLine(0xd6c3ff, 0.36);
    aircraftProjectedToDiskLines.push(aircraftProjectedToDiskLine);
    scene.add(aircraftProjectedToDiskLine);

    const aircraftPhaseSlotToClosestDiskLine = makeSceneLine(0xff7a66, 0.34);
    aircraftPhaseSlotToClosestDiskLines.push(aircraftPhaseSlotToClosestDiskLine);
    scene.add(aircraftPhaseSlotToClosestDiskLine);

    upperSegmentMeshes.push([]);
    upperNodeMeshes.push([]);
  }

  kiteMeshes.forEach((mesh, index) => {
    const visible = index < count;
    mesh.visible = visible;
    controlFeatureLayers.forEach((layer) => {
      const layerVisible = controlFeatureLayerVisible(layer.mode);
      const markerVisible = visible && controlFeaturesVisible() && layerVisible;
      const lineVisible = visible && controlFeatureLinesVisible() && layerVisible;
      layer.rabbitMeshes[index].visible = markerVisible;
      layer.lookaheadOnDiskMeshes[index].visible = markerVisible;
      layer.projectedPhaseMeshes[index].visible = markerVisible;
      layer.guidanceLines[index].visible = lineVisible;
      layer.lookaheadRadialOffsetLines[index].visible = lineVisible;
      layer.projectedToDiskLines[index].visible = lineVisible;
      layer.phaseSlotToClosestDiskLines[index].visible = lineVisible;
      const phaseSlotMesh = layer.phaseSlotMeshes[index];
      if (phaseSlotMesh) {
        phaseSlotMesh.visible = markerVisible;
      }
    });
    upperSegmentMeshes[index].forEach((segment) => {
      segment.visible = visible;
    });
    upperNodeMeshes[index].forEach((node) => {
      node.visible = visible && tetherNodesVisible();
    });
  });
  applyVisualizationScales();
}

function renderControlFeatureLayer(
  frame: ApiFrame,
  index: number,
  layer: ControlFeatureLayerMeshes,
  showAdaptiveSlots: boolean
): void {
  const rabbitMesh = layer.rabbitMeshes[index];
  const lookaheadOnDiskMesh = layer.lookaheadOnDiskMeshes[index];
  const projectedPhaseMesh = layer.projectedPhaseMeshes[index];
  const guidanceLine = layer.guidanceLines[index];
  const lookaheadRadialOffsetLine = layer.lookaheadRadialOffsetLines[index];
  const projectedToDiskLine = layer.projectedToDiskLines[index];
  const phaseSlotToClosestDiskLine = layer.phaseSlotToClosestDiskLines[index];
  if (
    !rabbitMesh ||
    !lookaheadOnDiskMesh ||
    !projectedPhaseMesh ||
    !guidanceLine ||
    !lookaheadRadialOffsetLine ||
    !projectedToDiskLine ||
    !phaseSlotToClosestDiskLine
  ) {
    return;
  }

  const layerVisible = controlFeatureLayerVisible(layer.mode);
  const showControlFeatures = controlFeaturesVisible() && layerVisible;
  const showControlLines = controlFeatureLinesVisible() && layerVisible;
  const rabbitPosition = rabbitTargetPosition(frame, index, layer.mode);

  rabbitMesh.position.copy(rabbitPosition);
  rabbitMesh.visible = showControlFeatures;
  setMaterialColor(rabbitMesh.material, controlLookaheadColor, 0.34);

  const lookaheadOnDisk = lookaheadOnDiskPosition(frame, index, layer.mode);
  lookaheadOnDiskMesh.position.copy(lookaheadOnDisk);
  lookaheadOnDiskMesh.visible = showControlFeatures;
  setMaterialColor(lookaheadOnDiskMesh.material, controlLookaheadOnDiskColor, 0.22);

  const projectedPhase = projectedPhasePosition(frame, index, layer.mode);
  const closestDisk = closestDiskPosition(frame, index, layer.mode);
  const diskPlaneProjection = diskPlaneProjectionPosition(frame, index, layer.mode);
  projectedPhaseMesh.position.copy(projectedPhase);
  projectedPhaseMesh.visible = showControlFeatures;
  setMaterialColor(projectedPhaseMesh.material, controlProjectedPhaseColor, 0.28);

  const phaseSlot = phaseSlotPosition(frame, index, layer.mode);

  updateFadedLine(
    guidanceLine,
    diskPlaneProjection,
    rabbitPosition,
    controlLookaheadColor,
    1.0,
    0.4,
    showControlLines
  );
  updateLine(
    lookaheadRadialOffsetLine,
    lookaheadOnDisk,
    rabbitPosition,
    controlRabbitRelationshipLineColor,
    0.62,
    showControlLines
  );
  updateLine(
    projectedToDiskLine,
    projectedPhase,
    closestDisk,
    controlClosestDiskColor,
    0.56,
    showControlLines
  );
  updateLine(
    phaseSlotToClosestDiskLine,
    phaseSlot,
    closestDisk,
    controlRabbitRelationshipLineColor,
    0.46,
    showControlLines && showAdaptiveSlots
  );
}

function renderFrame(frame: ApiFrame): void {
  lastRenderedFrame = frame;
  hasRenderedSimulationFrame = true;
  ensureKites(frame.kite_positions_n.length, frame);
  const showAdaptiveSlots = activeSummaryRequest?.phase_mode === "adaptive";
  payloadMesh.position.copy(toThree(frame.payload_position_n));
  payloadMesh.visible = true;
  payloadMesh.scale.setScalar(controlFeatureScale());
  const splitterPosition = toThree(frame.splitter_position_n);
  splitterMesh.position.copy(splitterPosition);
  splitterMesh.visible = false;
  if (shouldSnapOrbitTargetToFrame) {
    snapCameraTargetToFrame(frame);
    shouldSnapOrbitTargetToFrame = false;
  }
  applyCameraFollow(frame);
  updateControlRings(frame);
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
    const upperSegments = upperSegmentMeshes[index];
    const upperNodes = upperNodeMeshes[index];
    const quatData = frame.kite_quaternions_n2b[index];
    if (
      !mesh ||
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
    controlFeatureLayers.forEach((layer) => {
      renderControlFeatureLayer(frame, index, layer, showAdaptiveSlots);
    });

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
  if (!playbackReleased || playbackPaused) {
    return;
  }
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
  updateControlLabels();
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

function plotGroupHeight(group?: PlotGroupDefinition): number {
  return group?.height ?? PLOT_GROUP_HEIGHT_PX;
}

function plotGroupData(group: PlotGroupDefinition, frames: ApiFrame[]): PlotlyDatum[] {
  const frameTimes = frames.map((frame) => frame.time);
  return group.traces.map((trace) => ({
    type: "scatter",
    mode: "lines",
    name: trace.name,
    x: frameTimes,
    y: frames.map((frame) => trace.value(frame)),
    customdata: trace.hoverText
      ? frames.map((frame) => [trace.hoverText?.(frame)])
      : undefined,
    connectgaps: false,
    line: {
      color: trace.color,
      width: trace.width ?? 2,
      dash: trace.dash ?? "solid",
      shape: trace.shape ?? "linear"
    },
    visible: plotTraceVisible(trace),
    hovertemplate:
      trace.hoverTemplate ??
      (trace.hoverText
        ? `%{customdata[0]}<br><span style="color:#90a8ba">t = %{x:.2f} s</span><extra></extra>`
        : `${trace.name}<br>t=%{x:.2f}s<br>%{y:.4f}<extra></extra>`)
  })
  );
}

function plotGroupLayout(group: PlotGroupDefinition): PlotlyDatum {
  return {
    autosize: true,
    height: plotGroupHeight(group),
    margin: { l: 54, r: 18, t: 18, b: 48 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#081018",
    font: {
      family: "IBM Plex Sans, Helvetica Neue, sans-serif",
      color: "#d9ecfb"
    },
    showlegend: false,
    hoverlabel: {
      bgcolor: "#07131c",
      bordercolor: "#45d7a7",
      font: {
        family: "IBM Plex Sans, Helvetica Neue, sans-serif",
        color: "#eaf6ff",
        size: 13
      },
      align: "left"
    },
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
      automargin: true,
      tickmode: group.yTickVals ? "array" : undefined,
      tickvals: group.yTickVals,
      ticktext: group.yTickText,
      range: group.yRange
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

function plotTabLabel(definition: PlotSectionDefinition): string {
  return definition.title
    .replace(/^Controller\s*\/\s*(?:\d+\.\s*)?/i, "")
    .replace(/^Physics\s*\/\s*/i, "")
    .trim();
}

interface PlotTabEntry {
  key: string;
  button: HTMLButtonElement;
  host: HTMLElement;
  plots: HTMLElement[];
}

function activatePlotTab(key: string, entries: PlotTabEntry[]): void {
  activePlotTabKey = key;
  entries.forEach((entry) => {
    const active = entry.key === key;
    entry.button.classList.toggle("active", active);
    entry.button.setAttribute("aria-selected", String(active));
    entry.host.classList.toggle("tab-inactive", !active);
    if (active) {
      entry.plots.forEach((plot) => {
        if (plot.childElementCount > 0) {
          Plotly.Plots?.resize(plot);
        }
      });
    }
  });
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
  activePlotTabKey = null;
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
  const requestedTabKey = activePlotTabKey;
  plotsNode.innerHTML = "";
  ensurePlotKiteVisibility(kiteCount);

  const definitions = buildPlotSections(kiteCount);
  const tabEntries: PlotTabEntry[] = [];
  const tabs = document.createElement("div");
  tabs.className = "plot-tabs";
  tabs.setAttribute("role", "tablist");
  tabs.setAttribute("aria-label", "Plot sections");
  const tabPanels = document.createElement("div");
  tabPanels.className = "plot-tab-panels";
  plotsNode.append(tabs, tabPanels);

  const sectionPromises = definitions.map(async (definition) => {
    const host = document.createElement("section");
    host.className = "plot-section";
    const sectionKey = plotSectionKey(definition);
    host.dataset.plotSection = sectionKey;

    const tabButton = document.createElement("button");
    tabButton.className = "plot-tab";
    tabButton.type = "button";
    tabButton.textContent = plotTabLabel(definition);
    tabButton.setAttribute("role", "tab");
    tabButton.setAttribute("aria-selected", "false");
    tabButton.addEventListener("click", () => {
      activatePlotTab(sectionKey, tabEntries);
    });
    tabs.append(tabButton);

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
    if (definition.showKiteControls === false) {
      controls.classList.add("empty");
    } else {
      renderPlotKiteControls(controls, kiteCount);
    }
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
    tabPanels.append(host);

    const sectionPlots: HTMLElement[] = [];
    tabEntries.push({
      key: sectionKey,
      button: tabButton,
      host,
      plots: sectionPlots
    });

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
      if (group.showSignalLegend === false) {
        signalLegend.classList.add("empty");
      } else {
        renderPlotSignalLegend(signalLegend, group.traces);
      }
      const plot = document.createElement("div");
      plot.className = "plot-canvas";
      plot.style.height = `${plotGroupHeight(group)}px`;

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
  const nextActiveTabKey =
    (requestedTabKey && tabEntries.some((entry) => entry.key === requestedTabKey)
      ? requestedTabKey
      : tabEntries[0]?.key) ?? null;
  if (nextActiveTabKey) {
    activatePlotTab(nextActiveTabKey, tabEntries);
  }
  applyPlotKiteVisibility();
}

function queueFrames(frames: ApiFrame[]): void {
  if (frames.length === 0) {
    return;
  }
  frames.forEach(recordFrameTetherTensions);
  framesReceived += frames.length;
  pendingPlaybackFrames.push(...frames);
  refreshProgressSummary();
}

function releaseBufferedPlayback(): void {
  playbackReleased = true;
  playbackPaused = false;
  playbackStartWallTimeMs = currentPlaybackRate === null ? null : performance.now();
  playbackStartSimTime = pendingPlaybackFrames[0]?.time ?? lastRenderedFrame?.time ?? 0;
  setRunControls();
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

async function loadDefaultConfig(): Promise<void> {
  try {
    const response = await fetch("/api/default_config");
    if (!response.ok) {
      throw new Error(`server returned ${response.status}`);
    }
    simulationDefaults = (await response.json()) as SimulationDefaults;
    durationInput.value = String(simulationDefaults.duration);
    dtControlInput.value = compactNumberInputValue(simulationDefaults.dt_control);
    phaseModeSelect.value = simulationDefaults.phase_mode;
    bridleEnabledInput.checked = simulationDefaults.bridle_enabled;
    simNoiseInput.checked = simulationDefaults.sim_noise_enabled;
    drydenSeedInput.value = String(simulationDefaults.dryden.seed);
    drydenIntensityScaleInput.value = compactNumberInputValue(
      simulationDefaults.dryden.intensity_scale
    );
    drydenLengthScaleInput.value = compactNumberInputValue(simulationDefaults.dryden.length_scale);
    drydenAltitudeIntensityInput.checked = simulationDefaults.dryden.altitude_intensity_enabled;
    drydenAltitudeLengthInput.checked = simulationDefaults.dryden.altitude_length_scale_enabled;
    syncDrydenTuningVisibility();
    maxThrottleAltitudePitchInput.checked =
      simulationDefaults.longitudinal_mode === "max_throttle_altitude_pitch";
    rkAbsTolInput.value = toleranceInputValue(simulationDefaults.rk_abs_tol);
    rkRelTolInput.value = toleranceInputValue(simulationDefaults.rk_rel_tol);
    maxSubstepsInput.value = String(simulationDefaults.max_substeps);
    renderControllerTuningControls(simulationDefaults.controller_tuning);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    appendConsole(`warning: failed to load default sim config: ${message}`);
  }
}

async function loadPresets(): Promise<void> {
  const response = await fetch("/api/presets");
  const presets = (await response.json()) as PresetInfo[];
  presetInfoById = new Map(presets.map((preset) => [preset.preset, preset]));
  presetSelect.innerHTML = "";
  presets.forEach((preset) => {
    const option = document.createElement("option");
    option.value = preset.preset;
    option.dataset.kites = String(preset.kites);
    option.textContent = `${preset.name} — ${preset.description}`;
    presetSelect.append(option);
  });
  if (presetInfoById.has(DEFAULT_PRESET)) {
    presetSelect.value = DEFAULT_PRESET;
  }
  syncSwarmOptionsVisibility();
  applyPresetDefaults();
  updateCameraFollowOptions(presetKiteCount(presetSelect.value as Preset));
}

async function runSimulation(): Promise<void> {
  if (runInProgress) {
    if (runStreamComplete) {
      restartSimulation();
    } else {
      togglePlaybackPause();
    }
    return;
  }
  await startSimulation();
}

function restartSimulation(): void {
  if (!runInProgress) {
    void startSimulation();
    return;
  }
  streamAbortController?.abort();
  void startSimulation();
}

async function startSimulation(): Promise<void> {
  showRuntimeTab("console");
  clearConsole();
  clearPlots("Run starting. Plots will be replaced when the solver finishes.");
  setFailure(null);
  lastFailureConsoleKey = null;

  const selectedTimeDilation = timeDilationSelect.value as TimeDilationPreset;
  const playbackLabel = timeDilationLabel(selectedTimeDilation);
  const playbackRate = timeDilationRate(selectedTimeDilation);
  const durationSeconds = Number(durationInput.value);
  const dtControl = positiveInputValue(dtControlInput, simulationDefaults?.dt_control ?? 0.01);
  const rkAbsTol = positiveInputValue(rkAbsTolInput, simulationDefaults?.rk_abs_tol ?? 1.0e-6);
  const rkRelTol = positiveInputValue(rkRelTolInput, simulationDefaults?.rk_rel_tol ?? 1.0e-6);
  const maxSubsteps = positiveIntegerInputValue(
    maxSubstepsInput,
    simulationDefaults?.max_substeps ?? 1000
  );
  const swarmDiskAltitudeM = nonnegativeInputValue(swarmDiskAltitudeInput, 350);
  const swarmDiskRadiusM = positiveInputValue(swarmDiskRadiusInput, 70);
  const swarmAircraftAltitudeM = optionalInputValue(swarmAircraftAltitudeInput);
  const swarmUpperTetherLengthM = positiveInputValue(swarmUpperTetherLengthInput, 120);
  const swarmCommonTetherLengthM = positiveInputValue(swarmCommonTetherLengthInput, 150);
  const request = {
    preset: presetSelect.value,
    swarm_kites: selectedSwarmKiteCount(),
    swarm_disk_altitude_m: swarmDiskAltitudeM,
    swarm_disk_radius_m: swarmDiskRadiusM,
    swarm_aircraft_altitude_m: swarmAircraftAltitudeM,
    swarm_upper_tether_length_m: swarmUpperTetherLengthM,
    swarm_common_tether_length_m: swarmCommonTetherLengthM,
    duration: durationSeconds,
    dt_control: dtControl,
    phase_mode: phaseModeSelect.value as PhaseMode,
    longitudinal_mode: (maxThrottleAltitudePitchInput.checked
      ? "max_throttle_altitude_pitch"
      : "total_energy") as LongitudinalMode,
    payload_mass_kg: Number(payloadInput.value),
    wind_speed_mps: Number(windInput.value),
    bridle_enabled: bridleEnabledInput.checked,
    sim_noise_enabled: simNoiseInput.checked,
    dryden: drydenConfigFromInputs(),
    rk_abs_tol: rkAbsTol,
    rk_rel_tol: rkRelTol,
    max_substeps: maxSubsteps,
    controller_tuning: controllerTuningFromInputs(),
    sample_stride: 1
  };
  const runSequence = activeRunSequence + 1;
  activeRunSequence = runSequence;
  const abortController = new AbortController();
  streamAbortController = abortController;
  runInProgress = true;
  runStreamComplete = false;
  playbackPaused = false;
  playbackReleased = false;
  controllerTuningChangedDuringRun = false;
  setRunControls();
  activeSummaryRequest = {
    preset: request.preset,
    swarm_kites: request.swarm_kites,
    swarm_disk_altitude_m: request.swarm_disk_altitude_m,
    swarm_disk_radius_m: request.swarm_disk_radius_m,
    swarm_aircraft_altitude_m: request.swarm_aircraft_altitude_m,
    swarm_upper_tether_length_m: request.swarm_upper_tether_length_m,
    swarm_common_tether_length_m: request.swarm_common_tether_length_m,
    phase_mode: request.phase_mode,
    longitudinal_mode: request.longitudinal_mode,
    sim_noise_enabled: request.sim_noise_enabled,
    dryden: request.dryden,
    bridle_enabled: request.bridle_enabled,
    dt_control: request.dt_control,
    rk_abs_tol: request.rk_abs_tol,
    rk_rel_tol: request.rk_rel_tol,
    max_substeps: request.max_substeps,
    controller_tuning: request.controller_tuning
  };
  resetPlaybackState(playbackLabel, playbackRate);
  const kiteCount = presetKiteCount(request.preset as Preset);
  summaryNode.innerHTML = renderSummaryCard(
    "Queued",
    "Waiting for first solver update",
    [
      { label: "Preset", value: request.preset },
      { label: "Kites", value: String(kiteCount) },
      { label: "Disk Altitude", value: optionalMetersLabel(request.swarm_disk_altitude_m) },
      { label: "Disk Radius", value: optionalMetersLabel(request.swarm_disk_radius_m) },
      { label: "Aircraft Altitude", value: optionalMetersLabel(request.swarm_aircraft_altitude_m) },
      {
        label: "Tethers",
        value: `lower ${compactNumberInputValue(request.swarm_common_tether_length_m)} m / upper ${compactNumberInputValue(request.swarm_upper_tether_length_m)} m`
      },
      { label: "Phase Mode", value: request.phase_mode },
      { label: "Longitudinal", value: longitudinalModeLabel(request.longitudinal_mode) },
      { label: "Bridle", value: request.bridle_enabled ? "Enabled" : "CG attach" },
      { label: "Sim Noise", value: request.sim_noise_enabled ? "Dryden gusts" : "Off" },
      {
        label: "Dryden",
        value: request.sim_noise_enabled
          ? `intensity ${compactNumberInputValue(request.dryden.intensity_scale)}, length ${compactNumberInputValue(request.dryden.length_scale)}, seed ${request.dryden.seed}`
          : "disabled"
      },
      {
        label: "RK abs / rel tol",
        value: `${toleranceLabel(request.rk_abs_tol)} / ${toleranceLabel(request.rk_rel_tol)}`
      },
      {
        label: "Sample Period / Rate",
        value: `${compactNumberInputValue(request.dt_control)} s / ${(1 / request.dt_control).toFixed(1)} Hz`
      },
      { label: "Substep Budget", value: String(request.max_substeps) },
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
    `run requested: preset=${request.preset}, kites=${request.swarm_kites}, disk_alt=${optionalMetersLabel(request.swarm_disk_altitude_m)}, disk_radius=${optionalMetersLabel(request.swarm_disk_radius_m)}, aircraft_alt=${optionalMetersLabel(request.swarm_aircraft_altitude_m)}, lower_tether=${compactNumberInputValue(request.swarm_common_tether_length_m)}m, upper_tether=${compactNumberInputValue(request.swarm_upper_tether_length_m)}m, duration=${request.duration}s, dt_control=${compactNumberInputValue(request.dt_control)}s (${(1 / request.dt_control).toFixed(1)} Hz), phase=${request.phase_mode}, longitudinal=${request.longitudinal_mode}, bridle=${request.bridle_enabled ? "enabled" : "cg_attach"}, noise=${request.sim_noise_enabled ? "dryden" : "off"}, dryden_intensity=${compactNumberInputValue(request.dryden.intensity_scale)}, dryden_length=${compactNumberInputValue(request.dryden.length_scale)}, dryden_seed=${request.dryden.seed}, rk_abs_tol=${toleranceLabel(request.rk_abs_tol)}, rk_rel_tol=${toleranceLabel(request.rk_rel_tol)}, max_substeps=${request.max_substeps}, time_dilation=${playbackLabel}`
  );
  appendConsole(controllerTuningSnapshotLabel(request.controller_tuning));

  try {
    const response = await fetch("/api/run_stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request),
      signal: abortController.signal
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
      if (runSequence !== activeRunSequence) {
        return;
      }
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
        if (event.kind === "error") {
          setSolverFailure(event.message, latestProgressState?.time ?? null);
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
        logFailureIfNeeded(event.summary.failure ?? null);
      }
      queueFrames(sceneFramesInChunk);
    }

    if (buffer.trim()) {
      const event = JSON.parse(buffer) as StreamEvent;
      if (event.kind === "log") {
        appendConsole(event.message);
      } else if (event.kind === "error") {
        setSolverFailure(event.message, latestProgressState?.time ?? null);
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
        logFailureIfNeeded(event.summary.failure ?? null);
      }
    }

    if (finalPlotFrames && finalPlotFrames.length > 0) {
      appendConsole(`rendering plots from ${finalPlotFrames.length} samples`);
      try {
        showRuntimeTab("plots");
        await renderFinalPlots(finalPlotFrames, kiteCount);
        appendConsole("plots rendered; starting 3D playback");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        clearPlots(`Plot rendering failed: ${message}`);
        appendConsole(`plot rendering failed: ${message}`);
      }
    } else {
      clearPlots("No plot samples were returned for this run.");
      appendConsole("no final plot buffer received");
    }
    if (runSequence !== activeRunSequence) {
      return;
    }
    releaseBufferedPlayback();
    await waitForPlaybackDrain();
    if (runSequence !== activeRunSequence) {
      return;
    }
    if (pendingSummary) {
      const finalSummaryHtml = formatRunSummary(
        pendingSummary,
        framesReceived,
        framesRendered,
        currentPlaybackLabel
      );
      summaryNode.innerHTML = finalSummaryHtml;
      lastSummaryHtml = finalSummaryHtml;
      latestProgressState = null;
      activeSummaryRequest = null;
      summaryRefreshPending = false;
    }
    appendConsole(`received ${framesReceived} frames, rendered ${framesRendered}`);
    if (!summary) {
      appendConsole("run ended without a summary");
    }
  } catch (error) {
    if (runSequence !== activeRunSequence) {
      return;
    }
    const message = error instanceof Error ? error.message : String(error);
    if (error instanceof DOMException && error.name === "AbortError") {
      appendConsole("run aborted");
    } else {
      runStreamComplete = false;
      summaryNode.innerHTML = `<div class="summary-error">Run failed: ${escapeHtml(message)}</div>`;
      setFailure(null);
      appendConsole(`error: ${message}`);
    }
  } finally {
    if (runSequence === activeRunSequence) {
      runInProgress = false;
      runStreamComplete = false;
      playbackPaused = false;
      playbackReleased = false;
      streamAbortController = null;
      setRunControls();
    }
  }
}

runForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void runSimulation();
});

restartButton.addEventListener("click", () => {
  restartSimulation();
});

runtimeConsoleTab.addEventListener("click", () => {
  showRuntimeTab("console");
});

runtimePlotsTab.addEventListener("click", () => {
  showRuntimeTab("plots");
});

presetSelect.addEventListener("change", () => {
  syncSwarmOptionsVisibility();
  applyPresetDefaults();
  updateCameraFollowOptions(presetKiteCount(presetSelect.value as Preset));
  shouldSnapOrbitTargetToFrame = true;
  resetCameraFollowState();
});

swarmKitesSelect.addEventListener("change", () => {
  updateCameraFollowOptions(presetKiteCount(presetSelect.value as Preset));
  shouldSnapOrbitTargetToFrame = true;
  resetCameraFollowState();
});

phaseModeSelect.addEventListener("change", () => {
  renderControllerDocs();
});

simNoiseInput.addEventListener("change", () => {
  syncDrydenTuningVisibility();
});

payloadInput.addEventListener("input", () => {
  if (tetherTensionScaleMode() === "payload") {
    rerenderTetherColors();
  }
});

maxThrottleAltitudePitchInput.addEventListener("change", () => {
  noteControllerTuningEditedDuringRun();
  syncControllerTuningVisibility();
  renderControllerDocs();
});

function handleControllerTuningFieldEdit(event: Event): void {
  const target = event.target;
  if (!(target instanceof HTMLInputElement || target instanceof HTMLSelectElement)) {
    return;
  }
  if (!target.dataset.tuningKey) {
    return;
  }
  noteControllerTuningEditedDuringRun();
  syncControllerTuningVisibility();
}

controllerTuningFieldsNode.addEventListener("input", handleControllerTuningFieldEdit);
controllerTuningFieldsNode.addEventListener("change", handleControllerTuningFieldEdit);

timeDilationSelect.addEventListener("change", () => {
  applyTimeDilationSelection(true);
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

trackpadNavigationInput.addEventListener("change", () => {
  applyPointerNavigationMode();
});

controlLabelsEnabledInput.addEventListener("change", () => {
  updateControlLabels();
});

controlDiskEnabledInput.addEventListener("change", () => {
  applyVisualizationVisibility();
});

controlFeaturesEnabledInput.addEventListener("change", () => {
  applyVisualizationVisibility();
});

controlFeatureLinesEnabledInput.addEventListener("change", () => {
  applyVisualizationVisibility();
});

controlFeaturesAtTargetAltitudeInput.addEventListener("change", () => {
  applyVisualizationVisibility();
});

controlFeaturesAtAircraftAltitudeInput.addEventListener("change", () => {
  applyVisualizationVisibility();
});

controlFeatureScaleInput.addEventListener("input", () => {
  applyVisualizationScales();
});

tetherNodesEnabledInput.addEventListener("change", () => {
  applyVisualizationVisibility();
});

tetherNodeScaleInput.addEventListener("input", () => {
  applyVisualizationScales();
});

tetherTensionScaleModeSelect.addEventListener("change", () => {
  syncTetherTensionScaleVisibility();
  rerenderTetherColors();
});

tetherTensionPayloadMarginInput.addEventListener("input", () => {
  rerenderTetherColors();
});

tetherTensionFixedMinInput.addEventListener("input", () => {
  rerenderTetherColors();
});

tetherTensionFixedMaxInput.addEventListener("input", () => {
  rerenderTetherColors();
});

fogEnabledInput.addEventListener("change", () => {
  applyFogVisibility();
});

airParticlesEnabledInput.addEventListener("change", () => {
  if (!airParticlesVisible()) {
    ambientParticleCloud.visible = false;
    gustParticleCloud.visible = false;
  }
});

airParticleOpacityInput.addEventListener("input", () => {
  applyAirParticleOpacity();
});

wingtipTrailsEnabledInput.addEventListener("change", () => {
  if (!wingtipTrailsVisible()) {
    clearWingtipTrailParticles();
  }
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
  const controlsWidth = document.querySelector<HTMLElement>(".controls")!.getBoundingClientRect().width;
  setSidebarWidth(controlsWidth, false);
  resizeSceneRenderer();
  activePlotSections.forEach((section) => {
    Plotly.Plots?.resize(section.plot);
  });
});

void Promise.all([loadDefaultConfig(), loadPresets()]).then(() => {
  syncDrydenTuningVisibility();
  syncTetherTensionScaleVisibility();
  applyPresetDefaults();
  renderControllerDocs();
  return runSimulation();
});
