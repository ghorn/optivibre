use std::convert::Infallible;
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use axum::body::Bytes;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use multikite_sim::{
    COMMON_NODES, FREE_COMMON_NODES, FREE_UPPER_NODES, InitRequest, PhaseMode, Preset,
    RunSummary, SimulationConfig, SimulationFrame, SimulationProgress, UPPER_NODES,
    available_presets, simulate_free_flight1_with_callbacks, simulate_free_flight1_with_progress,
    simulate_simple_tether_with_callbacks, simulate_simple_tether_with_progress,
    simulate_star1_with_callbacks,
    simulate_star1_with_progress, simulate_star3_with_callbacks,
    simulate_star3_with_progress, simulate_star4_with_callbacks, simulate_star4_with_progress,
    simulate_y2_with_callbacks, simulate_y2_with_progress,
};
use nalgebra::UnitQuaternion;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

const TEXT_HTML_UTF8: &str = "text/html; charset=utf-8";
const TEXT_JAVASCRIPT_UTF8: &str = "text/javascript; charset=utf-8";
const TEXT_CSS_UTF8: &str = "text/css; charset=utf-8";
const IMAGE_SVG_XML: &str = "image/svg+xml";
const APPLICATION_NDJSON_UTF8: &str = "application/x-ndjson; charset=utf-8";
const GENERATED_APP_JS: &str = include_str!(concat!(env!("OUT_DIR"), "/app.js"));

static LAST_SUMMARY: OnceLock<Mutex<Option<RunSummary>>> = OnceLock::new();

#[derive(Debug, Parser)]
#[command(
    name = "multikite_webapp",
    about = "Local interactive web app for multikite simulation demos."
)]
struct Cli {
    #[arg(long, env = "PORT", default_value_t = 3010)]
    port: u16,
}

#[derive(Clone, Debug, Deserialize)]
struct RunRequest {
    preset: Preset,
    duration: f64,
    phase_mode: PhaseMode,
    payload_mass_kg: Option<f64>,
    wind_speed_mps: Option<f64>,
    sample_stride: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
struct ApiRunResponse {
    summary: RunSummary,
    frames: Vec<ApiFrame>,
}

#[derive(Clone, Debug, Serialize)]
struct ApiFrame {
    time: f64,
    payload_position_n: [f64; 3],
    splitter_position_n: [f64; 3],
    clean_wind_n: [f64; 3],
    kite_gust_n: Vec<[f64; 3]>,
    kite_airflow_n: Vec<[f64; 3]>,
    kite_ref_span: Vec<f64>,
    kite_ref_chord: Vec<f64>,
    kite_ref_area: Vec<f64>,
    kite_cad_offset_b: Vec<[f64; 3]>,
    kite_bridle_pivot_b: Vec<[f64; 3]>,
    kite_bridle_radius: Vec<f64>,
    control_ring_center_n: [f64; 3],
    control_ring_radius: f64,
    common_tether: Vec<[f64; 3]>,
    common_tether_tensions: Vec<f64>,
    upper_tethers: Vec<Vec<[f64; 3]>>,
    upper_tether_tensions: Vec<Vec<f64>>,
    kite_positions_n: Vec<[f64; 3]>,
    kite_quaternions_n2b: Vec<[f64; 4]>,
    kite_attitudes_rpy_deg: Vec<[f64; 3]>,
    rabbit_targets_n: Vec<[f64; 3]>,
    phase_error: Vec<f64>,
    speed_target: Vec<f64>,
    inertial_speed: Vec<f64>,
    airspeed: Vec<f64>,
    alpha_deg: Vec<f64>,
    beta_deg: Vec<f64>,
    body_omega_b: Vec<[f64; 3]>,
    orbit_radius: Vec<f64>,
    rabbit_radius: Vec<f64>,
    curvature_y_b: Vec<f64>,
    curvature_y_ref: Vec<f64>,
    curvature_y_est: Vec<f64>,
    omega_world_z_ref: Vec<f64>,
    omega_world_z: Vec<f64>,
    beta_ref_deg: Vec<f64>,
    roll_ref_deg: Vec<f64>,
    roll_ff_deg: Vec<f64>,
    pitch_ref_deg: Vec<f64>,
    curvature_z_b: Vec<f64>,
    curvature_z_ref: Vec<f64>,
    top_tension: Vec<f64>,
    total_force_b: Vec<[f64; 3]>,
    aero_force_b: Vec<[f64; 3]>,
    aero_force_drag_b: Vec<[f64; 3]>,
    aero_force_side_b: Vec<[f64; 3]>,
    aero_force_lift_b: Vec<[f64; 3]>,
    tether_force_b: Vec<[f64; 3]>,
    gravity_force_b: Vec<[f64; 3]>,
    motor_force_b: Vec<[f64; 3]>,
    total_moment_b: Vec<[f64; 3]>,
    aero_moment_b: Vec<[f64; 3]>,
    tether_moment_b: Vec<[f64; 3]>,
    cl_total: Vec<f64>,
    cl_0_term: Vec<f64>,
    cl_alpha_term: Vec<f64>,
    cl_elevator_term: Vec<f64>,
    cl_flap_term: Vec<f64>,
    cd_total: Vec<f64>,
    cd_0_term: Vec<f64>,
    cd_induced_term: Vec<f64>,
    cd_surface_term: Vec<f64>,
    cy_total: Vec<f64>,
    cy_beta_term: Vec<f64>,
    cy_rudder_term: Vec<f64>,
    roll_coeff_total: Vec<f64>,
    roll_beta_term: Vec<f64>,
    roll_p_term: Vec<f64>,
    roll_r_term: Vec<f64>,
    roll_aileron_term: Vec<f64>,
    pitch_coeff_total: Vec<f64>,
    pitch_0_term: Vec<f64>,
    pitch_alpha_term: Vec<f64>,
    pitch_q_term: Vec<f64>,
    pitch_elevator_term: Vec<f64>,
    pitch_flap_term: Vec<f64>,
    yaw_coeff_total: Vec<f64>,
    yaw_beta_term: Vec<f64>,
    yaw_p_term: Vec<f64>,
    yaw_r_term: Vec<f64>,
    yaw_rudder_term: Vec<f64>,
    aileron_cmd_deg: Vec<f64>,
    flap_cmd_deg: Vec<f64>,
    winglet_cmd_deg: Vec<f64>,
    elevator_cmd_deg: Vec<f64>,
    rudder_cmd_deg: Vec<f64>,
    motor_torque: Vec<f64>,
    total_work: f64,
    total_kinetic_energy: f64,
    total_potential_energy: f64,
    total_tether_strain_energy: f64,
    work_minus_potential: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum StreamEvent {
    Log { message: String },
    Progress { progress: SimulationProgress },
    Frame { frame: ApiFrame },
    Summary { summary: RunSummary },
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", cli.port)).await?;
    println!(
        "multikite_webapp listening on http://127.0.0.1:{}",
        cli.port
    );

    let app = Router::new()
        .route("/", get(index))
        .route("/app.js", get(app_js))
        .route("/styles.css", get(styles_css))
        .route("/favicon.svg", get(favicon))
        .route("/favicon.ico", get(favicon))
        .route("/api/presets", get(presets))
        .route("/api/run", post(run))
        .route("/api/run_stream", post(run_stream))
        .route("/api/last_summary", get(last_summary))
        .route("/healthz", get(healthz));

    axum::serve(listener, app).await?;
    Ok(())
}

fn static_text_response(content_type: &'static str, body: &'static str) -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, HeaderValue::from_static(content_type))],
        body,
    )
}

async fn index() -> impl IntoResponse {
    static_text_response(TEXT_HTML_UTF8, include_str!("../static/index.html"))
}

async fn app_js() -> impl IntoResponse {
    static_text_response(TEXT_JAVASCRIPT_UTF8, GENERATED_APP_JS)
}

async fn styles_css() -> impl IntoResponse {
    static_text_response(TEXT_CSS_UTF8, include_str!("../static/styles.css"))
}

async fn favicon() -> impl IntoResponse {
    static_text_response(IMAGE_SVG_XML, include_str!("../static/favicon.svg"))
}

async fn presets() -> Json<Vec<multikite_sim::PresetInfo>> {
    Json(available_presets())
}

async fn healthz() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn last_summary() -> Json<Option<RunSummary>> {
    Json(
        LAST_SUMMARY
            .get_or_init(|| Mutex::new(None))
            .lock()
            .map(|summary| summary.clone())
            .unwrap_or(None),
    )
}

fn config_from_request(request: &RunRequest) -> (InitRequest, SimulationConfig) {
    (
        InitRequest {
            preset: request.preset,
            payload_mass_kg: request.payload_mass_kg,
            wind_speed_mps: request.wind_speed_mps,
        },
        SimulationConfig {
            duration: request.duration,
            phase_mode: request.phase_mode,
            sample_stride: request.sample_stride.unwrap_or(1).max(1),
            ..SimulationConfig::default()
        },
    )
}

fn to_api_frame<const NK: usize, const N_COMMON: usize, const N_UPPER: usize>(
    frame: &SimulationFrame<f64, NK, N_COMMON, N_UPPER>,
) -> ApiFrame {
    let payload_position_n = vec3(frame.state.payload.pos_n);
    let splitter_position_n = vec3(frame.state.splitter.pos_n);
    let clean_wind_n = vec3(frame.clean_wind_n);
    let control_ring_center_n = vec3(frame.control_ring_center_n);
    let common_tether = std::iter::once(vec3(frame.state.payload.pos_n))
        .chain(
            frame
                .state
                .common_tether
                .iter()
                .map(|node| vec3(node.pos_n)),
        )
        .chain(std::iter::once(vec3(frame.state.splitter.pos_n)))
        .collect();
    let upper_tethers = frame
        .state
        .kites
        .iter()
        .enumerate()
        .map(|(index, kite)| {
            std::iter::once(vec3(frame.state.splitter.pos_n))
                .chain(kite.tether.iter().map(|node| vec3(node.pos_n)))
                .chain(std::iter::once(vec3(frame.kite_bridle_positions_n[index])))
                .collect::<Vec<_>>()
        })
        .collect();
    ApiFrame {
        time: frame.time,
        payload_position_n,
        splitter_position_n,
        clean_wind_n,
        kite_gust_n: frame.kite_gusts_n.iter().map(|gust| vec3(*gust)).collect(),
        kite_airflow_n: frame
            .kite_gusts_n
            .iter()
            .map(|gust| vec3(frame.clean_wind_n + *gust))
            .collect(),
        kite_ref_span: frame.kite_ref_spans.to_vec(),
        kite_ref_chord: frame.kite_ref_chords.to_vec(),
        kite_ref_area: frame.kite_ref_areas.to_vec(),
        kite_cad_offset_b: frame
            .kite_cad_offsets_b
            .iter()
            .map(|offset| vec3(*offset))
            .collect(),
        kite_bridle_pivot_b: frame
            .kite_bridle_pivots_b
            .iter()
            .map(|pivot| vec3(*pivot))
            .collect(),
        kite_bridle_radius: frame.kite_bridle_radii.to_vec(),
        control_ring_center_n,
        control_ring_radius: frame.control_ring_radius,
        common_tether,
        common_tether_tensions: frame.common_tether_tensions.clone(),
        upper_tethers,
        upper_tether_tensions: frame.upper_tether_tensions.clone(),
        kite_positions_n: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| vec3(diag.cad_position_n))
            .collect(),
        kite_quaternions_n2b: frame
            .state
            .kites
            .iter()
            .map(|kite| {
                [
                    kite.body.quat_n2b.coords[3],
                    kite.body.quat_n2b.coords[0],
                    kite.body.quat_n2b.coords[1],
                    kite.body.quat_n2b.coords[2],
                ]
            })
            .collect(),
        kite_attitudes_rpy_deg: frame
            .state
            .kites
            .iter()
            .map(|kite| quaternion_to_rpy_deg(kite.body.quat_n2b))
            .collect(),
        rabbit_targets_n: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| vec3(diag.rabbit_target_n))
            .collect(),
        phase_error: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.phase_error)
            .collect(),
        speed_target: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.speed_target)
            .collect(),
        inertial_speed: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.cad_velocity_n.norm())
            .collect(),
        airspeed: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.airspeed)
            .collect(),
        alpha_deg: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.alpha.to_degrees())
            .collect(),
        beta_deg: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.beta.to_degrees())
            .collect(),
        body_omega_b: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| vec3(diag.omega_b))
            .collect(),
        orbit_radius: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.orbit_radius)
            .collect(),
        rabbit_radius: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.rabbit_radius)
            .collect(),
        curvature_y_b: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.curvature_y_b)
            .collect(),
        curvature_y_ref: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.curvature_y_ref)
            .collect(),
        curvature_y_est: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.curvature_y_est)
            .collect(),
        omega_world_z_ref: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.omega_world_z_ref)
            .collect(),
        omega_world_z: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.omega_world_z)
            .collect(),
        beta_ref_deg: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.beta_ref.to_degrees())
            .collect(),
        roll_ref_deg: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.roll_ref.to_degrees())
            .collect(),
        roll_ff_deg: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.roll_ff.to_degrees())
            .collect(),
        pitch_ref_deg: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.pitch_ref.to_degrees())
            .collect(),
        curvature_z_b: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.curvature_z_b)
            .collect(),
        curvature_z_ref: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.curvature_z_ref)
            .collect(),
        top_tension: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.top_tension)
            .collect(),
        total_force_b: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| vec3(diag.total_force_b))
            .collect(),
        aero_force_b: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| vec3(diag.aero_force_b))
            .collect(),
        aero_force_drag_b: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| vec3(diag.aero_force_drag_b))
            .collect(),
        aero_force_side_b: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| vec3(diag.aero_force_side_b))
            .collect(),
        aero_force_lift_b: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| vec3(diag.aero_force_lift_b))
            .collect(),
        tether_force_b: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| vec3(diag.tether_force_b))
            .collect(),
        gravity_force_b: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| vec3(diag.gravity_force_b))
            .collect(),
        motor_force_b: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| vec3(diag.motor_force_b))
            .collect(),
        total_moment_b: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| vec3(diag.total_moment_b))
            .collect(),
        aero_moment_b: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| vec3(diag.aero_moment_b))
            .collect(),
        tether_moment_b: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| vec3(diag.tether_moment_b))
            .collect(),
        cl_total: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.cl_total)
            .collect(),
        cl_0_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.cl_0_term)
            .collect(),
        cl_alpha_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.cl_alpha_term)
            .collect(),
        cl_elevator_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.cl_elevator_term)
            .collect(),
        cl_flap_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.cl_flap_term)
            .collect(),
        cd_total: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.cd_total)
            .collect(),
        cd_0_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.cd_0_term)
            .collect(),
        cd_induced_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.cd_induced_term)
            .collect(),
        cd_surface_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.cd_surface_term)
            .collect(),
        cy_total: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.cy_total)
            .collect(),
        cy_beta_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.cy_beta_term)
            .collect(),
        cy_rudder_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.cy_rudder_term)
            .collect(),
        roll_coeff_total: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.roll_coeff_total)
            .collect(),
        roll_beta_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.roll_beta_term)
            .collect(),
        roll_p_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.roll_p_term)
            .collect(),
        roll_r_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.roll_r_term)
            .collect(),
        roll_aileron_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.roll_aileron_term)
            .collect(),
        pitch_coeff_total: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.pitch_coeff_total)
            .collect(),
        pitch_0_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.pitch_0_term)
            .collect(),
        pitch_alpha_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.pitch_alpha_term)
            .collect(),
        pitch_q_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.pitch_q_term)
            .collect(),
        pitch_elevator_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.pitch_elevator_term)
            .collect(),
        pitch_flap_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.pitch_flap_term)
            .collect(),
        yaw_coeff_total: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.yaw_coeff_total)
            .collect(),
        yaw_beta_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.yaw_beta_term)
            .collect(),
        yaw_p_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.yaw_p_term)
            .collect(),
        yaw_r_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.yaw_r_term)
            .collect(),
        yaw_rudder_term: frame
            .diagnostics
            .kites
            .iter()
            .map(|diag| diag.yaw_rudder_term)
            .collect(),
        aileron_cmd_deg: frame
            .controls
            .kites
            .iter()
            .map(|control| control.surfaces.aileron.to_degrees())
            .collect(),
        flap_cmd_deg: frame
            .controls
            .kites
            .iter()
            .map(|control| control.surfaces.flap.to_degrees())
            .collect(),
        winglet_cmd_deg: frame
            .controls
            .kites
            .iter()
            .map(|control| control.surfaces.winglet.to_degrees())
            .collect(),
        elevator_cmd_deg: frame
            .controls
            .kites
            .iter()
            .map(|control| control.surfaces.elevator.to_degrees())
            .collect(),
        rudder_cmd_deg: frame
            .controls
            .kites
            .iter()
            .map(|control| control.surfaces.rudder.to_degrees())
            .collect(),
        motor_torque: frame
            .controls
            .kites
            .iter()
            .map(|control| control.motor_torque)
            .collect(),
        total_work: frame.state.total_work,
        total_kinetic_energy: frame.diagnostics.total_kinetic_energy,
        total_potential_energy: frame.diagnostics.total_potential_energy,
        total_tether_strain_energy: frame.diagnostics.total_tether_strain_energy,
        work_minus_potential: frame.diagnostics.work_minus_potential,
    }
}

fn vec3(value: nalgebra::Vector3<f64>) -> [f64; 3] {
    [value[0], value[1], value[2]]
}

fn quaternion_to_rpy_deg(quat_n2b: nalgebra::Quaternion<f64>) -> [f64; 3] {
    let unit = UnitQuaternion::from_quaternion(quat_n2b);
    let (roll, pitch, yaw) = unit.euler_angles();
    [roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees()]
}

fn run_preset(request: &RunRequest) -> Result<ApiRunResponse> {
    let mut progress_cb = |_| {};
    run_preset_with_progress(request, &mut progress_cb)
}

fn run_preset_with_progress<F: FnMut(SimulationProgress)>(
    request: &RunRequest,
    progress_cb: &mut F,
) -> Result<ApiRunResponse> {
    let (init, config) = config_from_request(request);
    let (summary, frames) = match request.preset {
        Preset::FreeFlight1 => {
            let run = simulate_free_flight1_with_progress(&init, &config, progress_cb)?;
            (
                run.summary,
                run.frames
                    .into_iter()
                    .map(|frame| to_api_frame(&frame))
                    .collect(),
            )
        }
        Preset::Star1 => {
            let run = simulate_star1_with_progress(&init, &config, progress_cb)?;
            (
                run.summary,
                run.frames
                    .into_iter()
                    .map(|frame| to_api_frame(&frame))
                    .collect(),
            )
        }
        Preset::Y2 => {
            let run = simulate_y2_with_progress(&init, &config, progress_cb)?;
            (
                run.summary,
                run.frames
                    .into_iter()
                    .map(|frame| to_api_frame(&frame))
                    .collect(),
            )
        }
        Preset::Star3 => {
            let run = simulate_star3_with_progress(&init, &config, progress_cb)?;
            (
                run.summary,
                run.frames
                    .into_iter()
                    .map(|frame| to_api_frame(&frame))
                    .collect(),
            )
        }
        Preset::Star4 => {
            let run = simulate_star4_with_progress(&init, &config, progress_cb)?;
            (
                run.summary,
                run.frames
                    .into_iter()
                    .map(|frame| to_api_frame(&frame))
                    .collect(),
            )
        }
        Preset::SimpleTether => {
            let run = simulate_simple_tether_with_progress(&init, &config, progress_cb)?;
            (
                run.summary,
                run.frames
                    .into_iter()
                    .map(|frame| to_api_frame(&frame))
                    .collect(),
            )
        }
    };
    if let Ok(mut slot) = LAST_SUMMARY.get_or_init(|| Mutex::new(None)).lock() {
        *slot = Some(summary.clone());
    }
    Ok(ApiRunResponse { summary, frames })
}

async fn run(
    Json(request): Json<RunRequest>,
) -> Result<Json<ApiRunResponse>, (StatusCode, String)> {
    run_preset(&request)
        .map(Json)
        .map_err(|error| (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))
}

async fn run_stream(
    Json(request): Json<RunRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(64);
    tokio::spawn(async move {
        let _ = send_stream_event(
            &sender,
            StreamEvent::Log {
                message: format!(
                    "starting preset={:?} duration={:.1}s phase_mode={:?}",
                    request.preset, request.duration, request.phase_mode
                ),
            },
        )
        .await;

        let request_for_run = request.clone();
        let progress_sender = sender.clone();
        let outcome = tokio::task::spawn_blocking(move || {
            let mut progress_cb = |progress: SimulationProgress| {
                let _ = send_stream_event_blocking(
                    &progress_sender,
                    StreamEvent::Progress { progress },
                );
            };
            let mut frame_cb = |frame: ApiFrame| {
                let _ = send_stream_event_blocking(&progress_sender, StreamEvent::Frame { frame });
            };
            run_preset_streaming(&request_for_run, &mut progress_cb, &mut frame_cb)
        })
        .await;
        match outcome {
            Ok(Ok(summary)) => {
                let _ = send_stream_event(
                    &sender,
                    StreamEvent::Log {
                        message: "simulation complete".to_string(),
                    },
                )
                .await;
                let _ = send_stream_event(&sender, StreamEvent::Summary { summary }).await;
                let _ = send_stream_event(
                    &sender,
                    StreamEvent::Log {
                        message: "run finished".to_string(),
                    },
                )
                .await;
            }
            Ok(Err(error)) => {
                let _ = send_stream_event(
                    &sender,
                    StreamEvent::Log {
                        message: format!("run failed: {error}"),
                    },
                )
                .await;
            }
            Err(error) => {
                let _ = send_stream_event(
                    &sender,
                    StreamEvent::Log {
                        message: format!("worker task failed: {error}"),
                    },
                )
                .await;
            }
        }
    });
    Ok((
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static(APPLICATION_NDJSON_UTF8),
        )],
        axum::body::Body::from_stream(ReceiverStream::new(receiver)),
    ))
}

async fn send_stream_event(
    sender: &mpsc::Sender<Result<Bytes, Infallible>>,
    event: StreamEvent,
) -> Result<(), mpsc::error::SendError<Result<Bytes, Infallible>>> {
    sender.send(Ok(stream_event_bytes(&event))).await?;
    sender.send(Ok(Bytes::from_static(b"\n"))).await
}

fn send_stream_event_blocking(
    sender: &mpsc::Sender<Result<Bytes, Infallible>>,
    event: StreamEvent,
) -> Result<(), mpsc::error::SendError<Result<Bytes, Infallible>>> {
    sender.blocking_send(Ok(stream_event_bytes(&event)))?;
    sender.blocking_send(Ok(Bytes::from_static(b"\n")))
}

fn stream_event_bytes(event: &StreamEvent) -> Bytes {
    Bytes::from(serde_json::to_vec(event).expect("stream event should serialize"))
}

fn run_preset_streaming<P: FnMut(SimulationProgress), G: FnMut(ApiFrame)>(
    request: &RunRequest,
    progress_cb: &mut P,
    frame_cb: &mut G,
) -> Result<RunSummary> {
    let (init, config) = config_from_request(request);
    let summary = match request.preset {
        Preset::FreeFlight1 => {
            let mut send_frame =
                |frame: SimulationFrame<f64, 1, FREE_COMMON_NODES, FREE_UPPER_NODES>| {
                    frame_cb(to_api_frame(&frame));
                };
            simulate_free_flight1_with_callbacks(&init, &config, progress_cb, &mut send_frame)?
                .summary
        }
        Preset::Star1 => {
            let mut send_frame = |frame: SimulationFrame<f64, 1, COMMON_NODES, UPPER_NODES>| {
                frame_cb(to_api_frame(&frame));
            };
            simulate_star1_with_callbacks(&init, &config, progress_cb, &mut send_frame)?.summary
        }
        Preset::Y2 => {
            let mut send_frame = |frame: SimulationFrame<f64, 2, COMMON_NODES, UPPER_NODES>| {
                frame_cb(to_api_frame(&frame));
            };
            simulate_y2_with_callbacks(&init, &config, progress_cb, &mut send_frame)?.summary
        }
        Preset::Star3 => {
            let mut send_frame = |frame: SimulationFrame<f64, 3, COMMON_NODES, UPPER_NODES>| {
                frame_cb(to_api_frame(&frame));
            };
            simulate_star3_with_callbacks(&init, &config, progress_cb, &mut send_frame)?.summary
        }
        Preset::Star4 => {
            let mut send_frame = |frame: SimulationFrame<f64, 4, COMMON_NODES, UPPER_NODES>| {
                frame_cb(to_api_frame(&frame));
            };
            simulate_star4_with_callbacks(&init, &config, progress_cb, &mut send_frame)?.summary
        }
        Preset::SimpleTether => {
            let mut send_frame = |frame: SimulationFrame<f64, 0, COMMON_NODES, UPPER_NODES>| {
                frame_cb(to_api_frame(&frame));
            };
            simulate_simple_tether_with_callbacks(&init, &config, progress_cb, &mut send_frame)?
                .summary
        }
    };
    if let Ok(mut slot) = LAST_SUMMARY.get_or_init(|| Mutex::new(None)).lock() {
        *slot = Some(summary.clone());
    }
    Ok(summary)
}
