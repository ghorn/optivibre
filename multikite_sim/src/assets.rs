use anyhow::Result;
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct AssetManifest {
    pub reference_source_commits: ReferenceSourceCommits,
    pub exports: Vec<ExportEntry>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReferenceSourceCommits {
    pub physics: String,
    pub controller: String,
    pub initialization: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ExportEntry {
    pub id: String,
    pub path: String,
    pub description: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReferenceExport {
    pub name: String,
    pub source_paths: Vec<String>,
    pub rigid_body: RigidBodyExport,
    pub aero: AeroExport,
    pub rotor: RotorExport,
    pub bridle: BridleExport,
    pub tethers: TethersExport,
    pub controller: ControllerExport,
    pub environment: EnvironmentExport,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RigidBodyExport {
    pub mass: f64,
    pub inertia_diagonal: [f64; 3],
    pub cad_offset_b: [f64; 3],
}

#[derive(Clone, Debug, Deserialize)]
pub struct AeroExport {
    pub ref_area: f64,
    pub ref_span: f64,
    pub ref_chord: f64,
    pub cl0: f64,
    pub cl_alpha: f64,
    pub cl_elevator: f64,
    pub cl_flap: f64,
    pub cd0: f64,
    pub cd_induced: f64,
    pub cd_surface_abs: f64,
    pub cy_beta: f64,
    pub cy_rudder: f64,
    pub roll_beta: f64,
    pub roll_p: f64,
    pub roll_r: f64,
    pub roll_aileron: f64,
    pub pitch0: f64,
    pub pitch_alpha: f64,
    pub pitch_q: f64,
    pub pitch_elevator: f64,
    pub pitch_flap: f64,
    pub yaw_beta: f64,
    pub yaw_p: f64,
    pub yaw_r: f64,
    pub yaw_rudder: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RotorExport {
    pub axis_b: [f64; 3],
    pub radius: f64,
    pub torque_to_force: f64,
    pub force_to_power: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BridleExport {
    pub pivot_b: [f64; 3],
    pub radius: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TethersExport {
    pub upper: TetherExport,
    pub common: TetherExport,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TetherExport {
    pub natural_length: f64,
    pub total_mass: f64,
    pub ea: f64,
    pub viscous_damping_coeff: f64,
    pub cd_phi: f64,
    pub diameter: f64,
    pub contact: ContactExport,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ContactExport {
    pub zeta: f64,
    pub enable_length: f64,
    pub ground_altitude: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ControllerExport {
    pub trim_aileron: f64,
    pub trim_flap: f64,
    pub trim_winglet: f64,
    pub trim_elevator: f64,
    pub trim_rudder: f64,
    pub trim_motor_torque: f64,
    pub wx_to_ail: f64,
    pub wy_to_elev: f64,
    pub wz_to_rudder: f64,
    pub speed_to_torque_p: f64,
    pub speed_to_torque_i: f64,
    pub rabbit_distance: f64,
    pub phase_lag_to_radius: f64,
    pub vert_vel_to_rabbit_height: f64,
    pub gain_int_y: f64,
    pub gain_int_z: f64,
    pub speed_ref: f64,
    pub disk_center_n: [f64; 3],
    pub disk_radius: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct EnvironmentExport {
    pub rho: f64,
    pub g: f64,
    pub wind_n: [f64; 3],
}

pub fn asset_manifest() -> Result<AssetManifest> {
    Ok(serde_json::from_str(include_str!(
        "../assets/asset_manifest.json"
    ))?)
}

pub fn reference_export() -> Result<ReferenceExport> {
    Ok(serde_json::from_str(include_str!(
        "../assets/reference_export.json"
    ))?)
}
