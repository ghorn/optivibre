use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::sync::OnceLock;

static ASSET_MANIFEST_CACHE: OnceLock<Result<AssetManifest, String>> = OnceLock::new();
static REFERENCE_EXPORT_CACHE: OnceLock<Result<ReferenceExport, String>> = OnceLock::new();
static REFERENCE_AVL_FIT_CACHE: OnceLock<Result<AeroCoeffModelExport, String>> = OnceLock::new();
static REFERENCE_ROTOR_FIT_CACHE: OnceLock<Result<RotorCoeffFitsExport, String>> = OnceLock::new();

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
    pub position_b: [f64; 3],
    pub radius: f64,
    pub inertia: f64,
    pub sign: f64,
    pub initial_speed: f64,
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

#[derive(Clone, Debug, Deserialize)]
pub struct AeroCoeffModelExport {
    #[serde(rename = "acmNominal")]
    pub nominal: ForceMomentQuartic2Export,
    #[serde(rename = "acmPqr")]
    pub pqr: [ForceMomentQuartic2Export; 3],
    #[serde(rename = "acmFlaps")]
    pub flaps: AeroFlapsExport,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AeroFlapsExport {
    #[serde(rename = "uRAileron")]
    pub r_aileron: ForceMomentFlapPolynomialExport,
    #[serde(rename = "uFlap")]
    pub flap: ForceMomentFlapPolynomialExport,
    #[serde(rename = "uWinglet")]
    pub winglet: ForceMomentFlapPolynomialExport,
    #[serde(rename = "uElevator")]
    pub elevator: ForceMomentFlapPolynomialExport,
    #[serde(rename = "uRudder")]
    pub rudder: ForceMomentFlapPolynomialExport,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ForceMomentQuartic2Export {
    #[serde(rename = "fmForce")]
    pub force: [Quartic2Export; 3],
    #[serde(rename = "fmMoment")]
    pub moment: [Quartic2Export; 3],
}

#[derive(Clone, Debug, Deserialize)]
pub struct ForceMomentFlapPolynomialExport {
    #[serde(rename = "fmForce")]
    pub force: [FlapPolynomialExport; 3],
    #[serde(rename = "fmMoment")]
    pub moment: [FlapPolynomialExport; 3],
}

#[derive(Clone, Debug, Deserialize)]
pub struct Quartic2Export {
    pub p42x0y0: f64,
    pub p42x1y0: f64,
    pub p42x0y1: f64,
    pub p42x0y2: f64,
    pub p42x1y1: f64,
    pub p42x2y0: f64,
    pub p42x0y3: f64,
    pub p42x1y2: f64,
    pub p42x2y1: f64,
    pub p42x3y0: f64,
    pub p42x0y4: f64,
    pub p42x1y3: f64,
    pub p42x2y2: f64,
    pub p42x3y1: f64,
    pub p42x4y0: f64,
}

#[allow(non_snake_case)]
#[derive(Clone, Debug, Deserialize)]
pub struct FlapPolynomialExport {
    pub fpA0B0D1: f64,
    pub fpA0B0D2: f64,
    pub fpA1B0D1: f64,
    pub fpA0B1D1: f64,
    pub fpA0B0D3: f64,
    pub fpA0B1D2: f64,
    pub fpA1B0D2: f64,
    pub fpA1B1D1: f64,
    pub fpA0B2D1: f64,
    pub fpA2B0D1: f64,
    pub fpA0B0D4: f64,
    pub fpA1B0D3: f64,
    pub fpA0B1D3: f64,
    pub fpA2B0D2: f64,
    pub fpA0B2D2: f64,
    pub fpA1B1D2: f64,
    pub fpA3B0D1: f64,
    pub fpA2B1D1: f64,
    pub fpA1B2D1: f64,
    pub fpA0B3D1: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RotorCoeffFitsExport {
    #[serde(rename = "rcAeroThrust")]
    pub aero_thrust: Quartic2Export,
    #[serde(rename = "rcAeroTorque")]
    pub aero_torque: Quartic2Export,
}

pub fn asset_manifest() -> Result<AssetManifest> {
    match ASSET_MANIFEST_CACHE.get_or_init(|| {
        serde_json::from_str(include_str!("../assets/asset_manifest.json"))
            .map_err(|e| e.to_string())
    }) {
        Ok(manifest) => Ok(manifest.clone()),
        Err(error) => Err(anyhow!(error.clone())),
    }
}

pub fn reference_export() -> Result<ReferenceExport> {
    match REFERENCE_EXPORT_CACHE.get_or_init(|| {
        serde_json::from_str(include_str!("../assets/reference_export.json"))
            .map_err(|e| e.to_string())
    }) {
        Ok(export) => Ok(export.clone()),
        Err(error) => Err(anyhow!(error.clone())),
    }
}

pub fn reference_avl_fit_ref() -> &'static AeroCoeffModelExport {
    match REFERENCE_AVL_FIT_CACHE.get_or_init(|| {
        serde_json::from_str(include_str!("../assets/AVL_Reference_fit.json"))
            .map_err(|e| e.to_string())
    }) {
        Ok(fit) => fit,
        Err(error) => panic!("failed to parse AVL_Reference_fit.json: {error}"),
    }
}

pub fn reference_rotor_fit_ref() -> &'static RotorCoeffFitsExport {
    match REFERENCE_ROTOR_FIT_CACHE.get_or_init(|| {
        serde_json::from_str(include_str!("../assets/XROTOR_Reference_fit.json"))
            .map_err(|e| e.to_string())
    }) {
        Ok(fit) => fit,
        Err(error) => panic!("failed to parse XROTOR_Reference_fit.json: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::asset_manifest;

    #[test]
    fn manifest_pins_pre_removal_physics_commit() {
        let manifest = asset_manifest().expect("asset manifest");
        assert_eq!(
            manifest.reference_source_commits.physics,
            "2052ae8e69af45be8a2aee4eee14edd9c88ff68f"
        );
    }
}
