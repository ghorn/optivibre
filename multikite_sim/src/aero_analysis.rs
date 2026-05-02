use crate::assets::{reference_avl_fit_ref, reference_export};
use crate::model::{eval_aero_coeffs, eval_force_moment_flap, eval_force_moment_quartic};
use crate::types::{AeroParams, KiteControls};
use anyhow::Result;
use nalgebra::Vector3;
use serde::Serialize;

const GRID_COUNT_ALPHA_BETA: usize = 41;
const GRID_COUNT_ALPHA_DELTA: usize = 41;
const ALPHA_MIN_DEG: f64 = -15.0;
const ALPHA_MAX_DEG: f64 = 20.0;
const BETA_MIN_DEG: f64 = -30.0;
const BETA_MAX_DEG: f64 = 30.0;
const COEFF_ZERO_EPS: f64 = 1.0e-12;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AeroAnalysis {
    pub title: &'static str,
    pub note: &'static str,
    pub coefficient_grids: Vec<AeroSurfaceGrid>,
    pub rate_derivative_groups: Vec<AeroSurfaceGroup>,
    pub control_surface_groups: Vec<AeroSurfaceGroup>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AeroSurfaceGroup {
    pub id: String,
    pub title: String,
    pub note: String,
    pub grids: Vec<AeroSurfaceGrid>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AeroSurfaceGrid {
    pub id: String,
    pub title: String,
    pub x_label: &'static str,
    pub y_label: &'static str,
    pub z_label: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<Vec<f64>>,
}

pub fn build_aero_analysis() -> Result<AeroAnalysis> {
    let export = reference_export()?;
    let aero = AeroParams {
        ref_area: export.aero.ref_area,
        ref_span: export.aero.ref_span,
        ref_chord: export.aero.ref_chord,
        cl0: export.aero.cl0,
        cl_alpha: export.aero.cl_alpha,
        cl_elevator: export.aero.cl_elevator,
        cl_flap: export.aero.cl_flap,
        cd0: export.aero.cd0,
        cd_induced: export.aero.cd_induced,
        cd_surface_abs: export.aero.cd_surface_abs,
        cy_beta: export.aero.cy_beta,
        cy_rudder: export.aero.cy_rudder,
        roll_beta: export.aero.roll_beta,
        roll_p: export.aero.roll_p,
        roll_r: export.aero.roll_r,
        roll_aileron: export.aero.roll_aileron,
        pitch0: export.aero.pitch0,
        pitch_alpha: export.aero.pitch_alpha,
        pitch_q: export.aero.pitch_q,
        pitch_elevator: export.aero.pitch_elevator,
        pitch_flap: export.aero.pitch_flap,
        yaw_beta: export.aero.yaw_beta,
        yaw_p: export.aero.yaw_p,
        yaw_r: export.aero.yaw_r,
        yaw_rudder: export.aero.yaw_rudder,
    };

    Ok(AeroAnalysis {
        title: "Reference Aero Coefficient Analysis",
        note: "Surfaces are evaluated from the vendored Reference AVL-fit model. Force coefficients are shown in aerodynamic convention: C_L is positive lift, C_D is positive drag, and C_Yw is wind-frame side force. Moment coefficients are body/CAD-axis C_l, C_m, and C_n. Rate surfaces are derivatives with respect to normalized p-hat, q-hat, and r-hat. Control-surface plots use beta = 0.",
        coefficient_grids: base_coefficient_grids(&aero),
        rate_derivative_groups: rate_derivative_groups(),
        control_surface_groups: control_surface_groups(),
    })
}

fn base_coefficient_grids(aero: &AeroParams<f64>) -> Vec<AeroSurfaceGrid> {
    let components = coefficient_components();
    components
        .iter()
        .map(|component| {
            alpha_beta_grid(
                &format!("base-{}", component.id),
                &format!("{} over alpha/beta", component.label),
                component.label,
                |alpha, beta| {
                    let coeffs = eval_aero_coeffs(
                        alpha,
                        beta,
                        &Vector3::zeros(),
                        25.0,
                        aero,
                        &KiteControls::zero(),
                    );
                    (component.value)(&coeffs.total_force_w, &coeffs.total_moment_c)
                },
            )
        })
        .collect()
}

fn rate_derivative_groups() -> Vec<AeroSurfaceGroup> {
    let fit = reference_avl_fit_ref();
    let rates = [
        ("p_hat", "p-hat roll-rate derivatives", &fit.pqr[0]),
        ("q_hat", "q-hat pitch-rate derivatives", &fit.pqr[1]),
        ("r_hat", "r-hat yaw-rate derivatives", &fit.pqr[2]),
    ];
    let components = coefficient_components();

    rates
        .iter()
        .map(|(rate_id, title, fit)| AeroSurfaceGroup {
            id: (*rate_id).to_string(),
            title: (*title).to_string(),
            note: format!(
                "Coefficient derivatives with respect to normalized {}. The runtime contribution is this surface multiplied by the corresponding normalized body rate.",
                rate_id.replace('_', "-")
            ),
            grids: components
                .iter()
                .map(|component| {
                    alpha_beta_grid(
                        &format!("rate-{}-{}", rate_id, component.id),
                        &format!("d{}/d{}", component.label, rate_id),
                        format!("d{}/d{}", component.label, rate_id),
                        |alpha, beta| {
                            let (force_w, moment_c) = eval_force_moment_quartic(fit, alpha, beta);
                            (component.value)(&force_w, &moment_c)
                        },
                    )
                })
                .collect(),
        })
        .collect()
}

fn control_surface_groups() -> Vec<AeroSurfaceGroup> {
    let fit = reference_avl_fit_ref();
    let surfaces = [
        (
            "aileron",
            "Aileron surface contribution",
            15.0,
            &fit.flaps.r_aileron,
        ),
        ("flap", "Flap surface contribution", 15.0, &fit.flaps.flap),
        (
            "winglet",
            "Winglet surface contribution",
            15.0,
            &fit.flaps.winglet,
        ),
        (
            "elevator",
            "Elevator surface contribution",
            20.0,
            &fit.flaps.elevator,
        ),
        (
            "rudder",
            "Rudder surface contribution",
            25.0,
            &fit.flaps.rudder,
        ),
    ];
    let components = coefficient_components();

    surfaces
        .iter()
        .map(|(surface_id, title, limit_deg, fit)| AeroSurfaceGroup {
            id: (*surface_id).to_string(),
            title: (*title).to_string(),
            note: format!(
                "{} contribution only, swept over alpha and deflection at beta = 0 deg.",
                surface_title(surface_id)
            ),
            grids: components
                .iter()
                .map(|component| {
                    alpha_delta_grid(
                        &format!("surface-{}-{}", surface_id, component.id),
                        &format!(
                            "{} {} contribution",
                            surface_title(surface_id),
                            component.label
                        ),
                        component.label,
                        *limit_deg,
                        |alpha, delta| {
                            let (force_w, moment_c) =
                                eval_force_moment_flap(fit, alpha, 0.0, delta);
                            (component.value)(&force_w, &moment_c)
                        },
                    )
                })
                .collect(),
        })
        .collect()
}

fn alpha_beta_grid<F, S>(id: &str, title: &str, z_label: S, mut value: F) -> AeroSurfaceGrid
where
    F: FnMut(f64, f64) -> f64,
    S: Into<String>,
{
    let alpha_deg = linspace(ALPHA_MIN_DEG, ALPHA_MAX_DEG, GRID_COUNT_ALPHA_BETA);
    let beta_deg = linspace(BETA_MIN_DEG, BETA_MAX_DEG, GRID_COUNT_ALPHA_BETA);
    let z = alpha_deg
        .iter()
        .map(|alpha| {
            beta_deg
                .iter()
                .map(|beta| clean_coeff(value(alpha.to_radians(), beta.to_radians())))
                .collect()
        })
        .collect();

    AeroSurfaceGrid {
        id: id.to_string(),
        title: title.to_string(),
        x_label: "beta (deg)",
        y_label: "alpha (deg)",
        z_label: z_label.into(),
        x: beta_deg,
        y: alpha_deg,
        z,
    }
}

fn alpha_delta_grid<F, S>(
    id: &str,
    title: &str,
    z_label: S,
    delta_limit_deg: f64,
    mut value: F,
) -> AeroSurfaceGrid
where
    F: FnMut(f64, f64) -> f64,
    S: Into<String>,
{
    let alpha_deg = linspace(ALPHA_MIN_DEG, ALPHA_MAX_DEG, GRID_COUNT_ALPHA_DELTA);
    let delta_deg = linspace(-delta_limit_deg, delta_limit_deg, GRID_COUNT_ALPHA_DELTA);
    let z = alpha_deg
        .iter()
        .map(|alpha| {
            delta_deg
                .iter()
                .map(|delta| clean_coeff(value(alpha.to_radians(), delta.to_radians())))
                .collect()
        })
        .collect();

    AeroSurfaceGrid {
        id: id.to_string(),
        title: title.to_string(),
        x_label: "delta (deg)",
        y_label: "alpha (deg)",
        z_label: z_label.into(),
        x: delta_deg,
        y: alpha_deg,
        z,
    }
}

fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![start];
    }
    let step = (end - start) / (count as f64 - 1.0);
    (0..count)
        .map(|index| start + step * index as f64)
        .collect()
}

fn clean_coeff(value: f64) -> f64 {
    if value.abs() < COEFF_ZERO_EPS {
        0.0
    } else {
        value
    }
}

struct CoeffComponent {
    id: &'static str,
    label: &'static str,
    value: fn(&Vector3<f64>, &Vector3<f64>) -> f64,
}

fn coefficient_components() -> [CoeffComponent; 6] {
    [
        CoeffComponent {
            id: "clift",
            label: "C_L",
            value: |force_w, _| -force_w[2],
        },
        CoeffComponent {
            id: "cd",
            label: "C_D",
            value: |force_w, _| -force_w[0],
        },
        CoeffComponent {
            id: "cyw",
            label: "C_Yw",
            value: |force_w, _| force_w[1],
        },
        CoeffComponent {
            id: "cl",
            label: "C_l",
            value: |_, moment_c| moment_c[0],
        },
        CoeffComponent {
            id: "cm",
            label: "C_m",
            value: |_, moment_c| moment_c[1],
        },
        CoeffComponent {
            id: "cn",
            label: "C_n",
            value: |_, moment_c| moment_c[2],
        },
    ]
}

fn surface_title(surface_id: &str) -> &'static str {
    match surface_id {
        "aileron" => "Aileron",
        "flap" => "Flap",
        "winglet" => "Winglet",
        "elevator" => "Elevator",
        "rudder" => "Rudder",
        _ => "Surface",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficient_components_use_aerodynamic_force_signs() {
        let components = coefficient_components();
        let force_w = Vector3::new(-0.12, 0.03, -0.8);
        let moment_c = Vector3::new(0.01, -0.02, 0.03);

        assert_eq!(components[0].label, "C_L");
        assert_eq!((components[0].value)(&force_w, &moment_c), 0.8);
        assert_eq!(components[1].label, "C_D");
        assert_eq!((components[1].value)(&force_w, &moment_c), 0.12);
        assert_eq!(components[2].label, "C_Yw");
        assert_eq!((components[2].value)(&force_w, &moment_c), 0.03);
        assert_eq!(components[3].label, "C_l");
        assert_eq!((components[3].value)(&force_w, &moment_c), 0.01);
        assert_eq!(components[4].label, "C_m");
        assert_eq!((components[4].value)(&force_w, &moment_c), -0.02);
        assert_eq!(components[5].label, "C_n");
        assert_eq!((components[5].value)(&force_w, &moment_c), 0.03);
    }

    #[test]
    fn clean_coeff_removes_roundoff_only() {
        assert_eq!(clean_coeff(5.0e-13), 0.0);
        assert_eq!(clean_coeff(-5.0e-13), 0.0);
        assert_eq!(clean_coeff(2.0e-12), 2.0e-12);
        assert_eq!(clean_coeff(-2.0e-12), -2.0e-12);
    }
}
