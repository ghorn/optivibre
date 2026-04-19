use crate::types::Diagnostics;
use nalgebra::Vector3;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use std::f64::consts::PI;

const METERS_TO_FEET: f64 = 3.28084;

#[derive(Clone, Debug)]
struct DrydenState {
    #[expect(
        dead_code,
        reason = "Retained to mirror the Haskell DrydenState layout."
    )]
    x_noise: Vector3<f64>,
    x_noise_f_z1: Vector3<f64>,
    x_noise2_f_z1: Vector3<f64>,
}

impl DrydenState {
    fn zero() -> Self {
        Self {
            x_noise: Vector3::zeros(),
            x_noise_f_z1: Vector3::zeros(),
            x_noise2_f_z1: Vector3::zeros(),
        }
    }
}

#[derive(Debug)]
pub struct DrydenField<const NK: usize> {
    states: [DrydenState; NK],
    gusts_n: [Vector3<f64>; NK],
    rng: StdRng,
}

impl<const NK: usize> DrydenField<NK> {
    pub fn new(seed: u64) -> Self {
        Self {
            states: std::array::from_fn(|_| DrydenState::zero()),
            gusts_n: std::array::from_fn(|_| Vector3::zeros()),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn gusts_n(&self) -> [Vector3<f64>; NK] {
        self.gusts_n
    }

    pub fn advance(
        &mut self,
        dt_seconds: f64,
        diagnostics: &Diagnostics<f64, NK>,
        clean_wind_n: &Vector3<f64>,
        ground_altitude_m: f64,
    ) {
        for kite_index in 0..NK {
            let agl_m =
                (-diagnostics.kites[kite_index].cad_position_n[2] - ground_altitude_m).max(0.0);
            let airspeed_mps = diagnostics.kites[kite_index].airspeed.max(0.0);
            let std_noise = Vector3::new(
                StandardNormal.sample(&mut self.rng),
                StandardNormal.sample(&mut self.rng),
                StandardNormal.sample(&mut self.rng),
            );
            self.states[kite_index] = update_dryden_turb(
                dt_seconds,
                agl_m,
                airspeed_mps,
                std_noise,
                &self.states[kite_index],
            );
            self.gusts_n[kite_index] =
                calc_wind_turbulence(agl_m, clean_wind_n, &self.states[kite_index]);
        }
    }
}

fn smooth_saturate(bounds: (f64, f64), value: f64) -> f64 {
    let (lower, upper) = bounds;
    if (upper - lower).abs() < 1.0e-12 {
        return upper;
    }
    let mean = 0.5 * (lower + upper);
    let magnitude = 0.5 * (upper - lower);
    let slope = 1.0;
    let c = 2.0 * slope / magnitude;
    magnitude * 2.0 * (1.0 / (1.0 + (-c * (value - mean)).exp()) - 0.5) + mean
}

fn calc_dryden_turb_length_scale(agl_ft: f64) -> Vector3<f64> {
    let l_wg = smooth_saturate((10.0, 1000.0), agl_ft);
    let l_uvg = l_wg / (0.177 + 0.000823 * l_wg).powf(1.2);
    Vector3::new(l_uvg, l_uvg, l_wg)
}

fn calc_dryden_turb_intensities(agl_ft: f64, wind20_ft: f64) -> Vector3<f64> {
    let agl_sat_ft = smooth_saturate((0.0, 1000.0), agl_ft);
    let sigma_wg = 0.1 * wind20_ft;
    let sigma_uvg = sigma_wg / (0.177 + 0.000823 * agl_sat_ft).powf(0.4);
    Vector3::new(sigma_uvg, sigma_uvg, sigma_wg)
}

fn calc_dryden_turb(agl_m: f64, wind_n: &Vector3<f64>, dryden_state: &DrydenState) -> Vector3<f64> {
    let sigma_ft =
        calc_dryden_turb_intensities(METERS_TO_FEET * agl_m, METERS_TO_FEET * wind_n.norm());
    let noise = Vector3::new(
        dryden_state.x_noise_f_z1[0],
        dryden_state.x_noise2_f_z1[1],
        dryden_state.x_noise2_f_z1[2],
    );
    noise.component_mul(&sigma_ft) / METERS_TO_FEET
}

fn calc_wind_turbulence(
    agl_m: f64,
    wind_n: &Vector3<f64>,
    dryden_state: &DrydenState,
) -> Vector3<f64> {
    let wind_dir = (-wind_n[1]).atan2(-wind_n[0]);
    let dcm_w2n = [
        Vector3::new(wind_dir.cos(), wind_dir.sin(), 0.0),
        Vector3::new(-wind_dir.sin(), wind_dir.cos(), 0.0),
        Vector3::new(0.0, 0.0, 1.0),
    ];
    let wind_turb_w = calc_dryden_turb(agl_m, wind_n, dryden_state);
    dcm_w2n[0] * wind_turb_w[0] + dcm_w2n[1] * wind_turb_w[1] + dcm_w2n[2] * wind_turb_w[2]
}

fn update_dryden_turb(
    dt_seconds: f64,
    agl_m: f64,
    apparent_airspeed_mps: f64,
    std_noise: Vector3<f64>,
    state: &DrydenState,
) -> DrydenState {
    let v_app_norm_ft = (1.0 + (METERS_TO_FEET * apparent_airspeed_mps).powi(2)).sqrt();
    let l_ft = calc_dryden_turb_length_scale(METERS_TO_FEET * agl_m);
    let tc = l_ft / v_app_norm_ft;
    let noise_prime = std_noise * (PI / dt_seconds.max(1.0e-6)).sqrt();
    let too = Vector3::new(2.0, 1.0, 1.0);
    let tc_scaled = Vector3::new(
        (too[0] * tc[0] / PI).sqrt(),
        (too[1] * tc[1] / PI).sqrt(),
        (too[2] * tc[2] / PI).sqrt(),
    );
    let dnoise = (noise_prime.component_mul(&tc_scaled) - state.x_noise_f_z1).component_div(&tc);
    let noise_f_z1 = state.x_noise_f_z1 + dnoise * dt_seconds;
    let noise2_f_z1 = state.x_noise2_f_z1
        + (noise_f_z1 + dnoise.component_mul(&tc) * 3.0_f64.sqrt() - state.x_noise2_f_z1)
            .component_div(&tc)
            * dt_seconds;
    DrydenState {
        x_noise: noise_prime,
        x_noise_f_z1: noise_f_z1,
        x_noise2_f_z1: noise2_f_z1,
    }
}

#[cfg(test)]
mod tests {
    use super::{DrydenField, calc_wind_turbulence};
    use crate::model::blank_diagnostics;
    use nalgebra::Vector3;

    #[test]
    fn zero_state_produces_zero_turbulence() {
        let field = DrydenField::<1>::new(7);
        let gust = calc_wind_turbulence(100.0, &Vector3::new(5.0, 0.0, 0.0), &field.states[0]);
        assert!(gust.norm() < 1.0e-12);
    }

    #[test]
    fn dryden_update_stays_finite() {
        let mut field = DrydenField::<2>::new(9);
        let mut diagnostics = blank_diagnostics::<f64, 2>();
        diagnostics.kites[0].cad_position_n = Vector3::new(0.0, 0.0, -120.0);
        diagnostics.kites[1].cad_position_n = Vector3::new(0.0, 0.0, -180.0);
        diagnostics.kites[0].airspeed = 22.0;
        diagnostics.kites[1].airspeed = 28.0;
        field.advance(0.02, &diagnostics, &Vector3::new(5.0, 0.0, 0.0), 0.0);
        for gust in field.gusts_n() {
            assert!(gust.iter().all(|value| value.is_finite()));
        }
    }
}
