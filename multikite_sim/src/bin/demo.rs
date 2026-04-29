use anyhow::Result;
use clap::{Parser, ValueEnum};
use multikite_sim::{
    InitRequest, PhaseMode, Preset, SimulationConfig, simulate_free_flight1,
    simulate_free_flight1_with_callbacks, simulate_simple_tether, simulate_star1,
    simulate_star1_with_callbacks, simulate_star3, simulate_star4, simulate_y2,
    simulate_y2_reference, simulate_y2_reference_with_callbacks, simulate_y2_with_callbacks,
};
use nalgebra::Vector3;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PresetArg {
    FreeFlight1,
    Star1,
    Y2,
    Y2Reference,
    Star3,
    Star4,
    SimpleTether,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PhaseModeArg {
    Adaptive,
    OpenLoop,
}

#[derive(Debug, Parser)]
struct Cli {
    #[arg(long, default_value = "y2")]
    preset: PresetArg,
    #[arg(long, default_value_t = 10.0)]
    duration: f64,
    #[arg(long, default_value = "adaptive")]
    phase_mode: PhaseModeArg,
    #[arg(long)]
    payload_mass_kg: Option<f64>,
    #[arg(long)]
    wind_speed_mps: Option<f64>,
    #[arg(long, help = "Enable stochastic Dryden gusts; disabled by default")]
    sim_noise: bool,
    #[arg(
        long,
        help = "Attach upper tethers at the body CG instead of the bridle"
    )]
    disable_bridle: bool,
    #[arg(long)]
    trace_every: Option<usize>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = SimulationConfig {
        duration: cli.duration,
        phase_mode: match cli.phase_mode {
            PhaseModeArg::Adaptive => PhaseMode::Adaptive,
            PhaseModeArg::OpenLoop => PhaseMode::OpenLoop,
        },
        sim_noise_enabled: cli.sim_noise,
        bridle_enabled: !cli.disable_bridle,
        ..SimulationConfig::default()
    };
    let init = InitRequest {
        preset: match cli.preset {
            PresetArg::FreeFlight1 => Preset::FreeFlight1,
            PresetArg::Star1 => Preset::Star1,
            PresetArg::Y2 => Preset::Y2,
            PresetArg::Y2Reference => Preset::Y2Reference,
            PresetArg::Star3 => Preset::Star3,
            PresetArg::Star4 => Preset::Star4,
            PresetArg::SimpleTether => Preset::SimpleTether,
        },
        payload_mass_kg: cli.payload_mass_kg,
        wind_speed_mps: cli.wind_speed_mps,
    };
    let summary = match (cli.preset, cli.trace_every) {
        (PresetArg::FreeFlight1, Some(trace_every)) => {
            let mut frame_index = 0usize;
            simulate_free_flight1_with_callbacks(&init, &config, &mut |_| {}, &mut |frame| {
                if frame_index % trace_every == 0 {
                    print_trace_frame(frame_index, frame.time, &frame);
                }
                frame_index += 1;
            })?
            .summary
        }
        (PresetArg::Star1, Some(trace_every)) => {
            let mut frame_index = 0usize;
            simulate_star1_with_callbacks(&init, &config, &mut |_| {}, &mut |frame| {
                if frame_index % trace_every == 0 {
                    print_trace_frame(frame_index, frame.time, &frame);
                }
                frame_index += 1;
            })?
            .summary
        }
        (PresetArg::Y2, Some(trace_every)) => {
            let mut frame_index = 0usize;
            simulate_y2_with_callbacks(&init, &config, &mut |_| {}, &mut |frame| {
                if frame_index % trace_every == 0 {
                    print_trace_frame(frame_index, frame.time, &frame);
                }
                frame_index += 1;
            })?
            .summary
        }
        (PresetArg::Y2Reference, Some(trace_every)) => {
            let mut frame_index = 0usize;
            simulate_y2_reference_with_callbacks(&init, &config, &mut |_| {}, &mut |frame| {
                if frame_index % trace_every == 0 {
                    print_trace_frame(frame_index, frame.time, &frame);
                }
                frame_index += 1;
            })?
            .summary
        }
        (_, Some(_)) => {
            anyhow::bail!(
                "--trace-every is currently implemented for free-flight1, star1, y2, and y2-reference"
            );
        }
        (PresetArg::FreeFlight1, None) => simulate_free_flight1(&init, &config)?.summary,
        (PresetArg::Star1, None) => simulate_star1(&init, &config)?.summary,
        (PresetArg::Y2, None) => simulate_y2(&init, &config)?.summary,
        (PresetArg::Y2Reference, None) => simulate_y2_reference(&init, &config)?.summary,
        (PresetArg::Star3, None) => simulate_star3(&init, &config)?.summary,
        (PresetArg::Star4, None) => simulate_star4(&init, &config)?.summary,
        (PresetArg::SimpleTether, None) => simulate_simple_tether(&init, &config)?.summary,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn print_trace_frame<const NK: usize, const N_COMMON: usize, const N_UPPER: usize>(
    frame_index: usize,
    time: f64,
    frame: &multikite_sim::SimulationFrame<f64, NK, N_COMMON, N_UPPER>,
) {
    for kite_index in 0..NK {
        let kite = &frame.diagnostics.kites[kite_index];
        let controls = &frame.controls.kites[kite_index];
        let quat = &frame.state.kites[kite_index].body.quat_n2b;
        let roll = roll_angle_from_quat_n2b(quat);
        let pitch = pitch_angle_from_quat_n2b(quat);
        eprintln!(
            concat!(
                "trace frame={frame_index} t={time:.3} kite={kite} ",
                "air={air:.3} alpha={alpha:.3} beta={beta:.3} ",
                "roll={roll:.3} roll_ref={roll_ref:.3} pitch={pitch:.3} pitch_ref={pitch_ref:.3} ",
                "quat=({qw:.5},{qx:.5},{qy:.5},{qz:.5}) ",
                "alt={alt:.3} alt_ref={alt_ref:.3} phase_err={phase:.5} ",
                "p={p:.3} q={q:.3} r={r:.3} tension={tension:.3} ",
                "moment=({mx:.3},{my:.3},{mz:.3}) tether_m=({tmx:.3},{tmy:.3},{tmz:.3}) ",
                "ail={ail:.3} elev={elev:.3} rud={rud:.3} torque={torque:.3} ",
                "curv_y={curv_y:.5} curv_y_ref={curv_y_ref:.5} omega_z={omega_z:.5} omega_z_ref={omega_z_ref:.5}"
            ),
            frame_index = frame_index,
            time = time,
            kite = kite_index + 1,
            air = kite.airspeed,
            alpha = kite.alpha.to_degrees(),
            beta = kite.beta.to_degrees(),
            roll = roll.to_degrees(),
            roll_ref = kite.roll_ref.to_degrees(),
            pitch = pitch.to_degrees(),
            pitch_ref = kite.pitch_ref.to_degrees(),
            qw = quat.coords[3],
            qx = quat.coords[0],
            qy = quat.coords[1],
            qz = quat.coords[2],
            alt = kite.altitude,
            alt_ref = kite.altitude_ref,
            phase = kite.phase_error,
            p = kite.omega_b[0],
            q = kite.omega_b[1],
            r = kite.omega_b[2],
            tension = kite.top_tension,
            mx = kite.total_moment_b[0],
            my = kite.total_moment_b[1],
            mz = kite.total_moment_b[2],
            tmx = kite.tether_moment_b[0],
            tmy = kite.tether_moment_b[1],
            tmz = kite.tether_moment_b[2],
            ail = controls.surfaces.aileron.to_degrees(),
            elev = controls.surfaces.elevator.to_degrees(),
            rud = controls.surfaces.rudder.to_degrees(),
            torque = controls.motor_torque,
            curv_y = kite.curvature_y_est,
            curv_y_ref = kite.curvature_y_ref,
            omega_z = kite.omega_world_z,
            omega_z_ref = kite.omega_world_z_ref,
        );
    }
}

fn rotate_nav_to_body(
    quat_n2b: &nalgebra::Quaternion<f64>,
    value_n: &Vector3<f64>,
) -> Vector3<f64> {
    let pure = nalgebra::Quaternion::new(0.0, value_n[0], value_n[1], value_n[2]);
    let conjugate = quat_n2b.conjugate();
    let rotated = quat_n2b * pure * conjugate;
    Vector3::new(rotated.coords[0], rotated.coords[1], rotated.coords[2])
}

fn roll_angle_from_quat_n2b(quat_n2b: &nalgebra::Quaternion<f64>) -> f64 {
    let down_b = rotate_nav_to_body(quat_n2b, &Vector3::new(0.0, 0.0, 1.0));
    down_b[1].atan2(down_b[2])
}

fn pitch_angle_from_quat_n2b(quat_n2b: &nalgebra::Quaternion<f64>) -> f64 {
    let down_b = rotate_nav_to_body(quat_n2b, &Vector3::new(0.0, 0.0, 1.0));
    (-down_b[0]).atan2((down_b[1] * down_b[1] + down_b[2] * down_b[2]).sqrt())
}
