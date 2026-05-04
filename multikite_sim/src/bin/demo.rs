use anyhow::Result;
use clap::{Parser, ValueEnum};
use multikite_sim::{
    DEFAULT_SWARM_KITES, InitRequest, LongitudinalMode, MAX_SWARM_KITES, MIN_SWARM_KITES,
    PhaseMode, Preset, SimulationConfig, control_roll_pitch_rad_from_quat_n2b,
    simulate_free_flight1, simulate_free_flight1_with_callbacks, simulate_simple_tether,
    simulate_swarm, simulate_swarm_with_callbacks,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PresetArg {
    Swarm,
    FreeFlight1,
    SimpleTether,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PhaseModeArg {
    Adaptive,
    OpenLoop,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum LongitudinalModeArg {
    TotalEnergy,
    MaxThrottleAltitudePitch,
}

#[derive(Debug, Parser)]
struct Cli {
    #[arg(long, default_value = "swarm")]
    preset: PresetArg,
    #[arg(long, default_value_t = DEFAULT_SWARM_KITES)]
    swarm_kites: usize,
    #[arg(long)]
    swarm_payload_altitude_m: Option<f64>,
    #[arg(long)]
    swarm_disk_altitude_m: Option<f64>,
    #[arg(long)]
    swarm_aircraft_altitude_m: Option<f64>,
    #[arg(long)]
    swarm_disk_diameter_m: Option<f64>,
    #[arg(long)]
    swarm_upper_tether_length_m: Option<f64>,
    #[arg(long)]
    swarm_common_tether_length_m: Option<f64>,
    #[arg(long, default_value_t = 10.0)]
    duration: f64,
    #[arg(long, default_value = "adaptive")]
    phase_mode: PhaseModeArg,
    #[arg(long)]
    payload_mass_kg: Option<f64>,
    #[arg(long)]
    wind_speed_mps: Option<f64>,
    #[arg(long, default_value = "total-energy")]
    longitudinal_mode: LongitudinalModeArg,
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
        longitudinal_mode: match cli.longitudinal_mode {
            LongitudinalModeArg::TotalEnergy => LongitudinalMode::TotalEnergy,
            LongitudinalModeArg::MaxThrottleAltitudePitch => {
                LongitudinalMode::MaxThrottleAltitudePitch
            }
        },
        sim_noise_enabled: cli.sim_noise,
        bridle_enabled: !cli.disable_bridle,
        ..SimulationConfig::default()
    };
    let init = InitRequest {
        preset: match cli.preset {
            PresetArg::Swarm => Preset::Swarm,
            PresetArg::FreeFlight1 => Preset::FreeFlight1,
            PresetArg::SimpleTether => Preset::SimpleTether,
        },
        payload_mass_kg: cli.payload_mass_kg,
        wind_speed_mps: cli.wind_speed_mps,
        swarm_kites: cli.swarm_kites.clamp(MIN_SWARM_KITES, MAX_SWARM_KITES),
        swarm_payload_altitude_m: cli.swarm_payload_altitude_m,
        swarm_disk_altitude_m: cli.swarm_disk_altitude_m,
        swarm_aircraft_altitude_m: cli.swarm_aircraft_altitude_m,
        swarm_disk_diameter_m: cli.swarm_disk_diameter_m,
        swarm_upper_tether_length_m: cli.swarm_upper_tether_length_m,
        swarm_common_tether_length_m: cli.swarm_common_tether_length_m,
    };
    let summary = match (cli.preset, cli.trace_every) {
        (PresetArg::Swarm, Some(trace_every)) => simulate_swarm_trace(&init, &config, trace_every)?,
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
        (_, Some(_)) => {
            anyhow::bail!("--trace-every is implemented for swarm and free-flight1");
        }
        (PresetArg::Swarm, None) => simulate_swarm_summary(&init, &config)?,
        (PresetArg::FreeFlight1, None) => simulate_free_flight1(&init, &config)?.summary,
        (PresetArg::SimpleTether, None) => simulate_simple_tether(&init, &config)?.summary,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn simulate_swarm_summary(
    init: &InitRequest,
    config: &SimulationConfig,
) -> Result<multikite_sim::RunSummary> {
    macro_rules! run_swarm {
        ($nk:literal) => {
            Ok(simulate_swarm::<$nk>(init, config)?.summary)
        };
    }
    match init.swarm_kites {
        1 => run_swarm!(1),
        2 => run_swarm!(2),
        3 => run_swarm!(3),
        4 => run_swarm!(4),
        5 => run_swarm!(5),
        6 => run_swarm!(6),
        7 => run_swarm!(7),
        8 => run_swarm!(8),
        9 => run_swarm!(9),
        10 => run_swarm!(10),
        11 => run_swarm!(11),
        12 => run_swarm!(12),
        count => anyhow::bail!(
            "swarm kites must be in {MIN_SWARM_KITES}..={MAX_SWARM_KITES}, got {count}"
        ),
    }
}

fn simulate_swarm_trace(
    init: &InitRequest,
    config: &SimulationConfig,
    trace_every: usize,
) -> Result<multikite_sim::RunSummary> {
    macro_rules! run_swarm {
        ($nk:literal) => {{
            let mut frame_index = 0usize;
            Ok(simulate_swarm_with_callbacks::<$nk, _, _>(
                init,
                config,
                &mut |_| {},
                &mut |frame| {
                    if frame_index % trace_every == 0 {
                        print_trace_frame(frame_index, frame.time, &frame);
                    }
                    frame_index += 1;
                },
            )?
            .summary)
        }};
    }
    match init.swarm_kites {
        1 => run_swarm!(1),
        2 => run_swarm!(2),
        3 => run_swarm!(3),
        4 => run_swarm!(4),
        5 => run_swarm!(5),
        6 => run_swarm!(6),
        7 => run_swarm!(7),
        8 => run_swarm!(8),
        9 => run_swarm!(9),
        10 => run_swarm!(10),
        11 => run_swarm!(11),
        12 => run_swarm!(12),
        count => anyhow::bail!(
            "swarm kites must be in {MIN_SWARM_KITES}..={MAX_SWARM_KITES}, got {count}"
        ),
    }
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
        let [roll, pitch] = control_roll_pitch_rad_from_quat_n2b(quat);
        eprintln!(
            concat!(
                "trace frame={frame_index} t={time:.3} kite={kite} ",
                "air={air:.3} alpha={alpha:.3} beta={beta:.3} ",
                "roll={roll:.3} roll_ref={roll_ref:.3} pitch={pitch:.3} pitch_ref={pitch_ref:.3} ",
                "pos=({px:.3},{py:.3},{pz:.3}) vel=({vx:.3},{vy:.3},{vz:.3}) ",
                "quat=({qw:.5},{qx:.5},{qy:.5},{qz:.5}) ",
                "alt={alt:.3} alt_ref={alt_ref:.3} phase_err={phase:.5} ",
                "p={p:.3} q={q:.3} r={r:.3} tension={tension:.3} ",
                "acc_n=({anx:.3},{any:.3},{anz:.3}) force_b=({fbx:.3},{fby:.3},{fbz:.3}) ",
                "moment=({mx:.3},{my:.3},{mz:.3}) aero_m=({amx:.3},{amy:.3},{amz:.3}) ",
                "rudder_m=({rmx:.3},{rmy:.3},{rmz:.3}) tether_m=({tmx:.3},{tmy:.3},{tmz:.3}) ",
                "yaw_coeff={yaw_coeff:.5} yaw_beta={yaw_beta:.5} yaw_r={yaw_r:.5} yaw_rudder={yaw_rudder:.5} ",
                "cy={cy:.5} cy_beta={cy_beta:.5} cy_rudder={cy_rudder:.5} ",
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
            px = kite.cad_position_n[0],
            py = kite.cad_position_n[1],
            pz = kite.cad_position_n[2],
            vx = kite.cad_velocity_n[0],
            vy = kite.cad_velocity_n[1],
            vz = kite.cad_velocity_n[2],
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
            anx = kite.body_accel_n[0],
            any = kite.body_accel_n[1],
            anz = kite.body_accel_n[2],
            fbx = kite.total_force_b[0],
            fby = kite.total_force_b[1],
            fbz = kite.total_force_b[2],
            mx = kite.total_moment_b[0],
            my = kite.total_moment_b[1],
            mz = kite.total_moment_b[2],
            amx = kite.aero_moment_b[0],
            amy = kite.aero_moment_b[1],
            amz = kite.aero_moment_b[2],
            rmx = kite.rudder_moment_b[0],
            rmy = kite.rudder_moment_b[1],
            rmz = kite.rudder_moment_b[2],
            tmx = kite.tether_moment_b[0],
            tmy = kite.tether_moment_b[1],
            tmz = kite.tether_moment_b[2],
            yaw_coeff = kite.yaw_coeff_total,
            yaw_beta = kite.yaw_beta_term,
            yaw_r = kite.yaw_r_term,
            yaw_rudder = kite.yaw_rudder_term,
            cy = kite.cy_total,
            cy_beta = kite.cy_beta_term,
            cy_rudder = kite.cy_rudder_term,
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
