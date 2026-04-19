use anyhow::Result;
use clap::{Parser, ValueEnum};
use multikite_sim::{
    InitRequest, PhaseMode, Preset, SimulationConfig, simulate_free_flight1,
    simulate_simple_tether, simulate_star1, simulate_star3, simulate_star4, simulate_y2,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PresetArg {
    FreeFlight1,
    Star1,
    Y2,
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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = SimulationConfig {
        duration: cli.duration,
        phase_mode: match cli.phase_mode {
            PhaseModeArg::Adaptive => PhaseMode::Adaptive,
            PhaseModeArg::OpenLoop => PhaseMode::OpenLoop,
        },
        ..SimulationConfig::default()
    };
    let init = InitRequest {
        preset: match cli.preset {
            PresetArg::FreeFlight1 => Preset::FreeFlight1,
            PresetArg::Star1 => Preset::Star1,
            PresetArg::Y2 => Preset::Y2,
            PresetArg::Star3 => Preset::Star3,
            PresetArg::Star4 => Preset::Star4,
            PresetArg::SimpleTether => Preset::SimpleTether,
        },
        payload_mass_kg: cli.payload_mass_kg,
        wind_speed_mps: cli.wind_speed_mps,
    };
    let summary = match cli.preset {
        PresetArg::FreeFlight1 => simulate_free_flight1(&init, &config)?.summary,
        PresetArg::Star1 => simulate_star1(&init, &config)?.summary,
        PresetArg::Y2 => simulate_y2(&init, &config)?.summary,
        PresetArg::Star3 => simulate_star3(&init, &config)?.summary,
        PresetArg::Star4 => simulate_star4(&init, &config)?.summary,
        PresetArg::SimpleTether => simulate_simple_tether(&init, &config)?.summary,
    };
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
