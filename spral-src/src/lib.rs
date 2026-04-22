#![no_std]

pub const SPRAL_VERSION: &str = "2025.09.18";
pub const METIS_VERSION: &str = "5.2.1";
pub const OPENBLAS_VERSION: &str = "0.3.32";
pub const OPENMP_REQUIRED: bool = true;
pub const SYSTEM_SOLVER_MATH_FALLBACKS: bool = false;

pub fn config_summary() -> &'static str {
    "spral=2025.09.18;metis=5.2.1;openblas=0.3.32;openmp=required;system_solver_math_fallbacks=false"
}
