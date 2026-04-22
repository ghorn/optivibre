#![no_std]

pub const SPRAL_VERSION: &str = "2025.09.18";
pub const METIS_VERSION: &str = "5.2.1";
pub const OPENBLAS_VERSION: &str = "0.3.32";
pub const OPENBLAS_THREADING: &str = env!("SPRAL_SRC_OPENBLAS_THREADING");
pub const OPENMP_REQUIRED: bool = true;
pub const SYSTEM_SOLVER_MATH_FALLBACKS: bool = false;

pub fn config_summary() -> &'static str {
    match OPENBLAS_THREADING {
        "serial" => {
            "spral=2025.09.18;metis=5.2.1;openblas=0.3.32;openblas_threading=serial;openmp=required;system_solver_math_fallbacks=false"
        }
        "openmp" => {
            "spral=2025.09.18;metis=5.2.1;openblas=0.3.32;openblas_threading=openmp;openmp=required;system_solver_math_fallbacks=false"
        }
        _ => {
            "spral=2025.09.18;metis=5.2.1;openblas=0.3.32;openblas_threading=unknown;openmp=required;system_solver_math_fallbacks=false"
        }
    }
}
