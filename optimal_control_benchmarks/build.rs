use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=OPT_LEVEL");
    let opt_level = env::var("OPT_LEVEL").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=OPTIVIBRE_OPT_LEVEL={opt_level}");
}
