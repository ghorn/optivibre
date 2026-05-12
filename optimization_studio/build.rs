use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=OPT_LEVEL");
    let opt_level = env::var("OPT_LEVEL").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=OPTIVIBRE_OPT_LEVEL={opt_level}");

    for path in [
        "static/app.ts",
        "static/globals.d.ts",
        "tsconfig.json",
        "package.json",
        "package-lock.json",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }

    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR missing"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR missing"));
    fs::create_dir_all(&out_dir).unwrap_or_else(|error| {
        panic!(
            "failed to create frontend output dir {}: {error}",
            out_dir.display()
        )
    });
    let status = Command::new("npm")
        .args(["exec", "tsc", "--", "-p", "tsconfig.json", "--outDir"])
        .arg(&out_dir)
        .current_dir(&manifest_dir)
        .status()
        .unwrap_or_else(|error| {
            panic!(
                "failed to run `npm exec tsc -- -p tsconfig.json --outDir {}` in {}: {error}",
                out_dir.display(),
                manifest_dir.display()
            )
        });

    if !status.success() {
        panic!(
            "frontend TypeScript build failed in {} with status {status}",
            manifest_dir.display()
        );
    }
}
