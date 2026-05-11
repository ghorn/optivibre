use std::collections::hash_map::DefaultHasher;
use std::env;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let src_dir = manifest_dir.join("src");
    let mut files = Vec::new();
    collect_rs_files(&src_dir, &mut files);
    files.sort();

    let mut hasher = DefaultHasher::new();
    for file in files {
        println!("cargo:rerun-if-changed={}", file.display());
        let relative = file.strip_prefix(&manifest_dir).unwrap_or(&file);
        relative.to_string_lossy().hash(&mut hasher);
        let bytes = fs::read(&file).unwrap_or_else(|error| {
            panic!(
                "failed to read {} for test-problem fingerprint: {error}",
                file.display()
            )
        });
        bytes.hash(&mut hasher);
    }
    println!(
        "cargo:rustc-env=TEST_PROBLEMS_BUILD_FINGERPRINT={:016x}",
        hasher.finish()
    );
}

fn collect_rs_files(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("failed to read source dir {}: {error}", dir.display()))
    {
        let entry = entry.expect("source dir entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}
