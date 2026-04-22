use std::env;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use tar::Archive;

const IPOPT_VERSION: &str = "3.14.20";
const IPOPT_COMMIT: &str = "4667204c76e534d3e4df6b1462f258a4f9c681bd";
const IPOPT_ARCHIVE_ROOT: &str = "Ipopt-4667204c76e534d3e4df6b1462f258a4f9c681bd";
const IPOPT_URL: &str =
    "https://github.com/coin-or/Ipopt/archive/4667204c76e534d3e4df6b1462f258a4f9c681bd.tar.gz";
const IPOPT_SHA256: &str = "ca552a9ca9d457fd4ff72405a5fee3a70e8a52016d2d5dba58453980db66c667";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    validate_feature_mode();
    for var in [
        "CC",
        "CXX",
        "FC",
        "F77",
        "MAKE",
        "PKG_CONFIG_PATH",
        "PKG_CONFIG_LIBDIR",
        "PKG_CONFIG_SYSROOT_DIR",
        "LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rustc-env=IPOPT_SRC_VERSION={IPOPT_VERSION}");
    println!("cargo:rustc-env=IPOPT_SRC_COMMIT={IPOPT_COMMIT}");
    println!("cargo:rustc-env=IPOPT_SRC_SOLVER_FAMILY=spral");
    println!("cargo:rustc-env=IPOPT_SRC_OPENBLAS_OWNER=spral-src");
    println!("cargo:rustc-env=IPOPT_SRC_SYSTEM_SOLVER_MATH_FALLBACKS=false");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set"));
    let source_dir = out_dir.join("sources");
    let build_dir = out_dir.join("build").join("ipopt");
    let install_dir = out_dir.join("install");
    fs::create_dir_all(&source_dir).expect("failed to create IPOPT source directory");
    fs::create_dir_all(&build_dir).expect("failed to create IPOPT build directory");
    fs::create_dir_all(&install_dir).expect("failed to create IPOPT install directory");

    let spral = SpralMetadata::from_env();
    let source = download_and_unpack(
        &source_dir,
        "ipopt",
        IPOPT_URL,
        IPOPT_SHA256,
        IPOPT_ARCHIVE_ROOT,
    );
    let libipopt = install_dir.join("lib").join(static_archive_name("ipopt"));
    if !libipopt.exists() {
        configure_and_build_ipopt(&source, &build_dir, &install_dir, &spral);
    }

    assert_inside(&out_dir, &libipopt, "IPOPT archive");
    emit_link_metadata(&install_dir, &spral);
}

#[derive(Clone, Debug)]
struct SpralMetadata {
    lib_dir: PathBuf,
    metis_lib_dir: PathBuf,
    openblas_lib_dir: PathBuf,
    openmp_lib: String,
    openmp_lib_dir: Option<PathBuf>,
    cxx_stdlib: String,
    cc: String,
    cxx: String,
    fc: String,
    runtime_link_dirs: Vec<PathBuf>,
    openblas_extra_link_dirs: Vec<PathBuf>,
    openblas_extra_link_libs: Vec<String>,
    openblas_threading: String,
    spral_cflags: String,
    spral_lflags: String,
    lapack_lflags: String,
}

impl SpralMetadata {
    fn from_env() -> Self {
        Self {
            lib_dir: metadata_path("LIB"),
            metis_lib_dir: metadata_path("METIS_LIB"),
            openblas_lib_dir: metadata_path("OPENBLAS_LIB"),
            openmp_lib: metadata("OPENMP_LIB"),
            openmp_lib_dir: optional_metadata_path("OPENMP_LIB_DIR"),
            cxx_stdlib: metadata("CXX_STDLIB"),
            cc: metadata("CC"),
            cxx: metadata("CXX"),
            fc: metadata("FC"),
            runtime_link_dirs: metadata_paths("RUNTIME_LINK_DIRS"),
            openblas_extra_link_dirs: metadata_paths("OPENBLAS_EXTRA_LINK_DIRS"),
            openblas_extra_link_libs: metadata_list("OPENBLAS_EXTRA_LINK_LIBS"),
            openblas_threading: metadata("OPENBLAS_THREADING"),
            spral_cflags: metadata("SPRAL_CFLAGS"),
            spral_lflags: metadata("SPRAL_LFLAGS"),
            lapack_lflags: metadata("LAPACK_LFLAGS"),
        }
    }
}

fn validate_feature_mode() {
    let spral_serial = cfg!(feature = "source-built-spral");
    let spral_openmp = cfg!(feature = "source-built-spral-openmp");
    let compatibility = [
        ("mumps", cfg!(feature = "mumps")),
        ("openblas-static", cfg!(feature = "openblas-static")),
        ("openblas-system", cfg!(feature = "openblas-system")),
        ("intel-mkl", cfg!(feature = "intel-mkl")),
        ("intel-mkl-static", cfg!(feature = "intel-mkl-static")),
        ("intel-mkl-system", cfg!(feature = "intel-mkl-system")),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name))
    .collect::<Vec<_>>();

    assert!(
        spral_serial ^ spral_openmp,
        "ipopt-src requires exactly one SPRAL source-built feature: \
         `source-built-spral` or `source-built-spral-openmp`"
    );
    assert!(
        compatibility.is_empty(),
        "ipopt-src upstream-compatible feature(s) {} are reserved for \
         non-parity compatibility work and are not wired into this strict \
         SPRAL source-built fork yet",
        compatibility.join(", ")
    );
}

fn metadata(name: &str) -> String {
    env::var(format!("DEP_SPRAL_{name}"))
        .unwrap_or_else(|_| panic!("spral-src did not emit DEP_SPRAL_{name}"))
}

fn optional_metadata(name: &str) -> Option<String> {
    env::var(format!("DEP_SPRAL_{name}")).ok()
}

fn metadata_path(name: &str) -> PathBuf {
    PathBuf::from(metadata(name))
}

fn optional_metadata_path(name: &str) -> Option<PathBuf> {
    optional_metadata(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn metadata_paths(name: &str) -> Vec<PathBuf> {
    metadata_list(name).into_iter().map(PathBuf::from).collect()
}

fn metadata_list(name: &str) -> Vec<String> {
    optional_metadata(name)
        .unwrap_or_default()
        .split(';')
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn configure_and_build_ipopt(
    source: &Path,
    build_dir: &Path,
    install_dir: &Path,
    spral: &SpralMetadata,
) {
    if build_dir.exists() {
        fs::remove_dir_all(build_dir).expect("failed to clear stale IPOPT build directory");
    }
    fs::create_dir_all(build_dir).expect("failed to create IPOPT build directory");
    let empty_pkg_config = build_dir.join("empty-pkgconfig");
    fs::create_dir_all(&empty_pkg_config).expect("failed to create empty pkg-config directory");

    let configure = source.join("configure");
    let mut command = Command::new(&configure);
    command
        .current_dir(build_dir)
        .arg(format!("--prefix={}", install_dir.display()))
        .arg("--disable-shared")
        .arg("--enable-static")
        .arg("--with-pic")
        .arg("--disable-linear-solver-loader")
        .arg("--disable-pardisomkl")
        .arg("--without-mumps")
        .arg("--without-hsl")
        .arg("--without-pardiso")
        .arg("--with-spral")
        .arg(format!("--with-spral-cflags={}", spral.spral_cflags))
        .arg(format!("--with-spral-lflags={}", spral.spral_lflags))
        .arg(format!("--with-lapack-lflags={}", spral.lapack_lflags));
    configure_fail_closed_env(&mut command, &empty_pkg_config, spral);
    run(&mut command, "IPOPT configure");

    let make = env::var("MAKE").unwrap_or_else(|_| "make".to_string());
    let jobs = env::var("NUM_JOBS").unwrap_or_else(|_| "1".to_string());
    let mut build = Command::new(&make);
    build.current_dir(build_dir).arg(format!("-j{jobs}"));
    configure_fail_closed_env(&mut build, &empty_pkg_config, spral);
    run(&mut build, "IPOPT make");

    let mut install = Command::new(&make);
    install.current_dir(build_dir).arg("install");
    configure_fail_closed_env(&mut install, &empty_pkg_config, spral);
    run(&mut install, "IPOPT make install");
}

fn configure_fail_closed_env(
    command: &mut Command,
    empty_pkg_config: &Path,
    spral: &SpralMetadata,
) {
    command
        .env("PKG_CONFIG_LIBDIR", empty_pkg_config)
        .env_remove("PKG_CONFIG_PATH")
        .env_remove("PKG_CONFIG_SYSROOT_DIR")
        .env_remove("LIBRARY_PATH")
        .env_remove("DYLD_LIBRARY_PATH")
        .env_remove("DYLD_FALLBACK_LIBRARY_PATH")
        .env("CC", &spral.cc)
        .env("CXX", &spral.cxx)
        .env("FC", &spral.fc)
        .env("F77", &spral.fc);
}

fn download_and_unpack(
    source_dir: &Path,
    name: &str,
    url: &str,
    expected_sha256: &str,
    archive_root: &str,
) -> PathBuf {
    let archive_path = source_dir.join(format!("{name}.tar.gz"));
    if !archive_path.exists() {
        eprintln!("ipopt-src: downloading {name} from {url}");
        let mut response = ureq_agent()
            .get(url)
            .call()
            .unwrap_or_else(|error| panic!("failed to download {name} from {url}: {error}"))
            .into_body()
            .into_reader();
        let mut archive =
            File::create(&archive_path).expect("failed to create downloaded source archive");
        io::copy(&mut response, &mut archive).expect("failed to write downloaded source archive");
    }
    verify_sha256(&archive_path, expected_sha256);

    let unpacked = source_dir.join(archive_root);
    if !unpacked.exists() {
        let archive = File::open(&archive_path).expect("failed to open source archive");
        let decoder = GzDecoder::new(archive);
        let mut tar = Archive::new(decoder);
        tar.unpack(source_dir)
            .unwrap_or_else(|error| panic!("failed to unpack {name} archive: {error}"));
    }
    unpacked
}

fn ureq_agent() -> ureq::Agent {
    ureq::config::Config::builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::NativeTls)
                .build(),
        )
        .build()
        .new_agent()
}

fn verify_sha256(path: &Path, expected: &str) {
    let mut file = File::open(path).expect("failed to open archive for checksum");
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let bytes = file
            .read(&mut buffer)
            .expect("failed to read archive for checksum");
        if bytes == 0 {
            break;
        }
        hasher.update(&buffer[..bytes]);
    }
    let actual = format!("{:x}", hasher.finalize());
    assert!(
        actual == expected,
        "checksum mismatch for {}: expected {expected}, got {actual}",
        path.display()
    );
}

fn emit_link_metadata(install_dir: &Path, spral: &SpralMetadata) {
    let include_dir = install_dir.join("include").join("coin-or");
    let lib_dir = install_dir.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-search=native={}", spral.lib_dir.display());
    println!(
        "cargo:rustc-link-search=native={}",
        spral.metis_lib_dir.display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        spral.openblas_lib_dir.display()
    );
    for dir in &spral.runtime_link_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for dir in &spral.openblas_extra_link_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    if let Some(dir) = &spral.openmp_lib_dir {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    println!("cargo:rustc-link-lib=static=ipopt");
    println!("cargo:rustc-link-lib=static=spral");
    println!("cargo:rustc-link-lib=static=metis");
    println!("cargo:rustc-link-lib=static=openblas");
    println!("cargo:rustc-link-lib={}", spral.openmp_lib);
    println!("cargo:rustc-link-lib={}", spral.cxx_stdlib);
    for lib in &spral.openblas_extra_link_libs {
        println!("cargo:rustc-link-lib={lib}");
    }
    if cfg!(unix) && !cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
    }

    println!("cargo:root={}", install_dir.display());
    println!("cargo:include={}", include_dir.display());
    println!("cargo:lib={}", lib_dir.display());
    println!("cargo:spral_lib={}", spral.lib_dir.display());
    println!("cargo:metis_lib={}", spral.metis_lib_dir.display());
    println!("cargo:openblas_lib={}", spral.openblas_lib_dir.display());
    println!("cargo:openmp_lib={}", spral.openmp_lib);
    if let Some(dir) = &spral.openmp_lib_dir {
        println!("cargo:openmp_lib_dir={}", dir.display());
    }
    println!("cargo:cxx_stdlib={}", spral.cxx_stdlib);
    println!("cargo:cc={}", spral.cc);
    println!("cargo:cxx={}", spral.cxx);
    println!("cargo:fc={}", spral.fc);
    println!(
        "cargo:runtime_link_dirs={}",
        join_paths_for_metadata(&spral.runtime_link_dirs)
    );
    println!(
        "cargo:openblas_extra_link_dirs={}",
        join_paths_for_metadata(&spral.openblas_extra_link_dirs)
    );
    println!(
        "cargo:openblas_extra_link_libs={}",
        spral.openblas_extra_link_libs.join(";")
    );
    println!("cargo:openblas_owner=spral-src");
    println!("cargo:openblas_threading={}", spral.openblas_threading);
    println!(
        "cargo:rustc-env=IPOPT_SRC_OPENBLAS_THREADING={}",
        spral.openblas_threading
    );
    println!("cargo:version={IPOPT_VERSION}");
    println!("cargo:source_commit={IPOPT_COMMIT}");
    println!("cargo:spral_cflags={}", spral.spral_cflags);
    println!("cargo:spral_lflags={}", spral.spral_lflags);
    println!("cargo:lapack_lflags={}", spral.lapack_lflags);
    emit_dep_metadata("ROOT", &install_dir.display().to_string());
    emit_dep_metadata("INCLUDE", &include_dir.display().to_string());
    emit_dep_metadata("LIB", &lib_dir.display().to_string());
    emit_dep_metadata("SPRAL_LIB", &spral.lib_dir.display().to_string());
    emit_dep_metadata("METIS_LIB", &spral.metis_lib_dir.display().to_string());
    emit_dep_metadata(
        "OPENBLAS_LIB",
        &spral.openblas_lib_dir.display().to_string(),
    );
    emit_dep_metadata("OPENMP_LIB", &spral.openmp_lib);
    if let Some(dir) = &spral.openmp_lib_dir {
        emit_dep_metadata("OPENMP_LIB_DIR", &dir.display().to_string());
    }
    emit_dep_metadata("CXX_STDLIB", &spral.cxx_stdlib);
    emit_dep_metadata("CC", &spral.cc);
    emit_dep_metadata("CXX", &spral.cxx);
    emit_dep_metadata("FC", &spral.fc);
    emit_dep_metadata(
        "RUNTIME_LINK_DIRS",
        &join_paths_for_metadata(&spral.runtime_link_dirs),
    );
    emit_dep_metadata(
        "OPENBLAS_EXTRA_LINK_DIRS",
        &join_paths_for_metadata(&spral.openblas_extra_link_dirs),
    );
    emit_dep_metadata(
        "OPENBLAS_EXTRA_LINK_LIBS",
        &spral.openblas_extra_link_libs.join(";"),
    );
    emit_dep_metadata("OPENBLAS_OWNER", "spral-src");
    emit_dep_metadata("OPENBLAS_THREADING", &spral.openblas_threading);
    emit_dep_metadata("VERSION", IPOPT_VERSION);
    emit_dep_metadata("SOURCE_COMMIT", IPOPT_COMMIT);
    emit_dep_metadata("SPRAL_CFLAGS", &spral.spral_cflags);
    emit_dep_metadata("SPRAL_LFLAGS", &spral.spral_lflags);
    emit_dep_metadata("LAPACK_LFLAGS", &spral.lapack_lflags);
    println!(
        "cargo:config=ipopt={IPOPT_VERSION};spral=source-built;system_solver_math_fallbacks=false"
    );
}

fn emit_dep_metadata(key: &str, value: &str) {
    println!("cargo:{key}={value}");
    println!("cargo::metadata={key}={value}");
}

fn join_paths_for_metadata(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(";")
}

fn run(command: &mut Command, label: &str) {
    eprintln!("ipopt-src: running {label}: {command:?}");
    let status = command
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .unwrap_or_else(|error| panic!("failed to start {label}: {error}"));
    assert!(status.success(), "{label} failed with status {status}");
}

fn assert_inside(root: &Path, path: &Path, label: &str) {
    let root = root
        .canonicalize()
        .unwrap_or_else(|error| panic!("failed to canonicalize OUT_DIR: {error}"));
    let path = path
        .canonicalize()
        .unwrap_or_else(|error| panic!("failed to canonicalize {label}: {error}"));
    assert!(
        path.starts_with(&root),
        "{label} resolved outside OUT_DIR: {} (OUT_DIR={})",
        path.display(),
        root.display()
    );
}

fn static_archive_name(name: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{name}.lib")
    } else {
        format!("lib{name}.a")
    }
}
