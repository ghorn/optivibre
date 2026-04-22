use std::env;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use tar::Archive;

const SPRAL_VERSION: &str = "2025.09.18";
const SPRAL_ARCHIVE_ROOT: &str = "spral-2025.09.18";
const SPRAL_URL: &str = "https://github.com/ralna/spral/archive/refs/tags/v2025.09.18.tar.gz";
const SPRAL_SHA256: &str = "1358168ac95297049e4fc810e54a16e0c765796cfbaa156e09979e2620b7dae7";

const METIS_VERSION: &str = "5.2.1";
const METIS_ARCHIVE_ROOT: &str = "METIS-5.2.1";
const METIS_URL: &str = "https://github.com/KarypisLab/METIS/archive/refs/tags/v5.2.1.tar.gz";
const METIS_SHA256: &str = "1a4665b2cd07edc2f734e30d7460afb19c1217c2547c2ac7bf6e1848d50aff7a";

const GKLIB_ARCHIVE_ROOT: &str = "GKlib-e2856c2f595b153ca1ce9258c5301dbabc4f39f5";
const GKLIB_URL: &str =
    "https://github.com/KarypisLab/GKlib/archive/e2856c2f595b153ca1ce9258c5301dbabc4f39f5.tar.gz";
const GKLIB_SHA256: &str = "ece01338c55412f085910968832289fb08e6761f6ce7b94755477077ce449155";

const OPENBLAS_VERSION: &str = "0.3.32";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OpenBlasThreading {
    Serial,
    OpenMp,
}

impl OpenBlasThreading {
    fn from_features() -> Self {
        match (
            cfg!(feature = "openblas-serial"),
            cfg!(feature = "openblas-openmp"),
        ) {
            (true, false) => Self::Serial,
            (false, true) => Self::OpenMp,
            (true, true) => panic!(
                "spral-src features `openblas-serial` and `openblas-openmp` are mutually exclusive"
            ),
            (false, false) => panic!(
                "spral-src requires exactly one OpenBLAS threading feature: `openblas-serial` or `openblas-openmp`"
            ),
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::OpenMp => "openmp",
        }
    }
}

#[derive(Clone, Debug)]
struct NativeLibrary {
    include_dir: PathBuf,
    lib_dir: PathBuf,
    archive: PathBuf,
    extra_link_dirs: Vec<PathBuf>,
    extra_link_libs: Vec<String>,
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for var in [
        "CC",
        "CXX",
        "FC",
        "HOST_CC",
        "OPENBLAS_CC",
        "OPENBLAS_FC",
        "OPENBLAS_HOSTCC",
        "OPENBLAS_RANLIB",
        "OPENBLAS_TARGET",
        "SPRAL_SRC_OPENMP_LIB",
        "SPRAL_SRC_OPENMP_LIB_DIR",
        "MESON",
        "NINJA",
        "MAKE",
        "PKG_CONFIG_PATH",
        "PKG_CONFIG_LIBDIR",
        "PKG_CONFIG_SYSROOT_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set"));
    let openblas_threading = OpenBlasThreading::from_features();
    let source_dir = out_dir.join("sources");
    let build_dir = out_dir.join("build");
    let install_dir = out_dir.join("install");
    fs::create_dir_all(&source_dir).expect("failed to create native source directory");
    fs::create_dir_all(&build_dir).expect("failed to create native build directory");
    fs::create_dir_all(&install_dir).expect("failed to create native install directory");

    let tools = Toolchain::detect();
    let metis_source = download_and_unpack(
        &source_dir,
        "metis",
        METIS_URL,
        METIS_SHA256,
        METIS_ARCHIVE_ROOT,
    );
    let gklib_source = download_and_unpack(
        &source_dir,
        "gklib",
        GKLIB_URL,
        GKLIB_SHA256,
        GKLIB_ARCHIVE_ROOT,
    );
    let spral_source = download_and_unpack(
        &source_dir,
        "spral",
        SPRAL_URL,
        SPRAL_SHA256,
        SPRAL_ARCHIVE_ROOT,
    );

    let metis = build_metis(&metis_source, &gklib_source, &build_dir.join("metis"));
    let openblas = build_openblas(&build_dir.join("openblas"), &tools, openblas_threading);
    let pc_dir = build_dir.join("pkgconfig");
    write_private_pkg_config(&pc_dir, &metis, &openblas);
    let spral = build_spral(
        &spral_source,
        &build_dir.join("spral"),
        &install_dir,
        &pc_dir,
        &metis,
        &openblas,
        &tools,
    );

    assert_inside(&out_dir, &metis.archive, "METIS archive");
    assert_inside(&out_dir, &openblas.archive, "OpenBLAS archive");
    assert_inside(&out_dir, &spral.archive, "SPRAL archive");

    emit_link_metadata(&spral, &metis, &openblas, &tools, openblas_threading);
}

#[derive(Clone, Debug)]
struct Toolchain {
    cc: String,
    cxx: String,
    fc: String,
    host_cc: String,
    meson: String,
    ninja: String,
    make: String,
    openmp_lib: String,
    openmp_lib_dir: Option<PathBuf>,
    cxx_stdlib: String,
    runtime_lib_dirs: Vec<PathBuf>,
}

impl Toolchain {
    fn detect() -> Self {
        let target = env::var("TARGET").expect("TARGET must be set");
        let host = env::var("HOST").expect("HOST must be set");
        let is_cross = target != host;

        let fc = env::var("OPENBLAS_FC")
            .or_else(|_| env::var("FC"))
            .unwrap_or_else(|_| {
                if is_cross {
                    panic!(
                        "spral-src requires OPENBLAS_FC or FC for cross-compilation target {target}"
                    );
                }
                require_native_tool("Fortran compiler", &["gfortran", "flang", "ifort"])
            });
        let cc = env::var("OPENBLAS_CC")
            .or_else(|_| env::var("CC"))
            .unwrap_or_else(|_| {
                companion_c_compiler(&fc)
                    .unwrap_or_else(|| require_native_tool("cc", &["cc", "clang", "gcc"]))
            });
        let cxx = env::var("CXX").unwrap_or_else(|_| {
            companion_cxx_compiler(&cc, &fc).unwrap_or_else(|| {
                if is_cross {
                    panic!("spral-src requires CXX for cross-compilation target {target}");
                }
                require_native_tool("C++ compiler", &["c++", "clang++", "g++"])
            })
        });
        let host_cc = env::var("OPENBLAS_HOSTCC")
            .or_else(|_| env::var("HOST_CC"))
            .unwrap_or_else(|_| {
                if is_cross {
                    require_native_tool("host C compiler", &["cc", "clang", "gcc"])
                } else {
                    cc.clone()
                }
            });
        let meson = env::var("MESON").unwrap_or_else(|_| require_native_tool("meson", &["meson"]));
        let ninja = env::var("NINJA").unwrap_or_else(|_| require_native_tool("ninja", &["ninja"]));
        let make =
            env::var("MAKE").unwrap_or_else(|_| require_native_tool("make", &["make", "gmake"]));

        require_command(&cc, "C compiler");
        require_command(&cxx, "C++ compiler");
        require_command(&fc, "Fortran compiler");
        require_command(&host_cc, "host C compiler");
        require_command(&meson, "meson");
        require_command(&ninja, "ninja");
        require_command(&make, "make");

        let openmp_lib =
            env::var("SPRAL_SRC_OPENMP_LIB").unwrap_or_else(|_| infer_openmp_lib(&cxx));
        let cxx_stdlib = infer_cxx_stdlib(&target, &cxx);
        let mut runtime_lib_dirs =
            compiler_runtime_lib_dirs(&target, &fc, &cxx, &openmp_lib, &cxx_stdlib);
        let openmp_lib_dir = env::var_os("SPRAL_SRC_OPENMP_LIB_DIR")
            .map(PathBuf::from)
            .or_else(|| compiler_lib_dir(&target, &cxx, &openmp_lib));
        if let Some(dir) = &openmp_lib_dir {
            push_unique_path(&mut runtime_lib_dirs, dir.clone());
        }

        Self {
            cc,
            cxx,
            fc,
            host_cc,
            meson,
            ninja,
            make,
            openmp_lib,
            openmp_lib_dir,
            cxx_stdlib,
            runtime_lib_dirs,
        }
    }
}

fn require_native_tool(label: &str, candidates: &[&str]) -> String {
    for candidate in candidates {
        if command_exists(candidate) {
            return (*candidate).to_string();
        }
    }
    panic!(
        "spral-src requires {label}; tried: {}",
        candidates.join(", ")
    );
}

fn require_command(command: &str, label: &str) {
    let status = Command::new(command)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    match status {
        Ok(status) if status.success() => {}
        Ok(_) | Err(_) => panic!("spral-src could not run configured {label}: {command}"),
    }
}

fn command_exists(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn companion_c_compiler(fc: &str) -> Option<String> {
    let path = resolve_command_path(fc)?;
    let dir = path.parent()?;
    let name = path.file_name()?.to_string_lossy();
    let mut candidates = Vec::new();
    if name.contains("gfortran") {
        candidates.push(name.replace("gfortran", "gcc"));
        if name == "gfortran" {
            if let Some(major) = compiler_major_version(fc) {
                candidates.push(format!("gcc-{major}"));
            }
            candidates.push("gcc".to_string());
        }
    } else if name.contains("flang") {
        candidates.push(name.replace("flang", "clang"));
        candidates.push("clang".to_string());
    } else if name.contains("ifort") {
        candidates.push(name.replace("ifort", "icc"));
        candidates.push("icc".to_string());
    }
    find_sibling_command(dir, candidates)
}

fn companion_cxx_compiler(cc: &str, fc: &str) -> Option<String> {
    if let Some(path) = resolve_command_path(cc) {
        let dir = path.parent()?;
        let name = path.file_name()?.to_string_lossy();
        let mut candidates = Vec::new();
        if name.contains("gcc") {
            candidates.push(name.replace("gcc", "g++"));
        } else if name.contains("clang") {
            candidates.push(name.replace("clang", "clang++"));
        } else if name.contains("icc") {
            candidates.push(name.replace("icc", "icpc"));
        }
        if let Some(found) = find_sibling_command(dir, candidates) {
            return Some(found);
        }
    }

    let path = resolve_command_path(fc)?;
    let dir = path.parent()?;
    let name = path.file_name()?.to_string_lossy();
    let mut candidates = Vec::new();
    if name.contains("gfortran") {
        candidates.push(name.replace("gfortran", "g++"));
        if name == "gfortran" {
            if let Some(major) = compiler_major_version(fc) {
                candidates.push(format!("g++-{major}"));
            }
            candidates.push("g++".to_string());
        }
    } else if name.contains("flang") {
        candidates.push(name.replace("flang", "clang++"));
        candidates.push("clang++".to_string());
    }
    find_sibling_command(dir, candidates)
}

fn find_sibling_command(dir: &Path, candidates: Vec<String>) -> Option<String> {
    for candidate in candidates {
        let path = dir.join(&candidate);
        if path.exists() {
            return Some(path.display().to_string());
        }
    }
    None
}

fn resolve_command_path(command: &str) -> Option<PathBuf> {
    let path = Path::new(command);
    if path.components().count() > 1 {
        return path.exists().then(|| path.to_path_buf());
    }
    let path_var = env::var_os("PATH")?;
    for dir in env::split_paths(&path_var) {
        let candidate = dir.join(command);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn compiler_major_version(command: &str) -> Option<String> {
    let output = Command::new(command)
        .arg("-dumpfullversion")
        .output()
        .or_else(|_| Command::new(command).arg("-dumpversion").output())
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let version = String::from_utf8(output.stdout).ok()?;
    version
        .trim()
        .split('.')
        .next()
        .filter(|part| !part.is_empty())
        .map(str::to_string)
}

fn infer_openmp_lib(cxx: &str) -> String {
    let lower = cxx.to_ascii_lowercase();
    if lower.contains("clang") && (cfg!(target_os = "macos") || cfg!(target_os = "freebsd")) {
        "omp".to_string()
    } else if lower.contains("intel") || lower.contains("icx") || lower.contains("icpx") {
        "iomp5".to_string()
    } else {
        "gomp".to_string()
    }
}

fn infer_cxx_stdlib(target: &str, cxx: &str) -> String {
    let lower = cxx.to_ascii_lowercase();
    if lower.contains("clang") && (target.contains("apple") || target.contains("freebsd")) {
        "c++".to_string()
    } else {
        "stdc++".to_string()
    }
}

fn compiler_runtime_lib_dirs(
    target: &str,
    fc: &str,
    cxx: &str,
    openmp_lib: &str,
    cxx_stdlib: &str,
) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for lib in ["gfortran", "quadmath", "gcc_s.1.1", "gcc_s"] {
        if let Some(dir) = compiler_lib_dir(target, fc, lib) {
            push_unique_path(&mut dirs, dir);
        }
    }
    for lib in [openmp_lib, cxx_stdlib] {
        if let Some(dir) = compiler_lib_dir(target, cxx, lib) {
            push_unique_path(&mut dirs, dir);
        }
    }
    dirs
}

fn compiler_lib_dir(target: &str, compiler: &str, lib: &str) -> Option<PathBuf> {
    for name in library_file_names(target, lib) {
        let output = Command::new(compiler)
            .arg(format!("-print-file-name={name}"))
            .output()
            .ok()?;
        if !output.status.success() {
            continue;
        }
        let path = PathBuf::from(String::from_utf8(output.stdout).ok()?.trim());
        if path.is_absolute() && path.exists() {
            return path.parent().map(Path::to_path_buf);
        }
    }
    None
}

fn library_file_names(target: &str, lib: &str) -> Vec<String> {
    let mut names = Vec::new();
    if target.contains("apple") {
        names.push(format!("lib{lib}.dylib"));
    } else if target.contains("windows") {
        names.push(format!("{lib}.lib"));
    } else {
        names.push(format!("lib{lib}.so"));
    }
    names.push(format!("lib{lib}.a"));
    names
}

fn push_unique_path(paths: &mut Vec<PathBuf>, path: PathBuf) {
    if !paths.iter().any(|existing| existing == &path) {
        paths.push(path);
    }
}

fn push_unique_string(values: &mut Vec<String>, value: String) {
    if !values.iter().any(|existing| existing == &value) {
        values.push(value);
    }
}

fn append_link_flags(
    dirs: &mut Vec<PathBuf>,
    libs: &mut Vec<String>,
    flags: &openblas_build::LinkFlags,
) {
    for dir in &flags.search_paths {
        push_unique_path(dirs, dir.clone());
    }
    for lib in &flags.libs {
        push_unique_string(libs, lib.clone());
    }
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
        eprintln!("spral-src: downloading {name} from {url}");
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

fn build_metis(source: &Path, gklib_source: &Path, out: &Path) -> NativeLibrary {
    let include_dir = source.join("include");
    let libmetis_dir = source.join("libmetis");
    let gklib_include_dir = gklib_source.join("include");
    let gklib_src_dir = gklib_source.join("src");
    let lib_dir = out.join("lib");
    let archive = lib_dir.join(static_archive_name("metis"));
    if archive.exists() {
        return NativeLibrary {
            include_dir,
            lib_dir,
            archive,
            extra_link_dirs: Vec::new(),
            extra_link_libs: Vec::new(),
        };
    }

    fs::create_dir_all(&lib_dir).expect("failed to create METIS lib directory");
    let mut build = cc::Build::new();
    build
        .define("IDXTYPEWIDTH", Some("32"))
        .define("REALTYPEWIDTH", Some("32"))
        .include(&include_dir)
        .include(&libmetis_dir)
        .include(&gklib_include_dir)
        .out_dir(&lib_dir)
        .warnings(false)
        .flag_if_supported("-fPIC");

    if cfg!(target_os = "windows") {
        build.define("USE_GKREGEX", None).define("WIN32", None);
    } else if cfg!(target_os = "linux") {
        build
            .define("LINUX", None)
            .define("_FILE_OFFSET_BITS", Some("64"));
    } else if cfg!(target_os = "macos") {
        build.define("MACOS", None);
    }

    for file in GKLIB_SOURCES {
        build.file(gklib_src_dir.join(file));
    }
    for file in METIS_SOURCES {
        build.file(libmetis_dir.join(file));
    }
    build.compile("metis");

    assert!(
        archive.exists(),
        "METIS build did not produce {}",
        archive.display()
    );
    NativeLibrary {
        include_dir,
        lib_dir,
        archive,
        extra_link_dirs: Vec::new(),
        extra_link_libs: Vec::new(),
    }
}

fn build_openblas(out: &Path, tools: &Toolchain, threading: OpenBlasThreading) -> NativeLibrary {
    fs::create_dir_all(out).expect("failed to create OpenBLAS build directory");
    let source = openblas_build::download(out)
        .unwrap_or_else(|error| panic!("failed to download OpenBLAS source: {error}"));
    let archive = source.join(static_archive_name("openblas"));
    let makefile_conf = source.join("Makefile.conf");
    if !archive.exists() || !makefile_conf.exists() || !openblas_has_required_lapack(&archive) {
        if makefile_conf.exists() {
            fs::remove_file(&makefile_conf).expect("failed to remove stale OpenBLAS Makefile.conf");
        }
        run_openblas_make(&source, tools, threading);
    }
    let make_conf = openblas_build::MakeConf::new(&makefile_conf)
        .unwrap_or_else(|error| panic!("failed to parse OpenBLAS Makefile.conf: {error}"));
    assert!(
        !make_conf.no_fortran,
        "OpenBLAS fell back to NOFORTRAN/f2c LAPACK; configure OPENBLAS_FC or FC for the target"
    );
    assert!(
        archive.exists(),
        "OpenBLAS build did not produce {}",
        archive.display()
    );
    let mut extra_link_dirs = tools.runtime_lib_dirs.clone();
    let mut extra_link_libs = Vec::new();
    append_link_flags(
        &mut extra_link_dirs,
        &mut extra_link_libs,
        &make_conf.c_extra_libs,
    );
    append_link_flags(
        &mut extra_link_dirs,
        &mut extra_link_libs,
        &make_conf.f_extra_libs,
    );
    for lib in ["gfortran", "quadmath"] {
        if compiler_lib_dir(
            &env::var("TARGET").expect("TARGET must be set"),
            &tools.fc,
            lib,
        )
        .is_some()
        {
            push_unique_string(&mut extra_link_libs, lib.to_string());
        }
    }
    NativeLibrary {
        include_dir: source.clone(),
        lib_dir: source.clone(),
        archive,
        extra_link_dirs,
        extra_link_libs,
    }
}

fn run_openblas_make(source: &Path, tools: &Toolchain, threading: OpenBlasThreading) {
    let out_log = File::create(source.join("out.log")).expect("failed to create OpenBLAS out.log");
    let err_log = File::create(source.join("err.log")).expect("failed to create OpenBLAS err.log");
    let mut command = Command::new(&tools.make);
    command
        .current_dir(source)
        .stdout(out_log)
        .stderr(err_log)
        .env_remove("TARGET")
        .args(openblas_make_args(tools, threading))
        .arg("shared");

    eprintln!("spral-src: running OpenBLAS make shared: {command:?}");
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to start OpenBLAS make: {error}"));
    if !status.success() {
        let err = fs::read_to_string(source.join("err.log")).unwrap_or_default();
        panic!("OpenBLAS source build failed with status {status}:\n{err}");
    }
}

fn openblas_has_required_lapack(archive: &Path) -> bool {
    if !archive.exists() {
        return false;
    }
    let output = Command::new("nm").arg("-g").arg(archive).output();
    let Ok(output) = output else {
        return true;
    };
    if !output.status.success() {
        return true;
    }
    let symbols = String::from_utf8_lossy(&output.stdout);
    symbols.contains("dsytrf_")
}

fn openblas_make_args(tools: &Toolchain, threading: OpenBlasThreading) -> Vec<String> {
    let mut args = vec![
        "NO_SHARED=1".to_string(),
        "NO_LAPACKE=1".to_string(),
        "USE_LOCKING=1".to_string(),
        format!("CC={}", tools.cc),
        format!("FC={}", tools.fc),
        format!("HOSTCC={}", tools.host_cc),
    ];
    match threading {
        OpenBlasThreading::Serial => {
            args.push("USE_THREAD=0".to_string());
            args.push("USE_OPENMP=0".to_string());
        }
        OpenBlasThreading::OpenMp => {
            args.push("USE_THREAD=1".to_string());
            args.push("USE_OPENMP=1".to_string());
        }
    }
    if let Ok(ranlib) = env::var("OPENBLAS_RANLIB") {
        args.push(format!("RANLIB={ranlib}"));
    }
    if let Ok(target) = env::var("OPENBLAS_TARGET") {
        args.push(format!("TARGET={}", target.to_ascii_uppercase()));
    } else if env::var("TARGET").expect("TARGET must be set")
        != env::var("HOST").expect("HOST must be set")
    {
        let target = env::var("TARGET").expect("TARGET must be set");
        let openblas_target = generic_openblas_target(&target).unwrap_or_else(|| {
            panic!("spral-src requires OPENBLAS_TARGET for cross-compilation target {target}")
        });
        args.push(format!("TARGET={openblas_target}"));
    }
    args
}

fn generic_openblas_target(target: &str) -> Option<&'static str> {
    let arch = target.split('-').next()?;
    match arch {
        "aarch64" => Some("ARMV8"),
        "arm" | "armv6" => Some("ARMV6"),
        "armv7" => Some("ARMV7"),
        "x86_64" => Some("SSE_GENERIC"),
        "loongarch64" => Some("LOONGSONGENERIC"),
        "mips64" | "mips64el" => Some("MIPS64_GENERIC"),
        "riscv64gc" => Some("RISCV64_GENERIC"),
        "sparc" => Some("SPARCV7"),
        _ => None,
    }
}

fn write_private_pkg_config(pc_dir: &Path, metis: &NativeLibrary, openblas: &NativeLibrary) {
    fs::create_dir_all(pc_dir).expect("failed to create private pkg-config directory");
    fs::write(
        pc_dir.join("openblas.pc"),
        format!(
            "prefix={prefix}\nlibdir={libdir}\nincludedir={includedir}\n\nName: OpenBLAS\nDescription: source-built OpenBLAS for spral-src\nVersion: {version}\nLibs: -L${{libdir}} -lopenblas\nCflags: -I${{includedir}}\n",
            prefix = openblas.lib_dir.display(),
            libdir = openblas.lib_dir.display(),
            includedir = openblas.include_dir.display(),
            version = OPENBLAS_VERSION,
        ),
    )
    .expect("failed to write OpenBLAS pkg-config file");
    fs::write(
        pc_dir.join("metis.pc"),
        format!(
            "prefix={prefix}\nlibdir={libdir}\nincludedir={includedir}\n\nName: METIS\nDescription: vendored METIS for spral-src\nVersion: {version}\nLibs: -L${{libdir}} -lmetis\nCflags: -I${{includedir}}\n",
            prefix = metis.lib_dir.display(),
            libdir = metis.lib_dir.display(),
            includedir = metis.include_dir.display(),
            version = METIS_VERSION,
        ),
    )
    .expect("failed to write METIS pkg-config file");
}

fn build_spral(
    source: &Path,
    build_dir: &Path,
    install_dir: &Path,
    pc_dir: &Path,
    metis: &NativeLibrary,
    openblas: &NativeLibrary,
    tools: &Toolchain,
) -> NativeLibrary {
    patch_spral_cuda_probe(source);

    let lib_dir = install_dir.join("lib");
    let include_dir = install_dir.join("include");
    let archive = lib_dir.join(static_archive_name("spral"));
    if archive.exists() {
        return NativeLibrary {
            include_dir,
            lib_dir,
            archive,
            extra_link_dirs: Vec::new(),
            extra_link_libs: Vec::new(),
        };
    }

    if build_dir.exists() {
        fs::remove_dir_all(build_dir).expect("failed to clear stale SPRAL build directory");
    }
    fs::create_dir_all(build_dir).expect("failed to create SPRAL build directory");

    let mut setup = Command::new(&tools.meson);
    setup
        .arg("setup")
        .arg(build_dir)
        .arg(source)
        .arg("--prefix")
        .arg(install_dir)
        .arg("--libdir")
        .arg("lib")
        .arg("--default-library")
        .arg("static")
        .arg("--buildtype")
        .arg("release")
        .arg("-Dgpu=false")
        .arg("-Dopenmp=true")
        .arg("-Dexamples=false")
        .arg("-Dtests=false")
        .arg("-Dmodules=false")
        .arg("-Dlibblas=openblas")
        .arg(format!(
            "-Dlibblas_path={}",
            meson_array_path(&openblas.lib_dir)
        ))
        .arg(format!(
            "-Dlibblas_include={}",
            openblas.include_dir.display()
        ))
        .arg("-Dliblapack=openblas")
        .arg(format!(
            "-Dliblapack_path={}",
            meson_array_path(&openblas.lib_dir)
        ))
        .arg("-Dlibmetis=metis")
        .arg(format!(
            "-Dlibmetis_path={}",
            meson_array_path(&metis.lib_dir)
        ))
        .arg("-Dlibhwloc=__spral_src_no_hwloc__")
        .arg(format!("-Dlibhwloc_path={}", meson_array_path(build_dir)));
    configure_private_env(&mut setup, pc_dir, tools);
    run(&mut setup, "SPRAL meson setup");

    let mut compile = Command::new(&tools.meson);
    compile.arg("compile").arg("-C").arg(build_dir);
    configure_private_env(&mut compile, pc_dir, tools);
    run(&mut compile, "SPRAL meson compile");

    let mut install = Command::new(&tools.meson);
    install.arg("install").arg("-C").arg(build_dir);
    configure_private_env(&mut install, pc_dir, tools);
    run(&mut install, "SPRAL meson install");

    assert!(
        archive.exists(),
        "SPRAL build did not produce {}",
        archive.display()
    );
    NativeLibrary {
        include_dir,
        lib_dir,
        archive,
        extra_link_dirs: Vec::new(),
        extra_link_libs: Vec::new(),
    }
}

fn patch_spral_cuda_probe(source: &Path) {
    let meson = source.join("meson.build");
    let contents = fs::read_to_string(&meson).expect("failed to read SPRAL meson.build");
    let patched = contents.replace(
        "libcuda = dependency('cuda', version : '>=10', modules : ['cublas'], required : false)",
        "libcuda = declare_dependency()",
    );
    if patched != contents {
        fs::write(&meson, patched).expect("failed to patch SPRAL CUDA dependency probe");
    }
}

fn configure_private_env(command: &mut Command, pc_dir: &Path, tools: &Toolchain) {
    command
        .env("PKG_CONFIG_LIBDIR", pc_dir)
        .env_remove("PKG_CONFIG_PATH")
        .env_remove("PKG_CONFIG_SYSROOT_DIR")
        .env("CC", &tools.cc)
        .env("CXX", &tools.cxx)
        .env("FC", &tools.fc)
        .env("NINJA", &tools.ninja);
}

fn emit_link_metadata(
    spral: &NativeLibrary,
    metis: &NativeLibrary,
    openblas: &NativeLibrary,
    tools: &Toolchain,
    openblas_threading: OpenBlasThreading,
) {
    for dir in [&spral.lib_dir, &metis.lib_dir, &openblas.lib_dir] {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for dir in &openblas.extra_link_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    if let Some(dir) = &tools.openmp_lib_dir {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib=static=spral");
    println!("cargo:rustc-link-lib=static=metis");
    println!("cargo:rustc-link-lib=static=openblas");
    println!("cargo:rustc-link-lib={}", tools.openmp_lib);
    println!("cargo:rustc-link-lib={}", tools.cxx_stdlib);
    for lib in &openblas.extra_link_libs {
        println!("cargo:rustc-link-lib={lib}");
    }
    if cfg!(unix) && !cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=pthread");
    }

    println!("cargo:root={}", spral.lib_dir.parent().unwrap().display());
    println!("cargo:include={}", spral.include_dir.display());
    println!("cargo:lib={}", spral.lib_dir.display());
    println!("cargo:metis_include={}", metis.include_dir.display());
    println!("cargo:metis_lib={}", metis.lib_dir.display());
    println!("cargo:openblas_include={}", openblas.include_dir.display());
    println!("cargo:openblas_lib={}", openblas.lib_dir.display());
    println!("cargo:openmp_lib={}", tools.openmp_lib);
    if let Some(dir) = &tools.openmp_lib_dir {
        println!("cargo:openmp_lib_dir={}", dir.display());
    }
    println!("cargo:cxx_stdlib={}", tools.cxx_stdlib);
    println!(
        "cargo:runtime_link_dirs={}",
        join_paths_for_metadata(&tools.runtime_lib_dirs)
    );
    println!(
        "cargo:openblas_extra_link_dirs={}",
        join_paths_for_metadata(&openblas.extra_link_dirs)
    );
    println!(
        "cargo:openblas_extra_link_libs={}",
        openblas.extra_link_libs.join(";")
    );
    println!("cargo:openblas_threading={}", openblas_threading.label());
    println!("cargo:spral_cflags=-I{}", shell_path(&spral.include_dir));
    println!(
        "cargo:spral_lflags={}",
        spral_link_flags(spral, metis, openblas, tools)
    );
    println!("cargo:lapack_lflags={}", openblas_link_flags(openblas));
    println!("cargo:spral_version={SPRAL_VERSION}");
    println!("cargo:metis_version={METIS_VERSION}");
    println!("cargo:openblas_version={OPENBLAS_VERSION}");
    emit_dep_metadata(
        "ROOT",
        &spral.lib_dir.parent().unwrap().display().to_string(),
    );
    emit_dep_metadata("INCLUDE", &spral.include_dir.display().to_string());
    emit_dep_metadata("LIB", &spral.lib_dir.display().to_string());
    emit_dep_metadata("METIS_INCLUDE", &metis.include_dir.display().to_string());
    emit_dep_metadata("METIS_LIB", &metis.lib_dir.display().to_string());
    emit_dep_metadata(
        "OPENBLAS_INCLUDE",
        &openblas.include_dir.display().to_string(),
    );
    emit_dep_metadata("OPENBLAS_LIB", &openblas.lib_dir.display().to_string());
    emit_dep_metadata("OPENMP_LIB", &tools.openmp_lib);
    if let Some(dir) = &tools.openmp_lib_dir {
        emit_dep_metadata("OPENMP_LIB_DIR", &dir.display().to_string());
    }
    emit_dep_metadata("CXX_STDLIB", &tools.cxx_stdlib);
    emit_dep_metadata("CC", &tools.cc);
    emit_dep_metadata("CXX", &tools.cxx);
    emit_dep_metadata("FC", &tools.fc);
    emit_dep_metadata("HOST_CC", &tools.host_cc);
    emit_dep_metadata(
        "RUNTIME_LINK_DIRS",
        &join_paths_for_metadata(&tools.runtime_lib_dirs),
    );
    emit_dep_metadata(
        "OPENBLAS_EXTRA_LINK_DIRS",
        &join_paths_for_metadata(&openblas.extra_link_dirs),
    );
    emit_dep_metadata(
        "OPENBLAS_EXTRA_LINK_LIBS",
        &openblas.extra_link_libs.join(";"),
    );
    emit_dep_metadata("OPENBLAS_THREADING", openblas_threading.label());
    emit_dep_metadata(
        "SPRAL_CFLAGS",
        &format!("-I{}", shell_path(&spral.include_dir)),
    );
    emit_dep_metadata(
        "SPRAL_LFLAGS",
        &spral_link_flags(spral, metis, openblas, tools),
    );
    emit_dep_metadata("LAPACK_LFLAGS", &openblas_link_flags(openblas));
    emit_dep_metadata("SPRAL_VERSION", SPRAL_VERSION);
    emit_dep_metadata("METIS_VERSION", METIS_VERSION);
    emit_dep_metadata("OPENBLAS_VERSION", OPENBLAS_VERSION);
    println!(
        "cargo:config=spral={SPRAL_VERSION};metis={METIS_VERSION};openblas={OPENBLAS_VERSION};openblas_threading={};openmp=required;system_solver_math_fallbacks=false",
        openblas_threading.label()
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

fn shell_path(path: &Path) -> String {
    path.display().to_string()
}

fn append_link_dir_flag(flags: &mut Vec<String>, dir: &Path) {
    flags.push(format!("-L{}", shell_path(dir)));
}

fn append_link_lib_flag(flags: &mut Vec<String>, lib: &str) {
    flags.push(format!("-l{lib}"));
}

fn openblas_link_flags(openblas: &NativeLibrary) -> String {
    let mut flags = Vec::new();
    append_link_dir_flag(&mut flags, &openblas.lib_dir);
    for dir in &openblas.extra_link_dirs {
        append_link_dir_flag(&mut flags, dir);
    }
    append_link_lib_flag(&mut flags, "openblas");
    for lib in &openblas.extra_link_libs {
        append_link_lib_flag(&mut flags, lib);
    }
    flags.join(" ")
}

fn spral_link_flags(
    spral: &NativeLibrary,
    metis: &NativeLibrary,
    openblas: &NativeLibrary,
    tools: &Toolchain,
) -> String {
    let mut flags = Vec::new();
    append_link_dir_flag(&mut flags, &spral.lib_dir);
    append_link_dir_flag(&mut flags, &metis.lib_dir);
    append_link_dir_flag(&mut flags, &openblas.lib_dir);
    for dir in &openblas.extra_link_dirs {
        append_link_dir_flag(&mut flags, dir);
    }
    if let Some(dir) = &tools.openmp_lib_dir {
        append_link_dir_flag(&mut flags, dir);
    }
    append_link_lib_flag(&mut flags, "spral");
    append_link_lib_flag(&mut flags, "metis");
    append_link_lib_flag(&mut flags, "openblas");
    append_link_lib_flag(&mut flags, &tools.openmp_lib);
    append_link_lib_flag(&mut flags, &tools.cxx_stdlib);
    for lib in &openblas.extra_link_libs {
        append_link_lib_flag(&mut flags, lib);
    }
    if cfg!(unix) && !cfg!(target_os = "macos") {
        append_link_lib_flag(&mut flags, "m");
        append_link_lib_flag(&mut flags, "pthread");
    }
    flags.join(" ")
}

fn run(command: &mut Command, label: &str) {
    eprintln!("spral-src: running {label}: {command:?}");
    let status = command
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

fn meson_array_path(path: &Path) -> String {
    format!("['{}']", path.display())
}

const GKLIB_SOURCES: &[&str] = &[
    "b64.c",
    "blas.c",
    "cache.c",
    "csr.c",
    "error.c",
    "evaluate.c",
    "fkvkselect.c",
    "fs.c",
    "getopt.c",
    "gk_util.c",
    "gkregex.c",
    "graph.c",
    "htable.c",
    "io.c",
    "itemsets.c",
    "mcore.c",
    "memory.c",
    "pqueue.c",
    "random.c",
    "rw.c",
    "seq.c",
    "sort.c",
    "string.c",
    "timers.c",
    "tokenizer.c",
];

const METIS_SOURCES: &[&str] = &[
    "auxapi.c",
    "balance.c",
    "bucketsort.c",
    "checkgraph.c",
    "coarsen.c",
    "compress.c",
    "contig.c",
    "debug.c",
    "fm.c",
    "fortran.c",
    "frename.c",
    "gklib.c",
    "graph.c",
    "initpart.c",
    "kmetis.c",
    "kwayfm.c",
    "kwayrefine.c",
    "mcutil.c",
    "mesh.c",
    "meshpart.c",
    "minconn.c",
    "mincover.c",
    "mmd.c",
    "ometis.c",
    "options.c",
    "parmetis.c",
    "pmetis.c",
    "refine.c",
    "separator.c",
    "sfm.c",
    "srefine.c",
    "stat.c",
    "timing.c",
    "util.c",
    "wspace.c",
];
