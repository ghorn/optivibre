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
const IPOPT_SPRAL_PARITY_PATCH_VERSION: &str = "optivibre-ipopt-spral-dump-v12";
const IPOPT_LLVM_COVERAGE_ENV: &str = "IPOPT_SRC_LLVM_COVERAGE";
const IPOPT_LLVM_COVERAGE_CC_ENV: &str = "IPOPT_SRC_LLVM_COVERAGE_CC";
const IPOPT_LLVM_COVERAGE_CXX_ENV: &str = "IPOPT_SRC_LLVM_COVERAGE_CXX";
const IPOPT_LLVM_COVERAGE_CXXFLAGS_ENV: &str = "IPOPT_SRC_LLVM_COVERAGE_CXXFLAGS";
const IPOPT_LLVM_COVERAGE_LDFLAGS_ENV: &str = "IPOPT_SRC_LLVM_COVERAGE_LDFLAGS";
const IPOPT_LLVM_COVERAGE_FLAGS: &str = "-fprofile-instr-generate -fcoverage-mapping";
const IPOPT_LLVM_COVERAGE_BUILD_VERSION: &str = "llvm-cov-v2";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    validate_feature_mode();
    // These affect native toolchain discovery/linking and must rebuild IPOPT.
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
        IPOPT_LLVM_COVERAGE_ENV,
        IPOPT_LLVM_COVERAGE_CC_ENV,
        IPOPT_LLVM_COVERAGE_CXX_ENV,
        IPOPT_LLVM_COVERAGE_CXXFLAGS_ENV,
        IPOPT_LLVM_COVERAGE_LDFLAGS_ENV,
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    println!("cargo:rustc-env=IPOPT_SRC_VERSION={IPOPT_VERSION}");
    println!("cargo:rustc-env=IPOPT_SRC_COMMIT={IPOPT_COMMIT}");
    println!("cargo:rustc-env=IPOPT_SRC_SOLVER_FAMILY=spral");
    println!("cargo:rustc-env=IPOPT_SRC_OPENBLAS_OWNER=spral-src");
    println!("cargo:rustc-env=IPOPT_SRC_SYSTEM_SOLVER_MATH_FALLBACKS=false");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR must be set"));
    let llvm_coverage = llvm_coverage_requested();
    let source_dir = out_dir.join("sources");
    let build_dir = out_dir.join("build").join(if llvm_coverage {
        "ipopt-llvm-cov"
    } else {
        "ipopt"
    });
    let install_dir = out_dir.join(if llvm_coverage {
        "install-llvm-cov"
    } else {
        "install"
    });
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
    patch_ipopt_spral_solver_interface(&source);
    patch_ipopt_expansion_matrix(&source);
    patch_ipopt_dense_vector(&source);
    patch_ipopt_tsym_linear_solver(&source);
    patch_ipopt_pdfull_space_solver(&source);
    let libipopt = install_dir.join("lib").join(static_archive_name("ipopt"));
    let parity_patch_marker =
        install_dir.join(format!(".{}", parity_patch_marker_name(llvm_coverage)));
    if !libipopt.exists() || !parity_patch_marker.exists() {
        configure_and_build_ipopt(&source, &build_dir, &install_dir, &spral, llvm_coverage);
        fs::write(
            &parity_patch_marker,
            parity_patch_marker_name(llvm_coverage),
        )
        .expect("failed to write IPOPT SPRAL parity patch marker");
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
    spral_version: String,
    metis_version: String,
    openblas_version: String,
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
            spral_version: metadata("SPRAL_VERSION"),
            metis_version: metadata("METIS_VERSION"),
            openblas_version: metadata("OPENBLAS_VERSION"),
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
    llvm_coverage: bool,
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
    configure_fail_closed_env(&mut command, &empty_pkg_config, spral, llvm_coverage);
    run(&mut command, "IPOPT configure");

    let make = env::var("MAKE").unwrap_or_else(|_| "make".to_string());
    let jobs = env::var("NUM_JOBS").unwrap_or_else(|_| "1".to_string());
    let mut build = Command::new(&make);
    build.current_dir(build_dir).arg(format!("-j{jobs}"));
    configure_fail_closed_env(&mut build, &empty_pkg_config, spral, llvm_coverage);
    run(&mut build, "IPOPT make");

    let mut install = Command::new(&make);
    install.current_dir(build_dir).arg("install");
    configure_fail_closed_env(&mut install, &empty_pkg_config, spral, llvm_coverage);
    run(&mut install, "IPOPT make install");
}

fn configure_fail_closed_env(
    command: &mut Command,
    empty_pkg_config: &Path,
    spral: &SpralMetadata,
    llvm_coverage: bool,
) {
    command
        .env("PKG_CONFIG_LIBDIR", empty_pkg_config)
        .env_remove("PKG_CONFIG_PATH")
        .env_remove("PKG_CONFIG_SYSROOT_DIR")
        .env_remove("LIBRARY_PATH")
        .env_remove("DYLD_LIBRARY_PATH")
        .env_remove("DYLD_FALLBACK_LIBRARY_PATH")
        .env(
            "CC",
            coverage_tool_or_spral(IPOPT_LLVM_COVERAGE_CC_ENV, &spral.cc, llvm_coverage),
        )
        .env(
            "CXX",
            coverage_tool_or_spral(IPOPT_LLVM_COVERAGE_CXX_ENV, &spral.cxx, llvm_coverage),
        )
        .env("FC", &spral.fc)
        .env("F77", &spral.fc);
    if llvm_coverage {
        command.env(
            "CFLAGS",
            combined_flag_env("CFLAGS", IPOPT_LLVM_COVERAGE_FLAGS, &[]),
        );
        command.env(
            "CXXFLAGS",
            combined_flag_env(
                "CXXFLAGS",
                IPOPT_LLVM_COVERAGE_FLAGS,
                &[IPOPT_LLVM_COVERAGE_CXXFLAGS_ENV],
            ),
        );
        command.env(
            "LDFLAGS",
            combined_flag_env(
                "LDFLAGS",
                IPOPT_LLVM_COVERAGE_FLAGS,
                &[IPOPT_LLVM_COVERAGE_LDFLAGS_ENV],
            ),
        );
    }
}

fn coverage_tool_or_spral(env_name: &str, fallback: &str, llvm_coverage: bool) -> String {
    if llvm_coverage && let Ok(value) = env::var(env_name) {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    fallback.to_string()
}

fn llvm_coverage_requested() -> bool {
    env::var(IPOPT_LLVM_COVERAGE_ENV).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn parity_patch_marker_name(llvm_coverage: bool) -> String {
    if llvm_coverage {
        format!("{IPOPT_SPRAL_PARITY_PATCH_VERSION}-{IPOPT_LLVM_COVERAGE_BUILD_VERSION}")
    } else {
        IPOPT_SPRAL_PARITY_PATCH_VERSION.to_string()
    }
}

fn combined_flag_env(base_name: &str, required_flags: &str, optional_sources: &[&str]) -> String {
    let mut value = env::var(base_name).unwrap_or_default();
    append_flags(&mut value, required_flags);
    for source in optional_sources {
        if let Ok(flags) = env::var(source) {
            append_flags(&mut value, flags.trim());
        }
    }
    value
}

fn append_flags(value: &mut String, flags: &str) {
    let flags = flags.trim();
    if flags.is_empty() {
        return;
    }
    if !value.trim().is_empty() {
        value.push(' ');
    }
    value.push_str(flags);
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

fn replace_required(text: &mut String, path: &Path, needle: &str, replacement: &str) {
    assert!(
        text.contains(needle),
        "failed to patch {}: missing source anchor {needle:?}",
        path.display()
    );
    *text = text.replace(needle, replacement);
}

fn replace_required_once(text: &mut String, path: &Path, needle: &str, replacement: &str) {
    assert!(
        text.contains(needle),
        "failed to patch {}: missing source anchor {needle:?}",
        path.display()
    );
    *text = text.replacen(needle, replacement, 1);
}

fn patch_ipopt_spral_solver_interface(source: &Path) {
    let path = source
        .join("src")
        .join("Algorithm")
        .join("LinearSolvers")
        .join("IpSpralSolverInterface.cpp");
    let mut text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    if text.contains("OptivibreDumpSpralInterfaceState") {
        return;
    }

    replace_required(
        &mut text,
        &path,
        "#include <cinttypes>\n",
        "#include <cinttypes>\n#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n",
    );
    replace_required(
        &mut text,
        &path,
        "namespace Ipopt\n{\n\nSpralSolverInterface::~SpralSolverInterface()",
        r#"namespace Ipopt
{

namespace
{

Index OptivibreSpralCompressedNnz(
   Index        ndim,
   const Index* ia,
   int          array_base
)
{
   if( ia == NULL || ndim < 0 )
   {
      return 0;
   }
   return ia[ndim] - array_base;
}

FILE* OptivibreOpenSpralDump(
   const char* phase,
   int         call_index
)
{
   const char* dump_dir = std::getenv("GLIDER_PARITY_IPOPT_SPRAL_DUMP_DIR");
   if( dump_dir == NULL || dump_dir[0] == '\0' )
   {
      return NULL;
   }

   char path[4096];
   std::snprintf(path, sizeof(path), "%s/ipopt_spral_%06d_%s.txt", dump_dir, call_index, phase);
   return std::fopen(path, "w");
}

void OptivibrePrintDoubleBits(
   FILE*  file,
   double value
)
{
   std::uint64_t bits = 0;
   std::memcpy(&bits, &value, sizeof(bits));
   std::fprintf(file, "0x%016" PRIx64, bits);
}

void OptivibreDumpIndexArray(
   FILE*        file,
   const char*  name,
   Index        len,
   const Index* values
)
{
   std::fprintf(file, "%s=[", name);
   for( Index i = 0; i < len; i++ )
   {
      std::fprintf(file, "%s%lld", i == 0 ? "" : ",", static_cast<long long>(values[i]));
   }
   std::fprintf(file, "]\n");
}

void OptivibreDumpDoubleArray(
   FILE*         file,
   const char*   name,
   Index         len,
   const double* values
)
{
   std::fprintf(file, "%s=[", name);
   for( Index i = 0; i < len; i++ )
   {
      std::fprintf(file, "%s%.17e", i == 0 ? "" : ",", values[i]);
   }
   std::fprintf(file, "]\n");

   std::fprintf(file, "%s_bits=[", name);
   for( Index i = 0; i < len; i++ )
   {
      if( i != 0 )
      {
         std::fprintf(file, ",");
      }
      OptivibrePrintDoubleBits(file, values[i]);
   }
   std::fprintf(file, "]\n");
}

void OptivibreDumpSpralInterfaceState(
   const char*                       phase,
   int                               call_index,
   bool                              new_matrix,
   Index                             ndim,
   const Index*                      ia,
   const Index*                      ja,
   Index                             nrhs,
   const double*                     values,
   const double*                     rhs_vals,
   const double*                     scaling,
   const struct spral_ssids_options& control,
   const struct spral_ssids_inform*  info
)
{
   FILE* file = OptivibreOpenSpralDump(phase, call_index);
   if( file == NULL )
   {
      return;
   }

   const Index nonzeros = OptivibreSpralCompressedNnz(ndim, ia, control.array_base);
   std::fprintf(file, "version=1\n");
   std::fprintf(file, "phase=%s\n", phase);
   std::fprintf(file, "call_index=%d\n", call_index);
   std::fprintf(file, "new_matrix=%d\n", new_matrix ? 1 : 0);
   std::fprintf(file, "ndim=%lld\n", static_cast<long long>(ndim));
   std::fprintf(file, "nonzeros=%lld\n", static_cast<long long>(nonzeros));
   std::fprintf(file, "nrhs=%lld\n", static_cast<long long>(nrhs));
   std::fprintf(file, "control_array_base=%d\n", control.array_base);
   std::fprintf(file, "control_ordering=%d\n", control.ordering);
   std::fprintf(file, "control_scaling=%d\n", control.scaling);
   std::fprintf(file, "control_pivot_method=%d\n", control.pivot_method);
   std::fprintf(file, "control_small=%.17e\n", control.small);
   std::fprintf(file, "control_u=%.17e\n", control.u);
   std::fprintf(file, "control_nemin=%d\n", control.nemin);
   if( info != NULL )
   {
      std::fprintf(file, "info_flag=%d\n", info->flag);
      std::fprintf(file, "info_matrix_rank=%d\n", info->matrix_rank);
      std::fprintf(file, "info_num_delay=%d\n", info->num_delay);
      std::fprintf(file, "info_num_factor=%" PRId64 "\n", info->num_factor);
      std::fprintf(file, "info_num_flops=%" PRId64 "\n", info->num_flops);
      std::fprintf(file, "info_num_neg=%d\n", info->num_neg);
      std::fprintf(file, "info_num_sup=%d\n", info->num_sup);
      std::fprintf(file, "info_num_two=%d\n", info->num_two);
      std::fprintf(file, "info_maxfront=%d\n", info->maxfront);
      std::fprintf(file, "info_maxsupernode=%d\n", info->maxsupernode);
   }
   OptivibreDumpIndexArray(file, "ia", ndim + 1, ia);
   OptivibreDumpIndexArray(file, "ja", nonzeros, ja);
   OptivibreDumpDoubleArray(file, "values", nonzeros, values);
   OptivibreDumpDoubleArray(file, "rhs", ndim * nrhs, rhs_vals);
   if( scaling != NULL )
   {
      OptivibreDumpDoubleArray(file, "scaling", ndim, scaling);
   }
   std::fclose(file);
}

} // namespace

SpralSolverInterface::~SpralSolverInterface()"#,
    );
    replace_required(
        &mut text,
        &path,
        "   struct spral_ssids_inform info;\n\n   if( new_matrix || pivtol_changed_ )",
        "   struct spral_ssids_inform info;\n   const int optivibre_dump_call_index = fctidx_++;\n\n   OptivibreDumpSpralInterfaceState(\"before\", optivibre_dump_call_index, new_matrix, ndim_, ia, ja, nrhs, val_, rhs_vals, scaling_, control_, NULL);\n\n   if( new_matrix || pivtol_changed_ )",
    );
    replace_required(
        &mut text,
        &path,
        "         spral_ssids_analyse_ptr32(false, ndim_, NULL, ia, ja, val_, &akeep_, &control_, &info);\n\n         Jnlst().Printf",
        "         spral_ssids_analyse_ptr32(false, ndim_, NULL, ia, ja, val_, &akeep_, &control_, &info);\n         OptivibreDumpSpralInterfaceState(\"after_analyse\", optivibre_dump_call_index, new_matrix, ndim_, ia, ja, nrhs, val_, rhs_vals, scaling_, control_, &info);\n\n         Jnlst().Printf",
    );
    replace_required(
        &mut text,
        &path,
        "      spral_ssids_factor_ptr32(false, ia, ja, val_, scaling_, akeep_, &fkeep_, &control_, &info);\n\n      Jnlst().Printf",
        "      spral_ssids_factor_ptr32(false, ia, ja, val_, scaling_, akeep_, &fkeep_, &control_, &info);\n      OptivibreDumpSpralInterfaceState(\"after_factor\", optivibre_dump_call_index, new_matrix, ndim_, ia, ja, nrhs, val_, rhs_vals, scaling_, control_, &info);\n\n      Jnlst().Printf",
    );
    replace_required(
        &mut text,
        &path,
        "      spral_ssids_solve(0, nrhs, rhs_vals, ndim_, akeep_, fkeep_, &control_, &info);\n\n      if( HaveIpData() )",
        "      spral_ssids_solve(0, nrhs, rhs_vals, ndim_, akeep_, fkeep_, &control_, &info);\n      OptivibreDumpSpralInterfaceState(\"after_solve\", optivibre_dump_call_index, new_matrix, ndim_, ia, ja, nrhs, val_, rhs_vals, scaling_, control_, &info);\n\n      if( HaveIpData() )",
    );

    fs::write(&path, text)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
}

fn patch_ipopt_expansion_matrix(source: &Path) {
    let path = source
        .join("src")
        .join("LinAlg")
        .join("IpExpansionMatrix.cpp");
    let mut text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    if text.contains("synthetic_plus_source_like") {
        return;
    }
    if text.contains("OptivibreDumpSinvBlrmZMTdBr") {
        upgrade_ipopt_expansion_matrix_patch(&path, text);
        return;
    }

    replace_required(
        &mut text,
        &path,
        "#include \"IpDenseVector.hpp\"\n",
        "#include \"IpDenseVector.hpp\"\n\n#include <cinttypes>\n#include <cmath>\n#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n",
    );
    replace_required(
        &mut text,
        &path,
        "namespace Ipopt\n{\n\n#if IPOPT_VERBOSITY > 0",
        r#"namespace Ipopt
{

namespace
{

#if defined(_MSC_VER)
#define OPTIVIBRE_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define OPTIVIBRE_NOINLINE __attribute__((noinline))
#else
#define OPTIVIBRE_NOINLINE
#endif

OPTIVIBRE_NOINLINE Number OptivibreSourceLikePlus(
   Number r,
   Number z,
   Number d,
   Number s
)
{
   return (r + z * d) / s;
}

OPTIVIBRE_NOINLINE Number OptivibreSourceLikeMinus(
   Number r,
   Number z,
   Number d,
   Number s
)
{
   return (r - z * d) / s;
}

void OptivibrePrintDoubleBits(
   FILE*  file,
   double value
)
{
   std::uint64_t bits = 0;
   std::memcpy(&bits, &value, sizeof(bits));
   std::fprintf(file, "0x%016" PRIx64, bits);
}

bool OptivibreShouldDumpExpansionCall(
   int call_index
)
{
   const char* dump_dir = std::getenv("GLIDER_PARITY_IPOPT_EXPANSION_DUMP_DIR");
   if( dump_dir == NULL || dump_dir[0] == '\0' )
   {
      return false;
   }

   const char* requested_call = std::getenv("GLIDER_PARITY_IPOPT_EXPANSION_DUMP_CALL");
   if( requested_call == NULL || requested_call[0] == '\0' )
   {
      return true;
   }
   char* end = NULL;
   const long parsed = std::strtol(requested_call, &end, 10);
   return end != requested_call && *end == '\0' && parsed == call_index;
}

FILE* OptivibreOpenExpansionDump(
   int call_index
)
{
   const char* dump_dir = std::getenv("GLIDER_PARITY_IPOPT_EXPANSION_DUMP_DIR");
   if( dump_dir == NULL || dump_dir[0] == '\0' )
   {
      return NULL;
   }

   char path[4096];
   std::snprintf(path, sizeof(path), "%s/ipopt_expansion_%06d.txt", dump_dir, call_index);
   return std::fopen(path, "w");
}

void OptivibreDumpNumber(
   FILE*       file,
   const char* name,
   Number      value
)
{
   std::fprintf(file, "%s=%.17e\n", name, value);
   std::fprintf(file, "%s_bits=", name);
   OptivibrePrintDoubleBits(file, value);
   std::fprintf(file, "\n");
}

void OptivibreDumpSinvBlrmZMTdBr(
   int                call_index,
   Number             alpha,
   Index              ncols,
   Index              nrows,
   const Index*       exp_pos,
   const DenseVector* dense_S,
   const DenseVector* dense_R,
   const DenseVector* dense_Z,
   const DenseVector* dense_D,
   const DenseVector* dense_X
)
{
   if( !OptivibreShouldDumpExpansionCall(call_index) )
   {
      return;
   }
   FILE* file = OptivibreOpenExpansionDump(call_index);
   if( file == NULL )
   {
      return;
   }

   const bool r_homogeneous = dense_R->IsHomogeneous();
   const bool z_homogeneous = dense_Z->IsHomogeneous();
   const Number* vals_S = dense_S->Values();
   const Number* vals_D = dense_D->Values();
   const Number* vals_R = r_homogeneous ? NULL : dense_R->Values();
   const Number* vals_Z = z_homogeneous ? NULL : dense_Z->Values();
   const Number* vals_X = dense_X->Values();

   std::fprintf(file, "version=1\n");
   std::fprintf(file, "call_index=%d\n", call_index);
   std::fprintf(file, "alpha=%.17e\n", alpha);
   std::fprintf(file, "ncols=%lld\n", static_cast<long long>(ncols));
   std::fprintf(file, "nrows=%lld\n", static_cast<long long>(nrows));
   std::fprintf(file, "r_homogeneous=%d\n", r_homogeneous ? 1 : 0);
   std::fprintf(file, "z_homogeneous=%d\n", z_homogeneous ? 1 : 0);
   if( r_homogeneous )
   {
      OptivibreDumpNumber(file, "r_scalar", dense_R->Scalar());
   }
   if( z_homogeneous )
   {
      OptivibreDumpNumber(file, "z_scalar", dense_Z->Scalar());
   }

   const volatile Number synthetic_r_plus = -1.;
   const volatile Number synthetic_r_minus = 1.;
   const volatile Number synthetic_z = 1.e308;
   const volatile Number synthetic_d = 1.e-308;
   const volatile Number synthetic_s = 1.;
   const Number source_like_plus = OptivibreSourceLikePlus(
      synthetic_r_plus,
      synthetic_z,
      synthetic_d,
      synthetic_s);
   const volatile Number separate_plus_product = synthetic_z * synthetic_d;
   const volatile Number separate_plus_sum = synthetic_r_plus + separate_plus_product;
   const Number separate_plus = separate_plus_sum / synthetic_s;
   const Number fma_plus = std::fma(
      static_cast<Number>(synthetic_z),
      static_cast<Number>(synthetic_d),
      static_cast<Number>(synthetic_r_plus)) / static_cast<Number>(synthetic_s);
   const Number source_like_minus = OptivibreSourceLikeMinus(
      synthetic_r_minus,
      synthetic_z,
      synthetic_d,
      synthetic_s);
   const volatile Number separate_minus_product = synthetic_z * synthetic_d;
   const volatile Number separate_minus_sum = synthetic_r_minus - separate_minus_product;
   const Number separate_minus = separate_minus_sum / synthetic_s;
   const Number fma_minus = std::fma(
      -static_cast<Number>(synthetic_z),
      static_cast<Number>(synthetic_d),
      static_cast<Number>(synthetic_r_minus)) / static_cast<Number>(synthetic_s);
   OptivibreDumpNumber(file, "synthetic_plus_source_like", source_like_plus);
   OptivibreDumpNumber(file, "synthetic_plus_separate", separate_plus);
   OptivibreDumpNumber(file, "synthetic_plus_fma", fma_plus);
   OptivibreDumpNumber(file, "synthetic_minus_source_like", source_like_minus);
   OptivibreDumpNumber(file, "synthetic_minus_separate", separate_minus);
   OptivibreDumpNumber(file, "synthetic_minus_fma", fma_minus);

   std::fprintf(file, "rows=index,exp_pos,s,r,z,d,x,separate,fma,x_bits,separate_bits,fma_bits\n");
   Index x_matches_separate = 0;
   Index x_matches_fma = 0;
   Index separate_differs_from_fma = 0;
   for( Index i = 0; i < ncols; i++ )
   {
      const Index d_index = exp_pos[i];
      const Number s = vals_S[i];
      const Number r = r_homogeneous ? dense_R->Scalar() : vals_R[i];
      const Number z = z_homogeneous ? dense_Z->Scalar() : vals_Z[i];
      const Number d = vals_D[d_index];
      const Number x = vals_X[i];
      Number separate;
      Number fma;
      if( alpha == 1. )
      {
         const volatile Number product = z * d;
         const volatile Number sum = r + product;
         separate = sum / s;
         fma = std::fma(z, d, r) / s;
      }
      else if( alpha == -1. )
      {
         const volatile Number product = z * d;
         const volatile Number sum = r - product;
         separate = sum / s;
         fma = std::fma(-z, d, r) / s;
      }
      else
      {
         const volatile Number val = alpha * z;
         const volatile Number product = val * d;
         const volatile Number sum = r + product;
         separate = sum / s;
         fma = std::fma(val, d, r) / s;
      }

      if( std::memcmp(&x, &separate, sizeof(Number)) == 0 )
      {
         x_matches_separate++;
      }
      if( std::memcmp(&x, &fma, sizeof(Number)) == 0 )
      {
         x_matches_fma++;
      }
      if( std::memcmp(&separate, &fma, sizeof(Number)) != 0 )
      {
         separate_differs_from_fma++;
      }

      std::fprintf(file, "row=%lld,%lld,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,",
         static_cast<long long>(i),
         static_cast<long long>(d_index),
         s,
         r,
         z,
         d,
         x,
         separate,
         fma);
      OptivibrePrintDoubleBits(file, x);
      std::fprintf(file, ",");
      OptivibrePrintDoubleBits(file, separate);
      std::fprintf(file, ",");
      OptivibrePrintDoubleBits(file, fma);
      std::fprintf(file, "\n");
   }
   std::fprintf(file, "summary_x_matches_separate=%lld\n", static_cast<long long>(x_matches_separate));
   std::fprintf(file, "summary_x_matches_fma=%lld\n", static_cast<long long>(x_matches_fma));
   std::fprintf(file, "summary_separate_differs_from_fma=%lld\n", static_cast<long long>(separate_differs_from_fma));
   std::fclose(file);
}

} // namespace

#if IPOPT_VERBOSITY > 0"#,
    );
    replace_required(
        &mut text,
        &path,
        "   Number* vals_X = dense_X->Values();\n\n   if( dense_R->IsHomogeneous() )",
        "   Number* vals_X = dense_X->Values();\n   static int optivibre_expansion_call_index = 0;\n   const int optivibre_this_expansion_call = optivibre_expansion_call_index++;\n\n   if( dense_R->IsHomogeneous() )",
    );
    replace_required(
        &mut text,
        &path,
        "      }\n   }\n}\n\nvoid ExpansionMatrix::ComputeRowAMaxImpl",
        "      }\n   }\n\n   OptivibreDumpSinvBlrmZMTdBr(\n      optivibre_this_expansion_call,\n      alpha,\n      NCols(),\n      NRows(),\n      exp_pos,\n      dense_S,\n      dense_R,\n      dense_Z,\n      dense_D,\n      dense_X);\n}\n\nvoid ExpansionMatrix::ComputeRowAMaxImpl",
    );

    fs::write(&path, text)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
}

fn upgrade_ipopt_expansion_matrix_patch(path: &Path, mut text: String) {
    text = text.replace(
        "void OptivibrePrintDoubleBits(\n",
        r#"#if defined(_MSC_VER)
#define OPTIVIBRE_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define OPTIVIBRE_NOINLINE __attribute__((noinline))
#else
#define OPTIVIBRE_NOINLINE
#endif

OPTIVIBRE_NOINLINE Number OptivibreSourceLikePlus(
   Number r,
   Number z,
   Number d,
   Number s
)
{
   return (r + z * d) / s;
}

OPTIVIBRE_NOINLINE Number OptivibreSourceLikeMinus(
   Number r,
   Number z,
   Number d,
   Number s
)
{
   return (r - z * d) / s;
}

void OptivibrePrintDoubleBits(
"#,
    );
    text = text.replace(
        "         separate = (r + z * d) / s;\n         fma = std::fma(z, d, r) / s;",
        "         const volatile Number product = z * d;\n         const volatile Number sum = r + product;\n         separate = sum / s;\n         fma = std::fma(z, d, r) / s;",
    );
    text = text.replace(
        "         separate = (r - z * d) / s;\n         fma = std::fma(-z, d, r) / s;",
        "         const volatile Number product = z * d;\n         const volatile Number sum = r - product;\n         separate = sum / s;\n         fma = std::fma(-z, d, r) / s;",
    );
    text = text.replace(
        "         const Number val = alpha * z;\n         separate = (r + val * d) / s;\n         fma = std::fma(val, d, r) / s;",
        "         const volatile Number val = alpha * z;\n         const volatile Number product = val * d;\n         const volatile Number sum = r + product;\n         separate = sum / s;\n         fma = std::fma(val, d, r) / s;",
    );
    text = text.replace(
        r#"   if( z_homogeneous )
   {
      OptivibreDumpNumber(file, "z_scalar", dense_Z->Scalar());
   }

   std::fprintf(file, "rows=index,exp_pos,s,r,z,d,x,separate,fma,x_bits,separate_bits,fma_bits\n");"#,
        r#"   if( z_homogeneous )
   {
      OptivibreDumpNumber(file, "z_scalar", dense_Z->Scalar());
   }

   const volatile Number synthetic_r_plus = -1.;
   const volatile Number synthetic_r_minus = 1.;
   const volatile Number synthetic_z = 1.e308;
   const volatile Number synthetic_d = 1.e-308;
   const volatile Number synthetic_s = 1.;
   const Number source_like_plus = OptivibreSourceLikePlus(
      synthetic_r_plus,
      synthetic_z,
      synthetic_d,
      synthetic_s);
   const volatile Number separate_plus_product = synthetic_z * synthetic_d;
   const volatile Number separate_plus_sum = synthetic_r_plus + separate_plus_product;
   const Number separate_plus = separate_plus_sum / synthetic_s;
   const Number fma_plus = std::fma(
      static_cast<Number>(synthetic_z),
      static_cast<Number>(synthetic_d),
      static_cast<Number>(synthetic_r_plus)) / static_cast<Number>(synthetic_s);
   const Number source_like_minus = OptivibreSourceLikeMinus(
      synthetic_r_minus,
      synthetic_z,
      synthetic_d,
      synthetic_s);
   const volatile Number separate_minus_product = synthetic_z * synthetic_d;
   const volatile Number separate_minus_sum = synthetic_r_minus - separate_minus_product;
   const Number separate_minus = separate_minus_sum / synthetic_s;
   const Number fma_minus = std::fma(
      -static_cast<Number>(synthetic_z),
      static_cast<Number>(synthetic_d),
      static_cast<Number>(synthetic_r_minus)) / static_cast<Number>(synthetic_s);
   OptivibreDumpNumber(file, "synthetic_plus_source_like", source_like_plus);
   OptivibreDumpNumber(file, "synthetic_plus_separate", separate_plus);
   OptivibreDumpNumber(file, "synthetic_plus_fma", fma_plus);
   OptivibreDumpNumber(file, "synthetic_minus_source_like", source_like_minus);
   OptivibreDumpNumber(file, "synthetic_minus_separate", separate_minus);
   OptivibreDumpNumber(file, "synthetic_minus_fma", fma_minus);

   std::fprintf(file, "rows=index,exp_pos,s,r,z,d,x,separate,fma,x_bits,separate_bits,fma_bits\n");"#,
    );

    fs::write(path, text)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
}

fn patch_ipopt_dense_vector(source: &Path) {
    let path = source.join("src").join("LinAlg").join("IpDenseVector.cpp");
    let mut text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    if text.contains("OptivibreDumpDenseAddTwoVectors") {
        upgrade_ipopt_dense_vector_patch(&path, text);
        return;
    }

    replace_required(
        &mut text,
        &path,
        "#include \"IpDenseVector.hpp\"\n",
        "#include \"IpDenseVector.hpp\"\n\n#include <cinttypes>\n#include <cmath>\n#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n",
    );
    replace_required(
        &mut text,
        &path,
        "namespace Ipopt\n{\n\n#if IPOPT_VERBOSITY > 0",
        r#"namespace Ipopt
{

namespace
{

#if defined(_MSC_VER)
#define OPTIVIBRE_DENSE_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define OPTIVIBRE_DENSE_NOINLINE __attribute__((noinline))
#else
#define OPTIVIBRE_DENSE_NOINLINE
#endif

OPTIVIBRE_DENSE_NOINLINE Number OptivibreDenseSourceLikeAdd(
   Number v1,
   Number b,
   Number v2
)
{
   return v1 + b * v2;
}

void OptivibreDensePrintDoubleBits(
   FILE*  file,
   double value
)
{
   std::uint64_t bits = 0;
   std::memcpy(&bits, &value, sizeof(bits));
   std::fprintf(file, "0x%016" PRIx64, bits);
}

bool OptivibreShouldDumpDenseAddCall(
   int call_index
)
{
   const char* dump_dir = std::getenv("GLIDER_PARITY_IPOPT_DENSE_ADD_DUMP_DIR");
   if( dump_dir == NULL || dump_dir[0] == '\0' )
   {
      return false;
   }

   const char* requested_call = std::getenv("GLIDER_PARITY_IPOPT_DENSE_ADD_DUMP_CALL");
   if( requested_call == NULL || requested_call[0] == '\0' )
   {
      return true;
   }
   char* end = NULL;
   const long parsed = std::strtol(requested_call, &end, 10);
   return end != requested_call && *end == '\0' && parsed == call_index;
}

FILE* OptivibreOpenDenseAddDump(
   int call_index
)
{
   const char* dump_dir = std::getenv("GLIDER_PARITY_IPOPT_DENSE_ADD_DUMP_DIR");
   if( dump_dir == NULL || dump_dir[0] == '\0' )
   {
      return NULL;
   }

   char path[4096];
   std::snprintf(path, sizeof(path), "%s/ipopt_dense_add_%06d.txt", dump_dir, call_index);
   return std::fopen(path, "w");
}

void OptivibreDenseDumpNumber(
   FILE*       file,
   const char* name,
   Number      value
)
{
   std::fprintf(file, "%s=%.17e\n", name, value);
   std::fprintf(file, "%s_bits=", name);
   OptivibreDensePrintDoubleBits(file, value);
   std::fprintf(file, "\n");
}

void OptivibreDumpDenseAddTwoVectors(
   int           call_index,
   Index         dim,
   Number        a,
   Number        b,
   Number        c,
   const Number* values_v1,
   const Number* values_v2,
   const Number* values_result
)
{
   if( !(a == 1. && c == 0. && b != 0. && b != 1. && b != -1.) )
   {
      return;
   }
   if( values_v1 == NULL || values_v2 == NULL || values_result == NULL )
   {
      return;
   }
   if( !OptivibreShouldDumpDenseAddCall(call_index) )
   {
      return;
   }

   FILE* file = OptivibreOpenDenseAddDump(call_index);
   if( file == NULL )
   {
      return;
   }

   const volatile Number synthetic_v1 = -1.;
   const volatile Number synthetic_b = 1.e308;
   const volatile Number synthetic_v2 = 1.e-308;
   const Number synthetic_source_like = OptivibreDenseSourceLikeAdd(
      synthetic_v1,
      synthetic_b,
      synthetic_v2);
   const volatile Number synthetic_product = synthetic_b * synthetic_v2;
   const volatile Number synthetic_sum = synthetic_v1 + synthetic_product;
   const Number synthetic_separate = synthetic_sum;
   const Number synthetic_fma = std::fma(
      static_cast<Number>(synthetic_b),
      static_cast<Number>(synthetic_v2),
      static_cast<Number>(synthetic_v1));

   std::fprintf(file, "version=1\n");
   std::fprintf(file, "call_index=%d\n", call_index);
   std::fprintf(file, "dim=%lld\n", static_cast<long long>(dim));
   std::fprintf(file, "a=%.17e\n", a);
   std::fprintf(file, "b=%.17e\n", b);
   std::fprintf(file, "c=%.17e\n", c);
   OptivibreDenseDumpNumber(file, "synthetic_source_like", synthetic_source_like);
   OptivibreDenseDumpNumber(file, "synthetic_separate", synthetic_separate);
   OptivibreDenseDumpNumber(file, "synthetic_fma", synthetic_fma);

   Index result_matches_separate = 0;
   Index result_matches_fma = 0;
   Index separate_differs_from_fma = 0;
   Index printed_rows = 0;
   std::fprintf(file, "rows=index,v1,b,v2,result,separate,fma,result_bits,separate_bits,fma_bits\n");
   for( Index i = 0; i < dim; i++ )
   {
      const Number v1 = values_v1[i];
      const Number v2 = values_v2[i];
      const Number result = values_result[i];
      const volatile Number product = b * v2;
      const volatile Number sum = v1 + product;
      const Number separate = sum;
      const Number fma = std::fma(b, v2, v1);

      const bool matches_separate = std::memcmp(&result, &separate, sizeof(Number)) == 0;
      const bool matches_fma = std::memcmp(&result, &fma, sizeof(Number)) == 0;
      const bool candidates_differ = std::memcmp(&separate, &fma, sizeof(Number)) != 0;
      if( matches_separate )
      {
         result_matches_separate++;
      }
      if( matches_fma )
      {
         result_matches_fma++;
      }
      if( candidates_differ )
      {
         separate_differs_from_fma++;
      }

      if( candidates_differ && printed_rows < 64 )
      {
         std::fprintf(file, "row=%lld,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,",
            static_cast<long long>(i),
            v1,
            b,
            v2,
            result,
            separate,
            fma);
         OptivibreDensePrintDoubleBits(file, result);
         std::fprintf(file, ",");
         OptivibreDensePrintDoubleBits(file, separate);
         std::fprintf(file, ",");
         OptivibreDensePrintDoubleBits(file, fma);
         std::fprintf(file, "\n");
         printed_rows++;
      }
   }
   std::fprintf(file, "summary_result_matches_separate=%lld\n", static_cast<long long>(result_matches_separate));
   std::fprintf(file, "summary_result_matches_fma=%lld\n", static_cast<long long>(result_matches_fma));
   std::fprintf(file, "summary_separate_differs_from_fma=%lld\n", static_cast<long long>(separate_differs_from_fma));
   std::fprintf(file, "summary_printed_rows=%lld\n", static_cast<long long>(printed_rows));
   std::fclose(file);
}

} // namespace

#if IPOPT_VERBOSITY > 0"#,
    );
    replace_required(
        &mut text,
        &path,
        "   DBG_ASSERT(c == 0. || initialized_);\n   if( (c == 0. || homogeneous_) && homogeneous_v1 && homogeneous_v2 )",
        "   static int optivibre_dense_add_call_index = 0;\n   const int optivibre_this_dense_add_call = optivibre_dense_add_call_index++;\n\n   DBG_ASSERT(c == 0. || initialized_);\n   if( (c == 0. || homogeneous_) && homogeneous_v1 && homogeneous_v2 )",
    );
    insert_ipopt_dense_vector_dump_call(&path, &mut text);

    fs::write(&path, text)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
}

fn upgrade_ipopt_dense_vector_patch(path: &Path, mut text: String) {
    if insert_ipopt_dense_vector_dump_call(path, &mut text) {
        fs::write(path, text)
            .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
    }
}

fn insert_ipopt_dense_vector_dump_call(path: &Path, text: &mut String) -> bool {
    if text.contains("OptivibreDumpDenseAddTwoVectors(\n      optivibre_this_dense_add_call") {
        return false;
    }
    replace_required(
        text,
        path,
        "   initialized_ = true;\n}\n\nNumber DenseVector::FracToBoundImpl",
        "   OptivibreDumpDenseAddTwoVectors(\n      optivibre_this_dense_add_call,\n      Dim(),\n      a,\n      b,\n      c,\n      values_v1,\n      values_v2,\n      values_);\n   initialized_ = true;\n}\n\nNumber DenseVector::FracToBoundImpl",
    );
    true
}

fn patch_ipopt_tsym_linear_solver(source: &Path) {
    let path = source
        .join("src")
        .join("Algorithm")
        .join("LinearSolvers")
        .join("IpTSymLinearSolver.cpp");
    let mut text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    if text.contains("OptivibreDumpTSymLinearSolverState") {
        if upgrade_ipopt_tsym_linear_solver_patch(&mut text) {
            fs::write(&path, text)
                .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
        }
        return;
    }

    replace_required(
        &mut text,
        &path,
        "#include \"IpBlas.hpp\"\n",
        "#include \"IpBlas.hpp\"\n\n#include <cinttypes>\n#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n",
    );
    replace_required(
        &mut text,
        &path,
        "namespace Ipopt\n{\n#if IPOPT_VERBOSITY > 0",
        r#"namespace Ipopt
{

namespace
{

void OptivibreTSymPrintDoubleBits(
   FILE*  file,
   double value
)
{
   std::uint64_t bits = 0;
   std::memcpy(&bits, &value, sizeof(bits));
   std::fprintf(file, "0x%016" PRIx64, bits);
}

bool OptivibreShouldDumpTSymLinearSolverCall(
   int call_index
)
{
   const char* dump_dir = std::getenv("GLIDER_PARITY_IPOPT_TSYM_DUMP_DIR");
   if( dump_dir == NULL || dump_dir[0] == '\0' )
   {
      return false;
   }

   const char* requested_call = std::getenv("GLIDER_PARITY_IPOPT_TSYM_DUMP_CALL");
   if( requested_call == NULL || requested_call[0] == '\0' )
   {
      return true;
   }
   char* end = NULL;
   const long parsed = std::strtol(requested_call, &end, 10);
   return end != requested_call && *end == '\0' && parsed == call_index;
}

FILE* OptivibreOpenTSymLinearSolverDump(
   int         call_index,
   const char* phase
)
{
   const char* dump_dir = std::getenv("GLIDER_PARITY_IPOPT_TSYM_DUMP_DIR");
   if( dump_dir == NULL || dump_dir[0] == '\0' )
   {
      return NULL;
   }

   char path[4096];
   std::snprintf(path, sizeof(path), "%s/ipopt_tsym_%06d_%s.txt", dump_dir, call_index, phase);
   return std::fopen(path, "w");
}

void OptivibreTSymDumpNumberArray(
   FILE*         file,
   const char*   name,
   Index         len,
   const Number* values
)
{
   if( values == NULL )
   {
      return;
   }

   std::fprintf(file, "%s=[", name);
   for( Index i = 0; i < len; i++ )
   {
      std::fprintf(file, "%s%.17e", i == 0 ? "" : ",", values[i]);
   }
   std::fprintf(file, "]\n");

   std::fprintf(file, "%s_bits=[", name);
   for( Index i = 0; i < len; i++ )
   {
      if( i != 0 )
      {
         std::fprintf(file, ",");
      }
      OptivibreTSymPrintDoubleBits(file, values[i]);
   }
   std::fprintf(file, "]\n");
}

void OptivibreDumpTSymLinearSolverState(
   int           call_index,
   const char*   phase,
   Index         dim,
   Index         nrhs,
   bool          new_matrix,
   bool          use_scaling,
   const Number* scaling_factors,
   const Number* rhs_unscaled,
   const Number* rhs_scaled,
   const Number* solution_scaled,
   const Number* solution_unscaled
)
{
   if( !OptivibreShouldDumpTSymLinearSolverCall(call_index) )
   {
      return;
   }

   FILE* file = OptivibreOpenTSymLinearSolverDump(call_index, phase);
   if( file == NULL )
   {
      return;
   }

   std::fprintf(file, "version=1\n");
   std::fprintf(file, "call_index=%d\n", call_index);
   std::fprintf(file, "phase=%s\n", phase);
   std::fprintf(file, "dim=%lld\n", static_cast<long long>(dim));
   std::fprintf(file, "nrhs=%lld\n", static_cast<long long>(nrhs));
   std::fprintf(file, "new_matrix=%d\n", new_matrix ? 1 : 0);
   std::fprintf(file, "use_scaling=%d\n", use_scaling ? 1 : 0);
   OptivibreTSymDumpNumberArray(file, "scaling_factors", use_scaling ? dim : 0, scaling_factors);
   OptivibreTSymDumpNumberArray(file, "rhs_unscaled", dim * nrhs, rhs_unscaled);
   OptivibreTSymDumpNumberArray(file, "rhs_scaled", dim * nrhs, rhs_scaled);
   OptivibreTSymDumpNumberArray(file, "solution_scaled", dim * nrhs, solution_scaled);
   OptivibreTSymDumpNumberArray(file, "solution_unscaled", dim * nrhs, solution_unscaled);
   std::fclose(file);
}

} // namespace

#if IPOPT_VERBOSITY > 0"#,
    );
    replace_required(
        &mut text,
        &path,
        "   Index nrhs = (Index) rhsV.size();\n   Number* rhs_vals = new Number[dim_ * nrhs];",
        "   static int optivibre_tsym_call_index = 0;\n   const int optivibre_this_tsym_call = optivibre_tsym_call_index++;\n   const bool optivibre_dump_this_tsym = OptivibreShouldDumpTSymLinearSolverCall(optivibre_this_tsym_call);\n\n   Index nrhs = (Index) rhsV.size();\n   Number* rhs_vals = new Number[dim_ * nrhs];\n   Number* optivibre_tsym_rhs_unscaled = optivibre_dump_this_tsym ? new Number[dim_ * nrhs] : NULL;\n   Number* optivibre_tsym_solution_scaled = optivibre_dump_this_tsym ? new Number[dim_ * nrhs] : NULL;",
    );
    replace_required(
        &mut text,
        &path,
        "      TripletHelper::FillValuesFromVector(dim_, *rhsV[irhs], &rhs_vals[irhs * (dim_)]);\n      if( Jnlst().ProduceOutput",
        "      TripletHelper::FillValuesFromVector(dim_, *rhsV[irhs], &rhs_vals[irhs * (dim_)]);\n      if( optivibre_dump_this_tsym )\n      {\n         IpBlasCopy(dim_, &rhs_vals[irhs * (dim_)], 1, &optivibre_tsym_rhs_unscaled[irhs * (dim_)], 1);\n      }\n      if( Jnlst().ProduceOutput",
    );
    replace_required(
        &mut text,
        &path,
        "   bool done = false;\n   // Call the linear solver through the interface to solve the",
        "   OptivibreDumpTSymLinearSolverState(\n      optivibre_this_tsym_call,\n      \"before\",\n      dim_,\n      nrhs,\n      new_matrix,\n      use_scaling_,\n      scaling_factors_,\n      optivibre_tsym_rhs_unscaled,\n      rhs_vals,\n      NULL,\n      NULL);\n\n   bool done = false;\n   // Call the linear solver through the interface to solve the",
    );
    replace_required(
        &mut text,
        &path,
        "      for( Index irhs = 0; irhs < nrhs; irhs++ )\n      {\n         if( use_scaling_ )",
        "      for( Index irhs = 0; irhs < nrhs; irhs++ )\n      {\n         if( optivibre_dump_this_tsym )\n         {\n            IpBlasCopy(dim_, &rhs_vals[irhs * (dim_)], 1, &optivibre_tsym_solution_scaled[irhs * (dim_)], 1);\n         }\n         if( use_scaling_ )",
    );
    replace_required(
        &mut text,
        &path,
        "   delete[] rhs_vals;\n\n   return retval;",
        "   OptivibreDumpTSymLinearSolverState(\n      optivibre_this_tsym_call,\n      \"after\",\n      dim_,\n      nrhs,\n      new_matrix,\n      use_scaling_,\n      scaling_factors_,\n      optivibre_tsym_rhs_unscaled,\n      NULL,\n      optivibre_tsym_solution_scaled,\n      retval == SYMSOLVER_SUCCESS ? rhs_vals : NULL);\n\n   delete[] optivibre_tsym_solution_scaled;\n   delete[] optivibre_tsym_rhs_unscaled;\n   delete[] rhs_vals;\n\n   return retval;",
    );

    fs::write(&path, text)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
}

fn upgrade_ipopt_tsym_linear_solver_patch(text: &mut String) -> bool {
    let before = text.clone();

    *text = text.replace(
        "FILE* OptivibreOpenTSymLinearSolverDump(\n   int call_index\n)",
        "FILE* OptivibreOpenTSymLinearSolverDump(\n   int         call_index,\n   const char* phase\n)",
    );
    *text = text.replace(
        "std::snprintf(path, sizeof(path), \"%s/ipopt_tsym_%06d.txt\", dump_dir, call_index);",
        "std::snprintf(path, sizeof(path), \"%s/ipopt_tsym_%06d_%s.txt\", dump_dir, call_index, phase);",
    );
    *text = text.replace(
        "void OptivibreDumpTSymLinearSolverState(\n   int           call_index,\n   Index         dim,",
        "void OptivibreDumpTSymLinearSolverState(\n   int           call_index,\n   const char*   phase,\n   Index         dim,",
    );
    *text = text.replace(
        ")\n{\n   FILE* file = OptivibreOpenTSymLinearSolverDump(call_index);",
        ")\n{\n   if( !OptivibreShouldDumpTSymLinearSolverCall(call_index) )\n   {\n      return;\n   }\n\n   FILE* file = OptivibreOpenTSymLinearSolverDump(call_index, phase);",
    );
    *text = text.replace(
        "   std::fprintf(file, \"call_index=%d\\n\", call_index);\n   std::fprintf(file, \"dim=%lld\\n\", static_cast<long long>(dim));",
        "   std::fprintf(file, \"call_index=%d\\n\", call_index);\n   std::fprintf(file, \"phase=%s\\n\", phase);\n   std::fprintf(file, \"dim=%lld\\n\", static_cast<long long>(dim));",
    );
    *text = text.replace(
        "   OptivibreDumpTSymLinearSolverState(\n      optivibre_this_tsym_call,\n      dim_,\n      nrhs,\n      new_matrix,\n      use_scaling_,\n      scaling_factors_,\n      optivibre_tsym_rhs_unscaled,\n      rhs_vals,\n      NULL,\n      NULL);\n\n   bool done = false;",
        "   OptivibreDumpTSymLinearSolverState(\n      optivibre_this_tsym_call,\n      \"before\",\n      dim_,\n      nrhs,\n      new_matrix,\n      use_scaling_,\n      scaling_factors_,\n      optivibre_tsym_rhs_unscaled,\n      rhs_vals,\n      NULL,\n      NULL);\n\n   bool done = false;",
    );
    *text = text.replace(
        "   OptivibreDumpTSymLinearSolverState(\n      optivibre_this_tsym_call,\n      dim_,\n      nrhs,\n      new_matrix,\n      use_scaling_,\n      scaling_factors_,\n      optivibre_tsym_rhs_unscaled,\n      NULL,\n      optivibre_tsym_solution_scaled,\n      retval == SYMSOLVER_SUCCESS ? rhs_vals : NULL);",
        "   OptivibreDumpTSymLinearSolverState(\n      optivibre_this_tsym_call,\n      \"after\",\n      dim_,\n      nrhs,\n      new_matrix,\n      use_scaling_,\n      scaling_factors_,\n      optivibre_tsym_rhs_unscaled,\n      NULL,\n      optivibre_tsym_solution_scaled,\n      retval == SYMSOLVER_SUCCESS ? rhs_vals : NULL);",
    );

    *text != before
}

fn patch_ipopt_pdfull_space_solver(source: &Path) {
    let path = source
        .join("src")
        .join("Algorithm")
        .join("IpPDFullSpaceSolver.cpp");
    let mut text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    if text.contains("OptivibreDumpPdfullSpaceResidual") {
        let mut changed = false;
        changed |= insert_ipopt_pdfull_space_solver_residual_stage_dump(&path, &mut text);
        changed |= insert_ipopt_pdfull_space_solver_solve_once_dump(&path, &mut text);
        if changed {
            fs::write(&path, text)
                .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
        }
        return;
    }

    replace_required(
        &mut text,
        &path,
        "#include \"IpPDFullSpaceSolver.hpp\"\n#include \"IpDebug.hpp\"\n\n#include <cmath>\n",
        "#include \"IpPDFullSpaceSolver.hpp\"\n#include \"IpDebug.hpp\"\n#include \"IpDenseVector.hpp\"\n\n#include <cinttypes>\n#include <cmath>\n#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n",
    );
    replace_required(
        &mut text,
        &path,
        "namespace Ipopt\n{\n\n#if IPOPT_VERBOSITY > 0",
        r#"namespace Ipopt
{

namespace
{

struct OptivibrePdfsDenseView
{
   Index         count;
   const Number* values;
};

OptivibrePdfsDenseView OptivibrePdfsDenseVectorView(
   const SmartPtr<const Vector>& vector
)
{
   OptivibrePdfsDenseView view;
   view.count = 0;
   view.values = NULL;
   if( IsValid(vector) )
   {
      const DenseVector* dense = dynamic_cast<const DenseVector*>(GetRawPtr(vector));
      if( dense != NULL )
      {
         view.count = dense->Dim();
         view.values = dense->ExpandedValues();
      }
   }
   return view;
}

void OptivibrePdfsPrintDoubleBits(
   FILE*  file,
   double value
)
{
   std::uint64_t bits = 0;
   std::memcpy(&bits, &value, sizeof(bits));
   std::fprintf(file, "0x%016" PRIx64, bits);
}

void OptivibrePdfsDumpNumberArray(
   FILE*         file,
   const char*   name,
   Index         len,
   const Number* values
)
{
   std::fprintf(file, "%s=[", name);
   for( Index i = 0; i < len; i++ )
   {
      std::fprintf(file, "%s%.17e", i == 0 ? "" : ",", values[i]);
   }
   std::fprintf(file, "]\n");

   std::fprintf(file, "%s_bits=[", name);
   for( Index i = 0; i < len; i++ )
   {
      if( i != 0 )
      {
         std::fprintf(file, ",");
      }
      OptivibrePdfsPrintDoubleBits(file, values[i]);
   }
   std::fprintf(file, "]\n");
}

void OptivibrePdfsDumpVector(
   FILE*                         file,
   const char*                   name,
   const SmartPtr<const Vector>& vector
)
{
   const OptivibrePdfsDenseView view = OptivibrePdfsDenseVectorView(vector);
   char count_name[128];
   std::snprintf(count_name, sizeof(count_name), "%s_count", name);
   std::fprintf(file, "%s=%lld\n", count_name, static_cast<long long>(view.count));
   if( view.values != NULL )
   {
      OptivibrePdfsDumpNumberArray(file, name, view.count, view.values);
   }
}

void OptivibrePdfsDumpIteratesVector(
   FILE*                 file,
   const char*           prefix,
   const IteratesVector& values
)
{
   char name[128];
   std::snprintf(name, sizeof(name), "%s_x", prefix);
   OptivibrePdfsDumpVector(file, name, values.x());
   std::snprintf(name, sizeof(name), "%s_s", prefix);
   OptivibrePdfsDumpVector(file, name, values.s());
   std::snprintf(name, sizeof(name), "%s_y_c", prefix);
   OptivibrePdfsDumpVector(file, name, values.y_c());
   std::snprintf(name, sizeof(name), "%s_y_d", prefix);
   OptivibrePdfsDumpVector(file, name, values.y_d());
   std::snprintf(name, sizeof(name), "%s_z_L", prefix);
   OptivibrePdfsDumpVector(file, name, values.z_L());
   std::snprintf(name, sizeof(name), "%s_z_U", prefix);
   OptivibrePdfsDumpVector(file, name, values.z_U());
   std::snprintf(name, sizeof(name), "%s_v_L", prefix);
   OptivibrePdfsDumpVector(file, name, values.v_L());
   std::snprintf(name, sizeof(name), "%s_v_U", prefix);
   OptivibrePdfsDumpVector(file, name, values.v_U());
}

bool OptivibreShouldDumpPdfullSpaceResidual(
   int call_index
)
{
   const char* dump_dir = std::getenv("GLIDER_PARITY_IPOPT_RESIDUAL_DUMP_DIR");
   if( dump_dir == NULL || dump_dir[0] == '\0' )
   {
      return false;
   }

   const char* requested_call = std::getenv("GLIDER_PARITY_IPOPT_RESIDUAL_DUMP_CALL");
   if( requested_call == NULL || requested_call[0] == '\0' )
   {
      return true;
   }
   char* end = NULL;
   const long parsed = std::strtol(requested_call, &end, 10);
   return end != requested_call && *end == '\0' && parsed == call_index;
}

FILE* OptivibreOpenPdfullSpaceResidualDump(
   int call_index
)
{
   const char* dump_dir = std::getenv("GLIDER_PARITY_IPOPT_RESIDUAL_DUMP_DIR");
   if( dump_dir == NULL || dump_dir[0] == '\0' )
   {
      return NULL;
   }

   char path[4096];
   std::snprintf(path, sizeof(path), "%s/ipopt_pdfullspace_residual_%06d.txt", dump_dir, call_index);
   return std::fopen(path, "w");
}

void OptivibreDumpPdfullSpaceResidual(
   int                    call_index,
   Number                 delta_x,
   Number                 delta_s,
   Number                 delta_c,
   Number                 delta_d,
   const IteratesVector&  rhs,
   const IteratesVector&  res,
   const IteratesVector&  resid
)
{
   if( !OptivibreShouldDumpPdfullSpaceResidual(call_index) )
   {
      return;
   }

   FILE* file = OptivibreOpenPdfullSpaceResidualDump(call_index);
   if( file == NULL )
   {
      return;
   }

   std::fprintf(file, "version=1\n");
   std::fprintf(file, "call_index=%d\n", call_index);
   std::fprintf(file, "delta_x=%.17e\n", delta_x);
   std::fprintf(file, "delta_s=%.17e\n", delta_s);
   std::fprintf(file, "delta_c=%.17e\n", delta_c);
   std::fprintf(file, "delta_d=%.17e\n", delta_d);
   OptivibrePdfsDumpIteratesVector(file, "rhs", rhs);
   OptivibrePdfsDumpIteratesVector(file, "res", res);
   OptivibrePdfsDumpIteratesVector(file, "resid", resid);
   std::fclose(file);
}

} // namespace

#if IPOPT_VERBOSITY > 0"#,
    );
    replace_required(
        &mut text,
        &path,
        "   DBG_START_METH(\"PDFullSpaceSolver::ComputeResiduals\", dbg_verbosity);\n\n   DBG_PRINT_VECTOR(2, \"res\", res);",
        "   DBG_START_METH(\"PDFullSpaceSolver::ComputeResiduals\", dbg_verbosity);\n   static int optivibre_pdfullspace_residual_call_index = 0;\n   const int optivibre_this_residual_call = optivibre_pdfullspace_residual_call_index++;\n\n   DBG_PRINT_VECTOR(2, \"res\", res);",
    );
    replace_required(
        &mut text,
        &path,
        "   DBG_PRINT_VECTOR(2, \"resid\", resid);\n\n   if( Jnlst().ProduceOutput(J_MOREVECTOR, J_LINEAR_ALGEBRA) )",
        "   OptivibreDumpPdfullSpaceResidual(\n      optivibre_this_residual_call,\n      delta_x,\n      delta_s,\n      delta_c,\n      delta_d,\n      rhs,\n      res,\n      resid);\n\n   DBG_PRINT_VECTOR(2, \"resid\", resid);\n\n   if( Jnlst().ProduceOutput(J_MOREVECTOR, J_LINEAR_ALGEBRA) )",
    );
    insert_ipopt_pdfull_space_solver_residual_stage_dump(&path, &mut text);
    insert_ipopt_pdfull_space_solver_solve_once_dump(&path, &mut text);

    fs::write(&path, text)
        .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
}

fn insert_ipopt_pdfull_space_solver_solve_once_dump(path: &Path, text: &mut String) -> bool {
    let before = text.clone();

    if !text.contains("OptivibreShouldDumpPdfullSpaceSolveOnce") {
        replace_required_once(
            text,
            path,
            "\n} // namespace\n\n#if IPOPT_VERBOSITY > 0",
            r#"
void OptivibrePdfsDumpVectorRef(
   FILE*         file,
   const char*   name,
   const Vector& vector
)
{
   const DenseVector* dense = dynamic_cast<const DenseVector*>(&vector);
   char count_name[128];
   std::snprintf(count_name, sizeof(count_name), "%s_count", name);
   std::fprintf(file, "%s=%lld\n", count_name, dense == NULL ? 0LL : static_cast<long long>(dense->Dim()));
   if( dense != NULL )
   {
      OptivibrePdfsDumpNumberArray(file, name, dense->Dim(), dense->ExpandedValues());
   }
}

bool OptivibreShouldDumpPdfullSpaceSolveOnce(
   int call_index
)
{
   const char* dump_dir = std::getenv("GLIDER_PARITY_IPOPT_SOLVEONCE_RHS_DUMP_DIR");
   if( dump_dir == NULL || dump_dir[0] == '\0' )
   {
      return false;
   }

   const char* requested_call = std::getenv("GLIDER_PARITY_IPOPT_SOLVEONCE_RHS_DUMP_CALL");
   if( requested_call == NULL || requested_call[0] == '\0' )
   {
      return true;
   }
   char* end = NULL;
   const long parsed = std::strtol(requested_call, &end, 10);
   return end != requested_call && *end == '\0' && parsed == call_index;
}

FILE* OptivibreOpenPdfullSpaceSolveOnceDump(
   int call_index
)
{
   const char* dump_dir = std::getenv("GLIDER_PARITY_IPOPT_SOLVEONCE_RHS_DUMP_DIR");
   if( dump_dir == NULL || dump_dir[0] == '\0' )
   {
      return NULL;
   }

   char path[4096];
   std::snprintf(path, sizeof(path), "%s/ipopt_pdfullspace_solveonce_rhs_%06d.txt", dump_dir, call_index);
   return std::fopen(path, "w");
}

void OptivibreDumpPdfullSpaceSolveOnceRhs(
   int                    call_index,
   Number                 alpha,
   Number                 beta,
   const IteratesVector&  rhs,
   const Vector&          slack_x_L,
   const Vector&          slack_x_U,
   const Vector&          slack_s_L,
   const Vector&          slack_s_U,
   const SmartPtr<Vector>& augRhs_x,
   const SmartPtr<Vector>& augRhs_s
)
{
   if( !OptivibreShouldDumpPdfullSpaceSolveOnce(call_index) )
   {
      return;
   }

   FILE* file = OptivibreOpenPdfullSpaceSolveOnceDump(call_index);
   if( file == NULL )
   {
      return;
   }

   std::fprintf(file, "version=1\n");
   std::fprintf(file, "call_index=%d\n", call_index);
   std::fprintf(file, "alpha=%.17e\n", alpha);
   std::fprintf(file, "beta=%.17e\n", beta);
   OptivibrePdfsDumpIteratesVector(file, "rhs", rhs);
   OptivibrePdfsDumpVectorRef(file, "slack_x_L", slack_x_L);
   OptivibrePdfsDumpVectorRef(file, "slack_x_U", slack_x_U);
   OptivibrePdfsDumpVectorRef(file, "slack_s_L", slack_s_L);
   OptivibrePdfsDumpVectorRef(file, "slack_s_U", slack_s_U);
   SmartPtr<const Vector> const_augRhs_x = augRhs_x;
   SmartPtr<const Vector> const_augRhs_s = augRhs_s;
   OptivibrePdfsDumpVector(file, "aug_rhs_x", const_augRhs_x);
   OptivibrePdfsDumpVector(file, "aug_rhs_s", const_augRhs_s);
   std::fclose(file);
}

} // namespace

#if IPOPT_VERBOSITY > 0"#,
        );
    }

    if !text.contains("optivibre_pdfullspace_solveonce_call_index") {
        replace_required(
            text,
            path,
            "   IpData().TimingStats().PDSystemSolverSolveOnce().Start();\n\n   // Compute the right hand side for the augmented system formulation",
            "   IpData().TimingStats().PDSystemSolverSolveOnce().Start();\n   static int optivibre_pdfullspace_solveonce_call_index = 0;\n   const int optivibre_this_solveonce_call = optivibre_pdfullspace_solveonce_call_index++;\n\n   // Compute the right hand side for the augmented system formulation",
        );
    }

    if !text.contains("OptivibreDumpPdfullSpaceSolveOnceRhs(\n      optivibre_this_solveonce_call")
    {
        replace_required(
            text,
            path,
            "   Pd_U.AddMSinvZ(-1.0, slack_s_U, *rhs.v_U(), *augRhs_s);\n\n   // Get space into which we can put the solution of the augmented system",
            "   Pd_U.AddMSinvZ(-1.0, slack_s_U, *rhs.v_U(), *augRhs_s);\n   OptivibreDumpPdfullSpaceSolveOnceRhs(\n      optivibre_this_solveonce_call,\n      alpha,\n      beta,\n      rhs,\n      slack_x_L,\n      slack_x_U,\n      slack_s_L,\n      slack_s_U,\n      augRhs_x,\n      augRhs_s);\n\n   // Get space into which we can put the solution of the augmented system",
        );
    }

    *text != before
}

fn insert_ipopt_pdfull_space_solver_residual_stage_dump(path: &Path, text: &mut String) -> bool {
    if text.contains("resid_x_after_W") {
        return false;
    }
    let before = text.clone();

    replace_required(
        text,
        path,
        "void OptivibrePdfsDumpIteratesVector(\n   FILE*                 file,\n   const char*           prefix,\n   const IteratesVector& values\n)",
        r#"void OptivibrePdfsDumpVector(
   FILE*                   file,
   const char*             name,
   const SmartPtr<Vector>& vector
)
{
   SmartPtr<const Vector> const_vector = vector;
   OptivibrePdfsDumpVector(file, name, const_vector);
}

void OptivibrePdfsDumpIteratesVector(
   FILE*                 file,
   const char*           prefix,
   const IteratesVector& values
)"#,
    );

    replace_required(
        text,
        path,
        "   const IteratesVector&  rhs,\n   const IteratesVector&  res,\n   const IteratesVector&  resid\n)",
        "   const IteratesVector&  rhs,\n   const IteratesVector&  res,\n   const IteratesVector&  resid,\n   const SmartPtr<Vector>& resid_x_after_W,\n   const SmartPtr<Vector>& resid_x_after_Jc,\n   const SmartPtr<Vector>& resid_x_after_Jd,\n   const SmartPtr<Vector>& resid_x_after_PxL,\n   const SmartPtr<Vector>& resid_x_after_PxU,\n   const SmartPtr<Vector>& resid_x_after_AddTwoVectors,\n   const SmartPtr<Vector>& resid_s_after_PdU,\n   const SmartPtr<Vector>& resid_s_after_PdL,\n   const SmartPtr<Vector>& resid_s_after_AddTwoVectors,\n   const SmartPtr<Vector>& resid_s_after_delta\n)",
    );
    replace_required(
        text,
        path,
        "   OptivibrePdfsDumpIteratesVector(file, \"resid\", resid);\n   std::fclose(file);",
        "   OptivibrePdfsDumpIteratesVector(file, \"resid\", resid);\n   OptivibrePdfsDumpVector(file, \"resid_x_after_W\", resid_x_after_W);\n   OptivibrePdfsDumpVector(file, \"resid_x_after_Jc\", resid_x_after_Jc);\n   OptivibrePdfsDumpVector(file, \"resid_x_after_Jd\", resid_x_after_Jd);\n   OptivibrePdfsDumpVector(file, \"resid_x_after_PxL\", resid_x_after_PxL);\n   OptivibrePdfsDumpVector(file, \"resid_x_after_PxU\", resid_x_after_PxU);\n   OptivibrePdfsDumpVector(file, \"resid_x_after_AddTwoVectors\", resid_x_after_AddTwoVectors);\n   OptivibrePdfsDumpVector(file, \"resid_s_after_PdU\", resid_s_after_PdU);\n   OptivibrePdfsDumpVector(file, \"resid_s_after_PdL\", resid_s_after_PdL);\n   OptivibrePdfsDumpVector(file, \"resid_s_after_AddTwoVectors\", resid_s_after_AddTwoVectors);\n   OptivibrePdfsDumpVector(file, \"resid_s_after_delta\", resid_s_after_delta);\n   std::fclose(file);",
    );

    replace_required(
        text,
        path,
        "   static int optivibre_pdfullspace_residual_call_index = 0;\n   const int optivibre_this_residual_call = optivibre_pdfullspace_residual_call_index++;\n\n   DBG_PRINT_VECTOR(2, \"res\", res);",
        "   static int optivibre_pdfullspace_residual_call_index = 0;\n   const int optivibre_this_residual_call = optivibre_pdfullspace_residual_call_index++;\n   const bool optivibre_dump_this_residual = OptivibreShouldDumpPdfullSpaceResidual(optivibre_this_residual_call);\n   SmartPtr<Vector> optivibre_resid_x_after_W;\n   SmartPtr<Vector> optivibre_resid_x_after_Jc;\n   SmartPtr<Vector> optivibre_resid_x_after_Jd;\n   SmartPtr<Vector> optivibre_resid_x_after_PxL;\n   SmartPtr<Vector> optivibre_resid_x_after_PxU;\n   SmartPtr<Vector> optivibre_resid_x_after_AddTwoVectors;\n   SmartPtr<Vector> optivibre_resid_s_after_PdU;\n   SmartPtr<Vector> optivibre_resid_s_after_PdL;\n   SmartPtr<Vector> optivibre_resid_s_after_AddTwoVectors;\n   SmartPtr<Vector> optivibre_resid_s_after_delta;\n\n   DBG_PRINT_VECTOR(2, \"res\", res);",
    );
    replace_required(
        text,
        path,
        "   W.MultVector(1., *res.x(), 0., *resid.x_NonConst());\n",
        "   W.MultVector(1., *res.x(), 0., *resid.x_NonConst());\n   if( optivibre_dump_this_residual )\n   {\n      optivibre_resid_x_after_W = resid.x()->MakeNewCopy();\n   }\n",
    );
    replace_required(
        text,
        path,
        "   J_c.TransMultVector(1., *res.y_c(), 1., *resid.x_NonConst());\n",
        "   J_c.TransMultVector(1., *res.y_c(), 1., *resid.x_NonConst());\n   if( optivibre_dump_this_residual )\n   {\n      optivibre_resid_x_after_Jc = resid.x()->MakeNewCopy();\n   }\n",
    );
    replace_required(
        text,
        path,
        "   J_d.TransMultVector(1., *res.y_d(), 1., *resid.x_NonConst());\n",
        "   J_d.TransMultVector(1., *res.y_d(), 1., *resid.x_NonConst());\n   if( optivibre_dump_this_residual )\n   {\n      optivibre_resid_x_after_Jd = resid.x()->MakeNewCopy();\n   }\n",
    );
    replace_required(
        text,
        path,
        "   Px_L.MultVector(-1., *res.z_L(), 1., *resid.x_NonConst());\n",
        "   Px_L.MultVector(-1., *res.z_L(), 1., *resid.x_NonConst());\n   if( optivibre_dump_this_residual )\n   {\n      optivibre_resid_x_after_PxL = resid.x()->MakeNewCopy();\n   }\n",
    );
    replace_required(
        text,
        path,
        "   Px_U.MultVector(1., *res.z_U(), 1., *resid.x_NonConst());\n",
        "   Px_U.MultVector(1., *res.z_U(), 1., *resid.x_NonConst());\n   if( optivibre_dump_this_residual )\n   {\n      optivibre_resid_x_after_PxU = resid.x()->MakeNewCopy();\n   }\n",
    );
    replace_required(
        text,
        path,
        "   resid.x_NonConst()->AddTwoVectors(delta_x, *res.x(), -1., *rhs.x(), 1.);\n",
        "   resid.x_NonConst()->AddTwoVectors(delta_x, *res.x(), -1., *rhs.x(), 1.);\n   if( optivibre_dump_this_residual )\n   {\n      optivibre_resid_x_after_AddTwoVectors = resid.x()->MakeNewCopy();\n   }\n",
    );
    replace_required(
        text,
        path,
        "   Pd_U.MultVector(1., *res.v_U(), 0., *resid.s_NonConst());\n",
        "   Pd_U.MultVector(1., *res.v_U(), 0., *resid.s_NonConst());\n   if( optivibre_dump_this_residual )\n   {\n      optivibre_resid_s_after_PdU = resid.s()->MakeNewCopy();\n   }\n",
    );
    replace_required(
        text,
        path,
        "   Pd_L.MultVector(-1., *res.v_L(), 1., *resid.s_NonConst());\n",
        "   Pd_L.MultVector(-1., *res.v_L(), 1., *resid.s_NonConst());\n   if( optivibre_dump_this_residual )\n   {\n      optivibre_resid_s_after_PdL = resid.s()->MakeNewCopy();\n   }\n",
    );
    replace_required(
        text,
        path,
        "   resid.s_NonConst()->AddTwoVectors(-1., *res.y_d(), -1., *rhs.s(), 1.);\n",
        "   resid.s_NonConst()->AddTwoVectors(-1., *res.y_d(), -1., *rhs.s(), 1.);\n   if( optivibre_dump_this_residual )\n   {\n      optivibre_resid_s_after_AddTwoVectors = resid.s()->MakeNewCopy();\n   }\n",
    );
    replace_required(
        text,
        path,
        "   if( delta_s != 0. )\n   {\n      resid.s_NonConst()->Axpy(delta_s, *res.s());\n   }\n\n   // c",
        "   if( delta_s != 0. )\n   {\n      resid.s_NonConst()->Axpy(delta_s, *res.s());\n   }\n   if( optivibre_dump_this_residual )\n   {\n      optivibre_resid_s_after_delta = resid.s()->MakeNewCopy();\n   }\n\n   // c",
    );
    replace_required(
        text,
        path,
        "      rhs,\n      res,\n      resid);\n",
        "      rhs,\n      res,\n      resid,\n      optivibre_resid_x_after_W,\n      optivibre_resid_x_after_Jc,\n      optivibre_resid_x_after_Jd,\n      optivibre_resid_x_after_PxL,\n      optivibre_resid_x_after_PxU,\n      optivibre_resid_x_after_AddTwoVectors,\n      optivibre_resid_s_after_PdU,\n      optivibre_resid_s_after_PdL,\n      optivibre_resid_s_after_AddTwoVectors,\n      optivibre_resid_s_after_delta);\n",
    );

    *text != before
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
    emit_dep_metadata("SPRAL_VERSION", &spral.spral_version);
    emit_dep_metadata("METIS_VERSION", &spral.metis_version);
    emit_dep_metadata("OPENBLAS_VERSION", &spral.openblas_version);
    emit_dep_metadata("OPENBLAS_THREADING", &spral.openblas_threading);
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
