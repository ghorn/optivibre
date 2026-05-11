use std::env;
use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sx_codegen::{Instruction, LoweredFunction, Slot, ValueRef};
use sx_core::{BinaryOp, CCS, CallPolicy, CallPolicyConfig, SXFunction, UnaryOp};

use crate::LlvmOptimizationLevel;

const OPTIVIBRE_JIT_CACHE_ENV: &str = "OPTIVIBRE_JIT_CACHE_DIR";
const OPTIVIBRE_JIT_CACHE_SCHEMA_VERSION: u32 = 2;
const OPTIVIBRE_JIT_CACHE_CODEGEN_FORMAT_VERSION: u32 = 1;
const OPTIVIBRE_JIT_CACHE_COMPILE_MODE: &str = "jit";
const OPTIVIBRE_JIT_CACHE_VERSION_DIR: &str = "v2";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JitCacheReport {
    pub hit: bool,
    pub check_time: Duration,
    pub read_time: Duration,
    pub write_time: Duration,
    pub load_time: Duration,
    pub materialize_time: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CachedObject {
    pub object_bytes: Vec<u8>,
    pub entry_dir: PathBuf,
    pub check_time: Duration,
    pub read_time: Duration,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CacheKeyMetadata {
    lowered_name: String,
    lowered_fingerprint: String,
    symbolic_fingerprint: Option<String>,
    entry_key_kind: String,
    entry_hash: String,
    target_triple: String,
    cpu_name: String,
    cpu_features: String,
    opt_level: String,
    target_component: String,
    cpu_component: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CacheKeyFingerprintInput {
    lowered_name: String,
    lowered_fingerprint: String,
    symbolic_fingerprint: Option<String>,
    entry_key_kind: &'static str,
    entry_key_fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct JitCacheManifest {
    schema_version: u32,
    lowered_name: String,
    lowered_fingerprint: String,
    symbolic_fingerprint: Option<String>,
    llvm_ir_fingerprint: Option<String>,
    entry_key_kind: String,
    target_triple: String,
    cpu_name: String,
    cpu_features: String,
    opt_level: String,
    compile_mode: String,
    codegen_format_version: u32,
    crate_version: String,
    object_size_bytes: usize,
}

pub(crate) fn cache_key_metadata(
    lowered: &LoweredFunction,
    opt_level: LlvmOptimizationLevel,
    target_triple: String,
    cpu_name: String,
    cpu_features: String,
) -> CacheKeyMetadata {
    let lowered_fingerprint = lowered_function_fingerprint(lowered);
    build_cache_key_metadata(
        CacheKeyFingerprintInput {
            lowered_name: lowered.name.clone(),
            lowered_fingerprint: lowered_fingerprint.clone(),
            symbolic_fingerprint: None,
            entry_key_kind: "lowered_fingerprint",
            entry_key_fingerprint: lowered_fingerprint,
        },
        opt_level,
        target_triple,
        cpu_name,
        cpu_features,
    )
}

pub(crate) fn cache_key_metadata_for_function(
    _function: &SXFunction,
    lowered: &LoweredFunction,
    _call_policy: CallPolicyConfig,
    opt_level: LlvmOptimizationLevel,
    target_triple: String,
    cpu_name: String,
    cpu_features: String,
) -> CacheKeyMetadata {
    let lowered_fingerprint = lowered_function_fingerprint(lowered);
    build_cache_key_metadata(
        CacheKeyFingerprintInput {
            lowered_name: lowered.name.clone(),
            lowered_fingerprint: lowered_fingerprint.clone(),
            symbolic_fingerprint: None,
            entry_key_kind: "lowered_fingerprint",
            entry_key_fingerprint: lowered_fingerprint,
        },
        opt_level,
        target_triple,
        cpu_name,
        cpu_features,
    )
}

fn build_cache_key_metadata(
    fingerprint: CacheKeyFingerprintInput,
    opt_level: LlvmOptimizationLevel,
    target_triple: String,
    cpu_name: String,
    cpu_features: String,
) -> CacheKeyMetadata {
    let opt_level = opt_level.label().to_string();
    let entry_hash = hash_parts(&[
        OPTIVIBRE_JIT_CACHE_COMPILE_MODE.as_bytes(),
        target_triple.as_bytes(),
        cpu_name.as_bytes(),
        cpu_features.as_bytes(),
        opt_level.as_bytes(),
        fingerprint.entry_key_fingerprint.as_bytes(),
        &OPTIVIBRE_JIT_CACHE_SCHEMA_VERSION.to_le_bytes(),
        &OPTIVIBRE_JIT_CACHE_CODEGEN_FORMAT_VERSION.to_le_bytes(),
    ]);
    let cpu_component = hash_parts(&[cpu_name.as_bytes(), cpu_features.as_bytes()]);
    CacheKeyMetadata {
        lowered_name: fingerprint.lowered_name,
        lowered_fingerprint: fingerprint.lowered_fingerprint,
        symbolic_fingerprint: fingerprint.symbolic_fingerprint,
        entry_key_kind: fingerprint.entry_key_kind.to_string(),
        entry_hash,
        target_component: sanitize_path_component(&target_triple),
        cpu_component,
        target_triple,
        cpu_name,
        cpu_features,
        opt_level,
    }
}

impl CacheKeyMetadata {
    fn version_root(&self) -> anyhow::Result<PathBuf> {
        Ok(optivibre_jit_cache_base_dir()?.join(OPTIVIBRE_JIT_CACHE_VERSION_DIR))
    }

    fn entry_dir(&self) -> anyhow::Result<PathBuf> {
        let prefix = &self.entry_hash[..2];
        Ok(self
            .version_root()?
            .join(&self.target_component)
            .join(&self.cpu_component)
            .join(&self.opt_level)
            .join(prefix)
            .join(&self.entry_hash))
    }

    fn manifest(
        &self,
        object_size_bytes: usize,
        llvm_ir_fingerprint: Option<String>,
    ) -> JitCacheManifest {
        JitCacheManifest {
            schema_version: OPTIVIBRE_JIT_CACHE_SCHEMA_VERSION,
            lowered_name: self.lowered_name.clone(),
            lowered_fingerprint: self.lowered_fingerprint.clone(),
            symbolic_fingerprint: self.symbolic_fingerprint.clone(),
            llvm_ir_fingerprint,
            entry_key_kind: self.entry_key_kind.clone(),
            target_triple: self.target_triple.clone(),
            cpu_name: self.cpu_name.clone(),
            cpu_features: self.cpu_features.clone(),
            opt_level: self.opt_level.clone(),
            compile_mode: OPTIVIBRE_JIT_CACHE_COMPILE_MODE.to_string(),
            codegen_format_version: OPTIVIBRE_JIT_CACHE_CODEGEN_FORMAT_VERSION,
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            object_size_bytes,
        }
    }
}

pub(crate) fn try_load_cached_object(metadata: &CacheKeyMetadata) -> Option<CachedObject> {
    let check_started = Instant::now();
    let entry_dir = metadata.entry_dir().ok()?;
    let manifest_path = entry_dir.join("manifest.json");
    let object_path = entry_dir.join("object.o");
    let manifest_text = fs::read_to_string(&manifest_path).ok()?;
    let manifest = serde_json::from_str::<JitCacheManifest>(&manifest_text).ok()?;
    if !manifest_matches(&manifest, metadata) {
        return None;
    }
    let check_time = check_started.elapsed();
    let read_started = Instant::now();
    let object_bytes = fs::read(&object_path).ok()?;
    let read_time = read_started.elapsed();
    if object_bytes.is_empty() || object_bytes.len() != manifest.object_size_bytes {
        return None;
    }
    Some(CachedObject {
        object_bytes,
        entry_dir,
        check_time,
        read_time,
    })
}

pub(crate) fn write_cached_object(
    metadata: &CacheKeyMetadata,
    object_bytes: &[u8],
    llvm_ir_fingerprint: Option<&str>,
) -> anyhow::Result<()> {
    let entry_dir = metadata.entry_dir()?;
    if entry_dir.exists() {
        return Ok(());
    }
    let Some(parent) = entry_dir.parent() else {
        anyhow::bail!("cache entry dir has no parent");
    };
    fs::create_dir_all(parent)?;
    let temp_dir = create_temp_entry_dir(parent, &metadata.entry_hash)?;
    let write_result = (|| -> anyhow::Result<()> {
        let object_path = temp_dir.join("object.o");
        write_synced_file(&object_path, object_bytes)?;
        let manifest = metadata.manifest(
            object_bytes.len(),
            llvm_ir_fingerprint.map(ToOwned::to_owned),
        );
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
        write_synced_file(&temp_dir.join("manifest.json"), &manifest_bytes)?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_dir_all(&temp_dir);
        return Err(error);
    }

    match fs::rename(&temp_dir, &entry_dir) {
        Ok(()) => Ok(()),
        Err(_) if entry_dir.exists() => {
            let _ = fs::remove_dir_all(&temp_dir);
            Ok(())
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&temp_dir);
            Err(error.into())
        }
    }
}

fn create_temp_entry_dir(parent: &Path, entry_hash: &str) -> anyhow::Result<PathBuf> {
    for attempt in 0..32u32 {
        let temp_dir = parent.join(format!(
            ".tmp-{}-{}-{}-{}",
            entry_hash,
            process::id(),
            temp_suffix(),
            attempt
        ));
        match fs::create_dir(&temp_dir) {
            Ok(()) => return Ok(temp_dir),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    anyhow::bail!("failed to create unique cache temp dir after repeated collisions");
}

pub(crate) fn remove_cached_entry(entry_dir: &Path) {
    let _ = fs::remove_dir_all(entry_dir);
}

pub fn clear_optivibre_jit_cache() -> anyhow::Result<()> {
    let cache_root = optivibre_jit_cache_base_dir()?;
    if cache_root.exists() {
        let mut last_error = None;
        for _ in 0..8 {
            match fs::remove_dir_all(&cache_root) {
                Ok(()) => return Ok(()),
                Err(error)
                    if matches!(
                        error.kind(),
                        std::io::ErrorKind::DirectoryNotEmpty
                            | std::io::ErrorKind::PermissionDenied
                    ) =>
                {
                    last_error = Some(error);
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(error) => return Err(error.into()),
            }
        }
        if cache_root.exists() {
            return Err(last_error
                .unwrap_or_else(|| std::io::Error::other("failed to clear JIT cache"))
                .into());
        }
    }
    Ok(())
}

fn manifest_matches(manifest: &JitCacheManifest, metadata: &CacheKeyMetadata) -> bool {
    let fingerprint_matches = match metadata.entry_key_kind.as_str() {
        "lowered_fingerprint" => manifest.lowered_fingerprint == metadata.lowered_fingerprint,
        "symbolic_fingerprint" => manifest.symbolic_fingerprint == metadata.symbolic_fingerprint,
        _ => false,
    };
    manifest.schema_version == OPTIVIBRE_JIT_CACHE_SCHEMA_VERSION
        && manifest.lowered_name == metadata.lowered_name
        && manifest.entry_key_kind == metadata.entry_key_kind
        && fingerprint_matches
        && manifest.target_triple == metadata.target_triple
        && manifest.cpu_name == metadata.cpu_name
        && manifest.cpu_features == metadata.cpu_features
        && manifest.opt_level == metadata.opt_level
        && manifest.compile_mode == OPTIVIBRE_JIT_CACHE_COMPILE_MODE
        && manifest.codegen_format_version == OPTIVIBRE_JIT_CACHE_CODEGEN_FORMAT_VERSION
        && manifest.crate_version == env!("CARGO_PKG_VERSION")
}

fn write_synced_file(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn temp_suffix() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos())
}

fn sanitize_path_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

fn hash_parts(parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        let len = u64::try_from(part.len()).unwrap_or(u64::MAX);
        hasher.update(len.to_le_bytes());
        hasher.update(part);
    }
    hex_encode(&hasher.finalize())
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn lowered_function_fingerprint(lowered: &LoweredFunction) -> String {
    let mut hasher = Sha256::new();
    write_string(&mut hasher, &lowered.name);
    write_callable_program(
        &mut hasher,
        &lowered.inputs,
        &lowered.outputs,
        &lowered.instructions,
        &lowered.output_values,
    );
    write_usize(&mut hasher, lowered.subfunctions.len());
    for subfunction in &lowered.subfunctions {
        write_call_policy(&mut hasher, subfunction.call_policy);
        write_callable_program(
            &mut hasher,
            &subfunction.inputs,
            &subfunction.outputs,
            &subfunction.instructions,
            &subfunction.output_values,
        );
    }
    hex_encode(&hasher.finalize())
}

fn write_callable_program(
    hasher: &mut Sha256,
    inputs: &[Slot],
    outputs: &[Slot],
    instructions: &[Instruction],
    output_values: &[Vec<ValueRef>],
) {
    write_slots(hasher, inputs);
    write_slots(hasher, outputs);
    write_usize(hasher, instructions.len());
    for instruction in instructions {
        write_instruction(hasher, instruction);
    }
    write_usize(hasher, output_values.len());
    for values in output_values {
        write_usize(hasher, values.len());
        for &value in values {
            write_value_ref(hasher, value);
        }
    }
}

fn write_instruction(hasher: &mut Sha256, instruction: &Instruction) {
    match instruction {
        Instruction::Unary { temp, op, input } => {
            hasher.update([0]);
            write_usize(hasher, *temp);
            write_unary_op(hasher, *op);
            write_value_ref(hasher, *input);
        }
        Instruction::Binary { temp, op, lhs, rhs } => {
            hasher.update([1]);
            write_usize(hasher, *temp);
            write_binary_op(hasher, *op);
            write_value_ref(hasher, *lhs);
            write_value_ref(hasher, *rhs);
        }
        Instruction::Call {
            temps,
            callee,
            inputs,
        } => {
            hasher.update([2]);
            write_usize_slice(hasher, temps);
            write_usize(hasher, *callee);
            write_usize(hasher, inputs.len());
            for &input in inputs {
                write_value_ref(hasher, input);
            }
        }
    }
}

fn write_value_ref(hasher: &mut Sha256, value: ValueRef) {
    match value {
        ValueRef::Input { slot, offset } => {
            hasher.update([0]);
            write_usize(hasher, slot);
            write_usize(hasher, offset);
        }
        ValueRef::Temp(temp) => {
            hasher.update([1]);
            write_usize(hasher, temp);
        }
        ValueRef::Const(value) => {
            hasher.update([2]);
            hasher.update(value.to_le_bytes());
        }
    }
}

fn write_call_policy(hasher: &mut Sha256, policy: CallPolicy) {
    hasher.update([match policy {
        CallPolicy::InlineAtCall => 0,
        CallPolicy::InlineAtLowering => 1,
        CallPolicy::InlineInLLVM => 2,
        CallPolicy::NoInlineLLVM => 3,
    }]);
}

fn write_unary_op(hasher: &mut Sha256, op: UnaryOp) {
    hasher.update([match op {
        UnaryOp::Abs => 0,
        UnaryOp::Sign => 1,
        UnaryOp::Floor => 2,
        UnaryOp::Ceil => 3,
        UnaryOp::Round => 4,
        UnaryOp::Trunc => 5,
        UnaryOp::Sqrt => 6,
        UnaryOp::Exp => 7,
        UnaryOp::Log => 8,
        UnaryOp::Sin => 9,
        UnaryOp::Cos => 10,
        UnaryOp::Tan => 11,
        UnaryOp::Asin => 12,
        UnaryOp::Acos => 13,
        UnaryOp::Atan => 14,
        UnaryOp::Sinh => 15,
        UnaryOp::Cosh => 16,
        UnaryOp::Tanh => 17,
        UnaryOp::Asinh => 18,
        UnaryOp::Acosh => 19,
        UnaryOp::Atanh => 20,
    }]);
}

fn write_binary_op(hasher: &mut Sha256, op: BinaryOp) {
    hasher.update([match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Pow => 4,
        BinaryOp::Atan2 => 5,
        BinaryOp::Hypot => 6,
        BinaryOp::Mod => 7,
        BinaryOp::Copysign => 8,
    }]);
}

fn write_slots(hasher: &mut Sha256, slots: &[Slot]) {
    write_usize(hasher, slots.len());
    for slot in slots {
        write_string(hasher, &slot.name);
        write_ccs(hasher, &slot.ccs);
    }
}

fn write_ccs(hasher: &mut Sha256, ccs: &CCS) {
    write_usize(hasher, ccs.nrow());
    write_usize(hasher, ccs.ncol());
    write_usize_slice(hasher, ccs.col_ptrs());
    write_usize_slice(hasher, ccs.row_indices());
}

fn write_string(hasher: &mut Sha256, value: &str) {
    write_usize(hasher, value.len());
    hasher.update(value.as_bytes());
}

fn write_usize_slice(hasher: &mut Sha256, values: &[usize]) {
    write_usize(hasher, values.len());
    for value in values {
        write_usize(hasher, *value);
    }
}

fn write_usize(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn default_cache_base_dir() -> anyhow::Result<PathBuf> {
    let home = env::var_os("HOME").ok_or_else(|| anyhow::anyhow!("HOME is not set"))?;
    Ok(PathBuf::from(home)
        .join(".cache")
        .join("optivibre")
        .join("llvm_jit"))
}

pub(crate) fn optivibre_jit_cache_base_dir() -> anyhow::Result<PathBuf> {
    if let Some(override_dir) = env::var_os(OPTIVIBRE_JIT_CACHE_ENV) {
        Ok(PathBuf::from(override_dir))
    } else {
        default_cache_base_dir()
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::sync::{Mutex, OnceLock};

    use sx_codegen::lower_function;
    use sx_core::{NamedMatrix, SXFunction, SXMatrix};
    use tempfile::TempDir;

    use super::{
        JitCacheManifest, OPTIVIBRE_JIT_CACHE_ENV, clear_optivibre_jit_cache,
        optivibre_jit_cache_base_dir, try_load_cached_object, write_cached_object,
    };
    use crate::{
        CompiledJitFunction, FunctionCompileOptions, JitOptimizationLevel, host_cache_metadata,
        native_cache_target_info, optimized_jit_ir_fingerprint,
    };

    fn cache_env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn with_temp_cache_root<T>(callback: impl FnOnce(&TempDir) -> T) -> T {
        let _guard = cache_env_lock();
        let temp_dir = TempDir::new().expect("temp dir");
        unsafe { env::set_var(OPTIVIBRE_JIT_CACHE_ENV, temp_dir.path()) };
        let result = callback(&temp_dir);
        unsafe { env::remove_var(OPTIVIBRE_JIT_CACHE_ENV) };
        result
    }

    fn demo_lowered(name: &str, shift: f64) -> sx_codegen::LoweredFunction {
        let function = demo_function(name, shift);
        lower_function(&function).expect("lowered")
    }

    fn demo_function(name: &str, shift: f64) -> SXFunction {
        let x = SXMatrix::sym_dense("x", 2, 1).expect("x");
        let value = SXMatrix::scalar(x.nz(0) * x.nz(1) + shift);
        SXFunction::new(
            name,
            vec![NamedMatrix::new("x", x).expect("input")],
            vec![NamedMatrix::new("value", value).expect("output")],
        )
        .expect("function")
    }

    #[test]
    fn cache_miss_writes_manifest_and_object_entry() {
        with_temp_cache_root(|_| {
            let lowered = demo_lowered("cache_write_demo", 1.0);
            let metadata =
                host_cache_metadata(&lowered, JitOptimizationLevel::O0).expect("cache metadata");
            let bytes = crate::emit_object_bytes_lowered(
                &lowered,
                JitOptimizationLevel::O0,
                &crate::LlvmTarget::Native,
            )
            .expect("object bytes");
            write_cached_object(&metadata, &bytes, None).expect("write cache");
            let entry = metadata.entry_dir().expect("entry dir");
            assert!(entry.join("manifest.json").exists());
            assert!(entry.join("object.o").exists());
            let cached = try_load_cached_object(&metadata).expect("cache hit");
            assert_eq!(cached.object_bytes, bytes);
        });
    }

    #[test]
    fn repeated_compile_of_identical_lowered_function_hits_cache() {
        with_temp_cache_root(|_| {
            let lowered = demo_lowered("cache_hit_demo", 2.0);
            let first = CompiledJitFunction::compile_lowered(&lowered, JitOptimizationLevel::O0)
                .expect("first compile");
            assert!(!first.compile_report().cache.hit);
            let second = CompiledJitFunction::compile_lowered(&lowered, JitOptimizationLevel::O0)
                .expect("second compile");
            assert!(second.compile_report().cache.hit);
        });
    }

    #[test]
    fn fingerprint_changes_on_materially_different_lowered_functions() {
        let lhs = demo_lowered("fingerprint_a", 1.0);
        let rhs = demo_lowered("fingerprint_b", 2.0);
        let lhs_meta = host_cache_metadata(&lhs, JitOptimizationLevel::O0).expect("lhs metadata");
        let rhs_meta = host_cache_metadata(&rhs, JitOptimizationLevel::O0).expect("rhs metadata");
        assert_ne!(lhs_meta.lowered_fingerprint, rhs_meta.lowered_fingerprint);
        assert_ne!(lhs_meta.entry_hash, rhs_meta.entry_hash);
    }

    #[test]
    fn llvm_ir_fingerprint_changes_on_materially_different_lowered_functions() {
        let lhs = demo_lowered("ir_fingerprint_a", 1.0);
        let rhs = demo_lowered("ir_fingerprint_b", 2.0);
        let lhs_fp = optimized_jit_ir_fingerprint(
            &lhs,
            JitOptimizationLevel::O0,
            &crate::LlvmTarget::Native,
        )
        .expect("lhs ir fingerprint");
        let rhs_fp = optimized_jit_ir_fingerprint(
            &rhs,
            JitOptimizationLevel::O0,
            &crate::LlvmTarget::Native,
        )
        .expect("rhs ir fingerprint");
        assert_ne!(lhs_fp, rhs_fp);
    }

    #[test]
    fn manifest_records_ir_fingerprint_and_key_kind() {
        with_temp_cache_root(|_| {
            let lowered = demo_lowered("cache_manifest_demo", 7.0);
            let metadata =
                host_cache_metadata(&lowered, JitOptimizationLevel::O0).expect("metadata");
            let bytes = crate::emit_object_bytes_lowered(
                &lowered,
                JitOptimizationLevel::O0,
                &crate::LlvmTarget::Native,
            )
            .expect("object bytes");
            let ir_fingerprint = optimized_jit_ir_fingerprint(
                &lowered,
                JitOptimizationLevel::O0,
                &crate::LlvmTarget::Native,
            )
            .expect("ir fingerprint");
            write_cached_object(&metadata, &bytes, Some(&ir_fingerprint)).expect("write cache");
            let manifest_path = metadata
                .entry_dir()
                .expect("entry dir")
                .join("manifest.json");
            let manifest = serde_json::from_str::<JitCacheManifest>(
                &fs::read_to_string(manifest_path).expect("manifest text"),
            )
            .expect("manifest json");
            assert_eq!(manifest.entry_key_kind, "lowered_fingerprint");
            assert_eq!(
                manifest.llvm_ir_fingerprint.as_deref(),
                Some(ir_fingerprint.as_str())
            );
            assert!(manifest.symbolic_fingerprint.is_none());
        });
    }

    #[test]
    fn function_cache_manifest_uses_lowered_fingerprint_without_symbolic_walk() {
        with_temp_cache_root(|_| {
            let function = demo_function("cache_symbolic_demo", 8.0);
            let options = FunctionCompileOptions::from(JitOptimizationLevel::O0);
            let _compiled = CompiledJitFunction::compile_function_with_options(&function, options)
                .expect("compile function");
            let lowered = lower_function(&function).expect("lowered");
            let (target_triple, cpu_name, cpu_features) =
                native_cache_target_info(&crate::LlvmTarget::Native).expect("target info");
            let metadata = super::cache_key_metadata_for_function(
                &function,
                &lowered,
                options.call_policy,
                options.opt_level,
                target_triple,
                cpu_name,
                cpu_features,
            );
            let manifest_path = metadata
                .entry_dir()
                .expect("entry dir")
                .join("manifest.json");
            let manifest = serde_json::from_str::<JitCacheManifest>(
                &fs::read_to_string(manifest_path).expect("manifest text"),
            )
            .expect("manifest json");
            assert_eq!(manifest.entry_key_kind, "lowered_fingerprint");
            assert!(manifest.symbolic_fingerprint.is_none());
        });
    }

    #[test]
    fn target_or_options_mismatch_invalidates_entry() {
        with_temp_cache_root(|_| {
            let lowered = demo_lowered("cache_mismatch_demo", 3.0);
            let metadata_o0 =
                host_cache_metadata(&lowered, JitOptimizationLevel::O0).expect("o0 metadata");
            let bytes = crate::emit_object_bytes_lowered(
                &lowered,
                JitOptimizationLevel::O0,
                &crate::LlvmTarget::Native,
            )
            .expect("object bytes");
            write_cached_object(&metadata_o0, &bytes, None).expect("write cache");
            let metadata_o2 =
                host_cache_metadata(&lowered, JitOptimizationLevel::O2).expect("o2 metadata");
            assert!(try_load_cached_object(&metadata_o2).is_none());
        });
    }

    #[test]
    fn corrupt_manifest_or_object_is_treated_as_miss() {
        with_temp_cache_root(|_| {
            let lowered = demo_lowered("cache_corrupt_demo", 4.0);
            let metadata =
                host_cache_metadata(&lowered, JitOptimizationLevel::O0).expect("metadata");
            let bytes = crate::emit_object_bytes_lowered(
                &lowered,
                JitOptimizationLevel::O0,
                &crate::LlvmTarget::Native,
            )
            .expect("object bytes");
            write_cached_object(&metadata, &bytes, None).expect("write cache");
            let entry = metadata.entry_dir().expect("entry dir");
            fs::write(entry.join("manifest.json"), b"not json").expect("corrupt manifest");
            assert!(try_load_cached_object(&metadata).is_none());

            write_cached_object(&metadata, &bytes, None).expect("rewrite cache");
            fs::write(entry.join("object.o"), b"bad").expect("corrupt object");
            assert!(try_load_cached_object(&metadata).is_none());
        });
    }

    #[test]
    fn concurrent_populate_of_same_key_is_safe() {
        with_temp_cache_root(|_| {
            let lowered = demo_lowered("cache_race_demo", 5.0);
            let metadata =
                host_cache_metadata(&lowered, JitOptimizationLevel::O0).expect("metadata");
            let bytes = crate::emit_object_bytes_lowered(
                &lowered,
                JitOptimizationLevel::O0,
                &crate::LlvmTarget::Native,
            )
            .expect("object bytes");
            std::thread::scope(|scope| {
                for _ in 0..4 {
                    let metadata = metadata.clone();
                    let bytes = bytes.clone();
                    scope.spawn(move || {
                        write_cached_object(&metadata, &bytes, None).expect("cache write");
                    });
                }
            });
            let cached = try_load_cached_object(&metadata).expect("cache hit");
            assert_eq!(cached.object_bytes, bytes);
        });
    }

    #[test]
    fn clear_cache_removes_entire_cache_root() {
        with_temp_cache_root(|_| {
            let lowered = demo_lowered("cache_clear_demo", 6.0);
            let metadata =
                host_cache_metadata(&lowered, JitOptimizationLevel::O0).expect("metadata");
            let bytes = crate::emit_object_bytes_lowered(
                &lowered,
                JitOptimizationLevel::O0,
                &crate::LlvmTarget::Native,
            )
            .expect("object bytes");
            write_cached_object(&metadata, &bytes, None).expect("write cache");
            let cache_root = optivibre_jit_cache_base_dir().expect("cache root");
            assert!(cache_root.exists());
            clear_optivibre_jit_cache().expect("clear cache");
            assert!(!cache_root.exists());
        });
    }
}
