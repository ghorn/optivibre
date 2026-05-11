use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sx_codegen::{Instruction, LoweredFunction, LoweredSubfunction, Slot, ValueRef};
use sx_core::{
    BinaryOp, CCS, CallPolicyConfig, NamedMatrix, NodeView, SX, SXFunction, SXMatrix,
    lookup_function_ref,
};

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
    function: &SXFunction,
    lowered: &LoweredFunction,
    call_policy: CallPolicyConfig,
    opt_level: LlvmOptimizationLevel,
    target_triple: String,
    cpu_name: String,
    cpu_features: String,
) -> CacheKeyMetadata {
    let lowered_fingerprint = lowered_function_fingerprint(lowered);
    let symbolic_fingerprint = symbolic_function_fingerprint(function, call_policy);
    build_cache_key_metadata(
        CacheKeyFingerprintInput {
            lowered_name: lowered.name.clone(),
            lowered_fingerprint: lowered_fingerprint.clone(),
            symbolic_fingerprint: Some(symbolic_fingerprint.clone()),
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
    let mut subfunction_memo = vec![None; lowered.subfunctions.len()];
    callable_fingerprint(
        CallableFingerprintShape {
            name: Some(&lowered.name),
            inputs: &lowered.inputs,
            outputs: &lowered.outputs,
            instructions: &lowered.instructions,
            output_values: &lowered.output_values,
            call_policy: None,
        },
        &lowered.subfunctions,
        &mut subfunction_memo,
    )
}

struct CallableFingerprintShape<'a> {
    name: Option<&'a str>,
    inputs: &'a [Slot],
    outputs: &'a [Slot],
    instructions: &'a [Instruction],
    output_values: &'a [Vec<ValueRef>],
    call_policy: Option<&'a str>,
}

fn symbolic_function_fingerprint(function: &SXFunction, call_policy: CallPolicyConfig) -> String {
    let mut function_memo = HashMap::new();
    symbolic_function_fingerprint_recursive(function, call_policy, &mut function_memo)
}

fn symbolic_function_fingerprint_recursive(
    function: &SXFunction,
    call_policy: CallPolicyConfig,
    function_memo: &mut HashMap<usize, String>,
) -> String {
    if let Some(existing) = function_memo.get(&function.id()) {
        return existing.clone();
    }

    let input_bindings = function.input_bindings();
    let mut expr_memo = HashMap::new();
    let mut hasher = Sha256::new();
    write_string(&mut hasher, function.name());
    write_string(&mut hasher, &format!("{:?}", call_policy.default_policy));
    hasher.update([u8::from(call_policy.respect_function_overrides)]);
    write_string(
        &mut hasher,
        &format!("{:?}", function.call_policy_override()),
    );
    write_named_matrices(
        &mut hasher,
        function.inputs(),
        None,
        &input_bindings,
        &mut expr_memo,
        function_memo,
        call_policy,
    );
    write_named_matrices(
        &mut hasher,
        function.outputs(),
        Some(function),
        &input_bindings,
        &mut expr_memo,
        function_memo,
        call_policy,
    );
    let fingerprint = hex_encode(&hasher.finalize());
    function_memo.insert(function.id(), fingerprint.clone());
    fingerprint
}

fn write_named_matrices(
    hasher: &mut Sha256,
    matrices: &[NamedMatrix],
    owner: Option<&SXFunction>,
    input_bindings: &HashMap<SX, (usize, usize)>,
    expr_memo: &mut HashMap<SX, String>,
    function_memo: &mut HashMap<usize, String>,
    call_policy: CallPolicyConfig,
) {
    write_usize(hasher, matrices.len());
    for matrix in matrices {
        write_string(hasher, matrix.name());
        write_ccs(hasher, matrix.matrix().ccs());
        if owner.is_some() {
            write_usize(hasher, matrix.matrix().nnz());
            for &value in matrix.matrix().nonzeros() {
                let fingerprint = sx_value_fingerprint(
                    value,
                    input_bindings,
                    expr_memo,
                    function_memo,
                    call_policy,
                );
                write_string(hasher, &fingerprint);
            }
        }
    }
}

fn sx_matrix_fingerprint(
    matrix: &SXMatrix,
    input_bindings: &HashMap<SX, (usize, usize)>,
    expr_memo: &mut HashMap<SX, String>,
    function_memo: &mut HashMap<usize, String>,
    call_policy: CallPolicyConfig,
) -> String {
    let mut hasher = Sha256::new();
    write_ccs(&mut hasher, matrix.ccs());
    write_usize(&mut hasher, matrix.nnz());
    for &value in matrix.nonzeros() {
        let fingerprint =
            sx_value_fingerprint(value, input_bindings, expr_memo, function_memo, call_policy);
        write_string(&mut hasher, &fingerprint);
    }
    hex_encode(&hasher.finalize())
}

fn sx_value_fingerprint(
    value: SX,
    input_bindings: &HashMap<SX, (usize, usize)>,
    expr_memo: &mut HashMap<SX, String>,
    function_memo: &mut HashMap<usize, String>,
    call_policy: CallPolicyConfig,
) -> String {
    if let Some(existing) = expr_memo.get(&value) {
        return existing.clone();
    }

    let fingerprint = match value.inspect() {
        NodeView::Constant(value) => hash_parts(&[&[0], &value.to_le_bytes()]),
        NodeView::Symbol { name, serial } => {
            if let Some(&(slot, offset)) = input_bindings.get(&value) {
                hash_parts(&[
                    &[1],
                    &(slot as u64).to_le_bytes(),
                    &(offset as u64).to_le_bytes(),
                ])
            } else {
                hash_parts(&[&[2], name.as_bytes(), &(serial as u64).to_le_bytes()])
            }
        }
        NodeView::Unary { op, arg } => {
            let arg =
                sx_value_fingerprint(arg, input_bindings, expr_memo, function_memo, call_policy);
            hash_parts(&[&[3], format!("{op:?}").as_bytes(), arg.as_bytes()])
        }
        NodeView::Binary { op, lhs, rhs } => {
            let lhs =
                sx_value_fingerprint(lhs, input_bindings, expr_memo, function_memo, call_policy);
            let rhs =
                sx_value_fingerprint(rhs, input_bindings, expr_memo, function_memo, call_policy);
            let (lhs, rhs) =
                if matches!(op, BinaryOp::Add | BinaryOp::Mul | BinaryOp::Hypot) && rhs < lhs {
                    (rhs, lhs)
                } else {
                    (lhs, rhs)
                };
            hash_parts(&[
                &[4],
                format!("{op:?}").as_bytes(),
                lhs.as_bytes(),
                rhs.as_bytes(),
            ])
        }
        NodeView::Call {
            function_id,
            function_name,
            inputs,
            output_slot,
            output_offset,
        } => {
            let callee = lookup_function_ref(function_id)
                .expect("symbolic function fingerprint should only reference registered callees");
            let callee_fingerprint =
                symbolic_function_fingerprint_recursive(&callee, call_policy, function_memo);
            let mut parts: Vec<Vec<u8>> = vec![
                vec![5],
                function_name.into_bytes(),
                callee_fingerprint.into_bytes(),
                (output_slot as u64).to_le_bytes().to_vec(),
                (output_offset as u64).to_le_bytes().to_vec(),
            ];
            for input in &inputs {
                parts.push(
                    sx_matrix_fingerprint(
                        input,
                        input_bindings,
                        expr_memo,
                        function_memo,
                        call_policy,
                    )
                    .into_bytes(),
                );
            }
            let part_refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
            hash_parts(&part_refs)
        }
    };

    expr_memo.insert(value, fingerprint.clone());
    fingerprint
}

fn callable_fingerprint(
    shape: CallableFingerprintShape<'_>,
    subfunctions: &[LoweredSubfunction],
    subfunction_memo: &mut [Option<String>],
) -> String {
    let definitions = build_temp_definitions(shape.instructions);
    let mut temp_memo = vec![None; definitions.len()];
    let mut hasher = Sha256::new();
    if let Some(name) = shape.name {
        write_string(&mut hasher, name);
    }
    if let Some(call_policy) = shape.call_policy {
        write_string(&mut hasher, call_policy);
    }
    write_slots(&mut hasher, shape.inputs);
    write_slots(&mut hasher, shape.outputs);
    write_usize(&mut hasher, shape.output_values.len());
    for values in shape.output_values {
        write_usize(&mut hasher, values.len());
        for &value in values {
            let fingerprint = value_ref_fingerprint(
                value,
                &definitions,
                &mut temp_memo,
                subfunctions,
                subfunction_memo,
            );
            write_string(&mut hasher, &fingerprint);
        }
    }
    hex_encode(&hasher.finalize())
}

#[derive(Clone, Copy)]
struct TempDefinition<'a> {
    instruction: &'a Instruction,
    output_index: usize,
}

fn build_temp_definitions(instructions: &[Instruction]) -> Vec<Option<TempDefinition<'_>>> {
    let max_temp = instructions
        .iter()
        .flat_map(Instruction::output_temps)
        .copied()
        .max()
        .map_or(0, |temp| temp + 1);
    let mut definitions = vec![None; max_temp];
    for instruction in instructions {
        for (output_index, &temp) in instruction.output_temps().iter().enumerate() {
            definitions[temp] = Some(TempDefinition {
                instruction,
                output_index,
            });
        }
    }
    definitions
}

fn value_ref_fingerprint(
    value: ValueRef,
    definitions: &[Option<TempDefinition<'_>>],
    temp_memo: &mut [Option<String>],
    subfunctions: &[LoweredSubfunction],
    subfunction_memo: &mut [Option<String>],
) -> String {
    match value {
        ValueRef::Input { slot, offset } => hash_parts(&[
            &[0],
            &(slot as u64).to_le_bytes(),
            &(offset as u64).to_le_bytes(),
        ]),
        ValueRef::Const(value) => hash_parts(&[&[1], &value.to_le_bytes()]),
        ValueRef::Temp(temp) => {
            if let Some(existing) = &temp_memo[temp] {
                return existing.clone();
            }
            let definition =
                definitions[temp].expect("temporary should have a defining instruction");
            let fingerprint = match definition.instruction {
                Instruction::Unary { op, input, .. } => {
                    let input = value_ref_fingerprint(
                        *input,
                        definitions,
                        temp_memo,
                        subfunctions,
                        subfunction_memo,
                    );
                    hash_parts(&[&[2], format!("{op:?}").as_bytes(), input.as_bytes()])
                }
                Instruction::Binary { op, lhs, rhs, .. } => {
                    let lhs = value_ref_fingerprint(
                        *lhs,
                        definitions,
                        temp_memo,
                        subfunctions,
                        subfunction_memo,
                    );
                    let rhs = value_ref_fingerprint(
                        *rhs,
                        definitions,
                        temp_memo,
                        subfunctions,
                        subfunction_memo,
                    );
                    let (lhs, rhs) =
                        if matches!(op, BinaryOp::Add | BinaryOp::Mul | BinaryOp::Hypot)
                            && rhs < lhs
                        {
                            (rhs, lhs)
                        } else {
                            (lhs, rhs)
                        };
                    hash_parts(&[
                        &[3],
                        format!("{op:?}").as_bytes(),
                        lhs.as_bytes(),
                        rhs.as_bytes(),
                    ])
                }
                Instruction::Call { callee, inputs, .. } => {
                    let callee = subfunction_fingerprint(*callee, subfunctions, subfunction_memo);
                    let output_index = (definition.output_index as u64).to_le_bytes();
                    let mut parts: Vec<Vec<u8>> =
                        vec![vec![4], output_index.to_vec(), callee.into_bytes()];
                    for input in inputs {
                        parts.push(
                            value_ref_fingerprint(
                                *input,
                                definitions,
                                temp_memo,
                                subfunctions,
                                subfunction_memo,
                            )
                            .into_bytes(),
                        );
                    }
                    let part_refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
                    hash_parts(&part_refs)
                }
            };
            temp_memo[temp] = Some(fingerprint.clone());
            fingerprint
        }
    }
}

fn subfunction_fingerprint(
    subfunction_index: usize,
    subfunctions: &[LoweredSubfunction],
    subfunction_memo: &mut [Option<String>],
) -> String {
    if let Some(existing) = &subfunction_memo[subfunction_index] {
        return existing.clone();
    }
    let subfunction = &subfunctions[subfunction_index];
    let fingerprint = callable_fingerprint(
        CallableFingerprintShape {
            name: None,
            inputs: &subfunction.inputs,
            outputs: &subfunction.outputs,
            instructions: &subfunction.instructions,
            output_values: &subfunction.output_values,
            call_policy: Some(&format!("{:?}", subfunction.call_policy)),
        },
        subfunctions,
        subfunction_memo,
    );
    subfunction_memo[subfunction_index] = Some(fingerprint.clone());
    fingerprint
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
    fn function_cache_manifest_records_symbolic_fingerprint() {
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
            assert!(manifest.symbolic_fingerprint.is_some());
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
