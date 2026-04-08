use std::ffi::{CStr, CString};
use std::fs;
use std::mem;
use std::path::Path;
use std::ptr;
use std::slice;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use llvm_sys::LLVMRealPredicate::{LLVMRealOGT, LLVMRealOLT};
use llvm_sys::analysis::{LLVMVerifierFailureAction, LLVMVerifyModule};
use llvm_sys::core::{
    LLVMAddAttributeAtIndex, LLVMAddFunction, LLVMAppendBasicBlockInContext, LLVMBuildAlloca,
    LLVMBuildCall2, LLVMBuildFAdd, LLVMBuildFCmp, LLVMBuildFDiv, LLVMBuildFMul, LLVMBuildFSub,
    LLVMBuildInBoundsGEP2, LLVMBuildLoad2, LLVMBuildRetVoid, LLVMBuildSelect, LLVMBuildStore,
    LLVMConstInt, LLVMConstReal, LLVMContextCreate, LLVMContextDispose,
    LLVMCreateBuilderInContext, LLVMCreateEnumAttribute, LLVMDisposeBuilder,
    LLVMDisposeMemoryBuffer, LLVMDisposeMessage, LLVMDoubleTypeInContext, LLVMFunctionType,
    LLVMGetBufferSize, LLVMGetBufferStart, LLVMGetEnumAttributeKindForName, LLVMGetNamedFunction,
    LLVMGetParam, LLVMGlobalGetValueType, LLVMInt64TypeInContext,
    LLVMModuleCreateWithNameInContext, LLVMPointerType, LLVMPositionBuilderAtEnd, LLVMSetTarget,
    LLVMSetLinkage, LLVMVoidTypeInContext,
};
use llvm_sys::error::{LLVMDisposeErrorMessage, LLVMErrorRef, LLVMGetErrorMessage};
use llvm_sys::orc2::LLVMOrcExecutorAddress;
use llvm_sys::orc2::lljit::{
    LLVMOrcCreateLLJIT, LLVMOrcDisposeLLJIT, LLVMOrcLLJITAddObjectFile,
    LLVMOrcLLJITGetGlobalPrefix, LLVMOrcLLJITGetMainJITDylib, LLVMOrcLLJITLookup,
};
use llvm_sys::orc2::{
    LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess, LLVMOrcJITDylibAddGenerator,
};
use llvm_sys::prelude::{
    LLVMBuilderRef, LLVMContextRef, LLVMMemoryBufferRef, LLVMModuleRef, LLVMTypeRef, LLVMValueRef,
};
use llvm_sys::{LLVMAttributeFunctionIndex, LLVMLinkage};
use llvm_sys::target::{
    LLVM_InitializeNativeAsmPrinter, LLVM_InitializeNativeTarget, LLVMCreateTargetData,
    LLVMDisposeTargetData, LLVMSetModuleDataLayout,
};
use llvm_sys::target_machine::{
    LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetMachineOptions,
    LLVMCreateTargetMachineWithOptions, LLVMDisposeTargetMachine, LLVMDisposeTargetMachineOptions,
    LLVMGetDefaultTargetTriple, LLVMGetHostCPUFeatures, LLVMGetHostCPUName,
    LLVMGetTargetFromTriple, LLVMTargetMachineEmitToMemoryBuffer, LLVMTargetMachineOptionsSetCPU,
    LLVMTargetMachineOptionsSetCodeGenOptLevel, LLVMTargetMachineOptionsSetCodeModel,
    LLVMTargetMachineOptionsSetFeatures, LLVMTargetMachineRef,
};
use llvm_sys::transforms::pass_builder::{
    LLVMCreatePassBuilderOptions, LLVMDisposePassBuilderOptions, LLVMRunPasses,
};
use sx_codegen::{
    Instruction, LoweredFunction, LoweredSubfunction, ValueRef, format_rust_source,
    lower_function_with_policies, sanitize_ident, to_pascal_case,
};
use sx_core::{
    BinaryOp, CallPolicy, CallPolicyConfig, CompileStats, CompileWarning, SXFunction, UnaryOp,
};

type RawKernelFn = unsafe extern "C" fn(*const *const f64, *const *mut f64);

fn kernel_symbol_name(name: &str) -> String {
    format!("__sx_codegen_llvm_{}", sanitize_ident(name))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlvmOptimizationLevel {
    O0,
    O2,
    O3,
    Os,
}

impl LlvmOptimizationLevel {
    pub const fn label(self) -> &'static str {
        match self {
            Self::O0 => "O0",
            Self::O2 => "O2",
            Self::O3 => "O3",
            Self::Os => "Os",
        }
    }

    fn codegen_level(self) -> LLVMCodeGenOptLevel {
        match self {
            Self::O0 => LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
            Self::O2 | Self::Os => LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
            Self::O3 => LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
        }
    }

    fn pass_pipeline(self) -> &'static str {
        match self {
            Self::O0 => "default<O0>",
            Self::O2 => "default<O2>",
            Self::O3 => "default<O3>",
            Self::Os => "default<Os>",
        }
    }

    pub fn from_cargo_opt_level(opt_level: &str) -> Option<Self> {
        match opt_level {
            "0" => Some(Self::O0),
            "2" => Some(Self::O2),
            "3" => Some(Self::O3),
            "s" | "z" => Some(Self::Os),
            _ => None,
        }
    }
}

pub type JitOptimizationLevel = LlvmOptimizationLevel;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum LlvmTarget {
    #[default]
    Native,
    Triple(String),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AotWrapperOptions {
    pub emit_doc_comments: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionCompileOptions {
    pub opt_level: LlvmOptimizationLevel,
    pub call_policy: CallPolicyConfig,
}

impl FunctionCompileOptions {
    pub const fn new(opt_level: LlvmOptimizationLevel, call_policy: CallPolicyConfig) -> Self {
        Self {
            opt_level,
            call_policy,
        }
    }
}

impl From<LlvmOptimizationLevel> for FunctionCompileOptions {
    fn from(opt_level: LlvmOptimizationLevel) -> Self {
        Self {
            opt_level,
            call_policy: CallPolicyConfig {
                default_policy: CallPolicy::InlineAtLowering,
                respect_function_overrides: true,
            },
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FunctionCompileReport {
    pub lowering_time: Duration,
    pub llvm_time: Duration,
    pub stats: CompileStats,
    pub warnings: Vec<CompileWarning>,
}

#[derive(Debug)]
pub struct CompiledJitFunction {
    lowered: LoweredFunction,
    compile_report: FunctionCompileReport,
    lljit: llvm_sys::orc2::lljit::LLVMOrcLLJITRef,
    function: RawKernelFn,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JitExecutionContext {
    inputs: Vec<Vec<f64>>,
    outputs: Vec<Vec<f64>>,
    input_ptrs: Vec<*const f64>,
    output_ptrs: Vec<*mut f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LlvmCompileMode {
    Jit,
    Aot,
}

impl CompiledJitFunction {
    pub fn compile_function(
        function: &SXFunction,
        opt_level: LlvmOptimizationLevel,
    ) -> Result<Self> {
        Self::compile_function_with_options(function, FunctionCompileOptions::from(opt_level))
    }

    pub fn compile_function_with_options(
        function: &SXFunction,
        options: FunctionCompileOptions,
    ) -> Result<Self> {
        let lowering_started = Instant::now();
        let lowered = lower_function_with_policies(function, options.call_policy)?;
        let lowering_time = lowering_started.elapsed();
        Self::compile_lowered_with_report(&lowered, options.opt_level, lowering_time)
    }

    pub fn compile_lowered(
        lowered: &LoweredFunction,
        opt_level: LlvmOptimizationLevel,
    ) -> Result<Self> {
        Self::compile_lowered_with_report(lowered, opt_level, Duration::ZERO)
    }

    fn compile_lowered_with_report(
        lowered: &LoweredFunction,
        opt_level: LlvmOptimizationLevel,
        lowering_time: Duration,
    ) -> Result<Self> {
        ensure_native_llvm_initialized()?;

        let llvm_started = Instant::now();
        let object = build_object_buffer(
            lowered,
            opt_level,
            &LlvmTarget::Native,
            LlvmCompileMode::Jit,
        )?;
        let llvm_time = llvm_started.elapsed();
        let lljit = create_lljit()?;
        let main_dylib = unsafe { LLVMOrcLLJITGetMainJITDylib(lljit) };
        let prefix = unsafe { LLVMOrcLLJITGetGlobalPrefix(lljit) };
        attach_current_process_symbols(main_dylib, prefix)?;
        let add_error = unsafe { LLVMOrcLLJITAddObjectFile(lljit, main_dylib, object) };
        if let Err(error) = consume_llvm_error(add_error) {
            let _ = unsafe { LLVMOrcDisposeLLJIT(lljit) };
            return Err(error.context("failed to add object file to LLJIT"));
        }

        let address = lookup_symbol_address(lljit, &kernel_symbol_name(&lowered.name))?;
        let addr = usize::try_from(address)
            .map_err(|_| anyhow!("JIT symbol address does not fit into usize"))?;
        let function = unsafe { mem::transmute::<usize, RawKernelFn>(addr) };

        Ok(Self {
            lowered: lowered.clone(),
            compile_report: FunctionCompileReport {
                lowering_time,
                llvm_time,
                stats: lowered.stats.clone(),
                warnings: lowered.warnings.clone(),
            },
            lljit,
            function,
        })
    }

    pub fn lowered(&self) -> &LoweredFunction {
        &self.lowered
    }

    pub fn compile_report(&self) -> &FunctionCompileReport {
        &self.compile_report
    }

    pub fn create_context(&self) -> JitExecutionContext {
        JitExecutionContext::new(&self.lowered)
    }

    pub fn eval(&self, context: &mut JitExecutionContext) {
        unsafe {
            (self.function)(context.input_ptrs.as_ptr(), context.output_ptrs.as_ptr());
        }
    }
}

pub fn emit_object_file(
    path: impl AsRef<Path>,
    function: &SXFunction,
    opt_level: LlvmOptimizationLevel,
    target: &LlvmTarget,
) -> Result<()> {
    emit_object_file_with_options(path, function, FunctionCompileOptions::from(opt_level), target)
}

pub fn emit_object_file_with_options(
    path: impl AsRef<Path>,
    function: &SXFunction,
    options: FunctionCompileOptions,
    target: &LlvmTarget,
) -> Result<()> {
    emit_object_file_lowered(
        path,
        &lower_function_with_policies(function, options.call_policy)?,
        options.opt_level,
        target,
    )
}

pub fn emit_object_file_lowered(
    path: impl AsRef<Path>,
    lowered: &LoweredFunction,
    opt_level: LlvmOptimizationLevel,
    target: &LlvmTarget,
) -> Result<()> {
    fs::write(path, emit_object_bytes_lowered(lowered, opt_level, target)?).map_err(Into::into)
}

pub fn emit_object_bytes_lowered(
    lowered: &LoweredFunction,
    opt_level: LlvmOptimizationLevel,
    target: &LlvmTarget,
) -> Result<Vec<u8>> {
    ensure_native_llvm_initialized()?;
    let object = build_object_buffer(lowered, opt_level, target, LlvmCompileMode::Aot)?;
    let bytes = unsafe { memory_buffer_to_bytes(object) }?;
    unsafe { LLVMDisposeMemoryBuffer(object) };
    Ok(bytes)
}

impl Drop for CompiledJitFunction {
    fn drop(&mut self) {
        let _ = unsafe { LLVMOrcDisposeLLJIT(self.lljit) };
    }
}

impl JitExecutionContext {
    fn new(lowered: &LoweredFunction) -> Self {
        let mut inputs = lowered
            .inputs
            .iter()
            .map(|slot| vec![0.0; slot.ccs.nnz()])
            .collect::<Vec<_>>();
        let mut outputs = lowered
            .outputs
            .iter()
            .map(|slot| vec![0.0; slot.ccs.nnz()])
            .collect::<Vec<_>>();
        let input_ptrs = inputs
            .iter_mut()
            .map(|slot| slot.as_ptr())
            .collect::<Vec<_>>();
        let output_ptrs = outputs
            .iter_mut()
            .map(|slot| slot.as_mut_ptr())
            .collect::<Vec<_>>();
        Self {
            inputs,
            outputs,
            input_ptrs,
            output_ptrs,
        }
    }

    pub fn input_mut(&mut self, slot: usize) -> &mut [f64] {
        &mut self.inputs[slot]
    }

    pub fn output(&self, slot: usize) -> &[f64] {
        &self.outputs[slot]
    }

    pub fn output_mut(&mut self, slot: usize) -> &mut [f64] {
        &mut self.outputs[slot]
    }
}

fn ensure_native_llvm_initialized() -> Result<()> {
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| unsafe {
        if LLVM_InitializeNativeTarget() != 0 {
            return Err("LLVM failed to initialize the native target".into());
        }
        if LLVM_InitializeNativeAsmPrinter() != 0 {
            return Err("LLVM failed to initialize the native asm printer".into());
        }
        Ok(())
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

fn build_object_buffer(
    lowered: &LoweredFunction,
    opt_level: LlvmOptimizationLevel,
    target: &LlvmTarget,
    compile_mode: LlvmCompileMode,
) -> Result<LLVMMemoryBufferRef> {
    unsafe {
        let context = LLVMContextCreate();
        let (target_machine, triple) = create_target_machine(opt_level, target, compile_mode)?;
        let module = match build_module(lowered, context, target_machine, &triple) {
            Ok(module) => module,
            Err(error) => {
                LLVMDisposeTargetMachine(target_machine);
                LLVMContextDispose(context);
                return Err(error);
            }
        };

        let object = (|| -> Result<LLVMMemoryBufferRef> {
            verify_module(module, "before optimization")?;
            run_default_pass_pipeline(module, target_machine, opt_level)?;
            verify_module(module, "after optimization")?;
            emit_object_buffer(module, target_machine)
        })();

        llvm_sys::core::LLVMDisposeModule(module);
        LLVMDisposeTargetMachine(target_machine);
        LLVMContextDispose(context);
        object
    }
}

unsafe fn create_target_machine(
    opt_level: LlvmOptimizationLevel,
    target: &LlvmTarget,
    compile_mode: LlvmCompileMode,
) -> Result<(LLVMTargetMachineRef, CString)> {
    let host_triple = unsafe { take_llvm_message(LLVMGetDefaultTargetTriple()) }?;
    let triple = match target {
        LlvmTarget::Native => host_triple.clone(),
        LlvmTarget::Triple(triple) => triple.clone(),
    };
    let use_host_cpu = match target {
        LlvmTarget::Native => true,
        LlvmTarget::Triple(target_triple) => target_triple == &host_triple,
    };
    let cpu = if use_host_cpu {
        unsafe { take_llvm_message(LLVMGetHostCPUName()) }?
    } else {
        "generic".to_string()
    };
    let features = if use_host_cpu {
        unsafe { take_llvm_message(LLVMGetHostCPUFeatures()) }?
    } else {
        String::new()
    };
    let triple_c = CString::new(triple)?;
    let cpu_c = CString::new(cpu)?;
    let features_c = CString::new(features)?;

    let mut target = ptr::null_mut();
    let mut error_message = ptr::null_mut();
    if unsafe { LLVMGetTargetFromTriple(triple_c.as_ptr(), &mut target, &mut error_message) } != 0 {
        let message = unsafe { take_owned_message(error_message) }?;
        bail!("failed to resolve LLVM target from triple: {message}");
    }

    let options = unsafe { LLVMCreateTargetMachineOptions() };
    if options.is_null() {
        bail!("LLVMCreateTargetMachineOptions returned null");
    }
    unsafe {
        LLVMTargetMachineOptionsSetCPU(options, cpu_c.as_ptr());
        LLVMTargetMachineOptionsSetFeatures(options, features_c.as_ptr());
        LLVMTargetMachineOptionsSetCodeGenOptLevel(options, opt_level.codegen_level());
        LLVMTargetMachineOptionsSetCodeModel(
            options,
            match compile_mode {
                LlvmCompileMode::Jit => LLVMCodeModel::LLVMCodeModelJITDefault,
                LlvmCompileMode::Aot => LLVMCodeModel::LLVMCodeModelDefault,
            },
        );
    }
    let target_machine =
        unsafe { LLVMCreateTargetMachineWithOptions(target, triple_c.as_ptr(), options) };
    unsafe { LLVMDisposeTargetMachineOptions(options) };
    if target_machine.is_null() {
        bail!("LLVMCreateTargetMachineWithOptions returned null");
    }
    Ok((target_machine, triple_c))
}

unsafe fn build_module(
    lowered: &LoweredFunction,
    context: LLVMContextRef,
    target_machine: LLVMTargetMachineRef,
    triple: &CString,
) -> Result<LLVMModuleRef> {
    let module_name = CString::new(format!("{}_llvm_jit", lowered.name))?;
    let symbol_name = CString::new(kernel_symbol_name(&lowered.name))?;
    let module = unsafe { LLVMModuleCreateWithNameInContext(module_name.as_ptr(), context) };
    unsafe { LLVMSetTarget(module, triple.as_ptr()) };

    let layout_string = unsafe { llvm_data_layout_string(target_machine) }?;
    let layout_c = CString::new(layout_string)?;
    let layout = unsafe { LLVMCreateTargetData(layout_c.as_ptr()) };
    unsafe {
        LLVMSetModuleDataLayout(module, layout);
        LLVMDisposeTargetData(layout);
    }

    let builder = unsafe { LLVMCreateBuilderInContext(context) };
    let build_result = (|| -> Result<()> {
        let f64_ty = unsafe { LLVMDoubleTypeInContext(context) };
        let i64_ty = unsafe { LLVMInt64TypeInContext(context) };
        let f64_ptr_ty = unsafe { LLVMPointerType(f64_ty, 0) };
        let ptr_array_ty = unsafe { LLVMPointerType(f64_ptr_ty, 0) };
        let void_ty = unsafe { LLVMVoidTypeInContext(context) };
        let mut param_types = [ptr_array_ty, ptr_array_ty];
        let function_ty = unsafe {
            LLVMFunctionType(
                void_ty,
                param_types.as_mut_ptr(),
                param_types.len() as u32,
                0,
            )
        };
        let function = unsafe { LLVMAddFunction(module, symbol_name.as_ptr(), function_ty) };
        let subfunctions = lowered
            .subfunctions
            .iter()
            .enumerate()
            .map(|(index, subfunction)| {
                unsafe {
                    declare_subfunction(
                        module,
                        context,
                        index,
                        subfunction,
                        f64_ty,
                        f64_ptr_ty,
                        void_ty,
                    )
                }
            })
            .collect::<Result<Vec<_>>>()?;

        unsafe {
            emit_root_callable(
                builder,
                module,
                context,
                function,
                lowered,
                &subfunctions,
                f64_ty,
                f64_ptr_ty,
                i64_ty,
            )?
        };

        for (index, subfunction) in lowered.subfunctions.iter().enumerate() {
            unsafe {
                emit_internal_callable(
                    builder,
                    module,
                    context,
                    subfunctions[index].function,
                    subfunction,
                    &subfunctions,
                    f64_ty,
                    f64_ptr_ty,
                    i64_ty,
                )?
            };
        }
        Ok(())
    })();

    unsafe { LLVMDisposeBuilder(builder) };
    match build_result {
        Ok(()) => Ok(module),
        Err(error) => {
            unsafe { llvm_sys::core::LLVMDisposeModule(module) };
            Err(error)
        }
    }
}

struct DeclaredSubfunction {
    function: LLVMValueRef,
    function_ty: LLVMTypeRef,
}

enum CallableAbi<'a> {
    Root {
        inputs_param: LLVMValueRef,
    },
    Internal {
        function: LLVMValueRef,
        input_offsets: &'a [usize],
    },
}

unsafe fn declare_subfunction(
    module: LLVMModuleRef,
    context: LLVMContextRef,
    index: usize,
    subfunction: &LoweredSubfunction,
    f64_ty: LLVMTypeRef,
    f64_ptr_ty: LLVMTypeRef,
    void_ty: LLVMTypeRef,
) -> Result<DeclaredSubfunction> {
    let input_count = subfunction.inputs.iter().map(|slot| slot.ccs.nnz()).sum::<usize>();
    let output_count = subfunction.outputs.iter().map(|slot| slot.ccs.nnz()).sum::<usize>();
    let mut params = vec![f64_ty; input_count];
    params.extend((0..output_count).map(|_| f64_ptr_ty));
    let function_ty =
        unsafe { LLVMFunctionType(void_ty, params.as_mut_ptr(), params.len() as u32, 0) };
    let name = CString::new(format!("__sx_internal_{}_{}", index, subfunction.name))?;
    let function = unsafe { LLVMAddFunction(module, name.as_ptr(), function_ty) };
    unsafe { LLVMSetLinkage(function, LLVMLinkage::LLVMInternalLinkage) };
    if matches!(subfunction.call_policy, CallPolicy::NoInlineLLVM) {
        unsafe { add_noinline_attribute(context, function)? };
    }
    Ok(DeclaredSubfunction {
        function,
        function_ty,
    })
}

unsafe fn add_noinline_attribute(context: LLVMContextRef, function: LLVMValueRef) -> Result<()> {
    let kind = unsafe { LLVMGetEnumAttributeKindForName(c"noinline".as_ptr(), 8) };
    if kind == 0 {
        bail!("LLVM could not resolve the noinline attribute kind");
    }
    let attr = unsafe { LLVMCreateEnumAttribute(context, kind, 0) };
    unsafe { LLVMAddAttributeAtIndex(function, LLVMAttributeFunctionIndex, attr) };
    Ok(())
}

unsafe fn emit_root_callable(
    builder: LLVMBuilderRef,
    module: LLVMModuleRef,
    context: LLVMContextRef,
    function: LLVMValueRef,
    lowered: &LoweredFunction,
    subfunctions: &[DeclaredSubfunction],
    f64_ty: LLVMTypeRef,
    f64_ptr_ty: LLVMTypeRef,
    i64_ty: LLVMTypeRef,
) -> Result<()> {
    let entry = unsafe { LLVMAppendBasicBlockInContext(context, function, c"entry".as_ptr()) };
    unsafe { LLVMPositionBuilderAtEnd(builder, entry) };
    let abi = CallableAbi::Root {
        inputs_param: unsafe { LLVMGetParam(function, 0) },
    };
    let outputs_param = unsafe { LLVMGetParam(function, 1) };
    let temps = unsafe {
        emit_instruction_sequence(
            builder,
            module,
            &abi,
            &lowered.instructions,
            subfunctions,
            f64_ty,
            f64_ptr_ty,
            i64_ty,
        )?
    };
    for (slot_idx, values) in lowered.output_values.iter().enumerate() {
        let output_ptr = unsafe { load_slot_ptr(builder, outputs_param, slot_idx, f64_ptr_ty, i64_ty) };
        for (offset, value) in values.iter().copied().enumerate() {
            let value_ref = unsafe {
                emit_value(builder, &abi, &temps, value, f64_ty, f64_ptr_ty, i64_ty)
            };
            let cell_ptr = unsafe { gep_f64_ptr(builder, output_ptr, offset, f64_ty, i64_ty) };
            unsafe { LLVMBuildStore(builder, value_ref, cell_ptr) };
        }
    }
    unsafe { LLVMBuildRetVoid(builder) };
    Ok(())
}

unsafe fn emit_internal_callable(
    builder: LLVMBuilderRef,
    module: LLVMModuleRef,
    context: LLVMContextRef,
    function: LLVMValueRef,
    lowered: &LoweredSubfunction,
    subfunctions: &[DeclaredSubfunction],
    f64_ty: LLVMTypeRef,
    f64_ptr_ty: LLVMTypeRef,
    i64_ty: LLVMTypeRef,
) -> Result<()> {
    let entry = unsafe { LLVMAppendBasicBlockInContext(context, function, c"entry".as_ptr()) };
    unsafe { LLVMPositionBuilderAtEnd(builder, entry) };
    let input_offsets = lowered
        .inputs
        .iter()
        .scan(0usize, |offset, slot| {
            let current = *offset;
            *offset += slot.ccs.nnz();
            Some(current)
        })
        .collect::<Vec<_>>();
    let abi = CallableAbi::Internal {
        function,
        input_offsets: &input_offsets,
    };
    let temps = unsafe {
        emit_instruction_sequence(
            builder,
            module,
            &abi,
            &lowered.instructions,
            subfunctions,
            f64_ty,
            f64_ptr_ty,
            i64_ty,
        )?
    };
    let output_param_base = lowered.inputs.iter().map(|slot| slot.ccs.nnz()).sum::<usize>();
    let mut linear_output = 0usize;
    for values in &lowered.output_values {
        for value in values.iter().copied() {
            let value_ref = unsafe {
                emit_value(builder, &abi, &temps, value, f64_ty, f64_ptr_ty, i64_ty)
            };
            let output_ptr = unsafe { LLVMGetParam(function, (output_param_base + linear_output) as u32) };
            unsafe { LLVMBuildStore(builder, value_ref, output_ptr) };
            linear_output += 1;
        }
    }
    unsafe { LLVMBuildRetVoid(builder) };
    Ok(())
}

unsafe fn emit_instruction_sequence(
    builder: LLVMBuilderRef,
    module: LLVMModuleRef,
    abi: &CallableAbi<'_>,
    instructions: &[Instruction],
    subfunctions: &[DeclaredSubfunction],
    f64_ty: LLVMTypeRef,
    f64_ptr_ty: LLVMTypeRef,
    i64_ty: LLVMTypeRef,
) -> Result<Vec<LLVMValueRef>> {
    let mut temps = Vec::new();
    for instruction in instructions {
        match instruction {
            Instruction::Unary { temp, op, input } => {
                if temps.len() != *temp {
                    bail!("lowered temp order is not contiguous");
                }
                let input =
                    unsafe { emit_value(builder, abi, &temps, *input, f64_ty, f64_ptr_ty, i64_ty) };
                temps.push(unsafe { emit_unary_op(builder, module, *op, input, f64_ty) });
            }
            Instruction::Binary { temp, op, lhs, rhs } => {
                if temps.len() != *temp {
                    bail!("lowered temp order is not contiguous");
                }
                let lhs =
                    unsafe { emit_value(builder, abi, &temps, *lhs, f64_ty, f64_ptr_ty, i64_ty) };
                let rhs =
                    unsafe { emit_value(builder, abi, &temps, *rhs, f64_ty, f64_ptr_ty, i64_ty) };
                temps.push(unsafe { emit_binary_op(builder, module, *op, lhs, rhs, f64_ty) });
            }
            Instruction::Call {
                temps: output_temps,
                callee,
                inputs,
            } => {
                if output_temps
                    .iter()
                    .enumerate()
                    .any(|(index, temp)| *temp != temps.len() + index)
                {
                    bail!("lowered call temp order is not contiguous");
                }
                let declared = &subfunctions[*callee];
                let mut args = inputs
                    .iter()
                    .copied()
                    .map(|value| unsafe {
                        emit_value(builder, abi, &temps, value, f64_ty, f64_ptr_ty, i64_ty)
                    })
                    .collect::<Vec<_>>();
                let mut output_ptrs = Vec::with_capacity(output_temps.len());
                for _ in output_temps {
                    output_ptrs.push(unsafe { LLVMBuildAlloca(builder, f64_ty, c"".as_ptr()) });
                }
                args.extend(output_ptrs.iter().copied());
                unsafe {
                    LLVMBuildCall2(
                        builder,
                        declared.function_ty,
                        declared.function,
                        args.as_mut_ptr(),
                        args.len() as u32,
                        c"".as_ptr(),
                    )
                };
                for output_ptr in output_ptrs {
                    temps.push(unsafe { LLVMBuildLoad2(builder, f64_ty, output_ptr, c"".as_ptr()) });
                }
            }
        }
    }
    Ok(temps)
}

unsafe fn emit_unary_op(
    builder: LLVMBuilderRef,
    module: LLVMModuleRef,
    op: UnaryOp,
    input: LLVMValueRef,
    f64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    match op {
        UnaryOp::Abs => unsafe { call_unary_math(builder, module, c"fabs", input, f64_ty) },
        UnaryOp::Floor => unsafe { call_unary_math(builder, module, c"floor", input, f64_ty) },
        UnaryOp::Ceil => unsafe { call_unary_math(builder, module, c"ceil", input, f64_ty) },
        UnaryOp::Round => unsafe { call_unary_math(builder, module, c"round", input, f64_ty) },
        UnaryOp::Trunc => unsafe { call_unary_math(builder, module, c"trunc", input, f64_ty) },
        UnaryOp::Sqrt => unsafe { call_unary_math(builder, module, c"sqrt", input, f64_ty) },
        UnaryOp::Exp => unsafe { call_unary_math(builder, module, c"exp", input, f64_ty) },
        UnaryOp::Log => unsafe { call_unary_math(builder, module, c"log", input, f64_ty) },
        UnaryOp::Sin => unsafe { call_unary_math(builder, module, c"sin", input, f64_ty) },
        UnaryOp::Cos => unsafe { call_unary_math(builder, module, c"cos", input, f64_ty) },
        UnaryOp::Tan => unsafe { call_unary_math(builder, module, c"tan", input, f64_ty) },
        UnaryOp::Asin => unsafe { call_unary_math(builder, module, c"asin", input, f64_ty) },
        UnaryOp::Acos => unsafe { call_unary_math(builder, module, c"acos", input, f64_ty) },
        UnaryOp::Atan => unsafe { call_unary_math(builder, module, c"atan", input, f64_ty) },
        UnaryOp::Sinh => unsafe { call_unary_math(builder, module, c"sinh", input, f64_ty) },
        UnaryOp::Cosh => unsafe { call_unary_math(builder, module, c"cosh", input, f64_ty) },
        UnaryOp::Tanh => unsafe { call_unary_math(builder, module, c"tanh", input, f64_ty) },
        UnaryOp::Asinh => unsafe { call_unary_math(builder, module, c"asinh", input, f64_ty) },
        UnaryOp::Acosh => unsafe { call_unary_math(builder, module, c"acosh", input, f64_ty) },
        UnaryOp::Atanh => unsafe { call_unary_math(builder, module, c"atanh", input, f64_ty) },
        UnaryOp::Sign => unsafe { emit_sign(builder, input, f64_ty) },
    }
}

unsafe fn emit_binary_op(
    builder: LLVMBuilderRef,
    module: LLVMModuleRef,
    op: BinaryOp,
    lhs: LLVMValueRef,
    rhs: LLVMValueRef,
    f64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    match op {
        BinaryOp::Add => unsafe { LLVMBuildFAdd(builder, lhs, rhs, c"".as_ptr()) },
        BinaryOp::Sub => unsafe { LLVMBuildFSub(builder, lhs, rhs, c"".as_ptr()) },
        BinaryOp::Mul => unsafe { LLVMBuildFMul(builder, lhs, rhs, c"".as_ptr()) },
        BinaryOp::Div => unsafe { LLVMBuildFDiv(builder, lhs, rhs, c"".as_ptr()) },
        BinaryOp::Pow => unsafe { call_binary_math(builder, module, c"pow", lhs, rhs, f64_ty) },
        BinaryOp::Atan2 => unsafe { call_binary_math(builder, module, c"atan2", lhs, rhs, f64_ty) },
        BinaryOp::Hypot => unsafe { call_binary_math(builder, module, c"hypot", lhs, rhs, f64_ty) },
        BinaryOp::Mod => unsafe { call_binary_math(builder, module, c"fmod", lhs, rhs, f64_ty) },
        BinaryOp::Copysign => unsafe {
            call_binary_math(builder, module, c"copysign", lhs, rhs, f64_ty)
        },
    }
}

unsafe fn emit_sign(
    builder: LLVMBuilderRef,
    input: LLVMValueRef,
    f64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    let zero = unsafe { LLVMConstReal(f64_ty, 0.0) };
    let one = unsafe { LLVMConstReal(f64_ty, 1.0) };
    let minus_one = unsafe { LLVMConstReal(f64_ty, -1.0) };
    let positive = unsafe { LLVMBuildFCmp(builder, LLVMRealOGT, input, zero, c"".as_ptr()) };
    let negative = unsafe { LLVMBuildFCmp(builder, LLVMRealOLT, input, zero, c"".as_ptr()) };
    let negative_or_zero =
        unsafe { LLVMBuildSelect(builder, negative, minus_one, zero, c"".as_ptr()) };
    unsafe { LLVMBuildSelect(builder, positive, one, negative_or_zero, c"".as_ptr()) }
}

unsafe fn call_unary_math(
    builder: LLVMBuilderRef,
    module: LLVMModuleRef,
    name: &CStr,
    input: LLVMValueRef,
    f64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    let (function, function_ty) = unsafe { get_or_declare_math_fn(module, name, 1, f64_ty) };
    let mut args = [input];
    unsafe {
        LLVMBuildCall2(
            builder,
            function_ty,
            function,
            args.as_mut_ptr(),
            1,
            c"".as_ptr(),
        )
    }
}

unsafe fn call_binary_math(
    builder: LLVMBuilderRef,
    module: LLVMModuleRef,
    name: &CStr,
    lhs: LLVMValueRef,
    rhs: LLVMValueRef,
    f64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    let (function, function_ty) = unsafe { get_or_declare_math_fn(module, name, 2, f64_ty) };
    let mut args = [lhs, rhs];
    unsafe {
        LLVMBuildCall2(
            builder,
            function_ty,
            function,
            args.as_mut_ptr(),
            2,
            c"".as_ptr(),
        )
    }
}

unsafe fn get_or_declare_math_fn(
    module: LLVMModuleRef,
    name: &CStr,
    arity: usize,
    f64_ty: LLVMTypeRef,
) -> (LLVMValueRef, LLVMTypeRef) {
    let existing = unsafe { LLVMGetNamedFunction(module, name.as_ptr()) };
    if !existing.is_null() {
        return (existing, unsafe { LLVMGlobalGetValueType(existing) });
    }
    let mut params = vec![f64_ty; arity];
    let function_ty = unsafe { LLVMFunctionType(f64_ty, params.as_mut_ptr(), arity as u32, 0) };
    (
        unsafe { LLVMAddFunction(module, name.as_ptr(), function_ty) },
        function_ty,
    )
}

unsafe fn llvm_data_layout_string(target_machine: LLVMTargetMachineRef) -> Result<String> {
    let data_layout =
        unsafe { llvm_sys::target_machine::LLVMCreateTargetDataLayout(target_machine) };
    let string_ptr = unsafe { llvm_sys::target::LLVMCopyStringRepOfTargetData(data_layout) };
    let string = unsafe { take_llvm_message(string_ptr) }?;
    unsafe { LLVMDisposeTargetData(data_layout) };
    Ok(string)
}

unsafe fn emit_value(
    builder: LLVMBuilderRef,
    abi: &CallableAbi<'_>,
    temps: &[LLVMValueRef],
    value: ValueRef,
    f64_ty: LLVMTypeRef,
    f64_ptr_ty: LLVMTypeRef,
    i64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    match value {
        ValueRef::Const(value) => unsafe { LLVMConstReal(f64_ty, value) },
        ValueRef::Temp(temp) => temps[temp],
        ValueRef::Input { slot, offset } => match abi {
            CallableAbi::Root { inputs_param } => {
                let base_ptr =
                    unsafe { load_slot_ptr(builder, *inputs_param, slot, f64_ptr_ty, i64_ty) };
                let value_ptr = unsafe { gep_f64_ptr(builder, base_ptr, offset, f64_ty, i64_ty) };
                unsafe { LLVMBuildLoad2(builder, f64_ty, value_ptr, c"".as_ptr()) }
            }
            CallableAbi::Internal {
                function,
                input_offsets,
            } => unsafe { LLVMGetParam(*function, (input_offsets[slot] + offset) as u32) },
        },
    }
}

unsafe fn load_slot_ptr(
    builder: LLVMBuilderRef,
    slots_param: LLVMValueRef,
    slot: usize,
    f64_ptr_ty: LLVMTypeRef,
    i64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    let mut index = [unsafe { LLVMConstInt(i64_ty, slot as u64, 0) }];
    let slot_ptr = unsafe {
        LLVMBuildInBoundsGEP2(
            builder,
            f64_ptr_ty,
            slots_param,
            index.as_mut_ptr(),
            1,
            c"".as_ptr(),
        )
    };
    unsafe { LLVMBuildLoad2(builder, f64_ptr_ty, slot_ptr, c"".as_ptr()) }
}

unsafe fn gep_f64_ptr(
    builder: LLVMBuilderRef,
    base_ptr: LLVMValueRef,
    offset: usize,
    f64_ty: LLVMTypeRef,
    i64_ty: LLVMTypeRef,
) -> LLVMValueRef {
    let mut index = [unsafe { LLVMConstInt(i64_ty, offset as u64, 0) }];
    unsafe {
        LLVMBuildInBoundsGEP2(
            builder,
            f64_ty,
            base_ptr,
            index.as_mut_ptr(),
            1,
            c"".as_ptr(),
        )
    }
}

unsafe fn run_default_pass_pipeline(
    module: LLVMModuleRef,
    target_machine: LLVMTargetMachineRef,
    opt_level: JitOptimizationLevel,
) -> Result<()> {
    let options = unsafe { LLVMCreatePassBuilderOptions() };
    let passes = CString::new(opt_level.pass_pipeline())?;
    let result = consume_llvm_error(unsafe {
        LLVMRunPasses(module, passes.as_ptr(), target_machine, options)
    });
    unsafe { LLVMDisposePassBuilderOptions(options) };
    result.context("failed to optimize LLVM module")
}

unsafe fn emit_object_buffer(
    module: LLVMModuleRef,
    target_machine: LLVMTargetMachineRef,
) -> Result<LLVMMemoryBufferRef> {
    let mut error_message = ptr::null_mut();
    let mut buffer = ptr::null_mut();
    if unsafe {
        LLVMTargetMachineEmitToMemoryBuffer(
            target_machine,
            module,
            LLVMCodeGenFileType::LLVMObjectFile,
            &mut error_message,
            &mut buffer,
        )
    } != 0
    {
        let message = unsafe { take_owned_message(error_message) }?;
        bail!("failed to emit LLVM object file: {message}");
    }
    Ok(buffer)
}

unsafe fn memory_buffer_to_bytes(buffer: LLVMMemoryBufferRef) -> Result<Vec<u8>> {
    let start = unsafe { LLVMGetBufferStart(buffer) };
    if start.is_null() {
        bail!("LLVM returned a null object buffer");
    }
    let size = unsafe { LLVMGetBufferSize(buffer) };
    let bytes = unsafe { slice::from_raw_parts(start.cast::<u8>(), size) };
    Ok(bytes.to_vec())
}

pub fn generate_aot_wrapper_module(
    lowered: &LoweredFunction,
    options: &AotWrapperOptions,
) -> Result<String> {
    let mut out = String::new();
    let module_name = format!("{}_llvm_aot", sanitize_ident(&lowered.name));
    let context_name = format!("{}LlvmAotContext", to_pascal_case(&lowered.name));
    let has_context = lowered
        .inputs
        .iter()
        .chain(lowered.outputs.iter())
        .any(|slot| !slot.ccs.is_scalar());

    out.push_str(&format!("pub mod {module_name} {{\n"));
    if options.emit_doc_comments {
        out.push_str(&format!(
            "    //! Generated LLVM AOT wrapper for `{}`.\n",
            lowered.name
        ));
    }
    out.push_str(
        "    #[derive(Clone, Copy, Debug, PartialEq, Eq)]\n    pub struct GeneratedCCS { pub nrow: usize, pub ncol: usize, pub col_ptrs: &'static [usize], pub row_indices: &'static [usize] }\n",
    );
    out.push_str(
        "    #[derive(Clone, Copy, Debug, PartialEq, Eq)]\n    pub struct SlotSpec { pub name: &'static str, pub ccs: GeneratedCCS, pub nnz: usize }\n",
    );

    for slot in lowered.inputs.iter().chain(lowered.outputs.iter()) {
        emit_ccs(slot.name.as_str(), &slot.ccs, &mut out);
    }

    out.push_str("    pub const INPUT_SPECS: &[SlotSpec] = &[\n");
    for slot in &lowered.inputs {
        let upper = sanitize_ident(slot.name.as_str()).to_ascii_uppercase();
        out.push_str(&format!(
            "        SlotSpec {{ name: \"{}\", ccs: {upper}_CCS, nnz: {} }},\n",
            slot.name,
            slot.ccs.nnz()
        ));
    }
    out.push_str("    ];\n");
    out.push_str("    pub const OUTPUT_SPECS: &[SlotSpec] = &[\n");
    for slot in &lowered.outputs {
        let upper = sanitize_ident(slot.name.as_str()).to_ascii_uppercase();
        out.push_str(&format!(
            "        SlotSpec {{ name: \"{}\", ccs: {upper}_CCS, nnz: {} }},\n",
            slot.name,
            slot.ccs.nnz()
        ));
    }
    out.push_str("    ];\n");
    out.push_str("    unsafe extern \"C\" {\n");
    out.push_str(&format!(
        "        fn {}(inputs: *const *const f64, outputs: *const *mut f64);\n",
        kernel_symbol_name(&lowered.name)
    ));
    out.push_str("    }\n");

    let mut params = Vec::new();
    for input in &lowered.inputs {
        let ident = sanitize_ident(input.name.as_str());
        let ty = if input.ccs.is_scalar() {
            "f64"
        } else {
            "&[f64]"
        };
        params.push(format!("{ident}: {ty}"));
    }
    for output in &lowered.outputs {
        let ident = sanitize_ident(output.name.as_str());
        let ty = if output.ccs.is_scalar() {
            "&mut f64"
        } else {
            "&mut [f64]"
        };
        params.push(format!("{ident}: {ty}"));
    }
    out.push_str(&format!("    pub fn eval({}) {{\n", params.join(", ")));
    for input in &lowered.inputs {
        if !input.ccs.is_scalar() {
            let ident = sanitize_ident(input.name.as_str());
            out.push_str(&format!(
                "        debug_assert_eq!({ident}.len(), {});\n",
                input.ccs.nnz()
            ));
        }
    }
    for output in &lowered.outputs {
        if !output.ccs.is_scalar() {
            let ident = sanitize_ident(output.name.as_str());
            out.push_str(&format!(
                "        debug_assert_eq!({ident}.len(), {});\n",
                output.ccs.nnz()
            ));
        }
    }
    let mut input_ptrs = Vec::new();
    for input in &lowered.inputs {
        let ident = sanitize_ident(input.name.as_str());
        if input.ccs.is_scalar() {
            let buffer_ident = format!("{ident}_buffer");
            out.push_str(&format!("        let {buffer_ident} = [{ident}];\n"));
            input_ptrs.push(format!("{buffer_ident}.as_ptr()"));
        } else {
            input_ptrs.push(format!("{ident}.as_ptr()"));
        }
    }
    let mut output_ptrs = Vec::new();
    let mut scalar_output_assignments = Vec::new();
    for output in &lowered.outputs {
        let ident = sanitize_ident(output.name.as_str());
        if output.ccs.is_scalar() {
            let buffer_ident = format!("{ident}_buffer");
            out.push_str(&format!("        let mut {buffer_ident} = [0.0_f64];\n"));
            output_ptrs.push(format!("{buffer_ident}.as_mut_ptr()"));
            scalar_output_assignments.push(format!("        *{ident} = {buffer_ident}[0];\n"));
        } else {
            output_ptrs.push(format!("{ident}.as_mut_ptr()"));
        }
    }
    out.push_str(&format!(
        "        let input_ptrs: [*const f64; {}] = [{}];\n",
        input_ptrs.len(),
        input_ptrs.join(", ")
    ));
    out.push_str(&format!(
        "        let mut output_ptrs: [*mut f64; {}] = [{}];\n",
        output_ptrs.len(),
        output_ptrs.join(", ")
    ));
    out.push_str(&format!(
        "        unsafe {{ {}(input_ptrs.as_ptr(), output_ptrs.as_mut_ptr()); }}\n",
        kernel_symbol_name(&lowered.name)
    ));
    for assignment in scalar_output_assignments {
        out.push_str(&assignment);
    }
    out.push_str("    }\n");

    if has_context {
        out.push_str("    #[derive(Clone, Debug, PartialEq)]\n");
        out.push_str(&format!("    pub struct {context_name} {{\n"));
        for input in &lowered.inputs {
            let ident = sanitize_ident(input.name.as_str());
            let field_ty = if input.ccs.is_scalar() {
                "f64"
            } else {
                "Vec<f64>"
            };
            out.push_str(&format!("        pub {ident}: {field_ty},\n"));
        }
        for output in &lowered.outputs {
            let ident = sanitize_ident(output.name.as_str());
            let field_ty = if output.ccs.is_scalar() {
                "f64"
            } else {
                "Vec<f64>"
            };
            out.push_str(&format!("        pub {ident}: {field_ty},\n"));
        }
        out.push_str("    }\n");
        out.push_str(&format!("    impl Default for {context_name} {{\n"));
        out.push_str("        fn default() -> Self {\n");
        out.push_str("            Self {\n");
        for input in &lowered.inputs {
            let ident = sanitize_ident(input.name.as_str());
            if input.ccs.is_scalar() {
                out.push_str(&format!("                {ident}: 0.0,\n"));
            } else {
                out.push_str(&format!(
                    "                {ident}: vec![0.0; {}],\n",
                    input.ccs.nnz()
                ));
            }
        }
        for output in &lowered.outputs {
            let ident = sanitize_ident(output.name.as_str());
            if output.ccs.is_scalar() {
                out.push_str(&format!("                {ident}: 0.0,\n"));
            } else {
                out.push_str(&format!(
                    "                {ident}: vec![0.0; {}],\n",
                    output.ccs.nnz()
                ));
            }
        }
        out.push_str("            }\n");
        out.push_str("        }\n");
        out.push_str("    }\n");
        out.push_str(&format!("    impl {context_name} {{\n"));
        out.push_str("        pub fn new() -> Self { Self::default() }\n");
        out.push_str("        pub fn eval(&mut self) {\n");
        let mut call_args = Vec::new();
        for input in &lowered.inputs {
            let ident = sanitize_ident(input.name.as_str());
            if input.ccs.is_scalar() {
                call_args.push(format!("self.{ident}"));
            } else {
                call_args.push(format!("&self.{ident}"));
            }
        }
        for output in &lowered.outputs {
            let ident = sanitize_ident(output.name.as_str());
            call_args.push(format!("&mut self.{ident}"));
        }
        out.push_str(&format!("            eval({});\n", call_args.join(", ")));
        out.push_str("        }\n");
        out.push_str("    }\n");
    }

    out.push_str("}\n");
    format_rust_source(&out)
}

fn emit_ccs(name: &str, ccs: &sx_core::CCS, out: &mut String) {
    let upper = sanitize_ident(name).to_ascii_uppercase();
    let col_ptrs = format!("{upper}_COL_PTRS");
    let row_indices = format!("{upper}_ROW_INDICES");
    out.push_str(&format!(
        "    pub const {col_ptrs}: &[usize] = &{:?};\n",
        ccs.col_ptrs()
    ));
    out.push_str(&format!(
        "    pub const {row_indices}: &[usize] = &{:?};\n",
        ccs.row_indices()
    ));
    out.push_str(&format!(
        "    pub const {upper}_CCS: GeneratedCCS = GeneratedCCS {{ nrow: {}, ncol: {}, col_ptrs: {col_ptrs}, row_indices: {row_indices} }};\n",
        ccs.nrow(),
        ccs.ncol()
    ));
}

fn create_lljit() -> Result<llvm_sys::orc2::lljit::LLVMOrcLLJITRef> {
    let mut lljit = ptr::null_mut();
    let error = unsafe { LLVMOrcCreateLLJIT(&mut lljit, ptr::null_mut()) };
    consume_llvm_error(error).context("failed to create LLJIT")?;
    Ok(lljit)
}

fn attach_current_process_symbols(
    jit_dylib: llvm_sys::orc2::LLVMOrcJITDylibRef,
    global_prefix: i8,
) -> Result<()> {
    let mut generator = ptr::null_mut();
    let error = unsafe {
        LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
            &mut generator,
            global_prefix,
            None,
            ptr::null_mut(),
        )
    };
    consume_llvm_error(error).context("failed to create current-process search generator")?;
    unsafe { LLVMOrcJITDylibAddGenerator(jit_dylib, generator) };
    Ok(())
}

fn lookup_symbol_address(
    lljit: llvm_sys::orc2::lljit::LLVMOrcLLJITRef,
    symbol_name: &str,
) -> Result<LLVMOrcExecutorAddress> {
    let symbol = CString::new(symbol_name)?;
    let mut address = 0_u64;
    let first_attempt = unsafe { LLVMOrcLLJITLookup(lljit, &mut address, symbol.as_ptr()) };
    if first_attempt.is_null() {
        return Ok(address);
    }
    let first_error = llvm_error_message(first_attempt);
    let prefix = unsafe { LLVMOrcLLJITGetGlobalPrefix(lljit) };
    if prefix == 0 {
        return Err(first_error.context("failed to resolve JIT symbol"));
    }
    let prefixed_name = format!("{}{}", prefix as u8 as char, symbol_name);
    let prefixed = CString::new(prefixed_name)?;
    let retry = unsafe { LLVMOrcLLJITLookup(lljit, &mut address, prefixed.as_ptr()) };
    if retry.is_null() {
        Ok(address)
    } else {
        Err(llvm_error_message(retry).context(format!(
            "failed to resolve JIT symbol after retry; first lookup error: {}",
            first_error
        )))
    }
}

unsafe fn verify_module(module: LLVMModuleRef, phase: &str) -> Result<()> {
    let mut message = ptr::null_mut();
    if unsafe {
        LLVMVerifyModule(
            module,
            LLVMVerifierFailureAction::LLVMReturnStatusAction,
            &mut message,
        )
    } == 0
    {
        return Ok(());
    }
    let detail = unsafe { take_owned_message(message) }?;
    bail!("LLVM module verification failed {phase}: {detail}");
}

fn consume_llvm_error(error: LLVMErrorRef) -> Result<()> {
    if error.is_null() {
        Ok(())
    } else {
        Err(llvm_error_message(error))
    }
}

fn llvm_error_message(error: LLVMErrorRef) -> anyhow::Error {
    unsafe {
        let message = LLVMGetErrorMessage(error);
        let text = if message.is_null() {
            "unknown LLVM error".to_string()
        } else {
            let text = CStr::from_ptr(message).to_string_lossy().into_owned();
            LLVMDisposeErrorMessage(message);
            text
        };
        anyhow!(text)
    }
}

unsafe fn take_llvm_message(message: *mut i8) -> Result<String> {
    unsafe { take_owned_message(message) }
}

unsafe fn take_owned_message(message: *mut i8) -> Result<String> {
    if message.is_null() {
        bail!("LLVM returned a null message pointer");
    }
    let text = unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned();
    unsafe { LLVMDisposeMessage(message) };
    Ok(text)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::{
        AotWrapperOptions, CompiledJitFunction, FunctionCompileOptions, JitOptimizationLevel,
        LlvmTarget,
        emit_object_bytes_lowered, generate_aot_wrapper_module,
    };
    use sx_codegen::{Instruction, LoweredFunction, LoweredSubfunction, ValueRef, lower_function};
    use sx_core::{
        BinaryOp, CallPolicy, CallPolicyConfig, NamedMatrix, SXFunction, SXMatrix, UnaryOp,
    };

    fn named(name: &str, matrix: SXMatrix) -> NamedMatrix {
        NamedMatrix::new(name, matrix).expect("named matrix should be valid")
    }

    fn eval_instruction_sequence(
        instructions: &[Instruction],
        inputs: &[Vec<f64>],
        subfunctions: &[LoweredSubfunction],
    ) -> Vec<f64> {
        let mut temps = vec![
            0.0;
            instructions
                .iter()
                .flat_map(Instruction::output_temps)
                .copied()
                .max()
                .map_or(0, |max_temp| max_temp + 1)
        ];
        let resolve = |value: ValueRef, temps: &[f64], inputs: &[Vec<f64>]| match value {
            ValueRef::Const(value) => value,
            ValueRef::Temp(temp) => temps[temp],
            ValueRef::Input { slot, offset } => inputs[slot][offset],
        };
        for instruction in instructions {
            match instruction {
                Instruction::Unary { temp, op, input } => {
                    let input = resolve(*input, &temps, inputs);
                    temps[*temp] = match op {
                        UnaryOp::Abs => input.abs(),
                        UnaryOp::Sign => {
                            if input > 0.0 {
                                1.0
                            } else if input < 0.0 {
                                -1.0
                            } else {
                                0.0
                            }
                        }
                        UnaryOp::Floor => input.floor(),
                        UnaryOp::Ceil => input.ceil(),
                        UnaryOp::Round => input.round(),
                        UnaryOp::Trunc => input.trunc(),
                        UnaryOp::Sqrt => input.sqrt(),
                        UnaryOp::Exp => input.exp(),
                        UnaryOp::Log => input.ln(),
                        UnaryOp::Sin => input.sin(),
                        UnaryOp::Cos => input.cos(),
                        UnaryOp::Tan => input.tan(),
                        UnaryOp::Asin => input.asin(),
                        UnaryOp::Acos => input.acos(),
                        UnaryOp::Atan => input.atan(),
                        UnaryOp::Sinh => input.sinh(),
                        UnaryOp::Cosh => input.cosh(),
                        UnaryOp::Tanh => input.tanh(),
                        UnaryOp::Asinh => input.asinh(),
                        UnaryOp::Acosh => input.acosh(),
                        UnaryOp::Atanh => input.atanh(),
                    };
                }
                Instruction::Binary { temp, op, lhs, rhs } => {
                    let lhs = resolve(*lhs, &temps, inputs);
                    let rhs = resolve(*rhs, &temps, inputs);
                    temps[*temp] = match op {
                        BinaryOp::Add => lhs + rhs,
                        BinaryOp::Sub => lhs - rhs,
                        BinaryOp::Mul => lhs * rhs,
                        BinaryOp::Div => lhs / rhs,
                        BinaryOp::Pow => lhs.powf(rhs),
                        BinaryOp::Atan2 => lhs.atan2(rhs),
                        BinaryOp::Hypot => lhs.hypot(rhs),
                        BinaryOp::Mod => lhs % rhs,
                        BinaryOp::Copysign => lhs.copysign(rhs),
                    };
                }
                Instruction::Call {
                    temps: output_temps,
                    callee,
                    inputs: call_inputs,
                } => {
                    let callee = &subfunctions[*callee];
                    let mut callee_inputs = Vec::with_capacity(callee.inputs.len());
                    let mut linear = 0usize;
                    for slot in &callee.inputs {
                        let nnz = slot.ccs.nnz();
                        callee_inputs.push(
                            call_inputs[linear..linear + nnz]
                                .iter()
                                .map(|value| resolve(*value, &temps, inputs))
                                .collect::<Vec<_>>(),
                        );
                        linear += nnz;
                    }
                    let callee_outputs = eval_subfunction(callee, &callee_inputs, subfunctions);
                    for (temp, value) in output_temps
                        .iter()
                        .copied()
                        .zip(callee_outputs.into_iter().flatten())
                    {
                        temps[temp] = value;
                    }
                }
            }
        }
        temps
    }

    fn eval_subfunction(
        lowered: &LoweredSubfunction,
        inputs: &[Vec<f64>],
        subfunctions: &[LoweredSubfunction],
    ) -> Vec<Vec<f64>> {
        let temps = eval_instruction_sequence(&lowered.instructions, inputs, subfunctions);
        let resolve = |value: ValueRef, temps: &[f64], inputs: &[Vec<f64>]| match value {
            ValueRef::Const(value) => value,
            ValueRef::Temp(temp) => temps[temp],
            ValueRef::Input { slot, offset } => inputs[slot][offset],
        };
        lowered
            .output_values
            .iter()
            .map(|slot| {
                slot.iter()
                    .copied()
                    .map(|value| resolve(value, &temps, inputs))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn eval_lowered(lowered: &LoweredFunction, inputs: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let temps = eval_instruction_sequence(&lowered.instructions, inputs, &lowered.subfunctions);
        let resolve = |value: ValueRef, temps: &[f64]| match value {
            ValueRef::Const(value) => value,
            ValueRef::Temp(temp) => temps[temp],
            ValueRef::Input { slot, offset } => inputs[slot][offset],
        };
        lowered
            .output_values
            .iter()
            .map(|slot| {
                slot.iter()
                    .copied()
                    .map(|value| resolve(value, &temps))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn build_policy_bundle_function() -> SXFunction {
        let z = SXMatrix::sym_dense("z", 2, 1).expect("callee input should build");
        let phi = SXMatrix::scalar(z.nz(0).sqr() + z.nz(1).sin() + z.nz(0) * z.nz(1));
        let psi = SXMatrix::dense_column(vec![z.nz(0) * z.nz(1), z.nz(0) + z.nz(1)])
            .expect("callee output should build");
        let stage = SXFunction::new(
            "stage_cost",
            vec![named("z", z.clone())],
            vec![named("phi", phi), named("psi", psi)],
        )
        .expect("callee should build");

        let x = SXMatrix::sym_dense("x", 2, 1).expect("root input should build");
        let stage_a = stage.call(&[x.clone()]).expect("first call should build");
        let swapped = SXMatrix::dense_column(vec![x.nz(1), x.nz(0)]).expect("swap should build");
        let stage_b = stage.call(&[swapped]).expect("second call should build");

        let objective = SXMatrix::scalar(
            stage_a[0].nz(0) + stage_b[0].nz(0) + stage_a[1].nz(0).sqr() + stage_b[1].nz(1),
        );
        let constraints = SXMatrix::dense_column(vec![
            stage_a[1].nz(0) + stage_b[1].nz(1),
            stage_a[1].nz(1) - stage_b[1].nz(0),
        ])
        .expect("constraint bundle should build");
        let gradient = objective.gradient(&x).expect("gradient should build");
        let jacobian = constraints.jacobian(&x).expect("jacobian should build");
        let hessian = SXMatrix::scalar(objective.scalar_expr().expect("objective should be scalar"))
            .hessian(&x)
            .expect("hessian should build");

        SXFunction::new(
            "policy_bundle",
            vec![named("x", x)],
            vec![
                named("objective", objective),
                named("gradient", gradient),
                named("jacobian", jacobian),
                named("hessian", hessian),
            ],
        )
        .expect("root function should build")
    }

    fn compile_with_policy(function: &SXFunction, policy: CallPolicy) -> CompiledJitFunction {
        CompiledJitFunction::compile_function_with_options(
            function,
            FunctionCompileOptions {
                opt_level: JitOptimizationLevel::O0,
                call_policy: CallPolicyConfig {
                    default_policy: policy,
                    respect_function_overrides: true,
                },
            },
        )
        .expect("compilation should succeed")
    }

    #[test]
    fn llvm_jit_matches_lowered_reference_for_multi_io_kernel() {
        let x = SXMatrix::sym_dense("x", 2, 1).unwrap();
        let y = SXMatrix::sym_dense("y", 1, 1).unwrap();
        let output0 = SXMatrix::scalar((x.nz(0) + y.nz(0)) / (1.0 + x.nz(1)));
        let output1 = SXMatrix::dense_column(vec![x.nz(0) * x.nz(1), x.nz(0) - y.nz(0)]).unwrap();
        let function = SXFunction::new(
            "llvm_demo",
            vec![
                NamedMatrix::new("x", x).unwrap(),
                NamedMatrix::new("y", y).unwrap(),
            ],
            vec![
                NamedMatrix::new("objective", output0).unwrap(),
                NamedMatrix::new("vector_out", output1).unwrap(),
            ],
        )
        .unwrap();

        let lowered = lower_function(&function).unwrap();
        let compiled =
            CompiledJitFunction::compile_lowered(&lowered, JitOptimizationLevel::O3).unwrap();
        let mut context = compiled.create_context();
        context.input_mut(0).copy_from_slice(&[1.5, 0.25]);
        context.input_mut(1).copy_from_slice(&[0.75]);
        compiled.eval(&mut context);

        let expected = eval_lowered(&lowered, &[vec![1.5, 0.25], vec![0.75]]);
        assert_eq!(context.output(0).len(), expected[0].len());
        assert_eq!(context.output(1).len(), expected[1].len());
        for (lhs, rhs) in context.output(0).iter().zip(expected[0].iter()) {
            assert!((lhs - rhs).abs() <= 1e-12);
        }
        for (lhs, rhs) in context.output(1).iter().zip(expected[1].iter()) {
            assert!((lhs - rhs).abs() <= 1e-12);
        }
    }

    #[test]
    fn llvm_aot_object_emission_produces_nonempty_bytes() {
        let x = SXMatrix::sym_dense("x", 2, 1).unwrap();
        let value = SXMatrix::scalar(x.nz(0) * x.nz(1) + x.nz(0));
        let function = SXFunction::new(
            "llvm_aot_object",
            vec![NamedMatrix::new("x", x).unwrap()],
            vec![NamedMatrix::new("value", value).unwrap()],
        )
        .unwrap();
        let lowered = lower_function(&function).unwrap();
        let bytes =
            emit_object_bytes_lowered(&lowered, JitOptimizationLevel::O2, &LlvmTarget::Native)
                .unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn llvm_aot_wrapper_generation_includes_module_and_context() {
        let x = SXMatrix::sym_dense("x", 2, 1).unwrap();
        let output = SXMatrix::dense_column(vec![x.nz(0) + 1.0, x.nz(1) / 2.0]).unwrap();
        let function = SXFunction::new(
            "llvm_aot_wrapper",
            vec![NamedMatrix::new("x", x).unwrap()],
            vec![NamedMatrix::new("output", output).unwrap()],
        )
        .unwrap();
        let lowered = lower_function(&function).unwrap();
        let wrapper = generate_aot_wrapper_module(
            &lowered,
            &AotWrapperOptions {
                emit_doc_comments: true,
            },
        )
        .unwrap();

        assert!(wrapper.contains("pub mod llvm_aot_wrapper_llvm_aot"));
        assert!(wrapper.contains("unsafe extern \"C\""));
        assert!(wrapper.contains("fn __sx_codegen_llvm_llvm_aot_wrapper"));
        assert!(wrapper.contains("pub struct LlvmAotWrapperLlvmAotContext"));
    }

    #[test]
    fn llvm_jit_matches_lowered_reference_for_transcendentals() {
        let x = SXMatrix::sym_dense("x", 2, 1).unwrap();
        let output = SXMatrix::scalar(
            x.nz(0).sin()
                + x.nz(0).cos()
                + x.nz(0).tan()
                + x.nz(0).exp()
                + x.nz(0).log()
                + x.nz(0).sqrt()
                + x.nz(0).atan()
                + x.nz(0).sinh()
                + x.nz(0).tanh()
                + x.nz(0).powf(1.5)
                + x.nz(0).hypot(x.nz(1))
                + x.nz(0).atan2(x.nz(1))
                + x.nz(0).modulo(x.nz(1)),
        );
        let function = SXFunction::new(
            "llvm_transcendentals",
            vec![NamedMatrix::new("x", x).unwrap()],
            vec![NamedMatrix::new("y", output).unwrap()],
        )
        .unwrap();
        let lowered = lower_function(&function).unwrap();
        let compiled =
            CompiledJitFunction::compile_lowered(&lowered, JitOptimizationLevel::O2).unwrap();
        for point in [[1.7, 0.8], [0.9, 1.1], [2.3, 0.6]] {
            let expected = eval_lowered(&lowered, &[point.to_vec()]);
            let mut context = compiled.create_context();
            context.input_mut(0).copy_from_slice(&point);
            compiled.eval(&mut context);
            assert_eq!(context.output(0).len(), 1);
            assert!((context.output(0)[0] - expected[0][0]).abs() <= 1e-12);
        }
    }

    #[test]
    fn llvm_jit_matches_lowered_reference_for_transcendental_gradient_bundle() {
        let x = SXMatrix::sym_dense("x", 2, 1).unwrap();
        let objective = SXMatrix::scalar(
            x.nz(0).sin()
                + x.nz(0).exp()
                + x.nz(0).powf(1.5)
                + x.nz(0).hypot(x.nz(1))
                + x.nz(0).atan2(x.nz(1))
                + x.nz(1).log(),
        );
        let gradient = objective.gradient(&x).unwrap();
        let function = SXFunction::new(
            "llvm_transcendental_gradient_bundle",
            vec![NamedMatrix::new("x", x).unwrap()],
            vec![
                NamedMatrix::new("objective", objective).unwrap(),
                NamedMatrix::new("gradient", gradient).unwrap(),
            ],
        )
        .unwrap();
        let lowered = lower_function(&function).unwrap();
        let compiled =
            CompiledJitFunction::compile_lowered(&lowered, JitOptimizationLevel::O2).unwrap();

        for point in [[1.7, 0.8], [0.9, 1.1], [2.3, 0.6]] {
            let expected = eval_lowered(&lowered, &[point.to_vec()]);
            let mut context = compiled.create_context();
            context.input_mut(0).copy_from_slice(&point);
            compiled.eval(&mut context);
            assert_eq!(context.output(0).len(), expected[0].len());
            assert_eq!(context.output(1).len(), expected[1].len());
            for (lhs, rhs) in context.output(0).iter().zip(expected[0].iter()) {
                assert!((lhs - rhs).abs() <= 1e-12);
            }
            for (lhs, rhs) in context.output(1).iter().zip(expected[1].iter()) {
                assert!((lhs - rhs).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn llvm_call_policies_preserve_derivative_bundle_outputs() {
        let function = build_policy_bundle_function();
        let baseline = compile_with_policy(&function, CallPolicy::InlineAtCall);
        let policies = [
            CallPolicy::InlineAtCall,
            CallPolicy::InlineAtLowering,
            CallPolicy::InlineInLLVM,
            CallPolicy::NoInlineLLVM,
        ];

        for point in [[1.7, 0.8], [0.9, 1.1], [2.3, 0.6]] {
            let mut baseline_context = baseline.create_context();
            baseline_context.input_mut(0).copy_from_slice(&point);
            baseline.eval(&mut baseline_context);
            let expected = (0..baseline.lowered().outputs.len())
                .map(|slot| baseline_context.output(slot).to_vec())
                .collect::<Vec<_>>();

            for policy in policies {
                let compiled = compile_with_policy(&function, policy);
                let mut context = compiled.create_context();
                context.input_mut(0).copy_from_slice(&point);
                compiled.eval(&mut context);
                for (slot, expected_slot) in expected.iter().enumerate() {
                    assert_eq!(context.output(slot).len(), expected_slot.len());
                    for (lhs, rhs) in context.output(slot).iter().zip(expected_slot.iter()) {
                        assert!((lhs - rhs).abs() <= 1e-12, "policy={policy:?}, slot={slot}");
                    }
                }
            }
        }
    }

    #[test]
    fn llvm_call_policy_stats_match_lowering_shape() {
        let function = build_policy_bundle_function();

        let inline_at_call = compile_with_policy(&function, CallPolicy::InlineAtCall);
        assert!(inline_at_call.lowered().subfunctions.is_empty());
        assert!(inline_at_call.lowered().stats.inlines_at_call >= 2);
        assert_eq!(inline_at_call.lowered().stats.llvm_call_instructions_emitted, 0);

        let inline_at_lowering = compile_with_policy(&function, CallPolicy::InlineAtLowering);
        assert!(inline_at_lowering.lowered().subfunctions.is_empty());
        assert!(inline_at_lowering.lowered().stats.inlines_at_lowering >= 2);
        assert_eq!(
            inline_at_lowering.lowered().stats.llvm_call_instructions_emitted,
            0
        );

        for policy in [CallPolicy::InlineInLLVM, CallPolicy::NoInlineLLVM] {
            let compiled = compile_with_policy(&function, policy);
            assert!(!compiled.lowered().subfunctions.is_empty());
            assert!(
                compiled
                    .lowered()
                    .subfunctions
                    .iter()
                    .all(|subfunction| subfunction.call_policy == policy)
            );
            assert!(compiled.lowered().stats.call_site_count >= 2);
            assert_eq!(
                compiled.lowered().stats.llvm_subfunctions_emitted,
                compiled.lowered().subfunctions.len()
            );
            assert!(compiled.lowered().stats.llvm_call_instructions_emitted >= 2);
        }
    }
}
