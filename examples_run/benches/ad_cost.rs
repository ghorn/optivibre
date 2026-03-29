#![cfg(feature = "ad-bench-artifacts")]

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use examples_source::{
    AD_FORWARD_SIZE, AD_HESSIAN_SIZE, AD_JACOBIAN_SIZE, AD_REVERSE_SIZE, ad_cost_cases,
    build_forward_sweep_case, build_hessian_case, build_jacobian_case, build_reverse_gradient_case,
};
use sx_codegen::lower_function;

include!(concat!(env!("OUT_DIR"), "/generated_ad_cost_llvm_aot.rs"));

fn fill_sequence(values: &mut [f64]) {
    for (idx, value) in values.iter_mut().enumerate() {
        *value = 0.05 + idx as f64 * 0.0025;
    }
}

fn report_operation_counts() {
    for case in ad_cost_cases().unwrap() {
        let original_ops = lower_function(&case.original).unwrap().instructions.len();
        let augmented_ops = lower_function(&case.augmented).unwrap().instructions.len();
        let ratio = augmented_ops as f64 / original_ops as f64;
        let normalized_ratio = ratio / case.sweep_count as f64;
        println!(
            "ad-cost-report case={} scenario={:?} size={} original_ops={} augmented_ops={} ratio={ratio:.3} normalized_ratio={normalized_ratio:.3}",
            case.key, case.scenario, case.size, original_ops, augmented_ops,
        );
    }
}

fn benchmark_symbolic_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbolic_construction");

    group.bench_function(BenchmarkId::new("reverse_gradient", AD_REVERSE_SIZE), |b| {
        b.iter(|| black_box(build_reverse_gradient_case(AD_REVERSE_SIZE).unwrap()))
    });
    group.bench_function(BenchmarkId::new("forward_sweep", AD_FORWARD_SIZE), |b| {
        b.iter(|| black_box(build_forward_sweep_case(AD_FORWARD_SIZE).unwrap()))
    });
    group.bench_function(BenchmarkId::new("jacobian", AD_JACOBIAN_SIZE), |b| {
        b.iter(|| black_box(build_jacobian_case(AD_JACOBIAN_SIZE).unwrap()))
    });
    group.bench_function(BenchmarkId::new("hessian", AD_HESSIAN_SIZE), |b| {
        b.iter(|| black_box(build_hessian_case(AD_HESSIAN_SIZE).unwrap()))
    });

    group.finish();
}

fn benchmark_generated_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("generated_evaluation");

    let mut reverse_original =
        ad_reverse_gradient_original_llvm_aot::AdReverseGradientOriginalLlvmAotContext::default();
    fill_sequence(&mut reverse_original.x);
    group.bench_function(BenchmarkId::new("reverse_gradient", "original"), |b| {
        b.iter(|| {
            reverse_original.eval();
            black_box(reverse_original.value)
        })
    });

    let mut reverse_augmented = ad_reverse_gradient_with_gradient_llvm_aot::AdReverseGradientWithGradientLlvmAotContext::default();
    fill_sequence(&mut reverse_augmented.x);
    group.bench_function(BenchmarkId::new("reverse_gradient", "with_gradient"), |b| {
        b.iter(|| {
            reverse_augmented.eval();
            black_box(reverse_augmented.gradient[0])
        })
    });

    let mut forward_original =
        ad_forward_sweep_original_llvm_aot::AdForwardSweepOriginalLlvmAotContext {
            t: 0.75,
            ..Default::default()
        };
    group.bench_function(BenchmarkId::new("forward_sweep", "original"), |b| {
        b.iter(|| {
            forward_original.eval();
            black_box(forward_original.outputs[0])
        })
    });

    let mut forward_augmented =
        ad_forward_sweep_with_directional_llvm_aot::AdForwardSweepWithDirectionalLlvmAotContext {
            t: 0.75,
            ..Default::default()
        };
    group.bench_function(BenchmarkId::new("forward_sweep", "with_directional"), |b| {
        b.iter(|| {
            forward_augmented.eval();
            black_box(forward_augmented.directional[0])
        })
    });

    let mut jacobian_original =
        ad_jacobian_original_llvm_aot::AdJacobianOriginalLlvmAotContext::default();
    fill_sequence(&mut jacobian_original.x);
    group.bench_function(BenchmarkId::new("jacobian", "original"), |b| {
        b.iter(|| {
            jacobian_original.eval();
            black_box(jacobian_original.outputs[0])
        })
    });

    let mut jacobian_augmented =
        ad_jacobian_with_jacobian_llvm_aot::AdJacobianWithJacobianLlvmAotContext::default();
    fill_sequence(&mut jacobian_augmented.x);
    group.bench_function(BenchmarkId::new("jacobian", "with_jacobian"), |b| {
        b.iter(|| {
            jacobian_augmented.eval();
            black_box(jacobian_augmented.jacobian[0])
        })
    });

    let mut hessian_original =
        ad_hessian_original_llvm_aot::AdHessianOriginalLlvmAotContext::default();
    fill_sequence(&mut hessian_original.x);
    group.bench_function(BenchmarkId::new("hessian", "original"), |b| {
        b.iter(|| {
            hessian_original.eval();
            black_box(hessian_original.value)
        })
    });

    let mut hessian_augmented =
        ad_hessian_with_hessian_llvm_aot::AdHessianWithHessianLlvmAotContext::default();
    fill_sequence(&mut hessian_augmented.x);
    group.bench_function(BenchmarkId::new("hessian", "with_hessian"), |b| {
        b.iter(|| {
            hessian_augmented.eval();
            black_box(hessian_augmented.hessian[0])
        })
    });

    group.finish();
}

fn benchmark_costs(c: &mut Criterion) {
    report_operation_counts();
    benchmark_symbolic_construction(c);
    benchmark_generated_evaluation(c);
}

criterion_group!(benches, benchmark_costs);
criterion_main!(benches);
