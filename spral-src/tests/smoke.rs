use std::ffi::c_void;
use std::process::Command;
use std::ptr;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SpralSsidsOptions {
    array_base: i32,
    print_level: i32,
    unit_diagnostics: i32,
    unit_error: i32,
    unit_warning: i32,
    ordering: i32,
    nemin: i32,
    ignore_numa: bool,
    use_gpu: bool,
    min_gpu_work: i64,
    max_load_inbalance: f32,
    gpu_perf_coeff: f32,
    scaling: i32,
    small_subtree_threshold: i64,
    cpu_block_size: i32,
    action: bool,
    pivot_method: i32,
    small: f64,
    u: f64,
    unused: [u8; 80],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SpralSsidsInform {
    flag: i32,
    matrix_dup: i32,
    matrix_missing_diag: i32,
    matrix_outrange: i32,
    matrix_rank: i32,
    maxdepth: i32,
    maxfront: i32,
    num_delay: i32,
    num_factor: i64,
    num_flops: i64,
    num_neg: i32,
    num_sup: i32,
    num_two: i32,
    stat: i32,
    cuda_error: i32,
    cublas_error: i32,
    maxsupernode: i32,
    unused: [u8; 76],
}

impl Default for SpralSsidsInform {
    fn default() -> Self {
        Self {
            flag: 0,
            matrix_dup: 0,
            matrix_missing_diag: 0,
            matrix_outrange: 0,
            matrix_rank: 0,
            maxdepth: 0,
            maxfront: 0,
            num_delay: 0,
            num_factor: 0,
            num_flops: 0,
            num_neg: 0,
            num_sup: 0,
            num_two: 0,
            stat: 0,
            cuda_error: 0,
            cublas_error: 0,
            maxsupernode: 0,
            unused: [0; 76],
        }
    }
}

unsafe extern "C" {
    fn spral_ssids_default_options(options: *mut SpralSsidsOptions);
    fn spral_ssids_analyse(
        check: bool,
        n: i32,
        order: *mut i32,
        ptr: *const i64,
        row: *const i32,
        val: *const f64,
        akeep: *mut *mut c_void,
        options: *const SpralSsidsOptions,
        inform: *mut SpralSsidsInform,
    );
    fn spral_ssids_factor(
        posdef: bool,
        ptr: *const i64,
        row: *const i32,
        val: *const f64,
        scaling: *mut f64,
        akeep: *mut c_void,
        fkeep: *mut *mut c_void,
        options: *const SpralSsidsOptions,
        inform: *mut SpralSsidsInform,
    );
    fn spral_ssids_solve1(
        job: i32,
        rhs: *mut f64,
        akeep: *mut c_void,
        fkeep: *mut c_void,
        options: *const SpralSsidsOptions,
        inform: *mut SpralSsidsInform,
    );
    fn spral_ssids_free(akeep: *mut *mut c_void, fkeep: *mut *mut c_void) -> i32;
}

fn rerun_with_openmp_cancellation_if_needed(test_name: &str) -> bool {
    if std::env::var_os("SPRAL_SRC_SMOKE_CHILD").is_some()
        || std::env::var("OMP_CANCELLATION").is_ok_and(|value| value.eq_ignore_ascii_case("true"))
    {
        return false;
    }

    let status = Command::new(std::env::current_exe().expect("current test executable"))
        .arg("--exact")
        .arg(test_name)
        .arg("--nocapture")
        .env("SPRAL_SRC_SMOKE_CHILD", "1")
        .env("OMP_CANCELLATION", "true")
        .status()
        .expect("failed to rerun SPRAL smoke test with OMP_CANCELLATION=true");
    assert!(
        status.success(),
        "SPRAL smoke test child failed with status {status}"
    );
    true
}

#[test]
fn source_built_spral_ssids_factor_solve_smoke() {
    if rerun_with_openmp_cancellation_if_needed("source_built_spral_ssids_factor_solve_smoke") {
        return;
    }

    unsafe {
        let mut options = std::mem::zeroed::<SpralSsidsOptions>();
        spral_ssids_default_options(&mut options);
        options.array_base = 0;
        options.ignore_numa = true;
        options.use_gpu = false;
        options.print_level = -1;

        let ptr = [0_i64, 2, 3];
        let row = [0_i32, 1, 1];
        let val = [2.0_f64, -1.0, 2.0];
        let mut akeep = ptr::null_mut();
        let mut fkeep = ptr::null_mut();
        let mut inform = SpralSsidsInform::default();

        spral_ssids_analyse(
            false,
            2,
            ptr::null_mut(),
            ptr.as_ptr(),
            row.as_ptr(),
            val.as_ptr(),
            &mut akeep,
            &options,
            &mut inform,
        );
        assert!(
            inform.flag >= 0,
            "spral_ssids_analyse failed with flag {}",
            inform.flag
        );

        spral_ssids_factor(
            false,
            ptr.as_ptr(),
            row.as_ptr(),
            val.as_ptr(),
            ptr::null_mut(),
            akeep,
            &mut fkeep,
            &options,
            &mut inform,
        );
        assert!(
            inform.flag >= 0,
            "spral_ssids_factor failed with flag {}",
            inform.flag
        );

        let mut rhs = [1.0_f64, 0.0];
        spral_ssids_solve1(0, rhs.as_mut_ptr(), akeep, fkeep, &options, &mut inform);
        assert!(
            inform.flag >= 0,
            "spral_ssids_solve1 failed with flag {}",
            inform.flag
        );
        assert!((rhs[0] - 2.0 / 3.0).abs() < 1.0e-10, "rhs={rhs:?}");
        assert!((rhs[1] - 1.0 / 3.0).abs() < 1.0e-10, "rhs={rhs:?}");

        assert_eq!(spral_ssids_free(&mut akeep, &mut fkeep), 0);
    }
}

#[test]
fn public_config_declares_no_system_solver_math_fallbacks() {
    assert!(std::hint::black_box(spral_src::OPENMP_REQUIRED));
    assert!(!std::hint::black_box(
        spral_src::SYSTEM_SOLVER_MATH_FALLBACKS
    ));
    assert!(spral_src::config_summary().contains("system_solver_math_fallbacks=false"));
}
