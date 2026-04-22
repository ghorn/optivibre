use std::ffi::c_char;

use spral_src as _;

unsafe extern "C" {
    fn dtrsv_(
        uplo: *const c_char,
        trans: *const c_char,
        diag: *const c_char,
        n: *const i32,
        a: *const f64,
        lda: *const i32,
        x: *mut f64,
        incx: *const i32,
    );
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let delta = (actual - expected).abs();
        assert!(
            delta <= 1e-12,
            "dtrsv mismatch at {index}: actual={actual:?} expected={expected:?} delta={delta:e}"
        );
    }
}

#[test]
fn openblas_dtrsv_lower_trans_unit_matches_reference() {
    // SPRAL APP forward/backward solves reach OpenBLAS through dtrsv.
    // This guards the source-built OpenBLAS target/threading combination
    // before it is allowed into native SSIDS parity evidence.
    let n = 33_i32;
    let lda = n;
    let incx = 1_i32;
    let dimension = n as usize;
    let mut lower = vec![0.0; dimension * dimension];
    for col in 0..dimension {
        for row in col..dimension {
            lower[col * dimension + row] = if row == col {
                1.0
            } else {
                let raw = ((row * 17 + col * 31 + 7) % 23) as f64 - 11.0;
                raw / 1024.0
            };
        }
    }

    let expected = (0..dimension)
        .map(|index| f64::from((index % 11) as i16 - 5) / 8.0)
        .collect::<Vec<_>>();
    let mut rhs = vec![0.0; dimension];
    for row in 0..dimension {
        let mut sum = 0.0;
        for col in 0..dimension {
            sum += lower[row * dimension + col] * expected[col];
        }
        rhs[row] = sum;
    }

    let uplo = b"L";
    let trans = b"T";
    let diag = b"U";
    unsafe {
        dtrsv_(
            uplo.as_ptr().cast(),
            trans.as_ptr().cast(),
            diag.as_ptr().cast(),
            &n,
            lower.as_ptr(),
            &lda,
            rhs.as_mut_ptr(),
            &incx,
        );
    }

    assert_close(&rhs, &expected);
}
