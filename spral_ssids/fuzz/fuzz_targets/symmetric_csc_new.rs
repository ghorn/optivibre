#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let dimension = usize::from(data[0] % 24);
    let ptr_count = dimension.saturating_add(1);
    if data.len() < 1 + ptr_count {
        return;
    }

    let col_ptrs = data[1..1 + ptr_count]
        .iter()
        .map(|byte| usize::from(*byte % 64))
        .collect::<Vec<_>>();
    let row_indices = data[1 + ptr_count..]
        .iter()
        .map(|byte| usize::from(*byte % 64))
        .collect::<Vec<_>>();
    let values = if row_indices.is_empty() {
        None
    } else {
        Some(vec![1.0; row_indices.len().saturating_sub(1)])
    };
    let _ = spral_ssids::SymmetricCscMatrix::new(
        dimension,
        &col_ptrs,
        &row_indices,
        values.as_deref(),
    );
});
