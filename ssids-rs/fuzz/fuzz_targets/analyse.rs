#![no_main]

use libfuzzer_sys::fuzz_target;

fn lower_csc_from_pairs(dimension: usize, pairs: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
    let mut columns = vec![Vec::new(); dimension];
    for (index, rows) in columns.iter_mut().enumerate() {
        rows.push(index);
    }
    for &(lhs, rhs) in pairs {
        let row = lhs.max(rhs);
        let col = lhs.min(rhs);
        columns[col].push(row);
    }

    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::new();
    col_ptrs.push(0);
    for mut rows in columns {
        rows.sort_unstable();
        rows.dedup();
        row_indices.extend(rows);
        col_ptrs.push(row_indices.len());
    }
    (col_ptrs, row_indices)
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let dimension = usize::from((data[0] % 23) + 1);
    let edges = data[1..]
        .chunks_exact(2)
        .filter_map(|pair| {
            let lhs = usize::from(pair[0]) % dimension;
            let rhs = usize::from(pair[1]) % dimension;
            (lhs != rhs).then_some((lhs, rhs))
        })
        .collect::<Vec<_>>();
    let (col_ptrs, row_indices) = lower_csc_from_pairs(dimension, &edges);
    if let Ok(matrix) = ssids_rs::SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, None)
    {
        let _ = ssids_rs::analyse(matrix, &ssids_rs::SsidsOptions::default());
    }
});
