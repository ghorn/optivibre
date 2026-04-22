#![no_main]

use libfuzzer_sys::fuzz_target;

fn lower_csc_from_pairs(
    dimension: usize,
    pairs: &[(usize, usize, f64)],
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut columns = vec![std::collections::BTreeMap::new(); dimension];
    for (index, rows) in columns.iter_mut().enumerate() {
        rows.insert(index, 1.0);
    }
    for &(lhs, rhs, value) in pairs {
        let row = lhs.max(rhs);
        let col = lhs.min(rhs);
        columns[col].insert(row, value);
    }

    let mut col_ptrs = Vec::with_capacity(dimension + 1);
    let mut row_indices = Vec::new();
    let mut values = Vec::new();
    col_ptrs.push(0);
    for rows in columns {
        for (row, value) in rows {
            row_indices.push(row);
            values.push(value);
        }
        col_ptrs.push(row_indices.len());
    }
    (col_ptrs, row_indices, values)
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let dimension = usize::from((data[0] % 19) + 1);
    let entries = data[1..]
        .chunks_exact(3)
        .filter_map(|chunk| {
            let lhs = usize::from(chunk[0]) % dimension;
            let rhs = usize::from(chunk[1]) % dimension;
            let value = (f64::from(chunk[2]) - 128.0) / 64.0;
            (lhs != rhs).then_some((lhs, rhs, value))
        })
        .collect::<Vec<_>>();
    let (col_ptrs, row_indices, values) = lower_csc_from_pairs(dimension, &entries);
    if let Ok(matrix) =
        ssids_rs::SymmetricCscMatrix::new(dimension, &col_ptrs, &row_indices, Some(&values))
        && let Ok((symbolic, _)) = ssids_rs::analyse(matrix, &ssids_rs::SsidsOptions::default())
    {
        let _ = ssids_rs::factorize(
            matrix,
            &symbolic,
            &ssids_rs::NumericFactorOptions::default(),
        );
    }
});
