#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let vertex_count = usize::from(data[0] % 32);
    let offset_count = vertex_count.saturating_add(1);
    if data.len() < 1 + offset_count {
        return;
    }

    let offsets = data[1..1 + offset_count]
        .iter()
        .map(|byte| usize::from(*byte % 64))
        .collect::<Vec<_>>();
    let neighbors = data[1 + offset_count..]
        .iter()
        .map(|byte| usize::from(*byte % 64))
        .collect::<Vec<_>>();
    let _ = metis_ordering::CsrGraph::new(offsets, neighbors);
});
