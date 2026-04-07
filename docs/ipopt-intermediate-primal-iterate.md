# IPOPT Intermediate Callback Proposal

## Problem

The underlying IPOPT intermediate callback receives `IpoptData*` and
`IpoptCalculatedQuantities*`, which are enough to access the current primal
iterate. Our current `cnlp -> ipopt-sys -> ipopt` stack was only forwarding
scalar diagnostics such as objective value, infeasibilities, barrier parameter,
step norm, and line-search count.

That meant higher layers could not reconstruct typed trajectories during IPOPT
iterations, even though the information already existed in the backend.

## Proposed Upstream Change

Make the intermediate callback surface additive by forwarding the current primal
iterate `x` through the full stack:

1. `cnlp` C API
   - Extend `CNLP_Intermediate_CB` with:
     - `CNLP_Index x_count`
     - `const CNLP_Number* x`
2. `cnlp` C++ bridge
   - Read `ip_data->curr()->x()`
   - Materialize a dense view via `DenseVector::ExpandedValues()`
   - Pass `(x_count, x_ptr)` into the user callback
3. `ipopt-sys`
   - Regenerate bindgen output from the updated C header
4. `ipopt` Rust crate
   - Extend `IntermediateCallbackData` with `x: &'a [Number]`
5. Higher-level callers
   - Clone or project `x` as needed for iteration reporting

## Why This Shape

- It is additive and backward-compatible at the conceptual API level.
- It keeps ownership in IPOPT and borrows the iterate for the duration of the
  callback only.
- It avoids inventing a second callback or forcing every client to call back
  into opaque IPOPT internals.
- It is sufficient for typed wrappers to reconstruct trajectories, plot live
  iterates, and provide richer diagnostics.

## Current Local Patch

The local vendored patch in this repo implements exactly that pass-through:

- `third_party/ipopt-sys/cnlp/src/c_api.h`
- `third_party/ipopt-sys/cnlp/src/nlp.cpp`
- `third_party/ipopt-sys/build.rs`
- `third_party/ipopt-sys/src/lib.rs`
- `third_party/ipopt/src/lib.rs`

On top of that, the optimization and OCP layers now retain the IPOPT iterate
vector in `IpoptIterationSnapshot`, which enables typed OCP iteration callbacks
to stream live trajectories instead of only final results.
