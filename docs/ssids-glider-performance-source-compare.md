# SSIDS Glider Performance Source Comparison

This note compares the two scoped hot areas from the glider factor profile:

- small-leaf numeric subtrees;
- APP dense-front factor/update.

The goal is performance parity without changing observable factor/solve behavior.
Native references are from SPRAL `v2025.09.18` under the source-built
`spral-src` tree. Rust references are current `ssids-rs/src/lib.rs`.

## Current Glider Evidence

Latest focused in-process profile after enabling the source-shaped small-leaf
path, removing the small-leaf square-to-packed-to-square contribution detour,
removing Rust-only finite checks from the small-leaf TPP apply helpers, and
porting the hot small-leaf TPP/contribution loops to source-shaped pointer
microkernels, adding aligned small-leaf TPP native-kernel parity coverage, and
copying solve-panel column tails from node-local `lcol` in contiguous slices.
The current path also emits small-leaf factor output directly into subtree
output and packs generated contribution only at the subtree root boundary.
The native-kernel parity shim now discovers the `spral-src` upstream checkout
under `target/{release,debug}/build`, uses the same GCC-flavored toolchain shape
as the source-built SPRAL tree for source-clone kernel traces, and the
small-leaf path reuses a subtree-local numeric work buffer in the same spirit as
SPRAL `Workspace`. Hot small-leaf TPP helpers are forced inline; this preserves
the exact operation order but lets the compiler optimize the tight helper calls.

| path | factor median | solve median | notes |
| --- | ---: | ---: | --- |
| native SPRAL | `1.009ms` | `2.086ms` | source-built `spral-src`, glider in-process median |
| Rust `SpralMatching`, profiled | `1.416ms` | `2.374ms` | includes Rust bucket timers; use for attribution |
| Rust `SpralMatching`, unprofiled | `1.189ms` | `2.067ms` | NLIP-like total with no Rust bucket timers |

The in-process replay keeps the exact augmented solution delta at
`6.938894e-18`. Correctness is tight. The real NLIP-like solve path is now
comparable to native on this replay, while factor still trails native by about
`179.958us` (`1.178x`). The larger profiled gap is instrumentation overhead plus
real kernel cost, so Rust-only buckets below are attribution, not native
bucket comparisons.

Rust small-leaf telemetry on the same glider replay:

| metric | value |
| --- | ---: |
| small-leaf subtrees | `2` |
| small-leaf candidate fronts | `235` |
| small-leaf APP fronts | `77` |
| small-leaf columns | `3471` |
| small-leaf dense entries | `179082` |
| small-leaf TPP | `614.968us` |
| small-leaf pivot factor | `328.553us` |
| small-leaf pivot search | `76.265us` |
| small-leaf contribution GEMM | `84.250us` |
| small-leaf contribution pack | `0ns` |
| small-leaf solve panel extraction | `75.712us` |
| small-leaf output append | `4.627us` |

The native sample shows the factor path spending material time in
`SmallLeafNumericSubtree -> ldlt_tpp_factor -> host_gemm/dgemm`. Rust now routes
the same candidate fronts through an active source-shaped small-leaf path, but
its TPP/contribution kernels are scalar Rust implementations rather than
native's optimized kernel/BLAS-backed path.

## Part 1: Small-Leaf Numeric Subtrees

### Native Source Shape

`SymbolicSubtree.hxx:57-84` classifies small leaf subtrees:

- counts `sum_k (nrow-k)^2`;
- forces nodes with `contrib.size() > 0` above the small-subtree threshold;
- starts at leaves, walks parents until the accumulated flop threshold is reached;
- records `small_leafs_` and marks `nodes_[i].insmallleaf = true`.

`NumericSubtree.hxx:96-159` handles those subtrees before singleton nodes:

- creates one `SmallLeafNumericSubtree` task per small leaf;
- the task depends on the parent node storage;
- the later singleton loop skips `symb_[ni].insmallleaf`.

`SmallLeafNumericSubtree.hxx:192-219` runs a leaf chain in node order:

- `assemble_pre`;
- `factor_node`;
- `assemble_post`.

`SmallLeafNumericSubtree.hxx:223-343` assembles directly into node-local storage:

- allocates aligned contiguous `node.lcol` as `(ldl + 2) * ncol`;
- zeros only factor storage;
- allocates `node.contrib` separately and does not zero it on the normal path;
- imports child delays directly into the parent node;
- maps child contribution entries directly into `lcol` when they affect fully summed columns.

`SmallLeafNumericSubtree.hxx:346-379` factors each node:

- always calls `ldlt_tpp_factor` for the indefinite small-leaf path;
- forms contribution with `calcLD<OP_N>` and `host_gemm(OP_N, OP_T)`;
- stores delays and statistics in the existing node.

`SmallLeafNumericSubtree.hxx:401-435` assembles generated contribution into
`node.contrib`, then frees child contribution blocks.

### Rust Source Shape

`ssids-rs/src/lib.rs` now classifies SPRAL-style small-leaf candidates and uses
the subtree root to select active small-leaf execution.

Important limitation: native classification includes a `contrib.size() > 0`
threshold guard. The Rust classifier now has an explicit
`has_contribution_inputs` guard and a unit test for that branch, but production
front-tree construction still defaults it to false because Rust does not yet
model native parttree `contrib_idx` inputs. That is good enough for current
glider execution, but a future minimized witness with native parttree
contribution inputs should still add the exact source map.

`ssids-rs/src/lib.rs` now has an active source-shaped small-leaf path that:

- allocates aligned node-local `lcol` as `(ldl + 2) * ncol`;
- imports child delayed rows into the parent node-local row order before the
  parent pivots;
- assembles child contributions into parent `lcol` before the parent pivot;
- factors the `m x n` node with a Rust port of `ldlt_tpp_factor`;
- stores `D` after `n * ldl`;
- forms generated contribution with `calcLD`/`host_gemm` semantics only when
  `nelim > 0`;
- frees no-elimination leaf contribution buffers and zeroes no-elimination
  internal-node contribution buffers like native `factor_node`;
- frees child contribution blocks after `assemble_post` consumes them;
- post-assembles the remaining child generated contribution.

The generic recursive path still handles non-small-leaf fronts. Planned
small-leaf execution no longer silently falls back on internal layout guard
failures; those now fail closed so a missed source branch becomes a minimized
witness. There is also no private enable/disable gate around planned small-leaf
dispatch anymore:

- recursively factors children;
- moves child contributions as `ContributionBlock` values;
- rebuilds `local_rows` and a zero-filled dense front per node;
- accumulates child packed contribution blocks into that dense front;
- calls `factorize_dense_front`;
- appends `FactorColumn`, block records, solve panels, and contributions into shared output.

### Concrete Implementation Difference

Native small-leaf processing is a node-local in-place pipeline:

```text
node.lcol / node.contrib / node.perm
  -> assemble_pre
  -> ldlt_tpp_factor
  -> calcLD + host_gemm contribution
  -> assemble_post into parent-facing contribution
```

Rust processing for non-small-leaf fronts is still a generic front
materialization pipeline:

```text
child result objects
  -> local row-set construction
  -> zero-filled local dense front
  -> packed child contribution replay
  -> APP or TPP dense factor
  -> separate factor-column and solve-panel records
```

The major implementation difference is now narrower than before. Rust has the
same small-leaf execution branch and storage shape for current glider fronts,
including source-shaped delayed child import for minimized delayed witnesses.
Rust also now mirrors native `ldlt_tpp_factor`'s full trailing
`host_gemm(OP_N, OP_T)` rectangle after each accepted 1x1/2x2 TPP pivot in both
the small-leaf path and the generic/root TPP tail, including native's
normally-unused row-before-column writes. Native's remaining advantage is mostly
kernel/storage efficiency: optimized `ldlt_tpp_factor`, BLAS-backed contribution
formation and some Rust solve-panel conversion overhead. A gated deep Rust split with
`SPRAL_SSIDS_SMALL_LEAF_DEEP_PROFILE=1` currently attributes the small-leaf
pivot-factor bucket mostly to the trailing rank update (`~232us`) rather than
multiplier scaling (`~51us`); that deep mode adds per-pivot timer overhead and
should not be used for side-by-side totals.

### Recommended Port Boundary

The real Rust small-leaf numeric path is now active for current subtrees, with
delayed-row import, no-elimination contribution free/zero behavior, and
post-assembly child contribution freeing covered by deterministic witnesses. The
small-leaf TPP prefix-trace fixture and the dense TPP full-storage fixtures now
compare the full native `host_gemm(OP_N, OP_T)` target rectangle, so the previous
lower-only update shape is closed for both small-leaf and generic/root TPP. The
next boundary is performance-parity work inside that branch:

1. Continue optimizing the source-shaped full-rectangle TPP trailing update
   loop; it remains the largest small-leaf bucket after pointer-shaped rank
   updates and max-scan unrolling. A simple branch-split/row-unroll version was
   rejected because it preserved correctness but regressed the side-by-side
   glider factor profile. The stricter
   `small_leaf_aligned_tpp_prefix_trace_matches_native_kernel` fixture now
   localizes the full native update rectangle after each accepted pivot.
2. Revisit contribution formation only after TPP narrows further; the
   row-blocked pointer microkernel keeps the GEMM-equivalent bucket near
   `85us`, but native `host_gemm` is still faster.
3. Delayed-row import is now ported and covered by a deterministic parent/child
   witness. No-elimination contribution free/zero behavior and post-assembly
   child contribution freeing are also ported and covered. The remaining
   small-leaf implementation differences are native block/workspace storage
   details and any future source branch exposed by a minimized witness.
4. Keep exact replay, native bitwise parity, matching/scaling parity, and dense
   sweeps as the guardrails for every kernel change.

## Part 2: APP Dense-Front Factor/Update

### Native Source Shape

`factor.hxx:58-63` sends non-TPP indefinite nodes to `ldlt_app_factor`.
If APP does not eliminate the whole node, `factor.hxx:74-103` finishes the tail
with `ldlt_tpp_factor` and forms contribution with `calcLD + host_gemm`.

`ldlt_app.cxx:962-1023` factors the diagonal block:

- recurses until the inner block size is 32;
- for a full aligned 32-column block, calls `block_ldlt<32>`;
- for partial or unaligned blocks, calls `ldlt_tpp_factor`.

`block_ldlt.hxx:290-365` is the fixed 32-column APP kernel:

- repeatedly calls `find_maxloc`;
- chooses 1x1 or 2x2 pivots;
- swaps inside the block;
- updates the fixed-size block workspace.

`ldlt_app.cxx:1585-1695` drives block-column elimination:

- factor diagonal block;
- apply pivots to off-diagonal blocks;
- update passed pivot counts;
- adjust accepted columns;
- restore failed rows/columns from `CopyBackup`;
- update trailing block grid.

`ldlt_app.cxx:1085-1185` performs trailing and contribution updates with
`calcLD` plus `host_gemm`. The hot arithmetic is delegated to BLAS-like kernels
over native block tiles.

### Rust Source Shape

`ssids-rs/src/lib.rs:6232-6622` implements the same APP behavior in flattened
dense-front storage:

- backs up the trailing lower triangle for each 32-pivot panel;
- finds APP pivots with `dense_find_maxloc`;
- applies 1x1/2x2 pivots into a 32-column scratch area;
- applies block pivots to trailing rows;
- scans for failed trailing columns;
- restores the rejected suffix from backup;
- recomputes the accepted-prefix update;
- materializes factor columns from the dense front.

`ssids-rs/src/lib.rs:4415-4565` is the Rust accepted-prefix update:

- builds an `LD` workspace;
- updates the remaining lower triangle with scalar or NEON-specific loops;
- intentionally mirrors native rounding for known witnesses.

`ssids-rs/src/lib.rs:5526-5558` materializes `FactorColumn` values from dense
front columns. The latest cleanup moved TPP materialization to this same style,
which reduced glider factor time and solve-panel build time.

`ssids-rs/src/lib.rs:3245-3318` now mirrors the local non-AVX
`block_ldlt.hxx::find_maxloc` lane split without the old per-column division.
The optimized scan keeps strict-greater tie behavior and is pinned by a local
source-shape regression test.

### Concrete Implementation Difference

Native APP is block-grid oriented:

```text
Block::factor
  -> block_ldlt<32> or TPP
  -> apply_pivot_app on block tiles
  -> ColumnData adjust/restore
  -> Block::update via calcLD + host_gemm
```

Rust APP is flattened-front oriented:

```text
dense front Vec<f64>
  -> APP pivot loop over a 32-column prefix
  -> apply pivots to trailing rows
  -> scan/restore rejected suffix
  -> accepted-prefix update in Rust loops
  -> FactorColumn / solve-panel materialization
```

The Rust version is numerically pinned but structurally different from native.
After the small-leaf branch was enabled, glider APP is no longer the main factor
bottleneck:

| Rust bucket | best recent median |
| --- | ---: |
| APP pivot factor | `94.042us` |
| APP block apply | `13.582us` |
| APP accepted update | `17.625us` |
| APP column storage | `0ns` |
| solve panel build | `111.954us` |
| TPP factor | `687.834us` |
| TPP contribution pack | `136.921us` |

Native samples show corresponding work inside `block_ldlt`, `apply_pivot_app`,
and `Block::update` with `host_gemm`/`host_trsm` calls rather than Rust-side
flattened loops. For the current glider witness, the TPP/small-leaf path should
stay ahead of APP work in the queue.

### Recommended Port Boundary

For glider, the current small-leaf branch is enabled. It is exact, but still
slower than native (`1.310ms` Rust unprofiled factor versus `1.014ms` native in
the latest in-process profile). That means the next step is no longer branch
hunting; it is source-shaped kernel/storage optimization inside the active
small-leaf branch.

For dense APP case 58/59 and any remaining glider APP work, the next APP-specific
port should be a storage/kernel change, not more branch hunting:

1. Add a native-shaped block-grid APP representation for 32-column panels.
2. Preserve the existing witness-pinned pivot and rounding behavior.
3. Replace accepted-prefix flattened updates with block-tile `calcLD + GEMM`
   style updates where the shape is large enough to pay off.
4. Keep factor column and solve panel materialization as a final extraction step,
   not interleaved with pivot/update work.

## Priority

1. Optimize the active small-leaf TPP pivot/factor loops while preserving exact
   pivot order and block records.
2. Replace the scalar contribution multiply with a source-shaped faster kernel
   while preserving exact generated contribution bits.
3. Re-run serial and parallel side-by-side glider profiles after each retained
   change.
