# SPRAL Matching, Scaling, and METIS Port Audit

This file tracks the Rust port of SPRAL SSIDS `options%ordering = 2` and
`options%scaling = 3`. Production `ssids-rs` must not call native SPRAL or
native METIS; those libraries are oracles for parity tests only.

| Upstream routine | Rust location | Test coverage | Status |
| --- | --- | --- | --- |
| `ssids/anal.F90::expand_matrix` | `ssids-rs/src/match_order.rs::expand_lower_to_full_spral` | `expand_lower_to_full_matches_spral_order` | Tested |
| `match_order.f90::match_order_metis_*` zero removal and abs transform | `remove_explicit_zeroes_and_abs` | `zero_removal_and_abs_preserves_column_order` | Tested |
| `scaling.f90::hungarian_match` | `spral_hungarian_match`, heap helpers | `matching_scaling_parity.rs` native phase trace, `native_singular_mo_match_phase_tests` | Native-matching on full-rank dense and structurally singular witnesses |
| `match_order.f90::mo_match` / `mo_scale` | `mo_match`, `mo_scale` | `mo_scale_diagonal_full_rank_uses_symmetric_inverse_sqrt_scale`, `mo_scale_singular_unmatched_uses_spral_huge_sentinel`, native `mo_scale` trace | Native-matching on dense case 58 and singular isolated/path fixtures; unmatched rows use the shipped `match_order.f90` `-huge(scale)` sentinel, so saved scaling is exactly zero |
| `match_order.f90::mo_split` cycle splitting | `mo_split_trace` | `mo_split_keeps_singletons_and_pairs_adjacent`, native `mo_split` cperm trace | Native-matching on dense case 58 |
| `match_order.f90::mo_split` compressed graph | `compressed_lower_pattern` | `compressed_lower_pattern_preserves_spral_row_order`, native compressed graph trace | Native-matching on dense case 58 |
| `metis5_wrapper.F90::half_to_full_drop_diag` | `metis_ordering::spral_half_to_full_drop_diag` | `spral_half_to_full_drop_diag_preserves_source_row_order` | Tested |
| METIS `compress.c::CompressGraph` closed-neighborhood compression | `metis_ordering::compress_metis_node_nd_graph` | `metis_node_nd_compresses_complete_graph_with_gklib_tie_order`, dense native trace | Native-matching on dense case 58 |
| GKlib `gklib.c::ikvsorti` key-only tie behavior | `metis_ordering::gk_ikvsorti_by_key` | `metis_node_nd_compresses_complete_graph_with_gklib_tie_order`, dense native trace | Native-matching on dense case 58 complete compressed graph |
| METIS/GKlib `util.c::InitRandom` and `gk_mkrandom.h::irandArrayPermute` | `metis_ordering::MetisRng` | `native_metis_random_fixture_phase_tests`, `metis_irand_array_permute_matches_gklib_template_branches` | Phase-tested against native source-built libmetis on the macOS baseline and wired through NodeND/coarsening |
| METIS `mmd.c::genmmd` / `ometis.c::MMDOrder` leaf ordering | `metis_ordering::mmd::mmd_order` | `native_metis_mmd_fixture_phase_tests` direct `libmetis__genmmd` oracle | Native-matching on path, star, disconnected, and empty fixtures; wired into recursive NodeND leaves |
| METIS `coarsen.c::BucketSortKeysInc`, `Match_RM`, `Match_2Hop*`, `Match_SHEM`, `CreateCoarseGraph` | `bucket_sort_keys_inc`, `metis_match_rm_coarsen`, `metis_match_2hop`, `metis_match_shem_coarsen`, `create_coarse_graph_from_match` | `metis_l1_match_rm_coarsening_matches_native_path_54_fixture`, `native_metis_node_nd_fixture_phase_tests` | Native-matching for identity self-match path/star, `path_54_match_rm`, `path_121`, `path_300`, and `path_1000`; multi-constraint matching remains out of scope for SPRAL NodeND |
| METIS `initpart.c::RandomBisection` zero-edge branch | `random_bisection_edge_state`, `balance_2way_no_edge` | `metis_node_nd_orders_empty_three_fixture`, native `empty_3` top-separator and NodeND fixtures | Native-matching for the SPRAL-default zero-edge graph path reached by structurally singular matching graphs |
| METIS `initpart.c::GrowBisection` plus `refine.c::Compute2WayPartitionParams` and cut FM no-op branch | `metis_debug_l1_edge_bisection_from_lower_csc` | `metis_l1_edge_bisection_matches_native_*`, `native_metis_node_bisection_stage_phase_tests` | Phase-tested for path/star and `path_54_match_rm` edge partition state; active no-edge branch now dispatches to `RandomBisection` |
| METIS `separator.c::ConstructSeparator`, `srefine.c::Compute2WayNodePartitionParams`, `sfm.c::FM_2WayNodeRefine*` coarse separator refinement | `metis_debug_l1_construct_separator_from_lower_csc` | `metis_l1_construct_separator_matches_native_*`, `native_metis_node_bisection_stage_phase_tests` | Native-matching for path/star and the 54-vertex Match_RM coarse separator; `ctrl->compress` rollback limits are ported |
| METIS `srefine.c::Project2WayNodePartition`, `Refine2WayNode`, active `FM_2WayNodeBalance` | `project_node_separator_state_through_coarsening`, `metis_l1_projected_separator_trace`, `fm_2way_node_balance` | `metis_l1_projected_separator_matches_native_path_54_fixture`, `native_metis_node_nd_fixture_phase_tests` | Native-matching for path/star no-op projection plus multi-level path projections through `path_1000` and L2 projection on `path_5000` |
| METIS `ometis.c::MlevelNodeBisectionL2` large-graph separator path | `metis_l2_node_bisection_trace_with_rng`, `metis_coarsen_graph_nlevels_with_maps` | `metis_node_nd_orders_path_5000_l2_fixture`, `native_metis_node_nd_top_separator_compression_retry_phase_test`, `native_metis_node_nd_fixture_phase_tests` | Native-matching on `path_5000`: four-level pre-coarsening, five L1 retry runs, best-separator restore, and final projection are pinned |
| METIS `ometis.c::MlevelNestedDissection` recursive separator plus MMD leaves | `metis_recursive_node_nd_order`, `assign_nested_dissection_positions` | `native_metis_node_nd_fixture_phase_tests` | Native-matching on complete-compressed, empty zero-edge, path, star, `path_54`, `path_121`, `path_300`, `path_1000`, compressed `twin_path_2400`, and L2 `path_5000` |
| METIS `ometis.c::MlevelNestedDissectionCC`, `contig.c::FindSepInducedComponents`, `ometis.c::SplitGraphOrderCC` | `metis_recursive_node_nd_order_cc`, `find_separator_induced_components`, `split_graph_order_cc` | `native_metis_cc_component_phase_tests`, `native_metis_node_nd_non_default_option_phase_tests` | Native-matching for `ccorder=true` separator-induced component CSR, shuffled component order, split subgraph labels/CSR, final star order, compressed `twin_path_256`, and combined pruning+CC |
| METIS `compress.c::PruneGraph` and `ometis.c` pruned-order expansion | `prune_metis_node_nd_graph`, `MetisNodeNdExpansion::Pruned` | `native_metis_prune_phase_tests`, `native_metis_node_nd_non_default_option_phase_tests` | Native-matching for no-prune reset, all-pruned ignored reset, partial hub pruning, compression disablement after real pruning, and final expansion of pruned vertices |
| METIS `ometis.c::MlevelNodeBisectionMultiple` top separator state and `nseps` retry semantics | `metis_node_bisection_multiple_trace_with_rng`, `metis_debug_node_nd_top_separator_from_lower_csc` | `native_metis_separator_fixture_phase_tests`, `native_metis_node_nd_top_separator_compression_retry_phase_test` direct internal native oracle | Pinned for path/star, `path_54`, compressed `twin_path_2400`, and L2 `path_5000`: separator labels, boundary nodes, mincut, part weights, compression map, equal-mincut retry tie behavior, and L2 separator projection are fail-closed native expectations |
| METIS `ometis.c::METIS_NodeND` option branches `COMPRESS`, `CCORDER`, `PFACTOR` | `metis_node_nd_order_with_options`, `MetisNodeNdOptions` | `native_metis_node_nd_non_default_option_phase_tests` source-built C shim | Native-matching for SPRAL default regression, forced `compress=false`, `ccorder=true`, no-prune `pfactor`, partial-prune `pfactor`, all-pruned ignored `pfactor`, pruning+CC, and compression+CC |
| `metis5_wrapper.F90::metis_order` calling METIS `NodeND` | `metis_ordering::metis_node_nd_order` | `metis_node_nd_*` unit tests, dense native trace, direct native fixture oracle | Dense-compression, `<5000` L1 multilevel, compression retry, `>=5000` L2 path fixtures, and documented non-default NodeND option branches integrated; SPRAL's wrapper still uses METIS defaults |
| `ssids.f90` analyse saved scaling | `SymbolicFactor::saved_matching_scaling` | `spral_matching_analysis_requires_values_and_reports_kind` | Integrated behind `OrderingStrategy::SpralMatching` |
| `ssids.f90` factor `case(3)` scaling permutation | `numeric_scaling_for_symbolic` | `saved_matching_scaling_is_explicit_and_solves_in_original_coordinates` | Integrated behind `NumericScalingStrategy::SavedMatching` |
| `assemble.hxx::add_a_block` scaled values | `apply_permuted_symmetric_scaling` | `permuted_symmetric_scaling_uses_spral_multiplication_order`, dense native trace | Integrated |
| `fkeep.F90::inner_solve_cpu` RHS/solution scaling | `NumericFactor::solve_in_place_impl` | same numeric saved-scaling test | Integrated |

Branch telemetry status:

- `SpralMatchingTrace` now carries deterministic `branch_hits` for the
  matching/scaling ladder. `matching_scaling_parity.rs` keeps the scoped branch
  ledger fail-closed: every emitted branch must be classified as `hit`,
  `guarded`, `unreachable-for-SPRAL`, or `out-of-scope`, and no `needs-port`
  classification is allowed.
- Current emitted matching/scaling branches distinguish empty vs non-empty
  expansion, explicit-zero removal vs no zero removal, full-rank vs singular
  Hungarian matching, singleton/unmatched/two-cycle/long-cycle `mo_split`
  behavior, compressed graph construction, METIS `NodeND` entry, and saved
  scaling exponentiation.
- Rust `METIS_NodeND` now exposes a test-only branch-hit trace for default and
  configured option paths. The parity fixtures assert classified hits for the
  SPRAL default compression path, forced no-compression, `ccorder`, `pfactor`
  no-prune/all-pruned/partial-prune cases, identity/compressed/pruned
  expansion, recursive L1/L2 nested dissection, MMD leaves, compression retry,
  and the zero-edge bisection path.
- The existing factor and solve profiles now expose test-only branch-hit
  classifiers for the dense APP factor path and forward/diagonal/backward solve
  path. The branch fixture asserts that a dense APP witness hits block LDLT,
  maxloc, accepted-prefix update, and the combined solve kernel sequence.
- The same ledger records the existing native-oracle METIS, APP dense factor,
  and solve-kernel branch fixtures. Native METIS/APP branches are still proven
  by source-backed phase/output assertions rather than native branch counters;
  adding native internal branch counters remains optional unless a fixture
  begins to localize poorly.

Open parity work:

- METIS 5.2.1 `METIS_NodeND` internals are now source-ported for the SPRAL
  default path plus the non-default `COMPRESS`, `CCORDER`, and `PFACTOR`
  branches exposed by `MetisNodeNdOptions`. SPRAL's production wrapper still
  calls `METIS_SetDefaultOptions` and uses default `compress = true`,
  `ccorder = false`, and `pfactor = 0`; the configurable Rust API exists to
  keep the source branch audit fail-closed, not to change SPRAL defaults.
  Compression-driven `nseps` retry selection is pinned by
  `twin_path_2400_compression_nseps_retry`,
  including the source tie behavior that keeps the final equal-mincut trial.
  The zero-edge `RandomBisection` branch is pinned by `empty_3`, and the
  `MlevelNodeBisectionL2` branch is pinned by `path_5000_l2_projection`. The
  standalone `genmmd` leaf routine, `Match_RM`, `Match_2Hop`, `Match_SHEM`,
  multi-level projection, active node balance, recursive L1 nested dissection,
  L2 large-graph separator path, `MlevelNestedDissectionCC`, `PruneGraph`, and
  pruned-order expansion are now ported and native-oracle tested.
- Direct native `metis_order` fixture oracle status:
  `complete_69_compresses_to_single_component`, `path_6`, `star_7`, and
  path fixtures `54`, `121`, `300`, and `1000` now match. The compressed
  `twin_path_2400` fixture also matches after exercising `cfactor > 1.5`,
  `nseps = 2`, retry tie handling, and compressed FM rollback limits.
  `empty_3` matches after exercising the zero-edge `RandomBisection` path, and
  `path_5000` matches after exercising `MlevelNodeBisectionL2`. The additional
  source-built C `METIS_NodeND` shim now pins non-default forced
  no-compression, `ccorder`, pruning, all-pruned ignored, pruning+CC, and
  compression+CC fixtures.
- Direct native separator fixture status:
  `path_6` top separator is vertex `2` with `where=[1,1,2,0,0,0]`;
  `star_7` top separator is vertex `0` with `where=[2,0,1,0,1,1,0]`;
  `path_54` projects the coarse separator back to original vertex `26`.
  These expectations fail closed, and Rust now combines those separator states
  with Rust MMD leaves to reconstruct native NodeND order for all three
  fixtures. Larger recursive path fixtures are pinned by direct native NodeND
  comparisons rather than hard-coded separator vectors.
- Direct native node-bisection stage status:
  for `path_6`, the edge bisection boundary is `[2,3]` and
  `ConstructSeparator` removes vertex `3` from the separator; for `star_7`,
  the edge bisection boundary is `[0,1,3,6]` and `ConstructSeparator` reduces
  the separator to vertex `0`; for `path_54`, `Match_RM` contracts 54 vertices
  to 30 coarse vertices, then projects the final separator to original vertex
  `26`. Rust now matches the coarse graph, edge bisection, coarse separator,
  projected separator, and NodeND order for these fixtures; `path_121`,
  `path_300`, and `path_1000` additionally exercise `Match_2Hop`, `SHEM`,
  active node balance, multi-level projection, and recursive nested dissection.
  The compressed `twin_path_2400` top separator additionally confirms native
  compression maps, `nseps = 2`, equal-mincut retry behavior, and compressed
  FM rollback limits before final NodeND perm/invp comparison. `path_5000`
  confirms the native L2 top separator before final NodeND perm/invp comparison.
  The `empty_3` fixture pins the zero-edge path: native starts from
  `RandomBisection`, returns `where=[1,0,1]`, and expands to the same final
  NodeND perm/invp as Rust.
- Disconnected/empty METIS NodeND is no longer a hidden scaling mismatch for
  the SPRAL-default path. The singular `isolated_missing_column` fixture now
  asserts native `mo_match`, saved scaling, compressed METIS order, and final
  ordering because its compressed ordering graph reaches the pinned zero-edge
  branch.
- Broaden `matching_scaling_parity.rs` beyond dense-compressed witnesses; the
  dense case-58 path now has hard native-vs-Rust assertions.

Dense case-58 classification after the phase trace:

- Native analyse order equals the test-only direct `match_order_metis` wrapper.
- Rust expanded matrix, zero removal, `mo_match`, `mo_scale`, `mo_split` cperm,
  compressed graph, compressed METIS perm/invp, final order, and saved scaling
  match native SPRAL for this witness.
- Rust `SpralMatching` with saved matching scaling now matches native SPRAL
  matching/scaling solve bits on dense seed `0x706172697479`, case `58`.
- The factor-time scaling bit gap was `assemble.hxx::add_a_block` multiplication
  order: SPRAL evaluates `rscale * aval[src] * cscale`, not
  `aval[src] * (rscale * cscale)`.

Release performance notes on dense case 58:

- Rust `SpralMatching` analyse moved from roughly `18-19ms` to roughly `1.3ms`
  after reusing the initial symbolic pattern when SPRAL-style postorder is
  identity, replacing per-fill-edge linear membership checks with word-wise
  bitset fill propagation, and building permuted CSR graphs directly.
- Rust captured-order/no-scaling analyse moved further to roughly
  `0.5-0.65ms` on this witness after using the already-computed symbolic
  column counts in release builds and short-circuiting SPRAL's row-list result
  for the single full-rank supernode case. The debug/test build still checks
  the source-faithful `find_col_counts` port against the symbolic counts.
- Rust captured-order/no-scaling analyse then moved to roughly `0.4ms` after
  switching small/medium graph permutation to a sorted bitset construction and
  removing a per-column temporary allocation in symbolic fill simulation.
- `ssids-rs` now builds its analysis CSR graph directly from the validated
  lower CSC matrix accepted by `SymmetricCscMatrix`, avoiding the public
  `metis_ordering::CsrGraph::from_symmetric_csc` sort/dedup path in SSIDS
  analyse. Internally constructed SSIDS CSR graphs now use a hidden trusted
  constructor that still checks invariants in debug builds. Dense case-58
  release timings are now roughly `0.20-0.23ms` for captured-order/no-scaling
  analyse and roughly `0.79-0.86ms` for `SpralMatching` saved-scaling analyse,
  effectively matching native analyse timing on this witness.
- Rust `SpralMatching` analyse is now roughly native-speed on the same witness;
  the remaining matching/scaling overhead is still close to native's own
  matching/scaling overhead.
- `SPRAL_SSIDS_DEBUG_MATCHING=1` now splits the Rust SPRAL matching ladder into
  expand, zero-removal/abs compaction, `mo_match`, `mo_split`, and scaling-exp
  phases. On dense case 58 the `mo_match` phase is dominated by source-faithful
  cost construction plus the Hungarian walk; the CSC buffers now reserve the
  exact nonzero-sized capacity used by the upstream array path.
- The production and trace matching paths now feed the already compact
  absolute-value matrix directly into `mo_match`; the `mo_scale` boundary helper
  remains test-only for phase coverage of SPRAL's zero-removal/abs transform.
- Native SPRAL still analyses the captured explicit-order witness in roughly
  `0.2ms`, so the remaining performance gap is in the Rust symbolic-analysis
  implementation rather than in the matching/scaling phase itself.
- `SPRAL_SSIDS_DEBUG_FACTOR=1` now prints an env-gated Rust factor profile.
  Dense case 58 currently localizes most remaining Rust factor time to APP
  dense-front work, especially the accepted-prefix trailing update, rather than
  saved scaling or sparse front assembly.
- The APP accepted-prefix update now reuses the dense-front scratch buffer for
  its LD workspace in production; the allocating wrapper remains test-only. Its
  inner accepted-pivot dot keeps the source-faithful `mul_add` sequence but is
  unrolled by four, moving the dense case-58 accepted-update phase from roughly
  `0.17ms` to roughly `0.13ms` while preserving bitwise parity.
- The APP `host_trsm`-equivalent apply now specializes the common full
  group-of-four triangular solve while preserving each column's prior-dot order
  and the intra-group update sequence. Dense case-58 `app_triangular` moved
  from roughly `0.095ms` to roughly `0.041ms`.
- On AArch64, that same APP triangular apply now processes two trailing rows at
  a time with NEON while preserving each row's prior-column dot order and
  in-group triangular update order. Dense case-58 `app_triangular` now measures
  around `12-14us` in release profile runs.
- On AArch64, the in-block APP 1x1 and 2x2 update kernels now use two-row NEON
  updates while preserving each entry's source expression. Dense case-58
  `app_pivot` now measures around `51us` on the saved-scaling path.
- The APP factor profile now separates block backup/restore from the accepted
  trailing update. Dense case 58 shows backup/restore are small; the hot path is
  the source-equivalent `calcLD + host_gemm(OP_N, OP_T)` update. Rust now walks
  the target lower triangle in dense column order and caches each accepted
  column's L values while preserving the exact per-entry pivot accumulation
  order and the singleton incremental branch. Flattening the inner loop over
  accepted pivot columns trims the scalar accepted update to roughly `0.13ms`
  on the dense witness, with the remaining gap attributable to native SPRAL
  dispatching the same operation to BLAS.
- `SPRAL_SSIDS_DEBUG_FACTOR=1` now splits that accepted update into LD build and
  GEMM-equivalent buckets. Dense case 58 shows LD construction at roughly `4us`
  and the scalar GEMM-equivalent update at roughly `125us`, so further factor
  performance work should target the `host_gemm(OP_N, OP_T)` equivalent rather
  than matching/scaling, LD generation, or restore logic.
- On AArch64, the accepted-update kernel now processes four target rows with
  two NEON accumulators, then falls back to the two-row and scalar tails. Each
  lane keeps the same per-entry accepted-pivot accumulation order as the scalar
  path, while dense case 58 moves the GEMM-equivalent bucket to roughly `60us`
  and the full accepted-update bucket to roughly `65us` while keeping native
  matching/scaling solve bits exact.
- The AArch64 accepted-update kernel now also pairs adjacent target columns,
  reusing each LD row load for both columns while preserving the exact
  per-entry accepted-pivot FMA order. On the 160x160 dense witness
  `seed=0x7061726974792026, case=59`, the saved-scaling APP
  GEMM-equivalent bucket now measures roughly `53-58us`, and the saved-scaling
  factor path is roughly `0.32-0.36ms` in release profile runs with native
  matching/scaling solve bits still exact.
- Final lower-factor storage now reserves the exact number of off-diagonal
  entries before filling the CSC buffers. On the same 160x160 witness this
  trims the `lower_storage` profile bucket into the roughly `19-27us` range
  without changing numeric data or solve order.
- The AArch64 in-block APP 1x1 update now mirrors
  `block_ldlt.hxx::update_1x1`'s four-column source unroll, reusing each
  pivot-column NEON vector across four target columns while leaving each
  lower-triangle entry's FMA expression unchanged. On the same witness,
  saved-scaling `app_pivot` samples around `51-53us` and factor time around
  `0.33-0.35ms`, with native matching/scaling solve bits still exact.
- The APP maxloc scan now keeps the same source-shaped two-lane walk and
  strict tie behavior while indexing dense lower-triangle entries directly.
  An eleven-repeat release profile on the same witness moved saved-scaling
  `app_pivot` to about `45us` and factor time to about `274us`, with native
  matching/scaling solve bits still exact.
- Saved analyse scaling is now applied while filling the permuted CSC values,
  avoiding a second production pass over the same entries. A unit test pins
  the fused path bit-for-bit against the previous fill-then-scale path,
  including SPRAL's `row_scale * value * col_scale` multiplication order.
- Added `scripts/ssids_rs_release_profile.sh` as the repeatable release profile
  loop for dense APP witnesses and the glider exact replay. It reports medians
  for Rust/native analyse, factor, solve, and the Rust factor-profile buckets
  instead of relying on a single noisy timing sample.
- The remaining dense-front factor hot path is source-anchored to
  `ldlt_app.cxx::Block::update`, which builds an APP LD tile with
  `calcLD<OP_N>` and applies `host_gemm(OP_N, OP_T)`. The
  `app_accepted_update_dense_witnesses_match_native_host_gemm_tiles` unit test
  extracts dense witnesses and compares Rust's accepted-prefix trailing update
  against those native source kernels tile-by-tile before further kernel work.
- The AArch64 accepted-update kernel now also processes full four-column target
  groups, matching the source `host_gemm(OP_N, OP_T)` shape more closely while
  using scalar prelude rows for the lower-triangle diagonal boundary. A
  five-repeat release profile moved saved-scaling case 58 factor time to about
  `227us` and case 59 to about `313us`; case 59's saved-scaling
  `app_accepted_update` bucket dropped to about `48us`. The next measured
  dense-front buckets are APP pivot work and restore/storage overhead.
- The APP dense-front loop now reuses its row-order and packed trailing-lower
  backup buffers across 32-column blocks. The copied data and restore order are
  unchanged, but allocation churn is removed from the backup bucket. A short
  three-repeat release profile on the same case-59 witness measured native
  matching/scaling factor at about `226us` and Rust saved-scaling factor at
  about `290us`, with Rust's `app_backup` bucket reduced to about `10us`.
- APP factor-column capture and final lower-factor storage now use exact
  capacity plus direct initialized writes for already validated copy paths.
  Entry order, row mapping checks, and factor values are unchanged. Short
  five-repeat case-59 profiles put saved-scaling `app_column_storage` around
  `13-15us` and `lower_storage` around `13us`; total factor timing remains
  noisy but the storage buckets are smaller than the previous roughly
  `20us`-class copies.
- Dense-front solve panel construction now copies each accepted lower-column
  tail as one contiguous slice, matching the already column-major dense-front
  storage instead of indexing every row through `dense_lower_offset`. The
  generic factor-column equivalence test pins the exact emitted panel values;
  short case-59 profiles put `solve_panel_build` around `7us`.
- Factor setup now hoists invariant permutation lookups while building the
  permuted lower CSC pattern and uses a guarded identity leaf-front assembly
  path when a front has no child contributions or interface rows. The fast path
  writes the same lower-CSC values into the dense front with identity local
  row/column numbering; short case-59 profiles put `front_assembly` around
  `14us` and `permuted_pattern` around `38us`.
- The APP `find_maxloc` scan now keeps the source-faithful two-lane max/tie
  order but writes the fixed 32x32 lower-block scan directly instead of routing
  every candidate through a helper. A five-repeat case-59 release profile put
  saved-scaling `app_pivot` around `46us`, with native/Rust saved-scaling
  factor medians at about `232us`/`259us`.
- The AArch64 four-column accepted-update microkernel now has an eight-row
  path. Each lower-triangle entry still accumulates accepted pivots in the same
  order as the scalar/source-shaped path, but the kernel reuses each accepted L
  broadcast across twice as many target rows. A five-repeat profile measured
  case 59 native/Rust saved-scaling factor medians at about `257us`/`278us`,
  with Rust `app_accepted_gemm` around `40us` and `app_accepted_update` around
  `45us`; case 58 was effectively tied at about `226us`/`228us`.
- Dense solve-panel construction now initializes each panel column once instead
  of zero-filling the full panel and then overwriting the lower part. The
  existing bit-for-bit equivalence test pins it against the generic
  factor-column path. A glider exact augmented replay repeat measured Rust
  factor around `3.46ms` vs native `1.06ms`, with Rust `solve_panel_build`
  around `292us`; the remaining glider factor gap is still mostly dense-front
  work across many small fronts rather than this panel copy.
- APP accepted-update LD construction now iterates explicitly over
  `APP_INNER_BLOCK_SIZE` row tiles instead of recomputing the tile by division
  for every row. This preserves the source `calcLD<OP_N>` vector/scalar split
  and row order. A five-repeat case-59 release profile measured native/Rust
  saved-scaling factor medians at about `254us`/`264us`, with Rust
  `app_accepted_ld` around `3us` and `app_accepted_update` around `41us`.
- The APP pivot profile now splits the in-block bucket into `app_maxloc`,
  `app_swap`, and `app_pivot_update`. These are Rust-only attribution buckets
  under the existing `app_pivot` total; they do not change pivot selection,
  swap order, or update arithmetic, but they localize the remaining dense-front
  gap before another kernel change is attempted. The fine-grained timers are
  opt-in via `SPRAL_SSIDS_DEBUG_FACTOR_APP_SUBPHASES=1` so default side-by-side
  release profiles keep their existing timing overhead.
- APP factor profiles now also report low-overhead front-shape counters when
  profiling is enabled: APP front count, panel count, maxloc calls, symmetric
  swaps, 1x1/2x2/zero pivot counts, and an eight-bucket APP front-size
  histogram. These are counted only on profiled runs and are parsed by
  `scripts/ssids_rs_release_profile.sh` as integer attribution metrics, not as
  native timing comparisons.
- The glider exact augmented replay currently exercises 88 APP fronts and 88
  APP panels: 75 fronts in the 33-64 bucket and 13 in the 65-96 bucket. The
  observed pivot mix is all 1x1 (`2816` pivots), with `0` 2x2 or zero pivots,
  `2816` maxloc calls, and roughly `2420` symmetric swaps. That makes glider's
  remaining factor gap a many-small-front APP path rather than the dense
  case-59 two-by-two-heavy path.
