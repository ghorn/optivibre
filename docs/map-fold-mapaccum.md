# `SXFunction` Higher-Order Nodes: `map`, `fold`, `mapaccum`

## Goal

After reusable `SXFunction` call nodes and LLVM-visible subfunctions exist, the next useful abstraction is structured repeated application. The immediate targets are:

- `map`: independent application of one symbolic function over a batch axis
- `fold`: serial reduction with an explicit carry
- `mapaccum`: serial carry plus collected per-step outputs

The design should preserve the current strengths of `SX`:

- fixed sparsity
- pure acyclic graphs
- exact AD
- backend-neutral lowering

It should also leave room for parallel setup and parallel evaluation without forcing MX-style generality into the first version.

## Core Representation

### Shared principles

- The callee is always an `SXFunction`.
- Batch and carry sizes are part of the node metadata, not discovered dynamically.
- All shapes remain fixed-CCS.
- Nodes are still pure and acyclic.
- Higher-order nodes are graph nodes, not syntax sugar for eager scalar unrolling.

### `map`

`map` applies one function independently across `N` slices:

```text
Y[i] = f(X0[i], X1[i], ..., Xk[i]) for i in 0..N-1
```

Suggested node payload:

- `function_id`
- `batch_len`
- `input_partition`: how each formal input is sliced out of each actual argument
- `output_layout`: how per-iteration outputs are packed back into the aggregate outputs
- `execution_hint`: `Serial`, `Parallel`, `Unrolled`

### `fold`

`fold` models a carry-only recurrence:

```text
carry[i + 1] = f(carry[i], input[i])
```

Suggested node payload:

- `function_id`
- `batch_len`
- `carry_input_slots`
- `carry_output_slots`
- `scan_input_slots`

Constraints for v1:

- carry arity and output arity are fixed
- no parallel execution in the generic case
- no implicit associativity assumptions

### `mapaccum`

`mapaccum` is `fold` plus collected step outputs:

```text
(carry[i + 1], y[i]) = f(carry[i], input[i])
```

Suggested node payload:

- everything from `fold`
- `mapped_output_slots`

This should be the default abstraction for trajectory rollout and scan-like OCP helpers.

## AD Semantics

### `map`

`map` is the cleanest case.

- Forward mode: map the callee tangent helper independently over all iterations.
- Reverse mode: map the pullback independently and sum adjoints into the batched inputs.
- Jacobian/Hessian structure is block diagonal across iterations when inputs are iteration-local.

This is the main entry point for evaluation parallelism.

### `fold`

`fold` is serial by construction.

- Forward mode: propagate tangents through the carry chain.
- Reverse mode: reverse-scan adjoints through the carry chain.
- Hessians require the usual second-order scan logic; there is no cheap block-diagonal shortcut.

For v1, the derivative helpers should themselves be represented as folded helper functions rather than unrolled scalar graphs.

### `mapaccum`

`mapaccum` combines the two behaviors.

- mapped outputs produce per-step adjoints
- carry state produces serial backpropagation
- setup should cache tangent and pullback helpers once per callee, not once per step

## Lowering Strategy

### `map`

Lowering should support three execution styles:

1. `SerialLoop`
2. `ParallelLoop`
3. `Unrolled`

Policy guidance:

- default to `SerialLoop`
- allow `ParallelLoop` only when iterations are independent and the backend can prove there is no cross-iteration aliasing
- use `Unrolled` only for small fixed trip counts or when explicitly requested

`SerialLoop` and `ParallelLoop` need first-class loop IR in `sx_codegen`; otherwise the abstraction collapses back into scalar unrolling too early.

### `fold`

Lower as a loop kernel with explicit carry temporaries.

- No general parallel lowering in v1.
- An associative-reduction specialization can come later as a separate node or flagged variant.

### `mapaccum`

Lower as a loop kernel that:

- updates carry state each iteration
- stores mapped outputs into preallocated output slices

This should not default to scalar unrolling, even for moderate `N`, because preserving the loop structure is the entire point for setup/JIT scalability.

## LLVM Design

Once `sx_codegen` can represent loops, LLVM emission should mirror the execution hint.

### `map`

- `SerialLoop`: emit a compact counted loop
- `ParallelLoop`: emit an outer runtime parallel-for abstraction later; do not bake in a threading runtime in v1
- `Unrolled`: emit repeated calls or inlined bodies depending on call policy

### `fold` and `mapaccum`

- emit counted serial loops
- keep carry state in stack slots or SSA temporaries depending on what LLVM simplifies better

The important design point is that loop structure should survive down to LLVM for compact IR and future parallelization, instead of being expanded in the symbolic layer.

## Parallelization Roadmap

### Setup parallelism

First wins should come from the compile pipeline, not from symbolic graph mutation itself.

1. Build independent derivative kernels in parallel:
   - objective gradient
   - equality Jacobian
   - inequality Jacobian
   - Hessian
2. JIT independent kernels in parallel.
3. Keep subfunction lowering caches thread-safe so repeated callees are still shared.

The real blocker for deeper symbolic setup parallelism is the current global SX interner. Until that is split or sharded, parallel symbolic construction will fight a central lock and show poor scaling.

### Evaluation parallelism

Primary target: `map`.

- independent iterations can run concurrently
- each worker should get disjoint output slices
- input slices should be read-only

Secondary target: batched solver callbacks that evaluate multiple shooting or collocation residuals across intervals.

### What should stay serial

- generic `fold`
- generic `mapaccum`
- reverse-mode propagation through carry state

If later profiling shows a strong need, add a distinct associative-reduction node rather than weakening the semantics of `fold`.

## Recommended Implementation Order

1. Add loop-capable lowering IR for `map`, `fold`, and `mapaccum`.
2. Implement `map` first with serial lowering and exact AD.
3. Add `mapaccum` next for rollout-style recurrences.
4. Add `fold` on the same serial loop substrate.
5. Add LLVM loop emission.
6. Add compile-time parallelism for independent derivative/JIT stages.
7. Add `map` parallel evaluation.

## Non-Goals for V1

- arbitrary control flow
- dynamic trip counts
- dynamic sparsity
- generic parallel `fold`
- automatic dependence analysis beyond obvious iteration independence

This keeps the design aligned with the current `SX` model while opening a path to the setup and evaluation parallelism that motivated reusable function nodes in the first place.
