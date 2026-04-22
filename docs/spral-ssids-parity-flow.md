# SPRAL SSIDS Parity Flow

This diagram tracks the Rust SSIDS parity ladder against native SPRAL SSIDS.
Green nodes have active bitwise or exact metadata coverage. Yellow nodes are
newly passing in the current checkpoint. Orange nodes have partial coverage or a
known narrowed boundary. Red nodes are the next open bitwise mismatch target.

Current newly passing witness:
`dense_tpp_factor_dyadic_cases_0_to_6_match_native_kernel` proves the
standalone native `ldlt_tpp_factor` path matches Rust bitwise for dyadic cases
0 through 6. This checkpoint also pins native TPP's `m`/`n` update bound:
factor multipliers are formed for all rows, but trailing updates only touch
candidate columns. The next smaller open TPP witness is
`dense_tpp_factor_small_dyadic_cases_match_native_kernel`, where the remaining
sweep first differs at case 7, matrix row 5, col 0.

```mermaid
flowchart TD
    A["Input symmetric CSC matrix"] --> B["Analyse structure"]
    B --> B1["Ordering / permutation"]
    B --> B2["Elimination tree"]
    B --> B3["Supernodes / fronts"]

    B1 --> C["Numeric factorization"]
    B2 --> C
    B3 --> C

    C --> D["For each front"]
    D --> E["Assemble dense front from matrix + children"]
    E --> F{"Dense front path"}

    F -->|"APP block path"| G["block_ldlt APP panel"]
    F -->|"TPP / tail path"| H["dense TPP tail factorization"]
    H --> H1["Standalone TPP 4x4 factor state"]
    H1 --> H2["Dyadic TPP cases 0-6 factor state"]
    H2 --> H3["Dyadic TPP case 7 update-order"]

    G --> G0["Native align_lda panel layout"]
    G0 --> G1["find_maxloc"]
    G1 --> G2{"Pivot choice"}
    G2 -->|"1x1"| G3["swap_cols if needed"]
    G3 --> G4["scale pivot column"]
    G4 --> G5["update_1x1 trailing block"]

    G2 -->|"2x2"| G6["test_2x2"]
    G6 --> G7["swap_cols twice"]
    G7 --> G8["compute 2x2 inverse / multipliers"]
    G8 --> G9["update_2x2 trailing block"]

    G2 -->|"failed / delayed"| G10["delay pivot to parent front"]

    G5 --> I["APP accepted-prefix update"]
    G9 --> I
    G10 --> J["Record delayed pivots"]
    H --> I

    I --> K["Store L/D blocks, local perms, pivot records"]
    K --> K1["Factor order, inertia, pivot stats"]
    K --> K2["Seed6 APP prefix inverse-D bits"]
    K2 --> K3["TPP-tail inverse-D bits"]
    J --> K
    K1 --> L{"More fronts?"}
    K3 --> L
    L -->|"yes"| D
    L -->|"no"| M["Report inertia + pivot stats"]

    M --> N["Solve phase"]
    N --> O["Forward triangular solve"]
    O --> P["Block diagonal solve"]
    P --> Q["Backward triangular solve"]
    Q --> R["Solution bit patterns"]

    classDef match fill:#dff7df,stroke:#2f8f46,color:#102615,stroke-width:2px;
    classDef newly fill:#fff4b8,stroke:#b88a00,color:#2a2100,stroke-width:3px;
    classDef partial fill:#ffe3bf,stroke:#b76d12,color:#2b1800,stroke-width:2px;
    classDef open fill:#ffd8d8,stroke:#b43b3b,color:#2b0d0d,stroke-width:2px;

    class A,B,B1,B2,B3,G0,G1,G3,G5,G6,G8,G9,H1,K1,K2,M,O,P,Q,R match;
    class H2 newly;
    class C,D,E,F,G,G2,G4,G7,G10,H,I,J,K,L,N partial;
    class H3,K3 open;
```
