# Schittkowski 306 Port Coverage

This module ports the source-backed Schittkowski nonlinear programming cases from
the public `test_probs_src.zip` package.

The upstream `PROB.FOR` numbering is intentionally sparse. In the first block it
defines TP001 through TP119, with documented gaps TP082, TP094, and TP115. It then
jumps directly to TP201. There are no upstream TP120 through TP200 subroutines or
entries in the 2011 Fortran package.

So "ported up to 200" means every source-backed case below 200 is present:
TP001-TP081, TP083-TP093, TP095-TP114, and TP116-TP119.

The second block resumes at TP201. The native split includes every source-backed
case through TP306, with one Rust file per source problem or entry point.
