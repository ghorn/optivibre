#[test]
fn source_built_ipopt_metadata_smoke() {
    assert_eq!(ipopt_src::IPOPT_VERSION, "3.14.20");
    assert_eq!(
        ipopt_src::IPOPT_SOURCE_COMMIT,
        "4667204c76e534d3e4df6b1462f258a4f9c681bd"
    );
    assert_eq!(ipopt_src::SOLVER_FAMILY, "spral");
    assert_eq!(ipopt_src::OPENBLAS_OWNER, "spral-src");
    assert!(!std::hint::black_box(
        ipopt_src::SYSTEM_SOLVER_MATH_FALLBACKS
    ));
    assert!(
        matches!(ipopt_src::OPENBLAS_THREADING, "serial" | "openmp"),
        "IPOPT must re-emit the spral-src OpenBLAS threading mode"
    );
}
