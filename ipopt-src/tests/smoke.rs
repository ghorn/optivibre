#[test]
fn source_built_ipopt_metadata_smoke() {
    assert_eq!(ipopt_src::IPOPT_VERSION, "3.14.20");
    assert_eq!(
        ipopt_src::IPOPT_SOURCE_COMMIT,
        "4667204c76e534d3e4df6b1462f258a4f9c681bd"
    );
}
