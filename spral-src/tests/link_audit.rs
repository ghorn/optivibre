use std::process::Command;

#[test]
fn executable_has_no_dynamic_solver_math_fallbacks() {
    let exe = std::env::current_exe().expect("current test executable path");
    let output = if cfg!(target_os = "macos") {
        Command::new("otool").arg("-L").arg(&exe).output()
    } else if cfg!(target_os = "linux") {
        Command::new("readelf").arg("-d").arg(&exe).output()
    } else {
        return;
    }
    .expect("failed to run platform link audit tool");

    assert!(
        output.status.success(),
        "link audit command failed: status={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for forbidden in [
        "libspral",
        "libmetis",
        "libopenblas",
        "Accelerate.framework",
        "vecLib.framework",
    ] {
        assert!(
            !stdout.contains(forbidden),
            "found forbidden dynamic solver/math dependency {forbidden} in:\n{stdout}"
        );
    }
}
