#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPORT_DIR="${IPOPT_PARITY_COVERAGE_DIR:-target/reports/ipopt-parity-coverage}"
COMBINED_LCOV="$REPORT_DIR/combined.lcov"
SUMMARY_JSON="$REPORT_DIR/coverage-summary.json"
SUMMARY_TEXT="$REPORT_DIR/combined.txt"
IPOPT_LCOV="$REPORT_DIR/ipopt-algorithm.lcov"
RUST_CORE_LCOV="$REPORT_DIR/rust-core.lcov"
NATIVE_LCOV="$REPORT_DIR/ipopt-native.lcov"
NATIVE_PROFDATA="$REPORT_DIR/ipopt-native.profdata"
AUDIT_MD="$REPORT_DIR/coverage-audit.md"
BRANCH_LEDGER_MD="$REPORT_DIR/branch-ledger.md"

cleanup_stray_profraw() {
  find "$ROOT" -path "$ROOT/target" -prune -o -type f -name 'default_*.profraw' -exec rm -f {} +
}

cleanup_target_profraw() {
  local profile_root="${CARGO_LLVM_COV_TARGET_DIR:-$ROOT/target}"
  [[ -d "$profile_root" ]] || return
  find "$profile_root" -maxdepth 1 -type f \( -name 'optivibre-*.profraw' -o -name 'default_*.profraw' \) -exec rm -f {} +
}

if ! cargo llvm-cov --version >/dev/null 2>&1; then
  echo "cargo-llvm-cov is required. Install it with: cargo install cargo-llvm-cov" >&2
  exit 1
fi

find_tool() {
  local explicit="$1"
  local name="$2"
  if [[ -n "$explicit" ]]; then
    printf '%s\n' "$explicit"
    return
  fi
  if [[ -x "/opt/homebrew/opt/llvm/bin/$name" ]]; then
    printf '/opt/homebrew/opt/llvm/bin/%s\n' "$name"
    return
  fi
  if [[ -x "/usr/local/opt/llvm/bin/$name" ]]; then
    printf '/usr/local/opt/llvm/bin/%s\n' "$name"
    return
  fi
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
    return
  fi
  echo "$name"
}

find_gcc_cxx_root() {
  local root
  for root in /opt/homebrew/Cellar/gcc/*/include/c++/* /usr/local/Cellar/gcc/*/include/c++/*; do
    [[ -d "$root" ]] || continue
    if compgen -G "$root/*-apple-darwin*" >/dev/null; then
      printf '%s\n' "$root"
      return
    fi
  done
}

find_gcc_lib_dir() {
  local dir
  for dir in /opt/homebrew/Cellar/gcc/*/lib/gcc/current /usr/local/Cellar/gcc/*/lib/gcc/current; do
    [[ -d "$dir" ]] || continue
    printf '%s\n' "$dir"
    return
  done
}

latest_path() {
  python3 - "$1" "$2" "${3:-0}" <<'PY'
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
pattern = sys.argv[2]
mode = sys.argv[3]
matches = []
for path in root.glob(pattern):
    if mode == "dir":
        if not path.is_dir():
            continue
    elif not path.is_file():
        continue
    if mode == "1" and not os.access(path, os.X_OK):
        continue
    matches.append(path)
if not matches:
    sys.exit(1)
print(max(matches, key=lambda path: path.stat().st_mtime))
PY
}

mkdir -p "$REPORT_DIR"
cleanup_stray_profraw
cleanup_target_profraw
rm -f \
  "$COMBINED_LCOV" \
  "$SUMMARY_JSON" \
  "$SUMMARY_TEXT" \
  "$IPOPT_LCOV" \
  "$RUST_CORE_LCOV" \
  "$NATIVE_LCOV" \
  "$NATIVE_PROFDATA" \
  "$AUDIT_MD" \
  "$BRANCH_LEDGER_MD"

export OMP_CANCELLATION="${OMP_CANCELLATION:-true}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY="${AD_CODEGEN_REQUIRE_NATIVE_SPRAL_PARITY:-1}"
export IPOPT_SRC_LLVM_COVERAGE="${IPOPT_SRC_LLVM_COVERAGE:-1}"
export GLIDER_PARITY_IPOPT_PRINT_LEVEL="${GLIDER_PARITY_IPOPT_PRINT_LEVEL:-0}"
export LLVM_COV="$(find_tool "${LLVM_COV:-}" llvm-cov)"
export LLVM_PROFDATA="$(find_tool "${LLVM_PROFDATA:-}" llvm-profdata)"
export IPOPT_SRC_LLVM_COVERAGE_CC="$(find_tool "${IPOPT_SRC_LLVM_COVERAGE_CC:-}" clang)"
export IPOPT_SRC_LLVM_COVERAGE_CXX="$(find_tool "${IPOPT_SRC_LLVM_COVERAGE_CXX:-}" clang++)"
GCC_CXX_ROOT="$(find_gcc_cxx_root)"
GCC_LIB_DIR="$(find_gcc_lib_dir)"
if [[ -n "$GCC_CXX_ROOT" && -z "${IPOPT_SRC_LLVM_COVERAGE_CXXFLAGS:-}" ]]; then
  GCC_TARGET_ROOT="$(find "$GCC_CXX_ROOT" -maxdepth 1 -type d -name '*-apple-darwin*' | head -n 1)"
  IPOPT_COVERAGE_CXXFLAGS="-stdlib=libstdc++ -Wno-invalid-constexpr -I$GCC_CXX_ROOT"
  if [[ -n "$GCC_TARGET_ROOT" ]]; then
    IPOPT_COVERAGE_CXXFLAGS+=" -I$GCC_TARGET_ROOT"
  fi
  export IPOPT_SRC_LLVM_COVERAGE_CXXFLAGS="$IPOPT_COVERAGE_CXXFLAGS"
fi
if [[ -n "$GCC_LIB_DIR" && -z "${IPOPT_SRC_LLVM_COVERAGE_LDFLAGS:-}" ]]; then
  export IPOPT_SRC_LLVM_COVERAGE_LDFLAGS="-stdlib=libstdc++ -L$GCC_LIB_DIR"
fi

if [[ "${IPOPT_PARITY_COVERAGE_CLEAN:-1}" != "0" ]]; then
  cargo llvm-cov clean --workspace
fi
if [[ "${IPOPT_PARITY_COVERAGE_REBUILD_NATIVE:-0}" != "0" ]]; then
  cargo clean -p spral-src
  cargo clean -p ipopt-src
fi

cargo llvm-cov show-env --sh >"$REPORT_DIR/cargo-llvm-cov-env.sh"
# shellcheck disable=SC1090
source "$REPORT_DIR/cargo-llvm-cov-env.sh"
cleanup_target_profraw

cargo test \
  -p optimization \
  --features ipopt,native-spral-src \
  --test interior_point_compare \
  -- --nocapture

cargo test \
  -p optimization \
  --lib restoration \
  --features ipopt,native-spral-src \
  -- --nocapture

cargo test \
  -p optimization \
  --lib full_space \
  --features ipopt,native-spral-src \
  -- --nocapture

cargo test \
  -p optimal_control_problems \
  --release \
  --features ipopt,native-spral-src \
  print_current_glider_native_spral_ipopt_first_divergence \
  -- --ignored --nocapture

export CC="$IPOPT_SRC_LLVM_COVERAGE_CC"
export CXX="$IPOPT_SRC_LLVM_COVERAGE_CXX"
cargo llvm-cov report --include-ffi --json --summary-only --output-path "$SUMMARY_JSON"
cargo llvm-cov report --include-ffi --lcov --output-path "$COMBINED_LCOV"
cargo llvm-cov report --include-ffi --text --output-path "$SUMMARY_TEXT"

mapfile -t PROFRAW_FILES < <(find "${CARGO_LLVM_COV_TARGET_DIR:-target}" -maxdepth 1 -type f -name 'optivibre-*.profraw')
if [[ "${#PROFRAW_FILES[@]}" -eq 0 ]]; then
  echo "no raw profile files found for IPOPT native coverage" >&2
  exit 1
fi
"$LLVM_PROFDATA" merge -sparse "${PROFRAW_FILES[@]}" -o "$NATIVE_PROFDATA"

INTERIOR_BIN="$(latest_path target/debug/deps 'interior_point_compare-*' 1)"
GLIDER_BIN="$(latest_path target/release/deps 'optimal_control_problems-*' 1)"
DEBUG_IPOPT_LIB="$(latest_path target/debug/build 'ipopt-src-*/out/install-llvm-cov/lib/libipopt.a')"
RELEASE_IPOPT_LIB="$(latest_path target/release/build 'ipopt-src-*/out/install-llvm-cov/lib/libipopt.a')"
CANONICAL_IPOPT_SOURCE="$(latest_path target/release/build 'ipopt-src-*/out/sources/Ipopt-4667204c76e534d3e4df6b1462f258a4f9c681bd' dir)"
mapfile -t IPOPT_SOURCE_ROOTS < <(find target -path '*/out/sources/Ipopt-4667204c76e534d3e4df6b1462f258a4f9c681bd' -type d)
PATH_EQUIV_ARGS=()
for source_root in "${IPOPT_SOURCE_ROOTS[@]}"; do
  PATH_EQUIV_ARGS+=("--path-equivalence=$source_root,$CANONICAL_IPOPT_SOURCE")
done

"$LLVM_COV" export \
  --format=lcov \
  --instr-profile="$NATIVE_PROFDATA" \
  "${PATH_EQUIV_ARGS[@]}" \
  "$GLIDER_BIN" \
  -object "$INTERIOR_BIN" \
  -object "$RELEASE_IPOPT_LIB" \
  -object "$DEBUG_IPOPT_LIB" \
  >"$NATIVE_LCOV"

python3 - "$COMBINED_LCOV" "$NATIVE_LCOV" "$IPOPT_LCOV" "$RUST_CORE_LCOV" "$AUDIT_MD" "$BRANCH_LEDGER_MD" <<'PY'
import pathlib
import re
import sys

combined = pathlib.Path(sys.argv[1])
native = pathlib.Path(sys.argv[2])
ipopt_out = pathlib.Path(sys.argv[3])
rust_out = pathlib.Path(sys.argv[4])
audit_out = pathlib.Path(sys.argv[5])
branch_ledger_out = pathlib.Path(sys.argv[6])

def read_records(path):
    records = []
    current = []
    for line in path.read_text().splitlines():
        current.append(line)
        if line == "end_of_record":
            records.append(current)
            current = []
    if current:
        records.append(current)
    return records

def source(record):
    for line in record:
        if line.startswith("SF:"):
            return line[3:]
    return ""

def canonical_source(path):
    marker = "/Ipopt-4667204c76e534d3e4df6b1462f258a4f9c681bd/"
    if marker in path:
        return "Ipopt-4667204c76e534d3e4df6b1462f258a4f9c681bd/" + path.split(marker, 1)[1]
    cwd = str(pathlib.Path.cwd()) + "/"
    if path.startswith(cwd):
        return path[len(cwd):]
    return path

def merge_records(records):
    merged = {}
    display_path = {}
    for record in records:
        path = source(record)
        key = canonical_source(path)
        display_path.setdefault(key, path)
        entry = merged.setdefault(key, {"lines": {}, "branches": {}})
        for line in record:
            if line.startswith("DA:"):
                line_no, count = line[3:].split(",", 1)
                try:
                    entry["lines"][int(line_no)] = entry["lines"].get(int(line_no), 0) + int(count)
                except ValueError:
                    entry["lines"].setdefault(int(line_no), 0)
            elif line.startswith("BRDA:"):
                line_no, block, branch, taken = line[5:].split(",", 3)
                branch_key = (int(line_no), block, branch)
                previous = entry["branches"].get(branch_key, 0)
                if taken == "-":
                    entry["branches"].setdefault(branch_key, previous)
                else:
                    try:
                        entry["branches"][branch_key] = previous + int(taken)
                    except ValueError:
                        entry["branches"].setdefault(branch_key, previous)
    normalized = []
    for key, entry in sorted(merged.items()):
        record = [f"SF:{display_path[key]}"]
        for line_no, count in sorted(entry["lines"].items()):
            record.append(f"DA:{line_no},{count}")
        for (line_no, block, branch), taken in sorted(entry["branches"].items()):
            record.append(f"BRDA:{line_no},{block},{branch},{taken}")
        record.append("end_of_record")
        normalized.append(record)
    return normalized

rust_source_records = merge_records(read_records(combined))
native_source_records = merge_records(read_records(native))

def keep_ipopt(path):
    return (
        "/Ipopt-4667204c76e534d3e4df6b1462f258a4f9c681bd/src/Algorithm/" in path
        and "/src/Algorithm/LinearSolvers/" not in path
    )

def keep_rust_core(path):
    return path.endswith("/optimization/src/interior_point.rs") or path.endswith("/optimization/src/filter.rs")

def write_filtered(path, records, predicate):
    selected = [record for record in records if predicate(source(record))]
    path.write_text("".join("\n".join(record) + "\n" for record in selected))
    return selected

def stats(selected):
    line_total = line_hit = branch_total = branch_hit = 0
    for record in selected:
        for line in record:
            if line.startswith("DA:"):
                line_total += 1
                try:
                    if int(line.rsplit(",", 1)[1]) > 0:
                        line_hit += 1
                except ValueError:
                    pass
            elif line.startswith("BRDA:"):
                branch_total += 1
                taken = line.rsplit(",", 1)[1]
                if taken != "-":
                    try:
                        if int(taken) > 0:
                            branch_hit += 1
                    except ValueError:
                        pass
    return line_total, line_hit, branch_total, branch_hit

def record_stats(record):
    return stats([record])

def line_hits(record):
    hits = {}
    for line in record:
        if line.startswith("DA:"):
            line_no, count = line[3:].split(",", 1)
            try:
                hits[int(line_no)] = int(count)
            except ValueError:
                hits[int(line_no)] = 0
    return hits

def branch_lines(record):
    by_line = {}
    for line in record:
        if not line.startswith("BRDA:"):
            continue
        line_no_text, _block, _branch, taken = line[5:].split(",", 3)
        line_no = int(line_no_text)
        by_line.setdefault(line_no, [0, 0])
        by_line[line_no][0] += 1
        if taken == "-":
            continue
        try:
            if int(taken) > 0:
                by_line[line_no][1] += 1
        except ValueError:
            pass
    return by_line

def source_lines(path):
    try:
        return pathlib.Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return []

def source_text(path, line_no):
    lines = source_lines(path)
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1].strip()
    return ""

def rust_unhit_branch_like(records):
    branch_re = re.compile(r"\b(if|match|while|for|loop)\b|=>")
    rows = []
    for record in records:
        path = source(record)
        hits = line_hits(record)
        lines = source_lines(path)
        for line_no, text in enumerate(lines, start=1):
            stripped = text.strip()
            if not stripped or stripped.startswith("//"):
                continue
            if not branch_re.search(stripped):
                continue
            if hits.get(line_no, 0) != 0:
                continue
            rows.append((path, line_no, classify_rust_line(path, line_no, stripped), stripped))
    return rows

def classify_rust_line(path, line_no, text):
    file_name = pathlib.Path(path).name
    if file_name == "interior_point.rs":
        if "options.mehrotra_algorithm" in text:
            return "option/display reporting"
        if "options.adaptive_mu_restore_previous_iterate" in text:
            return "option/display reporting"
        if "let schedule = match options.schedule" in text:
            return "diagnostic linear replay"
        if "pivot_method: match" in text:
            return "linear solver option mapping"
        if "InteriorPointTermination::Acceptable | InteriorPointTermination::FeasiblePointFound" in text:
            return "termination display/reporting"
        if "InteriorPointTermination::Converged =>" in text:
            return "termination display/reporting"
        if 16680 <= line_no <= 16770:
            return "unit-test parity witness"
        if 17450 <= line_no <= 17680:
            return "unit-test parity witness"
        if 18180 <= line_no <= 18300:
            return "unit-test parity witness"
        if 16950 <= line_no <= 17320:
            return "unit-test parity witness"
        if 17680 <= line_no <= 17920:
            return "unit-test parity witness"
        if "let ineq_text = if has_inequality_like_constraints" in text:
            return "solve summary/reporting"
        if "let comp_text = if has_inequality_like_constraints" in text:
            return "solve summary/reporting"
        if "let count_cell = match count" in text:
            return "solve summary/reporting"
        if "Some(count) => format!" in text:
            return "solve summary/reporting"
        if "None => format!" in text:
            return "solve summary/reporting"
        if "log_boxed_section" in text:
            return "solve summary/reporting"
        if text == "if !matches!(":
            return "option validation"
        if "override length" in text:
            return "error handling"
        if "InteriorPointBoundMultiplierInitMethod::MuBased" in text:
            return "initialization option-profile branch"
        if "'restart_current_iteration: loop" in text:
            return "main solve loop"
        if "InteriorPointMuStrategy::Monotone" in text:
            return "monotone barrier update"
        if "InteriorPointMuStrategy::Adaptive" in text:
            return "adaptive barrier update"
        if "prepare_spral_workspace" in text or "prepare_native_spral_workspace" in text:
            return "linear-solver workspace setup"
        if "Ok(workspace)" in text:
            return "linear-solver workspace setup"
        if "options.linear_solver == InteriorPointLinearSolver::SpralSrc" in text:
            return "linear-solver workspace setup"
        if "tiny_step_unchecked_accept" in text:
            return "line-search tiny-step branch"
        if "watchdog_active" in text or "watchdog_reference" in text or "watchdog_assessment" in text or "stored_point" in text:
            return "watchdog globalization branch"
        if "step_kind == InteriorPointStepKind::Feasibility" in text:
            return "watchdog globalization branch"
        if "IpoptRestorationAlgorithmStatus::MaxIterationsExceeded" in text:
            return "restoration status mapping"
        if 24700 <= line_no <= 24740:
            return "restoration status mapping"
        if 25170 <= line_no <= 25250:
            return "restoration status mapping"
        if line_no < 760:
            return "option/display/reporting"
        if 1200 <= line_no <= 1560:
            return "diagnostic/restoration branch"
        if 1560 <= line_no <= 1685:
            return "fixed-variable elimination"
        if 1685 <= line_no <= 2030:
            return "diagnostic/restoration branch"
        if text == "_ => iteration,":
            return "diagnostic/restoration branch"
        if "barrier_parameter: if barrier_pair_count > 0" in text:
            return "solve summary/reporting"
        if 1040 <= line_no <= 1135:
            return "diagnostic/reporting"
        if 1930 <= line_no <= 2180 and "ssids_rs" in text:
            return "unreachable: non-parity linear solver/debug branch"
        if 2160 <= line_no <= 2200:
            return "restoration status mapping"
        if 2030 <= line_no <= 2180:
            return "linear-solver workspace setup"
        if 2200 <= line_no <= 2290:
            return "diagnostic linear replay"
        if 2320 <= line_no <= 3075:
            return "diagnostic linear replay"
        if 3075 <= line_no <= 3605:
            return "diagnostic factorization progress"
        if 3605 <= line_no <= 3745:
            return "diagnostic factorization progress"
        if 3745 <= line_no <= 3875:
            return "monotone barrier update"
        if "WatchdogStopKind::" in text:
            return "watchdog globalization branch"
        if 3850 <= line_no <= 3910 and "InteriorPointAlphaForYStrategy::" in text:
            return "covered by alpha_for_y option-profile witness"
        if 3900 <= line_no <= 4100:
            return "termination display/tolerance reporting"
        if 4100 <= line_no <= 4195:
            return "bounds and IPOPT dense-vector arithmetic helper"
        if 4200 <= line_no <= 5170:
            return "bounds, fixed-variable, and sparse-preprocessing helper"
        if 5180 <= line_no <= 5470:
            return "KKT pattern validation/preprocessing"
        if 5470 <= line_no <= 5550:
            return "bounds, fixed-variable, and sparse-preprocessing helper"
        if 5550 <= line_no <= 6990:
            return "full-space residual/refinement helper"
        if 6930 <= line_no <= 6945:
            return "full-space residual/refinement helper"
        if 7000 <= line_no <= 7170:
            return "unreachable: non-parity linear solver/debug branch"
        if 7170 <= line_no <= 8485:
            return "linear solver dispatch, retry, and diagnostics"
        if 8485 <= line_no <= 8970:
            return "linear solver fallback/retry path"
        if 8970 <= line_no <= 9565:
            return "iteration log formatting/reporting"
        if 9560 <= line_no <= 10030:
            return "bound/slack stationarity helper alternate branch"
        if 10030 <= line_no <= 10210:
            return "unit-test parity witness"
        if 10400 <= line_no <= 11050:
            return "unit-test parity witness"
        if 10200 <= line_no <= 10645:
            return "solve summary/reporting"
        if 11050 <= line_no <= 11340:
            return "solve summary/reporting"
        if 11360 <= line_no <= 11440:
            return "covered by reduced negative-curvature helper"
        if 11340 <= line_no <= 11840:
            return "initialization option-profile branch"
        if 11640 <= line_no <= 11790:
            return "linear solver workspace selection"
        if 11840 <= line_no <= 11990:
            return "solve summary/reporting"
        if 11920 <= line_no <= 11970:
            return "unreachable: non-parity linear solver/debug branch"
        if 11980 <= line_no <= 12120:
            return "watchdog globalization start gate"
        if 12650 <= line_no <= 12920:
            return "perturbation and inertia retry policy"
        if 13680 <= line_no <= 13795:
            return "full-space negative-curvature option branch"
        if 12120 <= line_no <= 14330:
            return "SOC/watchdog globalization branch"
        if 14330 <= line_no <= 14920:
            return "accepted-step and watchdog state bookkeeping"
        if 16200 <= line_no <= 16460:
            return "unit-test parity witness"
        if 14920 <= line_no <= 16730:
            return "iteration log formatting/reporting"
        if 16750 <= line_no <= 16795:
            return "iteration log formatting/reporting"
        if 16900 <= line_no <= 19020:
            return "unit-test parity witness"
        if 25945 <= line_no <= 26170:
            return "covered by acceptable restoration fallback helper"
        if 26420 <= line_no <= 26445:
            return "covered by acceptable restoration fallback helper"
        if 24800 <= line_no <= 24900:
            return "restoration status mapping"
        if 25240 <= line_no <= 25330:
            return "restoration status mapping"
        if 25480 <= line_no <= 25560:
            return "restoration status mapping"
        if 25690 <= line_no <= 25945:
            return "restoration status mapping"
        if 14720 <= line_no <= 14770:
            return "iteration log formatting/reporting"
    if any(token in text for token in ["SparseQdldl", "SsidsRs", "Auto", "compare_solvers"]):
        return "unreachable: non-parity linear solver/debug branch"
    if any(token in text for token in ["Self::", "impl Default", "label()", "as_str", "format_", "InteriorPointSpralPivotMethod"]):
        return "option/display mapping"
    if any(token in text for token in ["InteriorPointIterationEvent", "present_codes", "summary"]):
        return "diagnostic/reporting"
    if any(token in text for token in ["InteriorPointLinearDebugSchedule", "snapshot", "IpoptLinearRhsOrientation", "report.results", "report.notes", "matched_solvers", "mismatch", "compare_success"]):
        return "diagnostic linear replay"
    if any(token in text for token in ["linear_debug", "trace", "dump", "verbose", "debug"]):
        return "diagnostic only"
    if any(token in text for token in ["spral_action", "least_square_init", "mu_allow", "second_order_correction", "expect_infeasible_problem"]):
        return "option/reporting"
    if "restoration" in text or "Restoration" in text:
        return "diagnostic/restoration branch"
    if any(token in text for token in ["return Err", "Err(", "panic!", "assert!"]):
        return "error handling"
    return "needs audit"

def classify_ipopt_line(file_name, line_no, text):
    if file_name.startswith("IpResto"):
        if any(token in text for token in ["Jnlst", "Output", "Printf", "print_", "WallclockTime"]):
            return "diagnostic/restoration branch"
        return "restoration branch"
    if file_name == "IpPDPerturbationHandler.cpp":
        if 472 <= line_no <= 529:
            return "structural degeneracy finalization/status witness gap"
        if line_no in {386, 395}:
            return "covered by hessian-perturbation growth witness"
        if line_no in {435, 443}:
            return "covered by wrong-inertia fallback witness"
        if 158 <= line_no <= 338:
            return "covered by perturbation retry state-machine witnesses"
    if file_name == "IpPDFullSpaceSolver.cpp" and line_no < 320:
        return "diagnostic only"
    if file_name == "IpPDFullSpaceSolver.cpp":
        if line_no in {409, 414, 566}:
            return "option/error handling"
        if line_no in {423, 839, 849, 885, 900, 945}:
            return "linear-solver error path"
        if line_no in {524, 794}:
            return "linear-solver-failure witness gap"
        if line_no in {559, 560, 577, 578, 579}:
            return "covered by refinement option/reduced threshold witnesses"
        if line_no in {590, 595, 600, 621, 626, 651}:
            return "covered by reduced and live failed-refinement witnesses"
        if line_no == 656:
            return "covered by full-space final assembly witnesses"
        if line_no in {773, 819}:
            return "covered by live quality/pretend retry witnesses"
        if line_no in {843, 857, 866, 871, 882, 895}:
            return "covered by perturbation retry witnesses"
        if line_no in {831, 908, 913, 925, 939}:
            return "covered by negative-curvature option/helper witnesses"
        if 1030 <= line_no <= 1090:
            return "diagnostic only"
        if line_no == 1198:
            return "covered by reduced residual-ratio witness"
    if file_name == "IpFilterLSAcceptor.cpp" and 665 <= line_no <= 868:
        return "unreachable under parity options"
    if file_name == "IpFilterLSAcceptor.cpp":
        if line_no == 214:
            return "covered by filter theta option-rejection witness"
        if line_no in {283, 284}:
            return "covered by reduced filter witness"
        if line_no == 341:
            return "covered by theta-max filter witness"
        if line_no == 409:
            return "source-dead filter reset cap branch"
        if line_no == 487:
            return "covered by objective max-increase witness"
        if line_no == 576:
            return "covered by SOC method option witnesses"
        if line_no == 609:
            return "linear-solver-failure witness gap"
    if file_name == "IpBacktrackingLineSearch.cpp":
        if line_no == 203:
            return "covered by alpha_for_y option-profile witness"
        if line_no == 307:
            return "source-dead: stock BacktrackingLSAcceptor::DoFallback is false"
        if line_no in {568, 580, 581, 584, 611, 614}:
            return "covered by acceptable restoration fallback helper"
        if line_no == 403:
            return "covered by tiny-step eval-error fallback witness"
        if line_no == 422:
            return "covered by tiny-step dual-threshold witness"
        if line_no == 1261:
            return "covered by reduced tiny-step slack-step witness"
        if line_no in {455, 458, 546, 1144, 1167, 1172, 1182}:
            return "soft-restoration witness gap"
        if line_no == 520:
            return "unreachable under parity options"
        if line_no == 800:
            return "expect-infeasible restoration witness gap"
        if line_no in {902, 931, 933}:
            return "covered by alpha_for_y option-profile witness"
        if line_no == 1283:
            return "covered by acceptable-iterate backup witness"
        if line_no == 1300:
            return "covered by acceptable restoration fallback helper"
        if line_no == 1316:
            return "source fallback activation condition"
    if "GetCachedResult" in text or "_cache_" in text:
        return "cache bookkeeping"
    if "IsValid(add_cq_)" in text:
        return "unreachable under parity options"
    if any(token in text for token in ["CPUTIME_EXCEEDED", "WALLTIME_EXCEEDED", "DIVERGING", "USER_STOP", "INTERNAL_ERROR"]):
        return "termination/error status"
    if any(token in text for token in ["emergency_mode", "fallback_activated_", "ActivateFallbackMechanism", "STEP_COMPUTATION_FAILED"]):
        return "failure/fallback path"
    if any(token in text for token in ["expect_infeasible_problem", "start_with_resto", "soft_resto", "in_soft_resto", "goto_resto"]):
        return "restoration branch"
    if "fast_step_computation" in text:
        return "unreachable under parity options"
    if file_name == "IpDefaultIterateInitializer.cpp" and 360 <= line_no <= 470:
        return "unreachable under parity options"
    if file_name == "IpDefaultIterateInitializer.cpp":
        if line_no == 157:
            return "covered by least-square dual initializer witnesses"
        if line_no in {160, 702}:
            return "equality-multiplier calculator failure witness gap"
        if line_no == 192:
            return "TNLP initialization failure path"
        if line_no == 284:
            return "option/error handling"
        if line_no == 493:
            return "covered by initializer push option-rejection witness"
    if file_name == "IpDefaultIterateInitializer.cpp" and line_no in {208, 302}:
        return "unreachable under parity options"
    if file_name == "IpMonotoneMuUpdate.cpp":
        if line_no == 158:
            return "covered by tiny-step best-possible termination witness"
        if line_no == 176:
            return "covered by reduced monotone barrier helper"
        if line_no == 188:
            return "covered by reduced monotone line-search reset helper"
    if file_name == "IpPDSearchDirCalc.cpp":
        if line_no == 133:
            return "search-direction failure/status witness gap"
    if file_name == "IpIpoptCalculatedQuantities.cpp":
        if line_no == 474:
            return "covered by reduced adjusted-slack helper"
        if line_no == 479:
            return "covered by reduced adjusted-slack helper"
        if line_no in {702, 715}:
            return "objective-depends-on-mu TNLP feature gap"
        if line_no in {928, 1021}:
            return "covered by zero/nonzero kappa_d damping witnesses"
        if line_no in {1552, 1906, 1920}:
            return "unscaled constraint-violation dimension witness gap"
        if line_no in {1600, 1636, 1738, 1743, 1778, 1783}:
            return "original-bound violation/bound-relaxation witness gap"
        if line_no in {1811, 1832, 1855, 2781}:
            return "unreachable under parity options"
        if line_no in {1818, 1839, 1868, 3757, 3766}:
            return "cache bookkeeping"
        if line_no in {2528, 2563}:
            return "option/error handling"
        if line_no in {2543, 2544}:
            return "component norm option witness gap"
        if line_no == 2911:
            return "mu-target complementarity witness gap"
        if line_no in {2981, 2991, 3238, 3245, 3298, 3305, 3587, 3642}:
            return "zero-dimension component witness gap"
    if file_name == "IpIpoptData.cpp":
        if line_no in {95, 171, 271}:
            return "unreachable under parity options"
        if line_no == 135:
            return "initialization failure path"
        if line_no == 187:
            return "trial-container allocation bookkeeping"
    if file_name == "IpIpoptAlg.cpp":
        if line_no == 132:
            return "option/display reporting"
        if line_no == 450:
            return "termination/error status"
        if line_no == 456:
            return "termination watchdog cleanup witness gap"
        if line_no in {484, 542}:
            return "square feasibility multiplier postprocess witness gap"
        if line_no == 586:
            return "barrier-update status reporting"
        if line_no == 613:
            return "search-direction status reporting"
        if line_no == 657:
            return "skipped line-search witness gap"
        if line_no in {667, 670}:
            return "covered by reduced adjusted-slack helper"
        if line_no in {723, 733, 743, 753, 764}:
            return "covered by kappa_sigma bound-multiplier witness"
        if line_no in {777, 792, 797}:
            return "unreachable under parity options"
        if line_no == 823 or line_no >= 998:
            return "diagnostic problem-statistics reporting"
        if 857 <= line_no <= 990:
            return "square feasibility multiplier postprocess witness gap"
    if any(token in text for token in ["Optivibre", "dump", "Dump", "std::fprintf", "file", "Jnlst", "ProduceOutput"]):
        return "diagnostic only"
    if any(token in text for token in ["GetNumericValue", "GetBoolValue", "GetStringValue", "GetEnumValue", "ASSERT_EXCEPTION", "OPTION_INVALID", "FAILED_INITIALIZATION", "INTERNAL_ABORT"]):
        return "option/error handling"
    if any(token in text for token in ["warm_start", "least_square", "mehrotra", "LIMITED_MEMORY", "adaptive_mu", "accept_every_trial", "corrector_type", "recalc_y"]):
        return "unreachable under parity options"
    if any(token in text for token in ["Resto", "resto", "Restoration"]):
        return "restoration branch"
    return "active/watch"

def watched_branch_rows(records):
    watched = [
        ("DefaultIterateInitializer", ["IpDefaultIterateInitializer.cpp"], "initialization"),
        ("SetTrial/AcceptTrial/correct_bound_multiplier", ["IpIpoptData.cpp", "IpIpoptAlg.cpp"], "accepted-state mutation"),
        ("MonotoneMuUpdate", ["IpMonotoneMuUpdate.cpp"], "barrier update"),
        ("PDSearchDirCalc", ["IpPDSearchDirCalc.cpp"], "KKT direction"),
        ("PDFullSpaceSolver", ["IpPDFullSpaceSolver.cpp"], "full-space solve/refinement"),
        ("PDPerturbationHandler", ["IpPDPerturbationHandler.cpp"], "regularization/inertia"),
        ("BacktrackingLineSearch", ["IpBacktrackingLineSearch.cpp"], "globalization"),
        ("FilterLSAcceptor", ["IpFilterLSAcceptor.cpp"], "filter/SOC"),
        ("IpoptCalculatedQuantities", ["IpIpoptCalculatedQuantities.cpp"], "state/RHS components"),
        ("Restoration", ["IpResto"], "restoration"),
    ]
    rows = []
    for label, names, reason in watched:
        selected = []
        for record in records:
            file_name = pathlib.Path(source(record)).name
            if any(file_name == name or file_name.startswith(name) for name in names):
                selected.append(record)
        total = stats(selected)
        uncovered = []
        for record in selected:
            path = source(record)
            for line_no, (branch_total, branch_hit) in sorted(branch_lines(record).items()):
                missing = branch_total - branch_hit
                if missing <= 0:
                    continue
                file_name = pathlib.Path(path).name
                classification = classify_ipopt_line(file_name, line_no, source_text(path, line_no))
                uncovered.append(
                    (
                        file_name,
                        line_no,
                        missing,
                        branch_total,
                        classification,
                        source_text(path, line_no),
                    )
                )
        rows.append((label, reason, len(selected), total, uncovered))
    return rows

ipopt_records = write_filtered(ipopt_out, native_source_records, keep_ipopt)
rust_records = write_filtered(rust_out, rust_source_records, keep_rust_core)
ipopt_line_total, ipopt_line_hit, ipopt_branch_total, ipopt_branch_hit = stats(ipopt_records)
rust_line_total, rust_line_hit, rust_branch_total, rust_branch_hit = stats(rust_records)

audit_out.write_text(
    "\n".join(
        [
            "# IPOPT Parity Coverage Audit",
            "",
            "| Scope | Files | Lines hit / total | Branches hit / total |",
            "| --- | ---: | ---: | ---: |",
            f"| IPOPT Algorithm C++ | {len(ipopt_records)} | {ipopt_line_hit} / {ipopt_line_total} | {ipopt_branch_hit} / {ipopt_branch_total} |",
            f"| Rust NLIP core | {len(rust_records)} | {rust_line_hit} / {rust_line_total} | {rust_branch_hit} / {rust_branch_total} |",
            "",
            "Filtered reports:",
            f"- `{ipopt_out}`",
            f"- `{rust_out}`",
            f"- `{native}`",
            f"- `{branch_ledger_out}`",
        ]
    )
    + "\n"
)

ledger = [
    "# IPOPT Parity Branch Ledger",
    "",
    "This report is generated from the focused nonlinear parity coverage run.",
    "It is intentionally source-line oriented: uncovered branches here are either",
    "test gaps to close or branches that need an explicit unreachable/diagnostic",
    "classification in `docs/ipopt-parity-coverage.md`.",
    "",
    "## Watched IPOPT Branch Surfaces",
    "",
    "| Surface | Reason | Files | Lines hit / total | Branches hit / total | Active-watch / uncovered branch lines |",
    "| --- | --- | ---: | ---: | ---: | ---: |",
]

for label, reason, file_count, total, uncovered in watched_branch_rows(ipopt_records):
    line_total, line_hit, branch_total, branch_hit = total
    active_uncovered = sum(1 for row in uncovered if row[4] == "active/watch")
    ledger.append(
        f"| {label} | {reason} | {file_count} | {line_hit} / {line_total} | {branch_hit} / {branch_total} | {active_uncovered} / {len(uncovered)} |"
    )

ledger.extend(["", "## IPOPT Uncovered Branch Lines", ""])
for label, reason, _file_count, _total, uncovered in watched_branch_rows(ipopt_records):
    ledger.append(f"### {label}")
    if not uncovered:
        ledger.append("")
        ledger.append("No uncovered branch lines in the focused report.")
        ledger.append("")
        continue
    ledger.append("")
    ledger.append("| File | Line | Missing / total branches | Classification | Source |")
    ledger.append("| --- | ---: | ---: | --- | --- |")
    for file_name, line_no, missing, branch_total, classification, text in uncovered[:40]:
        text = text.replace("|", "\\|")
        ledger.append(f"| {file_name} | {line_no} | {missing} / {branch_total} | {classification} | `{text}` |")
    if len(uncovered) > 40:
        ledger.append(f"| ... | ... | ... | {len(uncovered) - 40} more omitted; inspect `{ipopt_out}` |")
    ledger.append("")

rust_unhit = rust_unhit_branch_like(rust_records)
ledger.extend(
    [
        "## Rust Core Unhit Branch-Like Lines",
        "",
        "| File | Line | Classification | Source |",
        "| --- | ---: | --- | --- |",
    ]
)
for path, line_no, classification, text in rust_unhit[:120]:
    rel = path.replace(str(pathlib.Path.cwd()) + "/", "")
    text = text.replace("|", "\\|")
    ledger.append(f"| {rel} | {line_no} | {classification} | `{text}` |")
if len(rust_unhit) > 120:
    ledger.append(f"| ... | ... | ... | {len(rust_unhit) - 120} more omitted; inspect `{rust_out}` |")
ledger.append("")
needs_audit = sum(1 for _path, _line, classification, _text in rust_unhit if classification == "needs audit")
ledger.extend(
    [
        "## Audit Counts",
        "",
        f"- Rust unhit branch-like lines: {len(rust_unhit)}",
        f"- Rust unhit branch-like lines still marked `needs audit`: {needs_audit}",
        "- Watched IPOPT rows show `active/watch / total` uncovered branch-line counts.",
    ]
)
branch_ledger_out.write_text("\n".join(ledger) + "\n")
if needs_audit:
    for path, line_no, classification, text in rust_unhit:
        if classification == "needs audit":
            rel = path.replace(str(pathlib.Path.cwd()) + "/", "")
            print(f"{rel}:{line_no}: {text}", file=sys.stderr)
    print(
        f"Rust core coverage still has {needs_audit} unclassified branch-like lines; see {branch_ledger_out}",
        file=sys.stderr,
    )
    sys.exit(2)
PY

cleanup_stray_profraw
echo "coverage reports written to $REPORT_DIR"
