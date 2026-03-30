#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "parity" / "casadi_v1.toml"
CASADI_ROOT = REPO_ROOT / "casadi_reference"
REPORT_DIR = REPO_ROOT / "target" / "reports"
REPORT_PATH = REPORT_DIR / "casadi_parity_audit.md"
JSON_PATH = REPORT_DIR / "casadi_parity_audit.json"
SCOPED_FILES = (
    "test/python/sx.py",
    "test/python/function.py",
    "test/python/ad.py",
    "test/python/sparsity.py",
)
TEST_RE = re.compile(r"^\s*def\s+(test_[A-Za-z0-9_]+)\s*\(")

STATUS_COLORS = {
    "exact_supported": "#2f9e44",
    "questionable_supported": "#f08c00",
    "intentionally_descoped": "#1971c2",
    "unsupported": "#fab005",
    "untracked": "#e03131",
}

LIKELY_IN_SCOPE_GAPS = {
    "test/python/sx.py::test_substitute": (
        "Symbolic substitution is the next missing transformation surface and would add a real graph-rewrite API."
    ),
    "test/python/sx.py::test_mtaylor": (
        "Multivariate Taylor expansion is still missing and would introduce a larger symbolic transformation layer."
    ),
    "test/python/sx.py::test_evalfail": (
        "The dynamic-language misuse path for symbolic inputs to numeric evaluation still needs a typed-Rust parity stance."
    ),
}

MANIFEST_SUPPORTED_ASSESSMENTS = {
    "test/python/sx.py::test_scalarSX": (
        "equivalent",
        "Local coverage now evaluates the same scalar unary/power pool at the upstream test "
        "point, including `log1p` and `expm1`.",
    ),
    "test/python/sx.py::test_gradient": (
        "equivalent",
        "Local coverage now checks the exact recursive `x**p` Jacobian chain used upstream.",
    ),
    "test/python/sx.py::test_gradient2": (
        "equivalent",
        "Local coverage now checks the symbolic-exponent recursive Jacobian chain used upstream.",
    ),
    "test/python/sx.py::test_SXJacobian": (
        "equivalent",
        "Local coverage now differentiates the same scalar unary/power pool at the upstream test "
        "point.",
    ),
    "test/python/sx.py::test_SXJac": (
        "equivalent",
        "Local coverage now checks the same scalar `jacobian(y[0], x[0])` construction used "
        "upstream.",
    ),
    "test/python/sx.py::test_SXJacobians": (
        "equivalent",
        "Local coverage now checks the same dense 3x1 unary Jacobian pool used upstream.",
    ),
    "test/python/sx.py::test_SXJacobians2": (
        "equivalent",
        "Local coverage now checks the same dense 1x3 unary Jacobian pool used upstream.",
    ),
    "test/python/sx.py::test_equivalence": (
        "equivalent",
        "Local coverage now checks the same `log1p`, `expm1`, and `hypot` equivalence cases.",
    ),
    "test/python/sx.py::test_const_folding_on_the_fly": (
        "equivalent",
        "Local simplification now collapses the same nested scalar products as the upstream test.",
    ),
    "test/python/sx.py::test_issue107": (
        "equivalent",
        "Local coverage now checks the same `z = x; z += y` aliasing regression on immutable SX "
        "handles.",
    ),
    "test/python/sx.py::test_SXbinary": (
        "equivalent",
        "Local coverage now checks the same dense matrix binary pool through a test-only dense "
        "adaptor over sparse-first `SXMatrix` primitives.",
    ),
    "test/python/sx.py::test_SXbinarySparse": (
        "equivalent",
        "Local coverage now checks the same sparse matrix binary pool through a test-only dense "
        "adaptor over sparse-first `SXMatrix` primitives.",
    ),
    "test/python/sx.py::test_SXbinary_diff": (
        "equivalent",
        "Local coverage now checks the same directional derivative pool through a test-only dense "
        "adaptor over sparse-first `SXMatrix` primitives and forward AD.",
    ),
    "test/python/sx.py::test_DMbinary": (
        "equivalent",
        "Local coverage now checks the same numeric dense binary pool through constant `SXMatrix` "
        "graphs and the same dense adaptor used for symbolic parity.",
    ),
    "test/python/sx.py::test_SXbinary_codegen": (
        "equivalent",
        "Local coverage now checks the same binary pool through an LLVM backend adaptor: AOT "
        "object emission succeeds and JIT evaluation matches the reference values.",
    ),
    "test/python/sx.py::test_SXsimplifications": (
        "equivalent",
        "Local coverage now checks the same simplification-on-the-fly regression pool, including "
        "the exact canonical string forms `x` and `(-x)`.",
    ),
    "test/python/sx.py::test_copysign": (
        "equivalent",
        "Local coverage now checks the same `copysign` values and first derivatives through a "
        "test-only branchless adaptor over the existing unary scalar ops.",
    ),
    "test/python/sx.py::test_depends_on": (
        "equivalent",
        "Local coverage now checks the same scalar and vector dependency cases through a "
        "test-only adaptor built from `free_symbols()` on sparse-first SX matrices.",
    ),
    "test/python/sx.py::test_symvar": (
        "equivalent",
        "Local coverage now checks the same free-symbol discovery order through a test-only "
        "`symvar` adaptor over `free_symbols()`.",
    ),
    "test/python/sx.py::test_contains": (
        "equivalent",
        "Local coverage now checks the same scalar membership helpers, including the same "
        "non-scalar error case, through test-only adaptors.",
    ),
    "test/python/sx.py::test_pow": (
        "equivalent",
        "Local coverage now checks the same dense `pow(..., 0)` nonzero-count regression "
        "through a test-only dense matrix adaptor.",
    ),
    "test/python/sx.py::test_primitivefunctions": (
        "equivalent",
        "Local coverage now checks the same primitive scalar utility pool through branchless "
        "adaptors built from the existing unary/binary math surface.",
    ),
    "test/python/sx.py::test_is_regular": (
        "equivalent",
        "Local coverage now checks the same regularity cases through a test-only adaptor that "
        "treats non-finite constants as irregular and symbolic-only expressions as undecidable.",
    ),
    "test/python/sx.py::test_SX_func": (
        "equivalent",
        "Local coverage now checks the same constructor shape invariants through the typed "
        "`NamedMatrix`/`SXFunction` API.",
    ),
    "test/python/sx.py::test_SX": (
        "equivalent",
        "Local coverage now checks the same dense matrix unary pool, including the 3-by-3 det "
        "and inverse tail, through test-only matrix adaptors over sparse-first primitives.",
    ),
    "test/python/sx.py::test_SXSparse": (
        "equivalent",
        "Local coverage now checks the same sparse matrix unary pool through a test-only adaptor "
        "that preserves structural zeros rather than densifying the runtime API.",
    ),
    "test/python/sx.py::test_SXslicing": (
        "equivalent",
        "Local coverage now checks the same dense and sparse indexing and `.nz[...]` selection "
        "cases through explicit test-only slice adaptors over sparse-first matrices.",
    ),
    "test/python/sx.py::test_SXconversion": (
        "equivalent",
        "Local coverage now checks the same scalar and matrix conversion surface through typed "
        "`SX` and `SXMatrix` constructors.",
    ),
    "test/python/sx.py::test_SX_func2": (
        "equivalent",
        "Local coverage now checks the same scalar typemap constructor roundtrip through "
        "transpose/transpose on 1-by-1 symbolic and numeric matrices.",
    ),
    "test/python/sx.py::test_SX_func3": (
        "equivalent",
        "Local coverage now checks the same vector constructor cases through a test-only vertical "
        "concatenation adaptor over sparse-first `SXMatrix` values.",
    ),
    "test/python/sx.py::test_SX1": (
        "equivalent",
        "Local coverage now checks the same function evaluation and Jacobian evaluation cases "
        "through a test-only LLVM JIT call adaptor.",
    ),
    "test/python/sx.py::test_SX2": (
        "equivalent",
        "Local coverage now checks the same symbolic function evaluation case through a test-only "
        "LLVM JIT call adaptor; Python-specific repr strings remain language-specific.",
    ),
    "test/python/sx.py::test_eval": (
        "equivalent",
        "Local coverage now checks the same function evaluation case through a test-only LLVM JIT "
        "call adaptor over sparse-first dense matrices.",
    ),
    "test/python/sx.py::test_symbolcheck": (
        "equivalent",
        "Local coverage now checks the same rejection of non-symbolic function inputs during "
        "`SXFunction` construction.",
    ),
    "test/python/sx.py::test_sparseconstr": (
        "equivalent",
        "Local coverage now checks the same lower-triangular and diagonal sparsity constructors "
        "through concrete `CCS` patterns.",
    ),
    "test/python/sx.py::test_null": (
        "equivalent",
        "Local coverage now checks the same empty-slot function behavior through a test-only LLVM "
        "JIT call adaptor and explicit empty `SXMatrix` slots.",
    ),
    "test/python/sx.py::test_evalchecking": (
        "equivalent",
        "Local coverage now checks the same evaluation shape checks, including the reshape-"
        "compatible 5-by-1 to 1-by-5 acceptance, through a test-only LLVM JIT call adaptor.",
    ),
    "test/python/ad.py::test_hessian": (
        "equivalent",
        "Local coverage now checks the same dense Hessian-by-Jacobian-chaining construction.",
    ),
    "test/python/ad.py::test_Jacobian": (
        "equivalent",
        "Local coverage now checks the same Jacobian cases via a test-only dense adaptor over the "
        "sparse slot-based `SXMatrix::jacobian` API.",
    ),
    "test/python/ad.py::test_jacobianSX": (
        "equivalent",
        "Local coverage now checks the same direct SX Jacobian cases via a test-only dense "
        "adaptor over the sparse slot-based API.",
    ),
    "test/python/ad.py::test_jacsparsity": (
        "equivalent",
        "Local coverage now checks the same Jacobian sparsity cases via a test-only dense "
        "adaptor over `SXMatrix::jacobian_ccs`.",
    ),
    "test/python/ad.py::test_bugshape": (
        "equivalent",
        "Local coverage now checks the same shape-preservation regression via a test-only "
        "adaptor from the de-scoped Function derivative API to direct `SXMatrix` differentiation.",
    ),
    "test/python/ad.py::test_bugglibc": (
        "equivalent",
        "Local coverage now checks the same regression that repeated Jacobian construction after "
        "evaluation remains stable and does not corrupt internal state.",
    ),
    "test/python/sparsity.py::test_NZ": (
        "equivalent",
        "Local coverage now checks the same triplet/nonzero constructor parity on concrete CCS.",
    ),
    "test/python/sparsity.py::test_union": (
        "equivalent",
        "Local coverage now checks the same sparsity union case through concrete CCS set-union "
        "helpers and a test-only adaptor for the old operator syntax.",
    ),
    "test/python/sparsity.py::test_intersection": (
        "equivalent",
        "Local coverage now checks the same sparsity intersection case through concrete CCS "
        "set-intersection helpers and a test-only adaptor for the old operator syntax.",
    ),
    "test/python/sparsity.py::test_rowcol": (
        "equivalent",
        "Local coverage now checks the same row/column cross-product constructor.",
    ),
    "test/python/sparsity.py::test_get_ccs": (
        "equivalent",
        "Local coverage now checks the same CCS array export and transpose/CRS relationship.",
    ),
    "test/python/sparsity.py::test_get_nzDense": (
        "equivalent",
        "Local coverage now checks the same dense linear-index roundtrip through `find()` on "
        "concrete CCS.",
    ),
    "test/python/sparsity.py::test_splower": (
        "equivalent",
        "Local coverage now checks the same lower-pattern extraction on concrete CCS.",
    ),
    "test/python/sparsity.py::test_inverse": (
        "equivalent",
        "Local coverage now checks the same complement-pattern construction as CasADi's "
        "`pattern_inverse()`.",
    ),
    "test/python/sparsity.py::test_kron": (
        "equivalent",
        "Local coverage now checks the same Kronecker-product sparsity construction on concrete "
        "CCS.",
    ),
    "test/python/sparsity.py::test_nz_method": (
        "equivalent",
        "Local coverage now checks the same sparse lookup-by-linear-index behavior through "
        "concrete `CCS::get_nz`.",
    ),
    "test/python/sparsity.py::test_serialize": (
        "equivalent",
        "Local coverage now checks the same concrete CCS serialization roundtrip cases as the "
        "upstream Sparsity test.",
    ),
    "test/python/sparsity.py::test_is_subset": (
        "equivalent",
        "Local coverage now checks the same concrete subset relation cases over dense, lower, "
        "diagonal, and empty CCS patterns.",
    ),
    "test/python/sparsity.py::test_find_nonzero": (
        "equivalent",
        "Local coverage now checks the same linear-index `find()` roundtrip through the concrete "
        "constructor.",
    ),
    "test/python/sparsity.py::test_reshape": (
        "equivalent",
        "Local coverage now checks the same column-major sparsity reshape mapping.",
    ),
    "test/python/sparsity.py::test_enlarge": (
        "equivalent",
        "Local coverage now checks the same dense and sparse enlarge/remap cases.",
    ),
    "test/python/sparsity.py::test_diag": (
        "equivalent",
        "Local coverage now checks the same diagonal extraction cases for matrices and vectors.",
    ),
    "test/python/sparsity.py::test_refcount": (
        "equivalent",
        "Local coverage now checks the same lower-triangular matrix-multiply shape regression "
        "through a test-only dense matrix multiplication adaptor.",
    ),
}

ASSESSMENT_LABELS = {
    "equivalent": "Semantically equivalent",
    "proxy_only": "Weaker local proxy",
    "design_divergence": "Current design diverges",
    "false_positive": "Overclaimed today",
}


@dataclass(frozen=True)
class ManifestEntry:
    case: str
    status: str
    rust_test: str | None


@dataclass(frozen=True)
class FileAudit:
    path: str
    total: int
    exact_supported: int
    questionable_supported: int
    intentionally_descoped: int
    unsupported: int
    untracked: int
    intentionally_descoped_cases: list[str]
    unsupported_cases: list[str]
    untracked_cases: list[str]


@dataclass(frozen=True)
class AuditSummary:
    casadi_commit: str
    total_upstream: int
    total_manifest_entries: int
    exact_supported: int
    questionable_supported: int
    intentionally_descoped: int
    unsupported: int
    untracked: int
    files: list[FileAudit]
    likely_in_scope_gaps: list[dict[str, str]]
    manifest_supported_notes: list[dict[str, str]]


def parse_upstream_cases(relative_path: str) -> list[str]:
    path = CASADI_ROOT / relative_path
    cases: list[str] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        match = TEST_RE.match(line)
        if match:
            case = f"{relative_path}::{match.group(1)}"
            if case not in seen:
                cases.append(case)
                seen.add(case)
    return cases


def load_manifest() -> tuple[str, list[ManifestEntry]]:
    payload = tomllib.loads(MANIFEST_PATH.read_text())
    entries = [
        ManifestEntry(
            case=entry["casadi_case"],
            status=entry["status"],
            rust_test=entry.get("rust_test"),
        )
        for entry in payload["entries"]
    ]
    return payload["casadi_commit"], entries


def classify_supported_case(case: str) -> str:
    assessment = MANIFEST_SUPPORTED_ASSESSMENTS.get(case)
    if assessment is None:
        return "questionable_supported"
    return "exact_supported" if assessment[0] == "equivalent" else "questionable_supported"


def relink(case: str) -> str:
    relative_path, test_name = case.split("::", 1)
    return f"`{relative_path}::{test_name}`"


def percent(count: int, total: int) -> float:
    return 0.0 if total == 0 else 100.0 * count / total


def render_badge(label: str, value: str, color: str) -> str:
    return (
        f"<span style=\"display:inline-block;padding:0.25rem 0.5rem;border-radius:999px;"
        f"background:{color};color:white;font-weight:600;margin-right:0.35rem;\">"
        f"{label}: {value}</span>"
    )


def render_stacked_bar(
    segments: list[tuple[str, int]],
    *,
    width: int = 860,
    height: int = 22,
) -> str:
    total = sum(count for _, count in segments)
    if total == 0:
        return ""

    x = 0.0
    rects: list[str] = []
    labels: list[str] = []
    for key, count in segments:
        if count == 0:
            continue
        segment_width = width * count / total
        rects.append(
            f"<rect x=\"{x:.3f}\" y=\"0\" width=\"{segment_width:.3f}\" height=\"{height}\" "
            f"rx=\"4\" ry=\"4\" fill=\"{STATUS_COLORS[key]}\"></rect>"
        )
        x += segment_width

    for key, count in segments:
        labels.append(
            render_badge(
                key.replace("_", " "),
                f"{count} ({percent(count, total):.1f}%)",
                STATUS_COLORS[key],
            )
        )

    svg = (
        f"<svg viewBox=\"0 0 {width} {height}\" width=\"100%\" height=\"{height}\" "
        f"role=\"img\" aria-label=\"stacked parity coverage bar\">{''.join(rects)}</svg>"
    )
    return f"{svg}<div style=\"margin-top:0.5rem;\">{''.join(labels)}</div>"


def render_card(title: str, value: str, detail: str, color: str) -> str:
    return (
        "<div style=\"flex:1;min-width:180px;padding:0.9rem 1rem;border-radius:12px;"
        f"border:1px solid {color};background:rgba(255,255,255,0.02);\">"
        f"<div style=\"font-size:0.85rem;opacity:0.8;\">{title}</div>"
        f"<div style=\"font-size:1.6rem;font-weight:700;color:{color};margin-top:0.2rem;\">{value}</div>"
        f"<div style=\"font-size:0.9rem;margin-top:0.2rem;opacity:0.85;\">{detail}</div>"
        "</div>"
    )


def build_summary() -> AuditSummary:
    casadi_commit, manifest_entries = load_manifest()
    entries_by_case = {entry.case: entry for entry in manifest_entries}
    upstream_cases_by_file = {
        relative_path: parse_upstream_cases(relative_path) for relative_path in SCOPED_FILES
    }

    file_audits: list[FileAudit] = []
    exact_supported = 0
    questionable_supported = 0
    intentionally_descoped = 0
    unsupported = 0
    untracked = 0

    for relative_path, cases in upstream_cases_by_file.items():
        file_exact = 0
        file_questionable = 0
        file_descoped = 0
        file_unsupported = 0
        file_untracked = 0
        file_descoped_cases: list[str] = []
        file_unsupported_cases: list[str] = []
        file_untracked_cases: list[str] = []

        for case in cases:
            entry = entries_by_case.get(case)
            if entry is None:
                file_untracked += 1
                file_untracked_cases.append(case)
                continue
            if entry.status == "intentionally_descoped":
                file_descoped += 1
                file_descoped_cases.append(case)
                continue
            if entry.status == "unsupported":
                file_unsupported += 1
                file_unsupported_cases.append(case)
                continue
            if classify_supported_case(case) == "exact_supported":
                file_exact += 1
            else:
                file_questionable += 1

        exact_supported += file_exact
        questionable_supported += file_questionable
        intentionally_descoped += file_descoped
        unsupported += file_unsupported
        untracked += file_untracked
        file_audits.append(
            FileAudit(
                path=relative_path,
                total=len(cases),
                exact_supported=file_exact,
                questionable_supported=file_questionable,
                intentionally_descoped=file_descoped,
                unsupported=file_unsupported,
                untracked=file_untracked,
                intentionally_descoped_cases=file_descoped_cases,
                unsupported_cases=file_unsupported_cases,
                untracked_cases=file_untracked_cases,
            )
        )

    supported_notes = []
    for case, (assessment_key, note) in sorted(MANIFEST_SUPPORTED_ASSESSMENTS.items()):
        supported_notes.append(
            {
                "case": case,
                "assessment_key": assessment_key,
                "assessment_label": ASSESSMENT_LABELS[assessment_key],
                "note": note,
            }
        )

    likely_gaps = [
        {
            "case": case,
            "note": note,
            "tracked": "yes" if case in entries_by_case else "no",
        }
        for case, note in LIKELY_IN_SCOPE_GAPS.items()
    ]

    return AuditSummary(
        casadi_commit=casadi_commit,
        total_upstream=sum(file.total for file in file_audits),
        total_manifest_entries=len(manifest_entries),
        exact_supported=exact_supported,
        questionable_supported=questionable_supported,
        intentionally_descoped=intentionally_descoped,
        unsupported=unsupported,
        untracked=untracked,
        files=file_audits,
        likely_in_scope_gaps=likely_gaps,
        manifest_supported_notes=supported_notes,
    )


def render_markdown(summary: AuditSummary) -> str:
    overall_segments = [
        ("exact_supported", summary.exact_supported),
        ("questionable_supported", summary.questionable_supported),
        ("intentionally_descoped", summary.intentionally_descoped),
        ("unsupported", summary.unsupported),
        ("untracked", summary.untracked),
    ]
    total_supported = summary.exact_supported + summary.questionable_supported
    truly_uncovered = summary.unsupported + summary.untracked
    questionable_fraction = (
        0.0
        if total_supported == 0
        else 100.0 * summary.questionable_supported / total_supported
    )

    cards = (
        "<div style=\"display:flex;gap:0.75rem;flex-wrap:wrap;\">"
        + render_card(
            "Scoped upstream tests",
            str(summary.total_upstream),
            "All test cases found in the four pinned CasADi files.",
            "#adb5bd",
        )
        + render_card(
            "Exact supported",
            str(summary.exact_supported),
            "Semantically equivalent case-level ports today.",
            STATUS_COLORS["exact_supported"],
        )
        + render_card(
            "Questionable supported",
            str(summary.questionable_supported),
            "Tracked as supported, but proxy/divergent/overclaimed.",
            STATUS_COLORS["questionable_supported"],
        )
        + render_card(
            "Intentionally de-scoped",
            str(summary.intentionally_descoped),
            "Tracked upstream cases we explicitly do not want in the public API.",
            STATUS_COLORS["intentionally_descoped"],
        )
        + render_card(
            "Truly uncovered",
            str(truly_uncovered),
            "Explicit unsupported plus completely untracked.",
            STATUS_COLORS["untracked"],
        )
        + "</div>"
    )

    lines = [
        "# CasADi Parity Audit",
        "",
        f"- Audit date: `2026-03-29`",
        f"- Pinned CasADi commit: `{summary.casadi_commit}`",
        f"- Report generator: `python3 scripts/casadi_parity_audit.py`",
        f"- Scoped upstream files: {', '.join(f'`{path}`' for path in SCOPED_FILES)}",
        "",
        cards,
        "",
        "## Coverage At A Glance",
        "",
        render_stacked_bar(overall_segments),
        "",
        (
            f"- Tracked manifest entries: **{summary.total_manifest_entries}**"
        ),
        (
            f"- Cases marked `supported` but not trustworthy enough to count as exact parity yet: "
            f"**{summary.questionable_supported}** "
            f"({questionable_fraction:.1f}% of tracked supported cases)."
        ),
        (
            f"- Intentionally de-scoped cases: **{summary.intentionally_descoped}**. These are "
            "tracked on purpose so they do not inflate the missing bucket."
        ),
        (
            "- Current enforcement still only checks that tracked `supported` entries point at a "
            "Rust test and that the CasADi commit matches. It does **not** yet enforce manifest "
            "completeness against the scoped upstream files."
        ),
        "",
        "## File Coverage",
        "",
        "| Upstream file | Visual | Exact | Questionable | De-scoped | Unsupported | Untracked |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for file_audit in summary.files:
        visual = render_stacked_bar(
            [
                ("exact_supported", file_audit.exact_supported),
                ("questionable_supported", file_audit.questionable_supported),
                ("intentionally_descoped", file_audit.intentionally_descoped),
                ("unsupported", file_audit.unsupported),
                ("untracked", file_audit.untracked),
            ],
            width=260,
            height=16,
        )
        lines.append(
            f"| `{file_audit.path}` | {visual} | {file_audit.exact_supported} / {file_audit.total} | "
            f"{file_audit.questionable_supported} | {file_audit.intentionally_descoped} | "
            f"{file_audit.unsupported} | {file_audit.untracked} |"
        )

    assessment_counter = Counter(
        note["assessment_key"] for note in summary.manifest_supported_notes
    )
    quality_segments = [
        ("exact_supported", assessment_counter["equivalent"]),
        ("questionable_supported", summary.questionable_supported),
    ]
    lines.extend(
        [
            "",
            "## Supported Entry Quality",
            "",
            render_stacked_bar(quality_segments, width=860, height=18),
            "",
            "| Manifest case | Assessment | Note |",
            "| --- | --- | --- |",
        ]
    )
    for note in summary.manifest_supported_notes:
        lines.append(
            f"| {relink(note['case'])} | {note['assessment_label']} | {note['note']} |"
        )

    lines.extend(
        [
            "",
            "## Likely In-Scope Gaps Worth Porting Next",
            "",
        ]
    )
    for item in summary.likely_in_scope_gaps:
        tracked = "already tracked" if item["tracked"] == "yes" else "missing from manifest"
        lines.append(f"- {relink(item['case'])}: {item['note']} Currently **{tracked}**.")

    lines.extend(
        [
            "",
            "## Drilldown: Actually Missing",
            "",
            "- `Unsupported` means the case is at least acknowledged in the manifest.",
            "- `Untracked` means the current parity harness does not mention the upstream case at all.",
            "- `Intentionally de-scoped` is tracked separately below and does not count as missing.",
            "",
        ]
    )

    for file_audit in summary.files:
        missing_total = file_audit.unsupported + file_audit.untracked
        lines.append(
            f"<details><summary><strong>{file_audit.path}</strong>: {missing_total} missing "
            f"({file_audit.unsupported} unsupported, {file_audit.untracked} untracked)</summary>"
        )
        lines.append("")
        if file_audit.unsupported_cases:
            lines.append("Unsupported cases:")
            lines.append("")
            for case in file_audit.unsupported_cases:
                lines.append(f"- {relink(case)}")
            lines.append("")
        if file_audit.untracked_cases:
            lines.append("Untracked cases:")
            lines.append("")
            for case in file_audit.untracked_cases:
                lines.append(f"- {relink(case)}")
            lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.extend(
        [
            "## Drilldown: Intentionally De-Scoped",
            "",
            "- These upstream cases are tracked explicitly because the local public API chooses a "
            "different surface on purpose.",
            "",
        ]
    )

    for file_audit in summary.files:
        if not file_audit.intentionally_descoped_cases:
            continue
        lines.append(
            f"<details><summary><strong>{file_audit.path}</strong>: "
            f"{file_audit.intentionally_descoped} intentionally de-scoped</summary>"
        )
        lines.append("")
        for case in file_audit.intentionally_descoped_cases:
            lines.append(f"- {relink(case)}")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- The sparse-only public API is still preserved. Where CasADi expects dense/full-shape "
            "derivative views, the exact parity ports now use a test-only adaptor layer rather "
            "than widening the runtime surface.",
            "- The missing bucket is now cleaner: intentionally rejected Function-derivative and "
            "legacy JIT-buffer APIs are tracked separately instead of being mixed in with real "
            "coverage gaps.",
            "- The main remaining work is breadth, not honesty. The supported bucket now reflects "
            "exact upstream-equivalent cases or exact test-only adaptors over intentional API "
            "differences.",
            "",
            "## Recommended Next Actions",
            "",
            "1. Expand `parity/casadi_v1.toml` until every scoped upstream case is explicitly "
            "classified as `supported`, `unsupported`, or `intentionally_descoped`.",
            "2. Port the next small in-scope cases first: binary-op parity, simplification parity, "
            "and early CCS utility cases.",
            "3. Turn this audit into CI enforcement so a scoped upstream test cannot remain absent "
            "from the manifest.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    JSON_PATH.write_text(json.dumps(asdict(summary), indent=2))
    REPORT_PATH.write_text(render_markdown(summary))
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
