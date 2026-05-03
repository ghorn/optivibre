#!/usr/bin/env python3
"""Parse SSIDS glider thread-scaling logs and render CSV/SVG summaries."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


MICRO = "\N{MICRO SIGN}"
DURATION_RE = re.compile(rf"^([0-9.]+)(ns|{MICRO}s|us|ms|s)$")
FACTOR_RE = re.compile(
    r"^\s*(rust_spral(?:_unprofiled)?|native_spral) factor=([^ ]+) solve=([^ ]+)"
)
META_RE = re.compile(r"^## ssids_thread_scaling (.*)$")


def seconds(raw: str) -> float:
    match = DURATION_RE.match(raw.strip())
    if not match:
        raise ValueError(f"not a duration: {raw!r}")
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "ns":
        return value / 1e9
    if unit in (f"{MICRO}s", "us"):
        return value / 1e6
    if unit == "ms":
        return value / 1e3
    return value


def parse_meta(line: str) -> dict[str, str]:
    match = META_RE.match(line.strip())
    if not match:
        return {}
    meta = {}
    for token in match.group(1).split():
        if "=" in token:
            key, value = token.split("=", 1)
            meta[key] = value
    return meta


def parse_log(path: Path) -> list[dict[str, str | float]]:
    meta: dict[str, str] = {}
    rows: list[dict[str, str | float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("## ssids_thread_scaling"):
            meta = parse_meta(line)
            continue
        match = FACTOR_RE.match(line)
        if not match:
            continue
        label = match.group(1)
        implementation = "native" if label == "native_spral" else "rust"
        profile_kind = "profiled"
        if label == "rust_spral_unprofiled":
            profile_kind = "unprofiled"
        elif label == "native_spral":
            profile_kind = "native"
        rows.append(
            {
                "mode": meta.get("mode", "unknown"),
                "threads": int(meta.get("threads", "0")),
                "rayon_threads": int(meta.get("rayon", "0")),
                "omp_threads": int(meta.get("omp", "0")),
                "openblas_threads": int(meta.get("openblas", "0")),
                "implementation": implementation,
                "profile_kind": profile_kind,
                "factor_s": seconds(match.group(2)),
                "solve_s": seconds(match.group(3)),
                "log": str(path),
            }
        )
    return rows


def parse_logs(log_dir: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for path in sorted(log_dir.glob("*.log")):
        rows.extend(parse_log(path))
    return rows


def write_csv(rows: list[dict[str, str | float]], path: Path) -> None:
    fieldnames = [
        "mode",
        "threads",
        "rayon_threads",
        "omp_threads",
        "openblas_threads",
        "implementation",
        "profile_kind",
        "factor_s",
        "solve_s",
        "log",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def nice_ms(value_s: float) -> str:
    return f"{value_s * 1e3:.3f}"


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_panel(
    rows: list[dict[str, str | float]],
    metric: str,
    mode: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> str:
    panel_rows = [row for row in rows if row["mode"] == mode]
    if not panel_rows:
        return ""
    xs = sorted({int(row["threads"]) for row in panel_rows})
    ys = [float(row[f"{metric}_s"]) for row in panel_rows]
    y_min = 0.0
    y_max = max(ys) * 1.12 if ys else 1.0
    y_max = max(y_max, 1e-9)
    x_min = min(xs)
    x_max = max(xs)
    x_span = max(x_max - x_min, 1)

    def sx(thread: int) -> float:
        return x0 + 44.0 + (thread - x_min) / x_span * (width - 68.0)

    def sy(value_s: float) -> float:
        return y0 + height - 32.0 - (value_s - y_min) / (y_max - y_min) * (height - 58.0)

    series_specs = [
        ("native", "native", "#111827"),
        ("rust", "unprofiled", "#2563eb"),
        ("rust", "profiled", "#dc2626"),
    ]
    out = [
        f'<g class="panel">',
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{width:.1f}" height="{height:.1f}" fill="#ffffff" stroke="#d1d5db"/>',
        f'<text x="{x0 + 10:.1f}" y="{y0 + 18:.1f}" font-size="13" font-weight="700">{svg_escape(mode)} {metric}</text>',
        f'<line x1="{x0 + 44:.1f}" y1="{y0 + height - 32:.1f}" x2="{x0 + width - 18:.1f}" y2="{y0 + height - 32:.1f}" stroke="#6b7280"/>',
        f'<line x1="{x0 + 44:.1f}" y1="{y0 + 26:.1f}" x2="{x0 + 44:.1f}" y2="{y0 + height - 32:.1f}" stroke="#6b7280"/>',
    ]
    for thread in xs:
        x = sx(thread)
        out.append(
            f'<text x="{x:.1f}" y="{y0 + height - 12:.1f}" text-anchor="middle" font-size="10">{thread}</text>'
        )
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = y_min + frac * (y_max - y_min)
        y = sy(value)
        out.append(
            f'<line x1="{x0 + 40:.1f}" y1="{y:.1f}" x2="{x0 + width - 18:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        out.append(
            f'<text x="{x0 + 38:.1f}" y="{y + 3:.1f}" text-anchor="end" font-size="9">{nice_ms(value)}</text>'
        )
    for implementation, profile_kind, color in series_specs:
        series = sorted(
            [
                row
                for row in panel_rows
                if row["implementation"] == implementation and row["profile_kind"] == profile_kind
            ],
            key=lambda row: int(row["threads"]),
        )
        if not series:
            continue
        points = [(sx(int(row["threads"])), sy(float(row[f"{metric}_s"]))) for row in series]
        dash = ' stroke-dasharray="4 3"' if profile_kind == "profiled" else ""
        out.append(
            f'<polyline points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="2"{dash}/>'
        )
        for x, y in points:
            out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
    out.append("</g>")
    return "\n".join(out)


def write_svg(rows: list[dict[str, str | float]], path: Path) -> None:
    modes = sorted({str(row["mode"]) for row in rows})
    if not modes:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"/>", encoding="utf-8")
        return
    panel_w = 360
    panel_h = 230
    margin = 24
    gap = 18
    width = margin * 2 + len(modes) * panel_w + (len(modes) - 1) * gap
    height = margin * 2 + 2 * panel_h + gap + 64
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f9fafb"/>',
        '<text x="24" y="28" font-size="18" font-weight="700">SSIDS glider timing vs thread count</text>',
        '<text x="24" y="50" font-size="12" fill="#374151">Native and Rust totals are side by side; Rust profiled includes instrumentation overhead.</text>',
    ]
    legend_y = 72
    legend = [("native", "#111827", ""), ("rust unprofiled", "#2563eb", ""), ("rust profiled", "#dc2626", "4 3")]
    legend_x = 24
    for label, color, dash in legend:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="2"{dash_attr}/>'
        )
        parts.append(
            f'<text x="{legend_x + 34}" y="{legend_y + 4}" font-size="12">{svg_escape(label)}</text>'
        )
        legend_x += 140
    top = margin + 72
    for row_index, metric in enumerate(["factor", "solve"]):
        for col_index, mode in enumerate(modes):
            x = margin + col_index * (panel_w + gap)
            y = top + row_index * (panel_h + gap)
            parts.append(render_panel(rows, metric, mode, x, y, panel_w, panel_h))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_markdown(rows: list[dict[str, str | float]], path: Path) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["mode"]),
            int(row["threads"]),
            str(row["implementation"]),
            str(row["profile_kind"]),
        ),
    )
    lines = [
        "# SSIDS Glider Thread Scaling",
        "",
        "| mode | threads | impl | profile | factor ms | solve ms | env |",
        "| --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in ordered:
        env = (
            f"rayon={row['rayon_threads']} "
            f"omp={row['omp_threads']} "
            f"openblas={row['openblas_threads']}"
        )
        lines.append(
            "| {mode} | {threads} | {implementation} | {profile_kind} | {factor} | {solve} | `{env}` |".format(
                mode=row["mode"],
                threads=row["threads"],
                implementation=row["implementation"],
                profile_kind=row["profile_kind"],
                factor=nice_ms(float(row["factor_s"])),
                solve=nice_ms(float(row["solve_s"])),
                env=env,
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    rows = parse_logs(args.log_dir)
    write_csv(rows, args.csv)
    write_svg(rows, args.svg)
    if args.markdown:
        write_markdown(rows, args.markdown)
    print(f"rows={len(rows)} csv={args.csv} svg={args.svg}")
    if args.markdown:
        print(f"markdown={args.markdown}")


if __name__ == "__main__":
    main()
