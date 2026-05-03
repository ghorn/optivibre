#!/usr/bin/env python3
"""Parse SSIDS glider thread-scaling logs and render CSV/HTML summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import urllib.request
from pathlib import Path


MICRO = "\N{MICRO SIGN}"
PLOTLY_VERSION = "2.35.2"
PLOTLY_CDN_URL = f"https://cdn.plot.ly/plotly-{PLOTLY_VERSION}.min.js"
PLOTLY_CACHE_PATH = Path("target/ssids-thread-scaling") / f"plotly-{PLOTLY_VERSION}.min.js"
DURATION_RE = re.compile(rf"^([0-9.]+)(ns|{MICRO}s|us|ms|s)$")
FACTOR_RE = re.compile(
    r"^\s*(rust_spral(?:_unprofiled)?|native_spral) factor=([^ ]+) solve=([^ ]+)"
)
SAMPLE_RE = re.compile(
    r"^\s*ssids_glider_sample index=([0-9]+) impl=(native|rust) "
    r"profile=(native|profiled|unprofiled) factor=([^ ]+) solve=([^ ]+)"
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
        sample_match = SAMPLE_RE.match(line)
        if sample_match:
            rows.append(
                {
                    "mode": meta.get("mode", "unknown"),
                    "threads": int(meta.get("threads", "0")),
                    "rayon_threads": int(meta.get("rayon", "0")),
                    "omp_threads": int(meta.get("omp", "0")),
                    "openblas_threads": int(meta.get("openblas", "0")),
                    "implementation": sample_match.group(2),
                    "profile_kind": sample_match.group(3),
                    "row_type": "sample",
                    "sample_index": int(sample_match.group(1)),
                    "factor_s": seconds(sample_match.group(4)),
                    "solve_s": seconds(sample_match.group(5)),
                    "log": str(path),
                }
            )
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
                "row_type": "summary",
                "sample_index": "",
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
        "row_type",
        "sample_index",
        "factor_s",
        "solve_s",
        "total_s",
        "log",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = {field: row[field] for field in fieldnames if field in row}
            output["total_s"] = float(row["factor_s"]) + float(row["solve_s"])
            writer.writerow(output)


def nice_ms(value_s: float) -> str:
    return f"{value_s * 1e3:.3f}"


def load_plotly_js(path: Path | None) -> str:
    if path:
        return path.read_text(encoding="utf-8")

    env_path = os.environ.get("SSIDS_PLOTLY_JS_PATH")
    if env_path:
        return Path(env_path).read_text(encoding="utf-8")

    if PLOTLY_CACHE_PATH.exists():
        return PLOTLY_CACHE_PATH.read_text(encoding="utf-8")

    PLOTLY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(PLOTLY_CDN_URL, timeout=30) as response:
        data = response.read().decode("utf-8")
    if "Plotly" not in data:
        raise RuntimeError(f"downloaded Plotly bundle from {PLOTLY_CDN_URL} did not look valid")
    PLOTLY_CACHE_PATH.write_text(data, encoding="utf-8")
    return data


def html_rows(rows: list[dict[str, str | float]]) -> list[dict[str, str | int | float]]:
    out: list[dict[str, str | int | float]] = []
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["mode"]),
            int(item["threads"]),
            str(item["implementation"]),
            str(item["profile_kind"]),
            str(item.get("row_type", "summary")),
            int(item["sample_index"]) if str(item.get("sample_index", "")).isdigit() else -1,
        ),
    ):
        env = (
            f"rayon={row['rayon_threads']} "
            f"omp={row['omp_threads']} "
            f"openblas={row['openblas_threads']}"
        )
        out.append(
            {
                "mode": str(row["mode"]),
                "threads": int(row["threads"]),
                "rayon_threads": int(row["rayon_threads"]),
                "omp_threads": int(row["omp_threads"]),
                "openblas_threads": int(row["openblas_threads"]),
                "implementation": str(row["implementation"]),
                "profile_kind": str(row["profile_kind"]),
                "row_type": str(row.get("row_type", "summary")),
                "sample_index": int(row["sample_index"])
                if str(row.get("sample_index", "")).isdigit()
                else None,
                "factor_ms": float(row["factor_s"]) * 1e3,
                "solve_ms": float(row["solve_s"]) * 1e3,
                "total_ms": (float(row["factor_s"]) + float(row["solve_s"])) * 1e3,
                "env": env,
                "log": str(row["log"]),
            }
        )
    return out


def write_html(rows: list[dict[str, str | float]], path: Path, plotly_js_path: Path | None) -> None:
    plotly_js = load_plotly_js(plotly_js_path).replace("</script>", "<\\/script>")
    data_json = json.dumps(html_rows(rows), separators=(",", ":")).replace("</script>", "<\\/script>")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SSIDS Glider Thread Scaling</title>
  <style>
    :root {{
      color-scheme: dark;
      --border: #263244;
      --muted: #94a3b8;
      --text: #e5e7eb;
      --bg: #090d14;
      --panel: #111827;
      --panel-2: #0f172a;
      --grid: #263244;
      --native: #f8fafc;
      --rust: #60a5fa;
      --profiled: #fb7185;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    main {{
      padding: 24px 28px 36px;
      max-width: 1500px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .notes {{
      max-width: 1180px;
      margin: 0 0 18px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }}
    .notes ul {{
      margin: 8px 0 0;
      padding-left: 20px;
    }}
    .explainer-grid,
    .insight-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
      margin: 14px 0 18px;
    }}
    .explain-card,
    .insight-card {{
      border: 1px solid var(--border);
      background: linear-gradient(180deg, #111827, #0b1220);
      border-radius: 10px;
      padding: 12px 14px;
    }}
    .explain-card h3,
    .insight-card h3 {{
      margin: 0 0 6px;
      font-size: 13px;
      letter-spacing: 0;
      color: #cbd5e1;
      text-transform: uppercase;
    }}
    .explain-card p,
    .insight-card p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }}
    .insight-card strong {{
      display: block;
      margin: 0 0 4px;
      color: var(--text);
      font-size: 20px;
      line-height: 1.15;
    }}
    .section-title {{
      margin: 20px 0 8px;
      font-size: 15px;
      color: #cbd5e1;
      letter-spacing: 0;
    }}
    .summary-wrap {{
      overflow-x: auto;
      margin: 12px 0 24px;
    }}
    .summary {{
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--border);
      margin: 0;
      font-size: 13px;
    }}
    .summary th,
    .summary td {{
      border-bottom: 1px solid var(--border);
      padding: 7px 10px;
      text-align: right;
      white-space: nowrap;
    }}
    .summary th:first-child,
    .summary td:first-child {{
      text-align: left;
    }}
    .summary th {{
      background: var(--panel-2);
      font-weight: 650;
    }}
    .mode-section {{
      margin: 0 0 24px;
      border: 1px solid var(--border);
      background: #0b1220;
      border-radius: 10px;
      padding: 16px;
    }}
    .mode-header {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin: 0 0 14px;
    }}
    .mode-header h2 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .mode-header p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }}
    .mode-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(min(100%, 360px), 1fr));
      gap: 14px;
    }}
    .chart-card {{
      min-height: 360px;
      border: 1px solid var(--border);
      background: var(--panel);
      border-radius: 8px;
      padding: 8px 8px 0;
      min-width: 0;
    }}
    .chart {{
      width: 100%;
      height: 345px;
    }}
  </style>
</head>
<body>
<main>
  <h1>SSIDS Glider Thread Scaling</h1>
  <div class="notes">
    <strong>Read this as timing evidence, not a correctness oracle.</strong>
    <ul>
      <li><strong>Rust unprofiled</strong> is the direct timing comparison against native SPRAL.</li>
      <li><strong>Rust profiled</strong> includes instrumentation overhead and is for attribution only.</li>
      <li>Boxes show per-repeat distributions when sample rows are available; the line is the median used for ratios.</li>
      <li>Total time is computed per repeat as factor plus solve, so its median may differ from factor median plus solve median.</li>
      <li>Extreme noisy samples are clipped visually to keep the normal timing band readable; clipped samples appear as top-edge triangles with exact values in hover.</li>
      <li>Busy-system runs can distort OpenMP/OpenBLAS rows; use a quiet machine for final ratios.</li>
    </ul>
  </div>
  <div class="explainer-grid">
    <section class="explain-card">
      <h3>What This Is</h3>
      <p>Paired replay of the same glider KKT system through source-built native SPRAL and pure Rust ssids-rs.</p>
    </section>
    <section class="explain-card">
      <h3>What Matters</h3>
      <p>Compare Rust unprofiled median total against native median total. The spread tells you whether a point is stable or scheduler-noisy.</p>
    </section>
    <section class="explain-card">
      <h3>Outlier Policy</h3>
      <p>No samples are discarded. Median is the center estimate; table stats use all raw samples, while chart outlier triangles keep extreme scheduler spikes visible without destroying scale.</p>
    </section>
  </div>
  <div class="insight-grid" id="insight-cards"></div>
  <h2 class="section-title">Paired Timing Summary</h2>
  <div class="summary-wrap">
    <table class="summary" id="summary-table"></table>
  </div>
  <h2 class="section-title">Mode Details</h2>
  <div id="charts"></div>
</main>
<script>
{plotly_js}
</script>
<script id="timing-data" type="application/json">{data_json}</script>
<script>
const rows = JSON.parse(document.getElementById("timing-data").textContent);
const summaryRows = rows.filter(row => row.row_type === "summary");
const sampleRows = rows.filter(row => row.row_type === "sample");
const modeOrder = [
  "serial",
  "ssids-rs-rayon",
  "spral-src-omp",
  "spral-src-openblas-pthreads",
  "spral-src-openblas-openmp",
  "mixed-rayon-omp",
];
const modeDescriptions = {{
  "serial": "Single-thread baseline: Rayon, OpenMP, and OpenBLAS are all pinned to 1.",
  "ssids-rs-rayon": "Varies Rayon threads for Rust ssids-rs while native SPRAL and OpenBLAS stay pinned.",
  "spral-src-omp": "Varies native SPRAL/OpenMP threads while Rust Rayon and OpenBLAS stay pinned.",
  "spral-src-openblas-pthreads": "Uses the pthreaded source-built OpenBLAS variant and varies OpenBLAS threads.",
  "spral-src-openblas-openmp": "Uses the OpenMP source-built OpenBLAS variant and varies OpenMP/OpenBLAS together.",
  "mixed-rayon-omp": "Oversubscription stress case: Rayon, OpenMP, and OpenBLAS thread counts move together.",
}};
const observedModes = new Set(rows.map(row => row.mode));
const modes = [
  ...modeOrder.filter(mode => observedModes.has(mode)),
  ...[...observedModes].filter(mode => !modeOrder.includes(mode)).sort(),
];
const metrics = [
  {{key: "factor_ms", label: "Factor Time", axis: "Factor time (ms)"}},
  {{key: "solve_ms", label: "Solve Time", axis: "Solve time (ms)"}},
  {{key: "total_ms", label: "Total Time", axis: "Total time (ms)"}},
];
const chartDivs = [];
const seriesSpecs = [
  {{implementation: "native", profile: "native", name: "native SPRAL", color: "#f8fafc", dash: "solid"}},
  {{implementation: "rust", profile: "unprofiled", name: "Rust ssids-rs", color: "#60a5fa", dash: "solid"}},
  {{implementation: "rust", profile: "profiled", name: "Rust profiled", color: "#fb7185", dash: "dash"}},
];

function formatMs(value) {{
  return `${{value.toFixed(3)}} ms`;
}}

function percentile(sortedValues, q) {{
  if (!sortedValues.length) return undefined;
  const index = Math.round(q * (sortedValues.length - 1));
  return sortedValues[Math.max(0, Math.min(sortedValues.length - 1, index))];
}}

function valueStats(values) {{
  const sorted = [...values].sort((a, b) => a - b);
  if (!sorted.length) return undefined;
  const midpoint = Math.floor(sorted.length / 2);
  const median = sorted.length % 2
    ? sorted[midpoint]
    : 0.5 * (sorted[midpoint - 1] + sorted[midpoint]);
  return {{
    n: sorted.length,
    min: sorted[0],
    p10: percentile(sorted, 0.10),
    p25: percentile(sorted, 0.25),
    median,
    p75: percentile(sorted, 0.75),
    p90: percentile(sorted, 0.90),
    max: sorted[sorted.length - 1],
  }};
}}

function rowsFor(mode, thread, spec, rowType) {{
  return rows.filter(row =>
    row.mode === mode &&
    row.threads === thread &&
    row.implementation === spec.implementation &&
    row.profile_kind === spec.profile &&
    row.row_type === rowType
  );
}}

function statsFor(mode, thread, spec, metric) {{
  let values = rowsFor(mode, thread, spec, "sample").map(row => row[metric.key]);
  const source = values.length ? "sample" : "summary";
  if (!values.length) {{
    values = rowsFor(mode, thread, spec, "summary").map(row => row[metric.key]);
  }}
  const stats = valueStats(values);
  return stats ? {{...stats, source}} : undefined;
}}

function statRange(stats) {{
  if (!stats) return "";
  return `${{stats.p10.toFixed(3)}}-${{stats.p90.toFixed(3)}}`;
}}

function robustAxis(modeRows, metric) {{
  const values = modeRows
    .filter(row => row.row_type === "sample")
    .map(row => row[metric.key])
    .filter(value => Number.isFinite(value))
    .sort((a, b) => a - b);
  if (!values.length) {{
    const summaryValues = modeRows
      .map(row => row[metric.key])
      .filter(value => Number.isFinite(value))
      .sort((a, b) => a - b);
    const maxSummary = summaryValues.length ? summaryValues[summaryValues.length - 1] : 1;
    return {{max: Math.max(maxSummary * 1.18, 1), clipped: 0, rawMax: maxSummary}};
  }}
  const rawMax = values[values.length - 1];
  const p90 = percentile(values, 0.90);
  const p95 = percentile(values, 0.95);
  const median = valueStats(values).median;
  const normal = Math.max(p95, p90 * 1.15, median * 2.0, 0.001);
  const hasExtremeTail = rawMax > Math.max(normal * 2.5, normal + 10.0);
  const max = hasExtremeTail ? normal * 1.25 : rawMax * 1.12;
  return {{
    max: Math.max(max, median * 1.5, 1),
    clipped: values.filter(value => value > max).length,
    rawMax,
  }};
}}

function seriesOffset(spec) {{
  if (spec.implementation === "native") return -0.24;
  if (spec.profile === "profiled") return 0.24;
  return 0;
}}

function sampleX(row, spec) {{
  return String(row.threads);
}}

function pairedRows() {{
  const pairs = [];
  for (const mode of modes) {{
    const threads = [...new Set(rows.filter(row => row.mode === mode).map(row => row.threads))].sort((a, b) => a - b);
    for (const thread of threads) {{
      const native = {{
        factor_ms: statsFor(mode, thread, seriesSpecs[0], metrics[0]),
        solve_ms: statsFor(mode, thread, seriesSpecs[0], metrics[1]),
        total_ms: statsFor(mode, thread, seriesSpecs[0], metrics[2]),
      }};
      const rust = {{
        factor_ms: statsFor(mode, thread, seriesSpecs[1], metrics[0]),
        solve_ms: statsFor(mode, thread, seriesSpecs[1], metrics[1]),
        total_ms: statsFor(mode, thread, seriesSpecs[1], metrics[2]),
      }};
      if (!native || !rust) continue;
      if (!native.factor_ms || !native.solve_ms || !native.total_ms || !rust.factor_ms || !rust.solve_ms || !rust.total_ms) continue;
      pairs.push({{
        mode,
        thread,
        native,
        rust,
        samples: Math.max(native.total_ms.n, rust.total_ms.n),
        factorRatio: rust.factor_ms.median / native.factor_ms.median,
        solveRatio: rust.solve_ms.median / native.solve_ms.median,
        totalRatio: rust.total_ms.median / native.total_ms.median,
      }});
    }}
  }}
  return pairs;
}}

function buildInsights() {{
  const pairs = pairedRows();
  const cards = document.getElementById("insight-cards");
  if (!pairs.length) {{
    cards.innerHTML = "";
    return;
  }}
  const bestRustTotal = [...pairs].sort((a, b) => a.rust.total_ms.median - b.rust.total_ms.median)[0];
  const bestNativeTotal = [...pairs].sort((a, b) => a.native.total_ms.median - b.native.total_ms.median)[0];
  const worstTotalRatio = [...pairs].sort((a, b) => b.totalRatio - a.totalRatio)[0];
  const serial = pairs.find(pair => pair.mode === "serial" && pair.thread === 1) || pairs[0];
  const rustRayonPairs = pairs.filter(pair => pair.mode === "ssids-rs-rayon");
  const bestRayon = rustRayonPairs.length
    ? [...rustRayonPairs].sort((a, b) => a.rust.total_ms.median - b.rust.total_ms.median)[0]
    : undefined;
  const maxThread = Math.max(...pairs.map(pair => pair.thread));
  const sampleCounts = sampleRows.length
    ? [...new Set(sampleRows.map(row => row.sample_index).filter(value => value !== null))].length
    : 0;

  const cardData = [
    [
      "Run Shape",
      `${{pairs.length}} cases`,
      `Thread counts 1-${{maxThread}} across ${{modes.length}} modes. ${{sampleCounts || "No"}} raw repeats per populated series.`,
    ],
    [
      "Distribution",
      sampleRows.length ? "raw samples" : "summary only",
      sampleRows.length
        ? "Boxes use raw repeats; median lines and ratios are derived from those repeats."
        : "This log predates sample output, so plots fall back to summary medians only.",
    ],
    [
      "Serial Baseline",
      `${{serial.totalRatio.toFixed(2)}}x total`,
      `Native ${{formatMs(serial.native.total_ms.median)}} vs Rust ${{formatMs(serial.rust.total_ms.median)}} at thread 1.`,
    ],
    [
      "Best Rust Total",
      `${{formatMs(bestRustTotal.rust.total_ms.median)}}`,
      `${{bestRustTotal.mode}}, threads=${{bestRustTotal.thread}}; ${{bestRustTotal.totalRatio.toFixed(2)}}x native for that case.`,
    ],
    [
      "Best Native Total",
      `${{formatMs(bestNativeTotal.native.total_ms.median)}}`,
      `${{bestNativeTotal.mode}}, threads=${{bestNativeTotal.thread}}; compare against Rust ${{formatMs(bestNativeTotal.rust.total_ms.median)}}.`,
    ],
    [
      "Worst Rust Ratio",
      `${{worstTotalRatio.totalRatio.toFixed(2)}}x total`,
      `${{worstTotalRatio.mode}}, threads=${{worstTotalRatio.thread}}. Treat noisy threaded rows cautiously.`,
    ],
  ];
  if (bestRayon) {{
    cardData.push([
      "Best Rust Rayon",
      `threads=${{bestRayon.thread}}`,
      `Rust total ${{formatMs(bestRayon.rust.total_ms.median)}}; ${{bestRayon.totalRatio.toFixed(2)}}x native in the Rayon-only mode.`,
    ]);
  }}
  cards.innerHTML = cardData
    .map(([title, value, body]) => `<section class="insight-card"><h3>${{title}}</h3><strong>${{value}}</strong><p>${{body}}</p></section>`)
    .join("");
}}

function buildSummary() {{
  const table = document.getElementById("summary-table");
  const header = [
    "mode",
    "threads",
    "samples",
    "factor native med",
    "factor Rust med",
    "factor ratio",
    "solve native med",
    "solve Rust med",
    "solve ratio",
    "total native med",
    "total Rust med",
    "Rust total p10-p90",
    "total ratio",
  ];
  const lines = [
    `<thead><tr>${{header.map(cell => `<th>${{cell}}</th>`).join("")}}</tr></thead>`,
    "<tbody>",
  ];
  for (const pair of pairedRows()) {{
    lines.push(
      `<tr>` +
      `<td>${{pair.mode}}</td>` +
      `<td>${{pair.thread}}</td>` +
      `<td>${{pair.samples}}</td>` +
      `<td>${{formatMs(pair.native.factor_ms.median)}}</td>` +
      `<td>${{formatMs(pair.rust.factor_ms.median)}}</td>` +
      `<td>${{pair.factorRatio.toFixed(2)}}x</td>` +
      `<td>${{formatMs(pair.native.solve_ms.median)}}</td>` +
      `<td>${{formatMs(pair.rust.solve_ms.median)}}</td>` +
      `<td>${{pair.solveRatio.toFixed(2)}}x</td>` +
      `<td>${{formatMs(pair.native.total_ms.median)}}</td>` +
      `<td>${{formatMs(pair.rust.total_ms.median)}}</td>` +
      `<td>${{statRange(pair.rust.total_ms)}} ms</td>` +
      `<td>${{pair.totalRatio.toFixed(2)}}x</td>` +
      `</tr>`
    );
  }}
  lines.push("</tbody>");
  table.innerHTML = lines.join("");
}}

function makeModeSection(mode) {{
  const section = document.createElement("section");
  section.className = "mode-section";
  const header = document.createElement("div");
  header.className = "mode-header";
  header.innerHTML =
    `<h2>${{mode}}</h2>` +
    `<p>${{modeDescriptions[mode] || "Thread scaling mode."}} Factor and solve are grouped here so runtime effects are easier to compare.</p>`;
  const grid = document.createElement("div");
  grid.className = "mode-grid";
  section.appendChild(header);
  section.appendChild(grid);
  document.getElementById("charts").appendChild(section);
  return grid;
}}

function makeChart(container, mode, metric) {{
  const card = document.createElement("section");
  card.className = "chart-card";
  const div = document.createElement("div");
  div.className = "chart";
  card.appendChild(div);
  container.appendChild(card);
  chartDivs.push(div);

  const modeRows = rows.filter(row => row.mode === mode);
  const threads = [...new Set(modeRows.map(row => row.threads))].sort((a, b) => a - b);
  const traces = [];
  const modeHasSamples = modeRows.some(row => row.row_type === "sample");
  const axis = robustAxis(modeRows, metric);
  for (const spec of seriesSpecs) {{
    const samples = modeRows
      .filter(row =>
        row.row_type === "sample" &&
        row.implementation === spec.implementation &&
        row.profile_kind === spec.profile
      )
      .sort((a, b) => a.threads - b.threads || (a.sample_index || 0) - (b.sample_index || 0));
    if (samples.length) {{
      const coreSamples = samples.filter(row => row[metric.key] <= axis.max);
      const clippedSamples = samples.filter(row => row[metric.key] > axis.max);
      traces.push({{
        x: coreSamples.map(row => String(row.threads)),
        y: coreSamples.map(row => row[metric.key]),
        customdata: coreSamples.map(row => [
          mode,
          metric.label,
          row.profile_kind,
          row.sample_index,
          row.rayon_threads,
          row.omp_threads,
          row.openblas_threads,
        ]),
        name: `${{spec.name}} samples`,
        type: "box",
        boxpoints: "all",
        jitter: 0.35,
        pointpos: 0,
        marker: {{color: spec.color, size: 4, opacity: 0.42}},
        line: {{color: spec.color, width: 1.5}},
        fillcolor: spec.color,
        opacity: 0.58,
        hovertemplate:
          "<b>%{{fullData.name}}</b><br>" +
          "Mode: %{{customdata[0]}}<br>" +
          "Metric: %{{customdata[1]}}<br>" +
          "Thread count: %{{x}}<br>" +
          "Sample: %{{customdata[3]}}<br>" +
          "Time: %{{y:.3f}} ms<br>" +
          "Rayon threads: %{{customdata[4]}}<br>" +
          "OMP threads: %{{customdata[5]}}<br>" +
          "OpenBLAS threads: %{{customdata[6]}}<extra></extra>",
      }});
      if (clippedSamples.length) {{
        traces.push({{
          x: clippedSamples.map(row => sampleX(row, spec)),
          y: clippedSamples.map(() => axis.max * 0.985),
          customdata: clippedSamples.map(row => [
            mode,
            metric.label,
            row.profile_kind,
            row.sample_index,
            row[metric.key],
            row.rayon_threads,
            row.omp_threads,
            row.openblas_threads,
          ]),
          name: `${{spec.name}} clipped outliers`,
          type: "scatter",
          mode: "markers",
          showlegend: false,
          marker: {{
            color: spec.color,
            size: 9,
            opacity: 0.95,
            symbol: "triangle-up",
            line: {{color: "#f8fafc", width: 0.8}},
          }},
          hovertemplate:
            "<b>Clipped outlier</b><br>" +
            "Mode: %{{customdata[0]}}<br>" +
            "Metric: %{{customdata[1]}}<br>" +
            "Thread count: %{{x}}<br>" +
            "Sample: %{{customdata[3]}}<br>" +
            "Actual time: %{{customdata[4]:.3f}} ms<br>" +
            "Shown at axis cap: %{{y:.3f}} ms<br>" +
            "Rayon threads: %{{customdata[5]}}<br>" +
            "OMP threads: %{{customdata[6]}}<br>" +
            "OpenBLAS threads: %{{customdata[7]}}<extra></extra>",
        }});
      }}
    }}

    const medianPoints = threads
      .map(thread => {{
        const stats = statsFor(mode, thread, spec, metric);
        if (!stats) return undefined;
        const nativeStats = statsFor(mode, thread, seriesSpecs[0], metric);
        const ratio = nativeStats && nativeStats.median > 0
          ? `${{(stats.median / nativeStats.median).toFixed(2)}}x native`
          : "native baseline";
        const exemplar = modeRows.find(row =>
          row.threads === thread &&
          row.implementation === spec.implementation &&
          row.profile_kind === spec.profile
        );
        return {{
          thread,
          y: stats.median,
          customdata: [
            mode,
            metric.label,
            spec.profile,
            ratio,
            stats.n,
            stats.p10,
            stats.p90,
            exemplar ? exemplar.rayon_threads : "",
            exemplar ? exemplar.omp_threads : "",
            exemplar ? exemplar.openblas_threads : "",
          ],
        }};
      }})
      .filter(point => point !== undefined);
    if (medianPoints.length) {{
      traces.push({{
        x: medianPoints.map(point => String(point.thread)),
        y: medianPoints.map(point => point.y),
        customdata: medianPoints.map(point => point.customdata),
        name: modeHasSamples ? `${{spec.name}} median` : spec.name,
        mode: "lines+markers",
        type: "scatter",
        showlegend: !modeHasSamples,
        line: {{color: spec.color, dash: spec.dash, width: 2.4}},
        marker: {{size: 7, color: spec.color}},
        hovertemplate:
          "<b>%{{fullData.name}}</b><br>" +
          "Mode: %{{customdata[0]}}<br>" +
          "Metric: %{{customdata[1]}}<br>" +
          "Thread count: %{{x}}<br>" +
          "Median: %{{y:.3f}} ms<br>" +
          "p10-p90: %{{customdata[5]:.3f}}-%{{customdata[6]:.3f}} ms<br>" +
          "Samples: %{{customdata[4]}}<br>" +
          "Ratio: %{{customdata[3]}}<br>" +
          "Rayon threads: %{{customdata[7]}}<br>" +
          "OMP threads: %{{customdata[8]}}<br>" +
          "OpenBLAS threads: %{{customdata[9]}}<extra></extra>",
      }});
    }}
  }}

  Plotly.newPlot(div, traces, {{
    autosize: true,
    height: 345,
    title: {{
      text: `${{mode}}: ${{metric.label}}${{axis.clipped ? ` (${{axis.clipped}} clipped)` : ""}}`,
      x: 0.02,
      xanchor: "left",
      font: {{size: 15, color: "#e5e7eb"}},
    }},
    margin: {{l: 66, r: 18, t: 54, b: 72}},
    paper_bgcolor: "#111827",
    plot_bgcolor: "#0f172a",
    font: {{color: "#e5e7eb"}},
    hovermode: "closest",
    dragmode: "pan",
    hoverlabel: {{
      bgcolor: "#020617",
      bordercolor: "#475569",
      font: {{color: "#e5e7eb", size: 12}},
      align: "left",
    }},
    legend: {{orientation: "h", x: 0, y: -0.28, font: {{size: 11, color: "#e5e7eb"}}}},
    boxmode: "group",
    xaxis: {{
      title: {{text: "Thread count"}},
      type: "category",
      categoryorder: "array",
      categoryarray: threads.map(thread => String(thread)),
      tickmode: "array",
      tickvals: threads.map(thread => String(thread)),
      ticktext: threads.map(thread => String(thread)),
      tickfont: {{size: 10}},
      automargin: true,
      zeroline: false,
      gridcolor: "#263244",
      linecolor: "#475569",
      tickcolor: "#475569",
    }},
    yaxis: {{
      title: {{text: metric.axis}},
      range: [0, axis.max],
      rangemode: "nonnegative",
      minallowed: 0,
      zeroline: false,
      gridcolor: "#263244",
      linecolor: "#475569",
      tickcolor: "#475569",
    }},
  }}, {{
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
    modeBarButtonsToRemove: ["select2d", "lasso2d"],
  }});
}}

function resizeCharts() {{
  for (const div of chartDivs) {{
    Plotly.Plots.resize(div);
  }}
}}

buildInsights();
buildSummary();
for (const mode of modes) {{
  const sectionGrid = makeModeSection(mode);
  for (const metric of metrics) {{
    makeChart(sectionGrid, mode, metric);
  }}
}}
requestAnimationFrame(resizeCharts);
window.addEventListener("load", resizeCharts);
window.addEventListener("resize", resizeCharts);
setTimeout(resizeCharts, 250);
</script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def write_markdown(rows: list[dict[str, str | float]], path: Path) -> None:
    ordered = sorted(
        [row for row in rows if row.get("row_type", "summary") == "summary"],
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
        "- Rust unprofiled is the direct timing comparison against native SPRAL.",
        "- Rust profiled includes instrumentation overhead and is for attribution only.",
        "- The HTML report shows per-repeat distributions when sample rows are present; this Markdown table stays on summary medians.",
        "- Busy-system runs can distort OpenMP/OpenBLAS rows; use a quiet machine for final ratios.",
        "",
        "| mode | threads | impl | profile | factor ms | solve ms | total ms | env |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in ordered:
        env = (
            f"rayon={row['rayon_threads']} "
            f"omp={row['omp_threads']} "
            f"openblas={row['openblas_threads']}"
        )
        lines.append(
            "| {mode} | {threads} | {implementation} | {profile_kind} | {factor} | {solve} | {total} | `{env}` |".format(
                mode=row["mode"],
                threads=row["threads"],
                implementation=row["implementation"],
                profile_kind=row["profile_kind"],
                factor=nice_ms(float(row["factor_s"])),
                solve=nice_ms(float(row["solve_s"])),
                total=nice_ms(float(row["factor_s"]) + float(row["solve_s"])),
                env=env,
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--plotly-js", type=Path)
    args = parser.parse_args()

    rows = parse_logs(args.log_dir)
    write_csv(rows, args.csv)
    write_html(rows, args.html, args.plotly_js)
    if args.markdown:
        write_markdown(rows, args.markdown)
    print(f"rows={len(rows)} csv={args.csv} html={args.html}")
    if args.markdown:
        print(f"markdown={args.markdown}")


if __name__ == "__main__":
    main()
