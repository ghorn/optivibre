use std::fs;
use std::path::Path;

use anyhow::Result;

use crate::runner::RunResults;

pub fn write_dashboard(results: &RunResults, path: &Path) -> Result<()> {
    fs::write(path, render_dashboard(results)?)?;
    Ok(())
}

fn render_dashboard(results: &RunResults) -> Result<String> {
    let records_json = serde_json::to_string(&results.records)?;
    Ok(format!(
        r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ad_codegen_rs solver dashboard</title>
  <script defer src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #0b1220;
      --border: #334155;
      --border-soft: #243244;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --blue: #93c5fd;
      --pass: #34d399;
      --reduced: #fbbf24;
      --fail: #f87171;
      --skip: #94a3b8;
      --sqp: #60a5fa;
      --ip: #a78bfa;
      --ipopt: #f472b6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 20px;
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    a, a:visited {{ color: #7dd3fc; text-decoration: underline; text-decoration-color: rgba(125, 211, 252, 0.45); text-underline-offset: 2px; }}
    a:hover {{ color: #bae6fd; text-decoration-color: rgba(186, 230, 253, 0.9); }}
    .page {{
      max-width: 1680px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      flex-wrap: wrap;
    }}
    .header h1 {{
      margin: 0;
      font-size: 42px;
      line-height: 1;
    }}
    .subtitle {{
      color: var(--muted);
      margin-top: 8px;
      font-size: 15px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.18);
    }}
    .card h2 {{
      margin: 0 0 12px;
      font-size: 20px;
    }}
    .controls-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 12px;
    }}
    .control label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .control select,
    .control input {{
      width: 100%;
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 9px 10px;
      font-size: 14px;
    }}
    .control-actions {{
      display: flex;
      gap: 8px;
      align-items: end;
    }}
    button {{
      background: #1d4ed8;
      color: white;
      border: 0;
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 14px;
      cursor: pointer;
    }}
    button.secondary {{
      background: #1f2937;
      border: 1px solid var(--border);
      color: var(--text);
    }}
    .chip-row {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    .chip {{
      display: inline-flex;
      gap: 6px;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      font-size: 12px;
    }}
    .chip .k {{ color: var(--muted); }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
    }}
    .metric {{
      background: var(--panel-2);
      border: 1px solid var(--border-soft);
      border-radius: 14px;
      padding: 14px;
    }}
    .metric .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 8px;
    }}
    .metric .value {{
      font-size: 28px;
      font-weight: 700;
      line-height: 1.1;
    }}
    .metric .sub {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }}
    .ok {{ color: var(--pass); }}
    .warn {{ color: var(--reduced); }}
    .bad {{ color: var(--fail); }}
    .skip {{ color: var(--skip); }}
    .solver-sqp {{ color: var(--sqp); }}
    .solver-nlip {{ color: var(--ip); }}
    .solver-ipopt {{ color: var(--ipopt); }}
    .grid-2 {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 18px;
    }}
    .legend {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 10px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .dot {{
      width: 11px;
      height: 11px;
      border-radius: 999px;
      display: inline-block;
    }}
    .square {{
      width: 11px;
      height: 11px;
      display: inline-block;
    }}
    .diamond {{
      width: 11px;
      height: 11px;
      display: inline-block;
      transform: rotate(45deg);
    }}
    .scatter-plot {{
      min-height: 320px;
      border-radius: 12px;
      overflow: hidden;
      background: linear-gradient(180deg, rgba(11, 18, 32, 0.88), rgba(8, 13, 24, 0.98));
      border: 1px solid var(--border-soft);
    }}
    .chart-point {{ cursor: pointer; }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      z-index: 1000;
      max-width: 420px;
      background: rgba(11, 18, 32, 0.98);
      color: var(--text);
      border: 1px solid #475569;
      border-radius: 10px;
      padding: 10px 12px;
      box-shadow: 0 8px 28px rgba(0, 0, 0, 0.35);
      font-size: 12px;
      line-height: 1.45;
      opacity: 0;
      transform: translate(12px, 12px);
      transition: opacity 0.08s ease;
      white-space: pre-line;
    }}
    .tooltip.visible {{ opacity: 1; }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
      font-size: 13px;
      vertical-align: top;
    }}
    th {{
      color: var(--blue);
      background: rgba(30, 41, 59, 0.55);
      position: sticky;
      top: 0;
    }}
    .table-scroll {{
      overflow: auto;
      max-height: 680px;
      border: 1px solid var(--border);
      border-radius: 12px;
    }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, monospace; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 54px;
      padding: 4px 10px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 12px;
    }}
    .pill.ok {{ background: rgba(52, 211, 153, 0.16); color: #d1fae5; }}
    .pill.warn {{ background: rgba(251, 191, 36, 0.16); color: #fde68a; }}
    .pill.bad {{ background: rgba(248, 113, 113, 0.16); color: #fecaca; }}
    .pill.skip {{ background: rgba(148, 163, 184, 0.16); color: #e2e8f0; }}
    .family-matrix td {{
      min-width: 150px;
    }}
    .family-matrix .summary-row td {{
      background: linear-gradient(180deg, rgba(30, 41, 59, 0.98), rgba(15, 23, 42, 0.94));
      border-bottom: 4px solid rgba(96, 165, 250, 0.82);
      font-weight: 700;
      position: sticky;
      top: 34px;
      z-index: 1;
    }}
    .family-matrix .summary-row td:first-child {{
      color: #e0f2fe;
      font-size: 14px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .family-matrix .summary-row .matrix-cell {{
      border-width: 2px;
      box-shadow: inset 0 1px 0 rgba(226, 232, 240, 0.08);
    }}
    .matrix-cell {{
      border-radius: 12px;
      border: 1px solid var(--border-soft);
      padding: 8px 10px;
      min-height: 54px;
      display: grid;
      gap: 4px;
      background: var(--panel-2);
    }}
    .matrix-cell.ok {{
      border-color: rgba(52, 211, 153, 0.35);
      background: rgba(52, 211, 153, 0.08);
    }}
    .matrix-cell.warn {{
      border-color: rgba(251, 191, 36, 0.35);
      background: rgba(251, 191, 36, 0.08);
    }}
    .matrix-cell.bad {{
      border-color: rgba(248, 113, 113, 0.35);
      background: rgba(248, 113, 113, 0.08);
    }}
    .matrix-main {{
      font-weight: 700;
      font-size: 14px;
    }}
    .matrix-sub {{
      color: var(--muted);
      font-size: 12px;
    }}
    .matrix-counts {{
      display: flex;
      flex-wrap: wrap;
      gap: 0 8px;
      align-items: center;
      font-size: 12px;
      line-height: 1.35;
    }}
    .matrix-count {{
      font-weight: 700;
      cursor: help;
      text-decoration: underline;
      text-decoration-thickness: 1px;
      text-underline-offset: 3px;
      text-decoration-color: rgba(148, 163, 184, 0.35);
    }}
    .matrix-count.pass {{ color: var(--pass); }}
    .matrix-count.reduced {{ color: var(--reduced); }}
    .matrix-count.panic {{ color: #fca5a5; }}
    .matrix-count.max_iters {{ color: #fb7185; }}
    .matrix-count.line_search {{ color: #f97316; }}
    .matrix-count.restoration {{ color: #ec4899; }}
    .matrix-count.step_computation {{ color: #c084fc; }}
    .matrix-count.local_infeasible {{ color: #ef4444; }}
    .matrix-count.non_finite {{ color: #a78bfa; }}
    .matrix-count.objective {{ color: #f43f5e; }}
    .matrix-count.primal {{ color: #e11d48; }}
    .matrix-count.dual {{ color: #be123c; }}
    .matrix-count.validation {{ color: #f87171; }}
    .matrix-count.solve_error {{ color: var(--fail); }}
    .matrix-count.skip {{ color: var(--skip); }}
    .matrix-bars {{
      display: flex;
      height: 9px;
      overflow: hidden;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.92);
      border: 1px solid rgba(148, 163, 184, 0.18);
    }}
    .matrix-segment.pass {{ background: var(--pass); }}
    .matrix-segment.reduced {{ background: var(--reduced); }}
    .matrix-segment.panic {{ background: #dc2626; }}
    .matrix-segment.max_iters {{ background: #fb7185; }}
    .matrix-segment.line_search {{ background: #f97316; }}
    .matrix-segment.restoration {{ background: #ec4899; }}
    .matrix-segment.step_computation {{ background: #c084fc; }}
    .matrix-segment.local_infeasible {{ background: #ef4444; }}
    .matrix-segment.non_finite {{ background: #a78bfa; }}
    .matrix-segment.objective {{ background: #f43f5e; }}
    .matrix-segment.primal {{ background: #e11d48; }}
    .matrix-segment.dual {{ background: #be123c; }}
    .matrix-segment.validation {{ background: #f87171; }}
    .matrix-segment.solve_error {{ background: var(--fail); }}
    .matrix-segment.skip {{ background: var(--skip); }}
    .detail-cell {{
      min-width: 240px;
      max-width: 420px;
      color: var(--muted);
      line-height: 1.35;
      white-space: normal;
    }}
    .detail-cell.error {{
      color: #fecaca;
    }}
    .empty-state {{
      color: var(--muted);
      padding: 20px 0 8px;
    }}
    @media (max-width: 1200px) {{
      .controls-grid,
      .metric-grid,
      .grid-2 {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    @media (max-width: 800px) {{
      body {{ padding: 12px; }}
      .controls-grid,
      .metric-grid,
      .grid-2 {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div>
        <h1>Solver Dashboard</h1>
        <div class="subtitle">Interactive filtering across solver, family, status, JIT level, constraint class, expected status, and more.</div>
      </div>
    </div>

    <div class="card">
      <h2>Filters</h2>
      <div class="controls-grid">
        <div class="control"><label for="filter-search">Problem search</label><input id="filter-search" type="text" placeholder="problem id or variant"></div>
        <div class="control"><label for="filter-solver">Solver</label><select id="filter-solver"></select></div>
        <div class="control"><label for="filter-test-set">Test set</label><select id="filter-test-set"></select></div>
        <div class="control"><label for="filter-family">Family</label><select id="filter-family"></select></div>
        <div class="control"><label for="filter-status">Status</label><select id="filter-status"></select></div>
        <div class="control"><label for="filter-jit">JIT</label><select id="filter-jit"></select></div>
        <div class="control"><label for="filter-expected">Expected</label><select id="filter-expected"></select></div>
        <div class="control"><label for="filter-constrained">Constraint class</label><select id="filter-constrained"></select></div>
        <div class="control"><label for="filter-parameterized">Parameterized</label><select id="filter-parameterized"></select></div>
        <div class="control"><label for="filter-source">Source</label><select id="filter-source"></select></div>
        <div class="control-actions">
          <button id="reset-filters" type="button" class="secondary">Reset filters</button>
        </div>
      </div>
      <div id="active-filters" class="chip-row"></div>
    </div>

    <div id="metrics" class="metric-grid"></div>

    <div class="grid-2">
      <div class="card">
        <h2>Iterations vs Variables</h2>
        <div class="legend">
          <span class="legend-item"><span class="dot" style="background:var(--pass)"></span>passed</span>
          <span class="legend-item"><span class="dot" style="background:var(--reduced)"></span>reduced</span>
          <span class="legend-item"><span class="dot" style="background:var(--fail)"></span>failed</span>
          <span class="legend-item"><span class="dot" style="background:var(--skip)"></span>skipped</span>
          <span class="legend-item"><span class="dot" style="background:#cbd5e1"></span>SQP</span>
          <span class="legend-item"><span class="square" style="background:#cbd5e1"></span>NLIP</span>
          <span class="legend-item"><span class="diamond" style="background:#cbd5e1"></span>IPOPT</span>
        </div>
        <div id="chart-vars-iters" class="scatter-plot"></div>
      </div>
      <div class="card">
        <h2>Total Time vs DOF</h2>
        <div id="chart-dof-time" class="scatter-plot"></div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h2>Slowest Filtered Runs</h2>
        <div id="slowest-table"></div>
      </div>
      <div class="card">
        <h2>Filtered Breakdown</h2>
        <div id="status-breakdown"></div>
      </div>
    </div>

    <div class="card">
      <h2>Family Summary</h2>
      <div class="subtitle" style="margin-top:0; margin-bottom:10px">One row per family, one column per solver. Cells are color-coded by aggregate result.</div>
      <div id="family-summary"></div>
    </div>

    <div class="card">
      <h2>All Runs</h2>
      <div class="subtitle" style="margin-top:0; margin-bottom:10px">Filtered problem/solver/JIT rows. Charts link to the same transcripts.</div>
      <div id="all-runs"></div>
    </div>

    <div id="tooltip" class="tooltip" aria-hidden="true"></div>
  </div>

  <script>
    const records = {records_json};
    const filterIds = [
      "filter-search",
      "filter-solver",
      "filter-test-set",
      "filter-family",
      "filter-status",
      "filter-jit",
      "filter-expected",
      "filter-constrained",
      "filter-parameterized",
      "filter-source",
    ];

    const statusOrder = ["passed", "reduced_accuracy", "failed_validation", "solve_error", "skipped"];
    const solverOrder = ["sqp", "nlip", "ipopt"];

    const byId = (id) => document.getElementById(id);

    const htmlEscape = (value) => String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");

    const statusLabel = (status) => {{
      switch (status) {{
        case "passed": return "passed";
        case "reduced_accuracy": return "reduced";
        case "failed_validation": return "failed_validation";
        case "solve_error": return "solve_error";
        case "skipped": return "skipped";
        default: return status;
      }}
    }};

    const statusClass = (status) => {{
      switch (status) {{
        case "passed": return "ok";
        case "reduced_accuracy": return "warn";
        case "failed_validation":
        case "solve_error": return "bad";
        case "skipped": return "skip";
        default: return "";
      }}
    }};

    const solverLabel = (solver) => {{
      switch (solver) {{
        case "sqp": return "SQP";
        case "nlip": return "NLIP";
        case "ipopt": return "IPOPT";
        default: return solver;
      }}
    }};

    const formatDuration = (seconds) => {{
      if (seconds >= 1) return `${{seconds.toFixed(2)}}s`;
      if (seconds >= 1e-3) return `${{(seconds * 1e3).toFixed(2)}}ms`;
      if (seconds >= 1e-6) return `${{(seconds * 1e6).toFixed(2)}}us`;
      return `${{(seconds * 1e9).toFixed(2)}}ns`;
    }};

    const numberText = (value) => value == null ? "--" : value.toExponential(3);
    const intText = (value) => value == null ? "--" : String(value);
    const boolText = (value) => value ? "yes" : "no";

    const constraintLabel = (record) => {{
      const d = record.descriptor;
      if (d.num_eq === 0 && d.num_ineq === 0 && d.num_box === 0) return "none";
      return `e${{d.num_eq}}/i${{d.num_ineq}}/b${{d.num_box}}`;
    }};

    const failureReason = (record) => {{
      switch (record.status) {{
        case "passed":
          return "--";
        case "reduced_accuracy":
        case "failed_validation":
          return record.validation.detail;
        case "solve_error":
          return record.error || "solve error";
        case "skipped":
          return "skipped";
        default:
          return "--";
      }}
    }};

    const elasticText = (record) => {{
      const activations = record.metrics.elastic_recovery_activations;
      const recoveryQps = record.metrics.elastic_recovery_qp_solves;
      if (activations == null || recoveryQps == null) return "--";
      return `${{activations}}/${{recoveryQps}}`;
    }};

    const failureCode = (record) => {{
      const text = failureReason(record).toLowerCase();
      if (record.status === "passed") return "--";
      if (record.status === "reduced_accuracy") return "reduced";
      if (text.includes("panicked")) return "panic";
      if (text.includes("max iteration") || text.includes("failed to converge")) return "max_iters";
      if (text.includes("line search") || text.includes("armijo")) return "line_search";
      if (text.includes("restoration")) return "restoration";
      if (text.includes("errorinstepcomputation") || text.includes("step computation")) return "step_computation";
      if (text.includes("primal infeasible") || text.includes("local infeasibility")) return "local_infeasible";
      if (text.includes("non-finite") || text.includes("nan") || text.includes("inf")) return "non_finite";
      if (record.status === "failed_validation" && text.includes("objective")) return "objective";
      if (record.status === "failed_validation" && text.includes("primal")) return "primal";
      if (record.status === "failed_validation" && text.includes("dual")) return "dual";
      if (text.includes("validation")) return "validation";
      return statusLabel(record.status);
    }};

    const resultSegment = (record) => {{
      if (record.status === "passed") return "pass";
      if (record.status === "reduced_accuracy") return "reduced";
      if (record.status === "skipped") return "skip";
      return failureCode(record);
    }};

    const segmentLabel = (segment) => ({{
      pass: "pass",
      reduced: "reduced",
      panic: "panic",
      max_iters: "max iters",
      line_search: "line search",
      restoration: "restoration",
      step_computation: "step comp",
      local_infeasible: "infeasible",
      non_finite: "non-finite",
      objective: "objective",
      primal: "primal",
      dual: "dual",
      validation: "validation",
      solve_error: "solve error",
      skip: "skip",
    }})[segment] || segment.replaceAll("_", " ");

    const segmentDescription = (segment) => ({{
      pass: "Solver returned success and strict validation passed.",
      reduced: "Accepted by the reduced-accuracy residual thresholds.",
      panic: "Solver panicked; the runner captured it as a solve error.",
      max_iters: "Solver stopped at its iteration limit.",
      line_search: "Line search or Armijo backtracking failed.",
      restoration: "Restoration phase failed before finding a feasible point.",
      step_computation: "IPOPT reported an error in step computation.",
      local_infeasible: "Solver reported local or primal infeasibility.",
      non_finite: "A non-finite value was encountered.",
      objective: "Objective validation failed.",
      primal: "Primal or constraint-residual validation failed.",
      dual: "Dual optimality validation failed.",
      validation: "Run completed but validation failed.",
      solve_error: "Solver returned an unclassified solve error.",
      skip: "Skipped by the manifest.",
    }})[segment] || segment.replaceAll("_", " ");

    const segmentOrder = [
      "pass",
      "reduced",
      "panic",
      "max_iters",
      "line_search",
      "restoration",
      "step_computation",
      "local_infeasible",
      "non_finite",
      "objective",
      "primal",
      "dual",
      "validation",
      "solve_error",
      "skip",
    ];

    const detailClass = (record) => record.status === "solve_error" ? "detail-cell error" : "detail-cell";

    const problemHref = (record) => record.console_output_path || null;

    const hoverText = (record, xLabel, yLabel, xValue, yValue) => {{
      return [
        `problem: ${{record.id}}`,
        `test set: ${{record.descriptor.test_set}}`,
        `family: ${{record.descriptor.family}} / ${{record.descriptor.variant}}`,
        `solver: ${{solverLabel(record.solver)}} / ${{record.options.jit_opt_level.toUpperCase()}} / ${{record.options.call_policy}}`,
        `status: ${{statusLabel(record.status)}}`,
        `expected: ${{record.expected}}`,
        `vars / dof: ${{record.descriptor.num_vars}} / ${{record.descriptor.dof}}`,
        `constraints: ${{constraintLabel(record)}}`,
        `${{xLabel}}: ${{xValue.toFixed(3)}}`,
        `${{yLabel}}: ${{yValue.toFixed(3)}}`,
        `iters: ${{intText(record.metrics.iterations)}}`,
        `elastic: ${{elasticText(record)}}`,
        `total: ${{formatDuration(record.timing.total_wall_time)}}`,
        `objective: ${{numberText(record.metrics.objective)}}`,
        `primal: ${{numberText(record.metrics.primal_inf)}}`,
        `dual: ${{numberText(record.metrics.dual_inf)}}`,
        `reason: ${{failureReason(record)}}`,
        `click: open transcript`,
      ].join("\n");
    }};

    function populateSelect(id, values, formatter = (value) => value) {{
      const select = byId(id);
      const current = select.value;
      const options = [''].concat(values);
      select.innerHTML = options.map((value) => {{
        const label = value === '' ? 'All' : formatter(value);
        const selected = value === current ? ' selected' : '';
        return `<option value="${{htmlEscape(value)}}"${{selected}}>${{htmlEscape(label)}}</option>`;
      }}).join('');
      if (!options.includes(current)) select.value = '';
    }}

    function initializeFilters() {{
      populateSelect('filter-solver', unique(records.map((record) => record.solver), solverOrder), solverLabel);
      populateSelect('filter-test-set', unique(records.map((record) => record.descriptor.test_set)));
      populateSelect('filter-family', unique(records.map((record) => record.descriptor.family)));
      populateSelect('filter-status', unique(records.map((record) => record.status), statusOrder), statusLabel);
      populateSelect('filter-jit', unique(records.map((record) => record.options.jit_opt_level.toUpperCase())));
      populateSelect('filter-expected', unique(records.map((record) => record.expected)));
      populateSelect('filter-constrained', ['constrained', 'unconstrained']);
      populateSelect('filter-parameterized', ['parameterized', 'non_parameterized']);
      populateSelect('filter-source', unique(records.map((record) => record.descriptor.source)));
    }}

    function unique(values, preferredOrder = []) {{
      const set = new Set(values.filter((value) => value != null));
      const rest = Array.from(set).filter((value) => !preferredOrder.includes(value)).sort();
      return preferredOrder.filter((value) => set.has(value)).concat(rest);
    }}

    function filteredRecords() {{
      const search = byId('filter-search').value.trim().toLowerCase();
      const solver = byId('filter-solver').value;
      const testSet = byId('filter-test-set').value;
      const family = byId('filter-family').value;
      const status = byId('filter-status').value;
      const jit = byId('filter-jit').value;
      const expected = byId('filter-expected').value;
      const constrained = byId('filter-constrained').value;
      const parameterized = byId('filter-parameterized').value;
      const source = byId('filter-source').value;

      return records.filter((record) => {{
        if (search) {{
          const haystack = `${{record.id}} ${{record.descriptor.variant}} ${{record.descriptor.description}}`.toLowerCase();
          if (!haystack.includes(search)) return false;
        }}
        if (solver && record.solver !== solver) return false;
        if (testSet && record.descriptor.test_set !== testSet) return false;
        if (family && record.descriptor.family !== family) return false;
        if (status && record.status !== status) return false;
        if (jit && record.options.jit_opt_level.toUpperCase() !== jit) return false;
        if (expected && record.expected !== expected) return false;
        if (source && record.descriptor.source !== source) return false;
        if (constrained) {{
          const want = constrained === 'constrained';
          if (record.descriptor.constrained !== want) return false;
        }}
        if (parameterized) {{
          const want = parameterized === 'parameterized';
          if (record.descriptor.parameterized !== want) return false;
        }}
        return true;
      }});
    }}

    function renderActiveFilters() {{
      const chips = [];
      for (const id of filterIds) {{
        const element = byId(id);
        const value = element.value.trim();
        if (!value) continue;
        const label = element.previousElementSibling.textContent;
        chips.push(`<span class="chip"><span class="k">${{htmlEscape(label)}}:</span> ${{htmlEscape(value)}}</span>`);
      }}
      byId('active-filters').innerHTML = chips.length ? chips.join('') : '<span class="chip"><span class="k">filters:</span> none</span>';
    }}

    function renderMetrics(records) {{
      const passed = records.filter((record) => record.status === 'passed').length;
      const reduced = records.filter((record) => record.status === 'reduced_accuracy').length;
      const failed = records.filter((record) => record.status === 'failed_validation' || record.status === 'solve_error').length;
      const skipped = records.filter((record) => record.status === 'skipped').length;
      const totalTime = records.reduce((sum, record) => sum + record.timing.total_wall_time, 0);
      const avgIters = records.filter((record) => record.metrics.iterations != null).length
        ? records.filter((record) => record.metrics.iterations != null).reduce((sum, record) => sum + record.metrics.iterations, 0)
          / records.filter((record) => record.metrics.iterations != null).length
        : 0;
      const total = records.length;
      byId('metrics').innerHTML = [
        metricCard('Filtered runs', String(total), total === records.length ? 'current slice' : ''),
        metricCard('Passed', String(passed), '', 'ok'),
        metricCard('Reduced', String(reduced), '', 'warn'),
        metricCard('Failed', String(failed), '', 'bad'),
        metricCard('Skipped', String(skipped), '', 'skip'),
        metricCard('Total time', formatDuration(totalTime), `avg iters ${{avgIters.toFixed(1)}}`),
      ].join('');
    }}

    function metricCard(label, value, sub = '', cls = '') {{
      return `<div class="metric"><div class="label">${{htmlEscape(label)}}</div><div class="value ${{cls}}">${{htmlEscape(value)}}</div><div class="sub">${{htmlEscape(sub)}}</div></div>`;
    }}

    const scatterStatusColor = (status) => ({{
      passed: '#34d399',
      reduced_accuracy: '#fbbf24',
      failed_validation: '#f87171',
      solve_error: '#f87171',
      skipped: '#94a3b8',
    }})[status] || '#94a3b8';

    const scatterSolverSymbol = (solver) => ({{
      sqp: 'circle',
      nlip: 'square',
      ipopt: 'diamond',
    }})[solver] || 'circle';

    function scatterTrace(records, solver, xLabel, yLabel, xValue, yValue) {{
      const solverRecords = records.filter((record) => record.solver === solver);
      return {{
        type: 'scatter',
        mode: 'markers',
        name: solverLabel(solver),
        x: solverRecords.map((record) => xValue(record)),
        y: solverRecords.map((record) => yValue(record)),
        text: solverRecords.map((record) => hoverText(record, xLabel, yLabel, xValue(record), yValue(record))),
        customdata: solverRecords.map((record) => problemHref(record) || ''),
        marker: {{
          color: solverRecords.map((record) => scatterStatusColor(record.status)),
          symbol: scatterSolverSymbol(solver),
          size: 11,
          line: {{ color: 'rgba(11, 18, 32, 0.95)', width: 1.2 }},
        }},
        hovertemplate: '%{{text}}<extra>' + htmlEscape(solverLabel(solver)) + '</extra>',
      }};
    }}

    function renderScatter(records, mountId, xLabel, yLabel, xValue, yValue) {{
      const mount = byId(mountId);
      if (!records.length) {{
        if (window.Plotly) {{
          window.Plotly.purge(mount);
        }}
        mount.innerHTML = '<div class="empty-state">No runs match the current filters.</div>';
        return;
      }}
      if (!window.Plotly) {{
        mount.innerHTML = '<div class="empty-state">Plotly is still loading.</div>';
        return;
      }}
      mount.innerHTML = '';
      const traces = solverOrder
        .map((solver) => scatterTrace(records, solver, xLabel, yLabel, xValue, yValue))
        .filter((trace) => trace.x.length > 0);
      const layout = {{
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(11, 18, 32, 0.88)',
        margin: {{ l: 58, r: 18, t: 14, b: 48 }},
        font: {{ color: '#e2e8f0', family: 'ui-sans-serif, system-ui, sans-serif', size: 12 }},
        legend: {{ orientation: 'h', y: -0.22, x: 0, font: {{ color: '#94a3b8', size: 11 }} }},
        hoverlabel: {{ bgcolor: 'rgba(11, 18, 32, 0.98)', bordercolor: '#334155', font: {{ color: '#e2e8f0' }} }},
        xaxis: {{
          title: xLabel,
          gridcolor: 'rgba(148, 163, 184, 0.12)',
          linecolor: '#475569',
          zeroline: false,
          ticks: 'outside',
          titlefont: {{ color: '#cbd5e1' }},
        }},
        yaxis: {{
          title: yLabel,
          gridcolor: 'rgba(148, 163, 184, 0.12)',
          linecolor: '#475569',
          zeroline: false,
          ticks: 'outside',
          titlefont: {{ color: '#cbd5e1' }},
        }},
      }};
      const config = {{ responsive: true, displaylogo: false, displayModeBar: false }};
      window.Plotly.react(mount, traces, layout, config).then(() => {{
        if (!mount.__problemLinkBound && typeof mount.on === 'function') {{
          mount.on('plotly_click', (event) => {{
            const href = event?.points?.[0]?.customdata;
            if (href) {{
              window.location.href = href;
            }}
          }});
          mount.__problemLinkBound = true;
        }}
      }});
    }}

    function renderSlowest(records) {{
      const rows = [...records].sort((a, b) => b.timing.total_wall_time - a.timing.total_wall_time).slice(0, 12);
      if (!rows.length) {{
        byId('slowest-table').innerHTML = '<div class="empty-state">No runs match the current filters.</div>';
        return;
      }}
      const html = `
        <div class="table-scroll">
          <table>
            <thead><tr><th>Problem</th><th>Solver</th><th>Status</th><th>Iters</th><th>Elastic</th><th>Total</th><th>Reason</th><th>Detail</th></tr></thead>
            <tbody>
              ${{rows.map((record) => `
                <tr>
                  <td class="mono">${{problemCell(record)}}</td>
                  <td class="solver-${{record.solver}}">${{solverLabel(record.solver)}} / ${{record.options.jit_opt_level.toUpperCase()}} / ${{record.options.call_policy}}</td>
                  <td><span class="pill ${{statusClass(record.status)}}">${{htmlEscape(statusLabel(record.status))}}</span></td>
                  <td>${{intText(record.metrics.iterations)}}</td>
                  <td>${{htmlEscape(elasticText(record))}}</td>
                  <td>${{formatDuration(record.timing.total_wall_time)}}</td>
                  <td>${{htmlEscape(failureCode(record))}}</td>
                  <td class="${{detailClass(record)}}">${{htmlEscape(failureReason(record))}}</td>
                </tr>`).join('')}}
            </tbody>
          </table>
        </div>`;
      byId('slowest-table').innerHTML = html;
    }}

    function renderStatusBreakdown(records) {{
      const solverGroups = unique(records.map((record) => record.solver), solverOrder);
      if (!solverGroups.length) {{
        byId('status-breakdown').innerHTML = '<div class="empty-state">No runs match the current filters.</div>';
        return;
      }}
      const html = solverGroups.map((solver) => {{
        const solverRecords = records.filter((record) => record.solver === solver);
        const counts = {{
          passed: solverRecords.filter((record) => record.status === 'passed').length,
          reduced: solverRecords.filter((record) => record.status === 'reduced_accuracy').length,
          failed: solverRecords.filter((record) => record.status === 'failed_validation' || record.status === 'solve_error').length,
          skipped: solverRecords.filter((record) => record.status === 'skipped').length,
        }};
        const total = Math.max(1, solverRecords.length);
        const parts = [
          ['passed', counts.passed, 'var(--pass)'],
          ['reduced', counts.reduced, 'var(--reduced)'],
          ['failed', counts.failed, 'var(--fail)'],
          ['skipped', counts.skipped, 'var(--skip)'],
        ];
        return `
          <div style="margin-bottom:14px">
            <div class="solver-${{solver}}" style="font-weight:700; margin-bottom:6px">${{solverLabel(solver)}}</div>
            <div style="display:flex; height:18px; border-radius:999px; overflow:hidden; background:#1f2937; border:1px solid var(--border-soft)">
              ${{parts.map(([label, count, color]) => `<div title="${{label}}: ${{count}}" style="width:${{(count / total * 100).toFixed(3)}}%; background:${{color}}"></div>`).join('')}}
            </div>
            <div style="margin-top:6px; color:var(--muted); font-size:12px">
              pass=${{counts.passed}} reduced=${{counts.reduced}} fail=${{counts.failed}} skipped=${{counts.skipped}}
            </div>
          </div>`;
      }}).join('');
      byId('status-breakdown').innerHTML = html;
    }}

    function renderFamilySummary(records) {{
      const families = unique(records.map((record) => record.descriptor.family));
      const solvers = unique(records.map((record) => record.solver), solverOrder);
      if (!families.length) {{
        byId('family-summary').innerHTML = '<div class="empty-state">No runs match the current filters.</div>';
        return;
      }}
      const html = `
        <div class="table-scroll">
          <table class="family-matrix">
            <thead>
              <tr>
                <th>Family</th>
                ${{solvers.map((solver) => `<th class="solver-${{solver}}">${{solverLabel(solver)}}</th>`).join('')}}
                <th>Total</th>
              </tr>
            </thead>
            <tbody>
              <tr class="summary-row">
                <td>Total</td>
                ${{solvers.map((solver) => familyMatrixCell(records.filter((record) => record.solver === solver))).join('')}}
                <td>${{formatDuration(records.reduce((sum, record) => sum + record.timing.total_wall_time, 0))}}</td>
              </tr>
              ${{families.map((family) => {{
                const familyRecords = records.filter((record) => record.descriptor.family === family);
                const totalTime = familyRecords.reduce((sum, record) => sum + record.timing.total_wall_time, 0);
                return `<tr>
                  <td class="mono">${{htmlEscape(family)}}</td>
                  ${{solvers.map((solver) => familyMatrixCell(familyRecords.filter((record) => record.solver === solver))).join('')}}
                  <td>${{formatDuration(totalTime)}}</td>
                </tr>`;
              }}).join('')}}
            </tbody>
          </table>
        </div>`;
      byId('family-summary').innerHTML = html;
    }}

    function familyMatrixCell(records) {{
      if (!records.length) return '<td>--</td>';
      const passed = records.filter((record) => record.status === 'passed').length;
      const reduced = records.filter((record) => record.status === 'reduced_accuracy').length;
      const total = records.length;
      const skipped = records.filter((record) => record.status === 'skipped').length;
      const accepted = passed + reduced;
      const fail = total - accepted - skipped;
      const cellClass = fail === 0 ? (reduced > 0 ? 'warn' : 'ok') : (accepted > 0 ? 'warn' : 'bad');
      const totalTime = records.reduce((sum, record) => sum + record.timing.total_wall_time, 0);
      const rate = total ? (accepted / total * 100).toFixed(0) : '0';
      const counts = new Map();
      for (const record of records) {{
        const segment = resultSegment(record);
        counts.set(segment, (counts.get(segment) || 0) + 1);
      }}
      const segments = segmentOrder
        .map((segment) => [segment, counts.get(segment) || 0, segmentLabel(segment)])
        .filter(([, count]) => count > 0);
      const countParts = segments
        .map(([segment, count, label]) => {{
          const title = `${{label}}: ${{count}}\\n${{segmentDescription(segment)}}`;
          return `<span class="matrix-count ${{segment}}" title="${{htmlEscape(title)}}">${{htmlEscape(label)}} ${{count}}</span>`;
        }})
        .join(' · ');
      const title = segments
        .map(([segment, count, label]) => `${{label}}: ${{count}} - ${{segmentDescription(segment)}}`)
        .join('\\n');
      return `<td><div class="matrix-cell ${{cellClass}}" title="${{htmlEscape(title)}}"><div class="matrix-bars">${{segments.map(([kind, count, label]) => `<div class="matrix-segment ${{kind}}" title="${{htmlEscape(`${{label}}: ${{count}}\\n${{segmentDescription(kind)}}`)}}" style="width:${{(count / total * 100).toFixed(3)}}%"></div>`).join('')}}</div><div class="matrix-main">${{rate}}% accepted</div><div class="matrix-counts">${{countParts || '<span class="matrix-sub">none</span>'}}</div><div class="matrix-sub">${{formatDuration(totalTime)}}</div></div></td>`;
    }}

    function renderAllRuns(records) {{
      if (!records.length) {{
        byId('all-runs').innerHTML = '<div class="empty-state">No runs match the current filters.</div>';
        return;
      }}
      const rows = [...records].sort((a, b) => (
        a.descriptor.family.localeCompare(b.descriptor.family)
        || a.id.localeCompare(b.id)
        || solverOrder.indexOf(a.solver) - solverOrder.indexOf(b.solver)
      ));
      const html = `
        <div class="table-scroll">
          <table>
            <thead>
              <tr>
	                <th>Problem</th>
	                <th>Set</th>
	                <th>Family</th>
                <th>Solver</th>
                <th>JIT</th>
                <th>Status</th>
                <th>Expected</th>
                <th>Vars</th>
                <th>DOF</th>
                <th>Constr</th>
                <th>Iters</th>
                <th>Elastic</th>
                <th>Total</th>
                <th>Reason</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              ${{rows.map((record) => `
                <tr>
	                  <td class="mono">${{problemCell(record)}}</td>
	                  <td class="mono">${{htmlEscape(record.descriptor.test_set)}}</td>
	                  <td class="mono">${{htmlEscape(record.descriptor.family)}}</td>
                  <td class="solver-${{record.solver}}">${{solverLabel(record.solver)}}</td>
                  <td>${{record.options.jit_opt_level.toUpperCase()}} / ${{record.options.call_policy}}</td>
                  <td><span class="pill ${{statusClass(record.status)}}">${{htmlEscape(statusLabel(record.status))}}</span></td>
                  <td>${{htmlEscape(record.expected)}}</td>
                  <td>${{record.descriptor.num_vars}}</td>
                  <td>${{record.descriptor.dof}}</td>
                  <td>${{htmlEscape(constraintLabel(record))}}</td>
                  <td>${{intText(record.metrics.iterations)}}</td>
                  <td>${{htmlEscape(elasticText(record))}}</td>
                  <td>${{formatDuration(record.timing.total_wall_time)}}</td>
                  <td>${{htmlEscape(failureCode(record))}}</td>
                  <td class="${{detailClass(record)}}">${{htmlEscape(failureReason(record))}}</td>
                </tr>`).join('')}}
            </tbody>
          </table>
        </div>`;
      byId('all-runs').innerHTML = html;
    }}

    function problemCell(record) {{
      const href = problemHref(record);
      const label = htmlEscape(record.id);
      return href ? `<a href="${{htmlEscape(href)}}">${{label}}</a>` : label;
    }}

    function renderAll() {{
      const filtered = filteredRecords();
      renderActiveFilters();
      renderMetrics(filtered);
      renderScatter(filtered, 'chart-vars-iters', 'vars', 'iterations', (record) => record.descriptor.num_vars, (record) => record.metrics.iterations || 0);
      renderScatter(filtered, 'chart-dof-time', 'dof', 'total time (ms)', (record) => record.descriptor.dof, (record) => record.timing.total_wall_time * 1e3);
      renderSlowest(filtered);
      renderStatusBreakdown(filtered);
      renderFamilySummary(filtered);
      renderAllRuns(filtered);
    }}

    function installTooltip() {{
      const tooltip = byId('tooltip');
      document.addEventListener('mouseover', (event) => {{
        const target = event.target.closest('[data-tip]');
        if (!target) return;
        tooltip.textContent = target.getAttribute('data-tip');
        tooltip.classList.add('visible');
        tooltip.setAttribute('aria-hidden', 'false');
      }});
      document.addEventListener('mousemove', (event) => {{
        if (!tooltip.classList.contains('visible')) return;
        const x = Math.min(window.innerWidth - tooltip.offsetWidth - 12, event.clientX + 14);
        const y = Math.min(window.innerHeight - tooltip.offsetHeight - 12, event.clientY + 14);
        tooltip.style.left = `${{Math.max(8, x)}}px`;
        tooltip.style.top = `${{Math.max(8, y)}}px`;
      }});
      document.addEventListener('mouseout', (event) => {{
        const target = event.target.closest('[data-tip]');
        if (!target) return;
        tooltip.classList.remove('visible');
        tooltip.setAttribute('aria-hidden', 'true');
      }});
    }}

    initializeFilters();
    for (const id of filterIds) {{
      byId(id).addEventListener('input', renderAll);
      byId(id).addEventListener('change', renderAll);
    }}
    byId('reset-filters').addEventListener('click', () => {{
      for (const id of filterIds) byId(id).value = '';
      renderAll();
    }});
    installTooltip();
    renderAll();
  </script>
</body>
</html>"##
    ))
}
