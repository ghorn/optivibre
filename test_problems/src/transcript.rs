use std::fs;
use std::path::Path;

use anyhow::Result;

use crate::model::{ProblemRunRecord, RunStatus, SolverKind};
use crate::runner::RunResults;

pub fn write_transcript_artifacts(results: &mut RunResults, output_dir: &Path) -> Result<()> {
    let transcript_dir = output_dir.join("transcripts");
    fs::create_dir_all(&transcript_dir)?;
    for record in &mut results.records {
        let Some(text) = record.console_output.as_deref() else {
            record.console_output_path = None;
            continue;
        };
        let basename = format!(
            "{}__{}__{}",
            slugify(&record.id),
            slugify(record.solver.label()),
            slugify(&record.options.label()),
        );
        let txt_filename = format!("{basename}.txt");
        let html_filename = format!("{basename}.html");
        let txt_path = transcript_dir.join(&txt_filename);
        let html_path = transcript_dir.join(&html_filename);
        fs::write(&txt_path, text)?;
        fs::write(
            &html_path,
            render_transcript_html(record, text, &txt_filename),
        )?;
        record.console_output_path = Some(format!("transcripts/{html_filename}"));
    }
    Ok(())
}

fn render_transcript_html(
    record: &ProblemRunRecord,
    transcript: &str,
    txt_filename: &str,
) -> String {
    let (status_class, status_text) = status_badge(record.status);
    let reason = transcript_reason(record);
    let elastic_stats = match (
        record.metrics.elastic_recovery_activations,
        record.metrics.elastic_recovery_qp_solves,
    ) {
        (Some(activations), Some(recovery_qps)) => {
            format!("{activations} activations / {recovery_qps} recovery_qps")
        }
        _ => "--".to_string(),
    };
    let filter_replay_json = record
        .filter_replay
        .as_ref()
        .and_then(|replay| serde_json::to_string(replay).ok());
    let filter_objective_label = match record.solver {
        SolverKind::Sqp | SolverKind::Nlip => "Objective",
        #[cfg(feature = "ipopt")]
        SolverKind::Ipopt => "Objective",
    };
    let filter_section = filter_replay_json.as_ref().map(|json| {
        format!(
            r#"
    <section class="filter-shell">
      <div class="filter-header">
        <div>
          <h2>Filter Replay</h2>
          <div class="filter-meta">Accepted path, current frontier, and rejected trials by iteration</div>
        </div>
        <div id="filter-frame-label" class="filter-frame-label"></div>
      </div>
      <div id="filter-plot" class="filter-plot"></div>
      <label class="filter-slider-shell" for="filter-frame">
        <span>Iteration</span>
        <input id="filter-frame" type="range" min="0" max="0" step="1" value="0" />
      </label>
    </section>
    <script defer src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
      const replay = {json};
      const mount = document.getElementById('filter-plot');
      const slider = document.getElementById('filter-frame');
      const label = document.getElementById('filter-frame-label');
      const config = {{ responsive: true, displaylogo: false, displayModeBar: false }};

      function pointViolation(point) {{
        return Math.max(point.violation, 1e-14);
      }}

      function frameLabel(frame) {{
        const mode = frame.accepted_mode ? ` · ${{frame.accepted_mode.replaceAll('_', ' ')}}` : '';
        return `iter ${{frame.iteration}} · ${{frame.phase.replaceAll('_', ' ')}}${{mode}}`;
      }}

      function renderFilterFrame(index) {{
        const frame = replay.frames[index];
        if (!frame || !window.Plotly) {{
          return;
        }}
        const pathFrames = replay.frames.slice(0, index + 1);
        const acceptedPath = {{
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Accepted path',
          x: pathFrames.map((entry) => pointViolation(entry.current)),
          y: pathFrames.map((entry) => entry.current.objective),
          line: {{ color: '#f7b267', width: 2.6 }},
          marker: {{ size: 5 }},
        }};
        const frontier = [...frame.frontier].sort((lhs, rhs) => lhs.violation - rhs.violation);
        const frontierTrace = {{
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Filter frontier',
          x: frontier.map((entry) => pointViolation(entry)),
          y: frontier.map((entry) => entry.objective),
          line: {{ color: '#5bd1b5', width: 2.4, dash: 'dot' }},
          marker: {{ size: 7 }},
        }};
        const currentTrace = {{
          type: 'scatter',
          mode: 'markers',
          name: 'Current iterate',
          x: [pointViolation(frame.current)],
          y: [frame.current.objective],
          marker: {{
            size: 12,
            color: '#f25f5c',
            line: {{ color: 'rgba(226, 232, 240, 0.92)', width: 1.4 }},
          }},
        }};
        const rejectedTrace = {{
          type: 'scatter',
          mode: 'markers',
          name: 'Rejected trials',
          x: frame.rejected_trials.map((entry) => pointViolation(entry)),
          y: frame.rejected_trials.map((entry) => entry.objective),
          marker: {{
            size: 8,
            color: '#93c5fd',
            symbol: 'x',
            line: {{ color: '#93c5fd', width: 1.2 }},
          }},
        }};
        const layout = {{
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: '#020617',
          margin: {{ l: 68, r: 18, t: 16, b: 54 }},
          font: {{ color: '#e2e8f0', family: 'ui-sans-serif, system-ui, sans-serif', size: 12 }},
          legend: {{ orientation: 'h', y: -0.26, x: 0, font: {{ color: '#94a3b8', size: 11 }} }},
          hoverlabel: {{ bgcolor: '#020617', bordercolor: '#334155', font: {{ color: '#e2e8f0' }} }},
          xaxis: {{
            title: 'Violation (∞-norm)',
            type: 'log',
            gridcolor: 'rgba(148, 163, 184, 0.12)',
            linecolor: '#475569',
            zeroline: false,
            ticks: 'outside',
            titlefont: {{ color: '#cbd5e1' }},
          }},
          yaxis: {{
            title: '{filter_objective_label} (-)',
            gridcolor: 'rgba(148, 163, 184, 0.12)',
            linecolor: '#475569',
            zeroline: false,
            ticks: 'outside',
            titlefont: {{ color: '#cbd5e1' }},
          }},
        }};
        label.textContent = frameLabel(frame);
        window.Plotly.react(mount, [acceptedPath, frontierTrace, currentTrace, rejectedTrace], layout, config);
      }}

      function startFilterReplay() {{
        if (!window.Plotly) {{
          setTimeout(startFilterReplay, 80);
          return;
        }}
        slider.max = String(Math.max(replay.frames.length - 1, 0));
        slider.value = slider.max;
        slider.addEventListener('input', () => renderFilterFrame(Number(slider.value)));
        renderFilterFrame(Number(slider.value));
      }}

      startFilterReplay();
    </script>"#,
            filter_objective_label = filter_objective_label,
        )
    }).unwrap_or_default();
    format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ margin: 24px; background: #0f172a; color: #e2e8f0; font-family: ui-sans-serif, system-ui, sans-serif; }}
    .card {{ background: #111827; border: 1px solid #334155; border-radius: 14px; padding: 20px; max-width: 1100px; }}
    h1 {{ margin: 0 0 10px; font-size: 28px; }}
    .meta {{ color: #94a3b8; margin: 0 0 16px; }}
    .links {{ margin: 0 0 16px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 0 0 18px; }}
    .summary-card {{ background: #0b1220; border: 1px solid #334155; border-radius: 12px; padding: 12px 14px; }}
    .summary-label {{ color: #94a3b8; font-size: 12px; margin-bottom: 4px; }}
    .summary-value {{ font-weight: 700; }}
    .badge {{ display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: 12px; font-weight: 700; }}
    .ok {{ background: #dcfce7; color: #166534; }}
    .warn {{ background: #fef3c7; color: #92400e; }}
    .bad {{ background: #fee2e2; color: #991b1b; }}
    .skip {{ background: #e5e7eb; color: #374151; }}
    .filter-shell {{
      margin: 0 0 18px;
      padding: 14px;
      background: #0b1220;
      border: 1px solid #334155;
      border-radius: 14px;
    }}
    .filter-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 10px;
    }}
    .filter-header h2 {{
      margin: 0;
      font-size: 18px;
    }}
    .filter-meta, .filter-frame-label {{
      color: #94a3b8;
      font-size: 13px;
    }}
    .filter-plot {{
      min-height: 320px;
      border-radius: 12px;
      overflow: hidden;
      background: #020617;
      border: 1px solid #1e293b;
    }}
    .filter-slider-shell {{
      display: grid;
      gap: 8px;
      margin-top: 12px;
      color: #94a3b8;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .filter-slider-shell input {{
      width: 100%;
    }}
    a {{ color: #93c5fd; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    pre {{
      margin: 0;
      padding: 16px;
      overflow-x: auto;
      white-space: pre;
      background: #020617;
      border: 1px solid #1e293b;
      border-radius: 12px;
      color: #e2e8f0;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      line-height: 1.5;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>{title}</h1>
    <p class="meta">{solver} / {jit}</p>
    <p class="links"><a href="../report.html">report</a> · <a href="../dashboard.html">dashboard</a> · <a href="{txt_filename}">raw txt</a></p>
    <div class="summary">
      <div class="summary-card"><div class="summary-label">Status</div><div class="summary-value"><span class="badge {status_class}">{status_text}</span></div></div>
      <div class="summary-card"><div class="summary-label">Reason</div><div class="summary-value">{reason}</div></div>
      <div class="summary-card"><div class="summary-label">Emergency restorations</div><div class="summary-value">{elastic_stats}</div></div>
      <div class="summary-card"><div class="summary-label">Termination thresholds</div><div class="summary-value">{solver_thresholds}</div></div>
      <div class="summary-card"><div class="summary-label">Validation thresholds</div><div class="summary-value">{validation_thresholds}</div></div>
    </div>
    {filter_section}
    <pre>{transcript}</pre>
  </div>
</body>
</html>"#,
        title = html_escape(&record.id),
        solver = html_escape(record.solver.label()),
        jit = html_escape(&record.options.label()),
        txt_filename = html_escape(txt_filename),
        status_class = status_class,
        status_text = status_text,
        reason = html_escape(&reason),
        elastic_stats = html_escape(&elastic_stats),
        solver_thresholds = html_escape(record.solver_thresholds.as_deref().unwrap_or("--")),
        validation_thresholds = html_escape(&record.validation.tolerance),
        filter_section = filter_section,
        transcript = html_escape(transcript),
    )
}

fn status_badge(status: RunStatus) -> (&'static str, &'static str) {
    match status {
        RunStatus::Passed => ("ok", "PASS"),
        RunStatus::ReducedAccuracy => ("warn", "REDUCED"),
        RunStatus::FailedValidation => ("bad", "FAIL"),
        RunStatus::SolveError => ("bad", "ERROR"),
        RunStatus::Skipped => ("skip", "SKIP"),
    }
}

fn transcript_reason(record: &ProblemRunRecord) -> String {
    match record.status {
        RunStatus::Passed => "--".to_string(),
        RunStatus::ReducedAccuracy | RunStatus::FailedValidation => {
            record.validation.detail.clone()
        }
        RunStatus::SolveError => record
            .error
            .clone()
            .unwrap_or_else(|| "solve error".to_string()),
        RunStatus::Skipped => "skipped".to_string(),
    }
}

fn slugify(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    out
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
