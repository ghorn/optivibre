type AnalysisMode = "base" | "rates" | "controls";
type PlotlyDatum = Record<string, unknown>;

interface PlotlyLike {
  newPlot(
    element: HTMLElement,
    data: PlotlyDatum[],
    layout: PlotlyDatum,
    config?: PlotlyDatum
  ): Promise<unknown>;
  purge(element: HTMLElement): void;
  Plots?: {
    resize(element: HTMLElement): void;
  };
}

declare const Plotly: PlotlyLike;

interface AeroSurfaceGrid {
  id: string;
  title: string;
  xLabel: string;
  yLabel: string;
  zLabel: string;
  x: number[];
  y: number[];
  z: number[][];
}

interface AeroSurfaceGroup {
  id: string;
  title: string;
  note: string;
  grids: AeroSurfaceGrid[];
}

interface AeroAnalysis {
  title: string;
  note: string;
  coefficientGrids: AeroSurfaceGrid[];
  rateDerivativeGroups: AeroSurfaceGroup[];
  controlSurfaceGroups: AeroSurfaceGroup[];
}

const noteNode = query<HTMLElement>("#analysis-note");
const statusNode = query<HTMLElement>("#analysis-status");
const sectionTitleNode = query<HTMLElement>("#analysis-section-title");
const sectionNoteNode = query<HTMLElement>("#analysis-section-note");
const gridNode = query<HTMLElement>("#analysis-grid");
const groupPicker = query<HTMLElement>("#analysis-group-picker");
const groupSelect = query<HTMLSelectElement>("#analysis-group-select");
const modeButtons = Array.from(
  document.querySelectorAll<HTMLButtonElement>("[data-analysis-mode]")
);

let analysis: AeroAnalysis | null = null;
let activeMode: AnalysisMode = "base";
let activePlots: HTMLElement[] = [];
const selectedGroupByMode = new Map<AnalysisMode, string>();

void loadAnalysis();

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const mode = button.dataset.analysisMode;
    if (mode === "base" || mode === "rates" || mode === "controls") {
      activeMode = mode;
      void renderActiveMode();
    }
  });
});

groupSelect.addEventListener("change", () => {
  selectedGroupByMode.set(activeMode, groupSelect.value);
  void renderActiveMode();
});

window.addEventListener("resize", () => {
  activePlots.forEach((plot) => Plotly.Plots?.resize(plot));
});

async function loadAnalysis(): Promise<void> {
  setStatus("Loading");
  try {
    const response = await fetch("/api/aero_analysis");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }
    analysis = (await response.json()) as AeroAnalysis;
    noteNode.textContent = analysis.note;
    setStatus("Ready");
    await renderActiveMode();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus("Failed");
    sectionNoteNode.textContent = message;
  }
}

async function renderActiveMode(): Promise<void> {
  if (!analysis) {
    return;
  }

  setStatus("Rendering");
  modeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.analysisMode === activeMode);
  });

  const selection = selectedSurfaces(analysis, activeMode);
  sectionTitleNode.textContent = selection.title;
  sectionNoteNode.textContent = selection.note;
  renderGroupPicker(selection.groups);
  await renderSurfaces(selection.grids);
  setStatus(`${selection.grids.length} surfaces`);
}

function selectedSurfaces(
  data: AeroAnalysis,
  mode: AnalysisMode
): {
  title: string;
  note: string;
  groups: AeroSurfaceGroup[];
  grids: AeroSurfaceGrid[];
} {
  if (mode === "base") {
    return {
      title: "Base Force and Moment Coefficients",
      note: "Zero-rate, zero-control coefficients swept over alpha and beta. Force coefficients use aerodynamic signs: positive lift, positive drag, and wind-frame side force C_Yw. Moments use body/CAD axes.",
      groups: [],
      grids: data.coefficientGrids
    };
  }

  const groups = mode === "rates" ? data.rateDerivativeGroups : data.controlSurfaceGroups;
  const fallback = groups[0];
  const selectedId = selectedGroupByMode.get(mode) ?? fallback?.id ?? "";
  const group = groups.find((candidate) => candidate.id === selectedId) ?? fallback;
  if (!group) {
    return {
      title: "No surfaces",
      note: "No aero-analysis surfaces were returned by the backend.",
      groups: [],
      grids: []
    };
  }
  selectedGroupByMode.set(mode, group.id);
  return {
    title: group.title,
    note: group.note,
    groups,
    grids: group.grids
  };
}

function renderGroupPicker(groups: AeroSurfaceGroup[]): void {
  groupPicker.hidden = groups.length === 0;
  groupSelect.innerHTML = "";
  groups.forEach((group) => {
    const option = document.createElement("option");
    option.value = group.id;
    option.textContent = group.title;
    groupSelect.append(option);
  });
  const selectedId = selectedGroupByMode.get(activeMode);
  if (selectedId) {
    groupSelect.value = selectedId;
  }
}

async function renderSurfaces(grids: AeroSurfaceGrid[]): Promise<void> {
  activePlots.forEach((plot) => Plotly.purge(plot));
  activePlots = [];
  gridNode.innerHTML = "";

  const plotPromises = grids.map(async (grid) => {
    const card = document.createElement("article");
    card.className = "analysis-plot-card";
    const title = document.createElement("h3");
    title.textContent = grid.title;
    const plot = document.createElement("div");
    plot.className = "analysis-plot";
    card.append(title, plot);
    gridNode.append(card);
    activePlots.push(plot);
    await Plotly.newPlot(plot, surfaceData(grid), surfaceLayout(grid), {
      responsive: true,
      displaylogo: false
    });
  });

  await Promise.all(plotPromises);
}

function surfaceData(grid: AeroSurfaceGrid): PlotlyDatum[] {
  return [
    {
      type: "surface",
      x: grid.x,
      y: grid.y,
      z: grid.z,
      colorscale: [
        [0.0, "#2563eb"],
        [0.45, "#dbeafe"],
        [0.5, "#f8fafc"],
        [0.55, "#fee2e2"],
        [1.0, "#dc2626"]
      ],
      cmid: 0,
      showscale: true,
      colorbar: {
        title: { text: grid.zLabel },
        thickness: 12,
        len: 0.72,
        outlinewidth: 0,
        tickfont: { color: "#d9ecfb" }
      },
      contours: {
        z: {
          show: true,
          usecolormap: true,
          project: { z: true },
          width: 1
        }
      },
      lighting: {
        ambient: 0.72,
        diffuse: 0.58,
        specular: 0.18,
        roughness: 0.85
      },
      hovertemplate: `${grid.xLabel}=%{x:.2f}<br>${grid.yLabel}=%{y:.2f}<br>${grid.zLabel}=%{z:.5f}<extra></extra>`
    }
  ];
}

function surfaceLayout(grid: AeroSurfaceGrid): PlotlyDatum {
  return {
    autosize: true,
    height: 580,
    margin: { l: 8, r: 8, t: 8, b: 8 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "#081018",
    font: {
      family: "IBM Plex Sans, Helvetica Neue, sans-serif",
      color: "#d9ecfb"
    },
    scene: {
      bgcolor: "#081018",
      xaxis: axisLayout(grid.xLabel),
      yaxis: axisLayout(grid.yLabel),
      zaxis: axisLayout(grid.zLabel),
      camera: {
        eye: { x: 1.45, y: -1.65, z: 1.15 }
      },
      aspectmode: "cube"
    }
  };
}

function axisLayout(title: string): PlotlyDatum {
  return {
    title: { text: title },
    gridcolor: "#243b4a",
    zerolinecolor: "#527086",
    linecolor: "#345266",
    tickfont: { color: "#d9ecfb" },
    titlefont: { color: "#d9ecfb" },
    showbackground: true,
    backgroundcolor: "#081018"
  };
}

function setStatus(message: string): void {
  statusNode.textContent = message;
}

function query<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`missing required element ${selector}`);
  }
  return element;
}
