type PlotlyPrimitive = string | number | boolean | null;
type PlotlyValue = PlotlyPrimitive | PlotlyObject | PlotlyValue[];

interface PlotlyObject {
  [key: string]: PlotlyValue | undefined;
}

type PlotlyRelayoutPayload = PlotlyObject;

interface PlotlyHostElement extends HTMLDivElement {
  on?: (eventName: "plotly_relayout", callback: (eventData: PlotlyRelayoutPayload) => void) => void;
}

interface PlotlyStatic {
  purge(element: PlotlyHostElement): void;
  relayout(
    element: PlotlyHostElement,
    update: PlotlyObject,
  ): Promise<void>;
  react(
    element: PlotlyHostElement,
    data: PlotlyObject[],
    layout: PlotlyObject,
    config: PlotlyObject,
  ): Promise<void>;
  newPlot(
    element: PlotlyHostElement,
    data: PlotlyObject[],
    layout: PlotlyObject,
    config: PlotlyObject,
  ): Promise<void>;
  extendTraces(
    element: PlotlyHostElement,
    update: PlotlyObject,
    traces: number[],
    maxPoints?: number,
  ): Promise<void>;
  restyle(
    element: PlotlyHostElement,
    update: PlotlyObject,
    traces?: number[],
  ): Promise<void>;
}

interface MathJaxStatic {
  startup?: {
    defaultReady(): void;
  };
  typesetClear?(roots: Element[]): void;
  typesetPromise?(roots: Element[]): Promise<void>;
}

interface Window {
  MathJax?: MathJaxStatic;
  Plotly?: PlotlyStatic;
}
