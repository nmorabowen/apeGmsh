// The first tool of ADR 0137 (H1, H2b): the model window's results step, read from `ToolContext`,
// buttons that ask the store for the previous or next step, and (H2b) the picked node's history of
// one nodal component over the stage, asked of main's data broker and drawn by the shared chart
// with the step on screen marked. A step change moves the mark only; a new stage, node or
// component asks again, and the newest request wins.

import { fmt, xyPlot } from "../ui/chart.ts";
import type { ToolAnswer, ToolContext, ToolDefinition, ToolHost } from "./contract.ts";

/** The component a new stage starts on: the first x displacement, else the first component. */
export function defaultComponent(components: readonly string[]): string | null {
  return components.find((c) => /(^|[_/])(displacement_x|Ux)$/i.test(c)) ?? components[0] ?? null;
}

function mount(host: ToolHost) {
  const doc = host.root.ownerDocument;
  const line = (cls: string, parent: HTMLElement = host.root) => {
    const d = doc.createElement("div");
    d.className = cls;
    parent.append(d);
    return d;
  };
  const model = line("tool-model");
  const stage = line("tool-stage");
  const step = line("tool-step");
  const picked = line("tool-picked");
  const bar = line("tool-bar");
  const button = (label: string, title: string) => {
    const b = doc.createElement("button");
    b.textContent = label;
    b.title = title;
    bar.append(b);
    return b;
  };
  const prev = button("◀ Previous step", "ask the model window for the previous step");
  const next = button("Next step ▶", "ask the model window for the next step");
  const history = line("tool-history");
  const pickRow = line("tool-history-bar", history);
  const select = doc.createElement("select");
  select.title = "the nodal component to chart over the stage";
  pickRow.append(select);
  const range = line("tool-range", history);
  const note = line("tool-history-note", history);
  const plotBox = line("tool-history-plot", history);

  let ctx: ToolContext | null = null;
  let at: ToolContext["step"] = null;
  let component: string | null = null;
  /** what was asked last, and what came back for it */
  let asked: string | null = null;
  let data: { key: string; values: ArrayLike<number> } | null = null;
  let error: string | null = null;

  prev.addEventListener("click", () => at && at.step > 0 && host.send({ type: "setResultStep", step: at.step - 1 }));
  next.addEventListener("click", () => at && at.step < at.steps - 1 && host.send({ type: "setResultStep", step: at.step + 1 }));
  select.addEventListener("change", () => {
    component = select.value;
    if (ctx) draw(ctx);
  });

  const nodeOf = (c: ToolContext) => (c.targets.nodes > 0 ? c.targets.nodeTags[0]! : null);

  /** Ask for the history the context and the chosen component name, when it is not the one asked last. */
  const ask = (c: ToolContext) => {
    const node = nodeOf(c);
    if (!c.step || node === null || component === null) return;
    const key = JSON.stringify([c.step.stage, node, component]);
    if (key === asked) return;
    asked = key;
    error = null;
    void host.request({ kind: "nodeHistory", stage: c.step.stage, node, component }).then((a: ToolAnswer) => {
      if (key !== asked) return;
      if (!a.ok) {
        if (a.superseded) return;
        error = a.error;
      } else if (a.kind === "nodeHistory") data = { key, values: a.values };
      if (ctx) draw(ctx);
    });
  };

  /**
   * H3: the chosen component's range over the stage's nodes at the step on screen. A second reader of
   * each step, which main serves from the viewport's read of it (the shared step cache).
   */
  let rangeAsked: string | null = null;
  const askRange = (c: ToolContext) => {
    if (!c.step || component === null) {
      range.textContent = "";
      return;
    }
    const s = c.step;
    const key = JSON.stringify([s.stage, s.step, component]);
    if (key === rangeAsked) return;
    rangeAsked = key;
    const comp = component;
    void host.request({ kind: "stepRange", stage: s.stage, step: s.step, component: comp }).then((a: ToolAnswer) => {
      if (key !== rangeAsked) return;
      if (!a.ok) {
        if (!a.superseded) range.textContent = a.error;
        return;
      }
      if (a.kind !== "stepRange") return;
      range.textContent =
        a.min === null || a.max === null
          ? `At step ${a.step + 1}: no ${comp} value at any node`
          : `At step ${a.step + 1}: ${comp} from ${fmt(a.min)} to ${fmt(a.max)} over ${a.nodes.toLocaleString("en-US")} nodes`;
    });
  };

  const draw = (c: ToolContext) => {
    const s = c.step;
    const node = nodeOf(c);
    history.hidden = !s;
    if (!s) return;
    // A stage's components: keep the choice while the stage offers it.
    const comps = s.components;
    if (component === null || !comps.includes(component)) component = defaultComponent(comps);
    if (select.options.length !== comps.length || [...select.options].some((o, i) => o.value !== comps[i])) {
      select.replaceChildren(
        ...comps.map((cName) => {
          const o = doc.createElement("option");
          o.value = cName;
          o.textContent = cName;
          return o;
        }),
      );
    }
    if (component !== null) select.value = component;
    ask(c);
    askRange(c);
    const key = node === null || component === null ? null : JSON.stringify([s.stage, node, component]);
    if (node === null) {
      note.textContent = "Pick a node in the model window (the selection bar's Nodes) to chart its history here.";
      plotBox.replaceChildren();
    } else if (error !== null) {
      note.textContent = error;
      plotBox.replaceChildren();
    } else if (data === null || data.key !== key) {
      note.textContent = `Reading node ${node}'s ${component} over the stage…`;
    } else {
      note.textContent = `Node ${node}, ${s.steps} steps; the step on screen is marked.`;
      const xs = Array.from({ length: data.values.length }, (_, i) => i + 1);
      plotBox.replaceChildren(xyPlot({ xs, ys: data.values, xName: "step", yName: `${component} at node ${node}`, title: `${component} at node ${node} over stage ${s.stageName}`, mark: s.step }));
    }
  };

  return {
    update(c: ToolContext) {
      ctx = c;
      at = c.step;
      model.textContent = c.model ?? "No model open";
      stage.textContent = c.step ? `Stage ${c.step.stageName}` : "No results stage shown";
      step.textContent = c.step ? `Step ${c.step.step + 1} of ${c.step.steps}${c.step.playing ? " · playing" : ""}` : "";
      const t = c.targets;
      picked.textContent = t.elements + t.nodes === 0 ? "Nothing picked" : `Picked: ${t.elements.toLocaleString("en-US")} elements, ${t.nodes.toLocaleString("en-US")} nodes`;
      prev.disabled = !c.step || c.step.step === 0;
      next.disabled = !c.step || c.step.step >= c.step.steps - 1;
      draw(c);
    },
  };
}

export const stepTool: ToolDefinition = { id: "step", title: "Step readout", cost: "light", mount };
