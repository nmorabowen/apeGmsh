// Node history (ADR 0137 H2b, its own tool since H4): the picked node's history of one nodal
// component over the stage, asked of main's data broker and drawn by the shared chart with the step
// on screen marked. A step change moves the mark only; a new stage, node or component asks again,
// and the newest request of this tile wins. Several Node history tiles may chart other components.

import { xyPlot } from "../ui/chart.ts";
import { componentPicker } from "./components.ts";
import type { ToolAnswer, ToolContext, ToolDefinition, ToolHost } from "./contract.ts";

function mount(host: ToolHost) {
  const doc = host.root.ownerDocument;
  const line = (cls: string) => {
    const d = doc.createElement("div");
    d.className = cls;
    host.root.append(d);
    return d;
  };
  const pickRow = line("tool-history-bar");
  const note = line("tool-history-note");
  const plotBox = line("tool-history-plot");

  let ctx: ToolContext | null = null;
  const picker = componentPicker(doc, "the nodal component to chart over the stage", () => ctx && draw(ctx));
  pickRow.append(picker.select);
  /** what was asked last, and what came back for it */
  let asked: string | null = null;
  let data: { key: string; values: ArrayLike<number> } | null = null;
  let error: string | null = null;

  const nodeOf = (c: ToolContext) => (c.targets.nodes > 0 ? c.targets.nodeTags[0]! : null);

  const ask = (stage: string, node: number, component: string, key: string) => {
    if (key === asked) return;
    asked = key;
    error = null;
    void host.request({ kind: "nodeHistory", stage, node, component }).then((a: ToolAnswer) => {
      if (key !== asked) return;
      if (!a.ok) {
        if (a.superseded) return;
        error = a.error;
      } else if (a.kind === "nodeHistory") data = { key, values: a.values };
      if (ctx) draw(ctx);
    });
  };

  const draw = (c: ToolContext) => {
    const s = c.step;
    pickRow.hidden = !s;
    if (!s) {
      note.textContent = "No results stage shown in the model window.";
      plotBox.replaceChildren();
      return;
    }
    const component = picker.sync(s.components);
    const node = nodeOf(c);
    if (node === null || component === null) {
      note.textContent = "Pick a node in the model window (the selection bar's Nodes) to chart its history here.";
      plotBox.replaceChildren();
      return;
    }
    const key = JSON.stringify([s.stage, node, component]);
    ask(s.stage, node, component, key);
    if (error !== null) {
      note.textContent = error;
      plotBox.replaceChildren();
    } else if (data === null || data.key !== key) {
      note.textContent = `Reading node ${node}'s ${component} over the stage…`;
    } else {
      note.textContent = `Node ${node}, ${s.steps} steps; the step on screen is marked.`;
      const xs = Array.from({ length: data.values.length }, (_, i) => i + 1);
      plotBox.replaceChildren(xyPlot({ xs, ys: data.values, xName: "step", yName: `${component} at node ${node}`, title: `${component} at node ${node} over stage ${s.stageName}`, mark: s.step, fit: true }));
    }
  };

  return {
    update(c: ToolContext) {
      ctx = c;
      draw(c);
    },
  };
}

export const historyTool: ToolDefinition = { id: "history", title: "Node history", cost: "light", mount };
