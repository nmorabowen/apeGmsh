// Step readout (ADR 0137 H1, H3): the model window's results step, read from `ToolContext`, buttons
// that ask the store for the previous or next step, and (H3) the chosen nodal component's range over
// the stage's nodes at the step on screen, a second reader of each step that main serves from the
// viewport's read of it (the shared step cache). The picked node's history is its own tool since H4
// (./history.ts).

import { fmt } from "../ui/chart.ts";
import { componentPicker } from "./components.ts";
import type { ToolAnswer, ToolContext, ToolDefinition, ToolHost } from "./contract.ts";

export { defaultComponent } from "./components.ts";

function mount(host: ToolHost) {
  const doc = host.root.ownerDocument;
  const line = (cls: string) => {
    const d = doc.createElement("div");
    d.className = cls;
    host.root.append(d);
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
  const rangeRow = line("tool-history-bar");
  const range = line("tool-range");

  let ctx: ToolContext | null = null;
  let at: ToolContext["step"] = null;
  const picker = componentPicker(doc, "the nodal component whose range over the nodes is shown", () => ctx && askRange(ctx));
  rangeRow.append(picker.select);

  prev.addEventListener("click", () => at && at.step > 0 && host.send({ type: "setResultStep", step: at.step - 1 }));
  next.addEventListener("click", () => at && at.step < at.steps - 1 && host.send({ type: "setResultStep", step: at.step + 1 }));

  let rangeAsked: string | null = null;
  const askRange = (c: ToolContext) => {
    const s = c.step;
    const component = s ? picker.sync(s.components) : null;
    if (!s || component === null) {
      range.textContent = "";
      return;
    }
    const key = JSON.stringify([s.stage, s.step, component]);
    if (key === rangeAsked) return;
    rangeAsked = key;
    void host.request({ kind: "stepRange", stage: s.stage, step: s.step, component }).then((a: ToolAnswer) => {
      if (key !== rangeAsked) return;
      if (!a.ok) {
        if (!a.superseded) range.textContent = a.error;
        return;
      }
      if (a.kind !== "stepRange") return;
      range.textContent =
        a.min === null || a.max === null
          ? `At step ${a.step + 1}: no ${component} value at any node`
          : `At step ${a.step + 1}: ${component} from ${fmt(a.min)} to ${fmt(a.max)} over ${a.nodes.toLocaleString("en-US")} nodes`;
    });
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
      rangeRow.hidden = !c.step;
      askRange(c);
    },
  };
}

export const stepTool: ToolDefinition = { id: "step", title: "Step readout", cost: "light", mount };
