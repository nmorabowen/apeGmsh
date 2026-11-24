// The placeholder tool of ADR 0137 H1: the model window's results step, read from `ToolContext`,
// and buttons that ask the store for the previous or next step. It proves the host: a separate
// sandboxed window, the one-way context, and an event back. Real tools (the material tester's
// surface, H2's chart) replace it as the first tool.

import type { ToolContext, ToolDefinition, ToolHost } from "./contract.ts";

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
  let at: ToolContext["step"] = null;
  prev.addEventListener("click", () => at && at.step > 0 && host.send({ type: "setResultStep", step: at.step - 1 }));
  next.addEventListener("click", () => at && at.step < at.steps - 1 && host.send({ type: "setResultStep", step: at.step + 1 }));
  return {
    update(ctx: ToolContext) {
      at = ctx.step;
      model.textContent = ctx.model ?? "No model open";
      stage.textContent = ctx.step ? `Stage ${ctx.step.stageName}` : "No results stage shown";
      step.textContent = ctx.step ? `Step ${ctx.step.step + 1} of ${ctx.step.steps}${ctx.step.playing ? " · playing" : ""}` : "";
      const t = ctx.targets;
      picked.textContent = t.elements + t.nodes === 0 ? "Nothing picked" : `Picked: ${t.elements.toLocaleString("en-US")} elements, ${t.nodes.toLocaleString("en-US")} nodes`;
      prev.disabled = !ctx.step || ctx.step.step === 0;
      next.disabled = !ctx.step || ctx.step.step >= ctx.step.steps - 1;
    },
  };
}

export const stepTool: ToolDefinition = { id: "step", title: "Step readout", cost: "light", mount };
