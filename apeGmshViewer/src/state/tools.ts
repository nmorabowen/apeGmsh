// The model window's side of the tool contract (ADR 0137 D10, H1), as pure functions of the state:
// the context a tool is sent, the check on each event a tool sends back, and the banner's lines
// for tool windows. renderer/tools.ts carries the context and the events over each tool's port.

import { TARGETS_SHOWN, type ToolContext, type ToolEvent } from "../tools/contract.ts";
import type { Event, State, ToolsState } from "./types.ts";

export const initialTools: ToolsState = { windows: {}, paused: false, request: null };

/** The base name of a path, for the context (a tool is not told where files live). */
function fileName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

/** The context every tool is sent now: small and plain (D10). */
export function toolContextOf(s: State): ToolContext {
  const at = s.phase.at;
  const stage = at?.kind === "results" ? s.results?.stages[at.stage] : undefined;
  const sel = s.selection;
  return {
    model: s.artifacts.model ? fileName(s.artifacts.model.path) : null,
    step:
      at?.kind === "results" && stage
        ? { stage: stage.id, stageName: stage.name, step: at.step, steps: stage.time.length, playing: s.playing, components: [...stage.nodal] }
        : null,
    targets: { elements: sel.decls.length, nodes: sel.nodes.length, decls: sel.decls.slice(0, TARGETS_SHOWN), nodeTags: sel.nodes.slice(0, TARGETS_SHOWN) },
    params: {},
    theme: s.theme === "light" ? "light" : "dark",
  };
}

/**
 * The store event a tool's event asks for, when the state allows it now; null otherwise (the
 * event is dropped). Anything a tool sends is checked here first: a tool is another process, and
 * the reducer throws on a step outside its stage.
 */
export function checkToolEvent(s: State, e: unknown): Event | null {
  const ev = e as ToolEvent | null;
  if (ev === null || typeof ev !== "object") return null;
  if (ev.type === "setResultStep") {
    const at = s.phase.at;
    if (at?.kind !== "results" || typeof ev.step !== "number" || !Number.isInteger(ev.step)) return null;
    const n = s.results?.stages[at.stage]?.time.length ?? 0;
    if (ev.step < 0 || ev.step >= n) return null;
    return { type: "setResultStep", step: ev.step };
  }
  return null;
}

/** The banner's lines for tool windows that stopped, were refused, or are paused. */
export function toolNotices(s: State): string[] {
  const out: string[] = [];
  for (const t of Object.values(s.tools.windows)) {
    if (t.status === "stopped") out.push(`${t.title} stopped: ${t.reason ?? "its renderer exited"}. The model window is unaffected; open a new one from Spotlight (Ctrl+K).`);
    else if (t.status === "unresponsive") out.push(`${t.title} is not responding. The model window is unaffected.`);
    else if (t.status === "refused") out.push(`${t.title} was not opened: ${t.reason ?? "refused"}.`);
  }
  if (s.tools.paused) out.push("Tool windows are paused: under 1.0 GB of RAM is available. They update again when memory frees.");
  return out;
}
