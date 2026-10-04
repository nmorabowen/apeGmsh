// Header panel: the model file's name and counts, and the sibling
// artifacts whose paths are known. Renders from selectors alone.

import { modelSummary } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import { byId, el } from "../ui/dom.ts";

export function mountHeader(store: Store): () => void {
  const header = byId("header");
  const name = byId("file-name");
  const stats = byId("file-stats");
  const render = (s: State, prev: State | null) => {
    if (prev && s.artifacts === prev.artifacts) return;
    const summary = modelSummary(s);
    header.hidden = summary === null;
    if (!summary) return;
    name.textContent = summary.name;
    stats.replaceChildren(summary.stats);
    for (const k of ["geometry", "results"] as const) {
      const a = s.artifacts[k];
      if (!a) continue;
      const file = a.path.split(/[\\/]/).pop() ?? a.path;
      stats.append(" · ", el("span", "sibling", `${k}: ${file}${a.status === "ready" ? "" : " (not read)"}`));
    }
    document.title = `${summary.name} - apeGmshViewer`;
  };
  render(store.get(), null);
  return store.subscribe(render);
}
