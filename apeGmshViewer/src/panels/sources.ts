// Sources panel: every /provenance record of the model, in capture order,
// each with its go-to-source button (`requestSource`). Objects apeGmsh
// synthesised inside a verb the user called (a stage's HOLD series and
// support pattern) are listed by default and marked `synthesised`, with the
// verb; their source is the user's call of that verb (maintainer ruling on
// #1378). It dispatches events and reads selectors, nothing else.

import { sourceFailure, sourcesOf } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import { byId, el, listen } from "../ui/dom.ts";

export function mountSources(store: Store): () => void {
  const panel = byId("sources");
  let unlisten: (() => void)[] = [];
  const render = (s: State, prev: State | null) => {
    if (prev && s.provenance === prev.provenance && s.decls === prev.decls && s.source === prev.source) return;
    for (const u of unlisten) u();
    unlisten = [];
    const rows = sourcesOf(s);
    panel.hidden = rows.length === 0;
    const synthesised = rows.filter((r) => r.origin === "synthesised").length;
    const head = el("h3", undefined, "Sources");
    head.append(el("span", "legend-count", synthesised ? `${rows.length} (${synthesised} synthesised)` : String(rows.length)));
    const parts: HTMLElement[] = [head];
    for (const r of rows) {
      const row = el("div", "source-row" + (r.origin === "synthesised" ? " synthesised" : ""));
      row.append(el("span", "source-key", r.key));
      if (r.origin === "synthesised") {
        const mark = el("span", "tag-synth", "synthesised");
        mark.title = r.verb ? `made by apeGmsh inside your ${r.verb}() call` : "made by apeGmsh inside a verb you called";
        row.append(mark);
      }
      const go = el("button", "source", r.label ?? "source");
      go.title = r.off ?? `open ${r.label} in the editor`;
      if (r.off === null) {
        const key = r.key;
        unlisten.push(listen(go, "click", () => store.dispatch({ type: "requestSource", decl: key })));
      } else go.disabled = true;
      row.append(go);
      parts.push(row);
      const failed = sourceFailure(s, r.key);
      if (failed) parts.push(el("div", "source-failed", `go to source: ${failed}`));
    }
    panel.replaceChildren(...parts);
  };
  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  return () => {
    unsubscribe();
    for (const u of unlisten) u();
  };
}
