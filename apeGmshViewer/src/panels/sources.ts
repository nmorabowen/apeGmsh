// Sources panel: every /provenance record of the model, in capture order,
// each with its go-to-source button (`requestSource`). Objects apeGmsh
// synthesised inside a verb the user called (a stage's HOLD series and
// support pattern) are listed by default and marked `synth`; their source is
// the user's call of that verb (maintainer ruling on #1378). It shares the
// right rail with the inspector, below it (design brief on #1426). It
// dispatches events and reads selectors, nothing else.

import { sourceFailure, sourcesHeaderOf, sourcesOf } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import { byId, el, listen } from "../ui/dom.ts";

export function mountSources(store: Store): () => void {
  const panel = byId("sources");
  let unlisten: (() => void)[] = [];
  const render = (s: State, prev: State | null) => {
    if (prev && s.provenance === prev.provenance && s.decls === prev.decls && s.source === prev.source && s.artifacts === prev.artifacts) return;
    for (const u of unlisten) u();
    unlisten = [];
    const header = sourcesHeaderOf(s);
    // Nothing to list (no model, no /provenance, no record): no box at all.
    panel.hidden = header === null;
    if (header === null) {
      panel.replaceChildren();
      return;
    }
    const head = el("h3", undefined, "Sources");
    head.append(el("span", "legend-count", header.synthesised ? `${header.count} · ${header.synthesised} synthesised` : String(header.count)));
    const parts: HTMLElement[] = [head];
    if (header.notice) parts.push(el("div", "sources-notice", header.notice));
    for (const r of sourcesOf(s)) {
      const synth = r.origin === "synthesised";
      const row = el("div", "source-row" + (synth ? " synthesised" : ""));
      // The key wraps only after a `/`: a <wbr> after each segment, and the
      // CSS never breaks inside one.
      const key = el("span", "source-key");
      r.segments.forEach((seg, i) => {
        key.append(el("span", i === 0 ? "source-zone" : "source-seg", seg));
        if (i < r.segments.length - 1) key.append(el("wbr"));
      });
      row.append(key);
      if (synth) {
        const mark = el("span", "synth", "synth");
        mark.title = r.verb ? `made by apeGmsh inside your ${r.verb}() call` : "made by apeGmsh inside a verb you called";
        row.append(mark);
      } else row.append(el("span"));
      const go = el("button", "source", r.label ?? "—");
      go.title = r.title;
      if (r.off === null) {
        const decl = r.key;
        unlisten.push(listen(go, "click", () => store.dispatch({ type: "requestSource", decl })));
      } else go.disabled = true;
      row.append(go);
      if (r.off !== null) row.append(el("div", "f-src source-off", r.off));
      const failed = sourceFailure(s, r.key);
      if (failed) row.append(el("div", "source-failed", `go to source: ${failed}`));
      parts.push(row);
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
