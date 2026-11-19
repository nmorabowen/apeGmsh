// Sources panel: every /provenance record of the model, in capture order,
// each with its go-to-source button (`requestSource`). Objects apeGmsh
// synthesised inside a verb the user called (a stage's HOLD series and
// support pattern) are listed by default and marked `synth`; their source is
// the user's call of that verb (maintainer ruling on #1378). It shares the
// right rail with the inspector, below it (design brief on #1426). It
// dispatches events and reads selectors, nothing else. The rows are added a
// page at a time: a large model has tens of thousands, and building and laying
// out every one cost seconds at load (#2838).

import { sourceFailure, sourcesHeaderOf, sourcesOf, type SourceRow } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import { byId, el, listen } from "../ui/dom.ts";

/** Rows added at a time. */
export const SOURCES_PAGE = 200;
/** How close to the end of the list (px) scrolling adds the next page. */
const NEAR_END = 400;

export function mountSources(store: Store): () => void {
  const panel = byId("sources");
  let unlisten: (() => void)[] = [];
  // How many rows are built, kept across a redraw of the same listing.
  let shown = SOURCES_PAGE;
  let listing: State["provenance"] | null = null;
  let showMore = () => {};
  const offScroll = listen(panel, "scroll", () => {
    if (panel.scrollTop + panel.clientHeight >= panel.scrollHeight - NEAR_END) showMore();
  });
  const render = (s: State, prev: State | null) => {
    if (prev && s.provenance === prev.provenance && s.decls === prev.decls && s.source === prev.source && s.artifacts === prev.artifacts) return;
    for (const u of unlisten) u();
    unlisten = [];
    showMore = () => {};
    if (s.provenance !== listing) {
      listing = s.provenance;
      shown = SOURCES_PAGE;
    }
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
    const rowOf = (r: SourceRow): HTMLElement => {
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
      return row;
    };
    const rows = sourcesOf(s);
    parts.push(...rows.slice(0, shown).map(rowOf));
    if (shown < rows.length) {
      const more = el("button", "sources-more");
      const label = () => {
        more.textContent = `show ${Math.min(SOURCES_PAGE, rows.length - shown)} more (${rows.length - shown} left)`;
      };
      label();
      showMore = () => {
        const from = shown;
        shown = Math.min(from + SOURCES_PAGE, rows.length);
        more.before(...rows.slice(from, shown).map(rowOf));
        if (shown < rows.length) label();
        else {
          more.remove();
          showMore = () => {};
        }
      };
      unlisten.push(listen(more, "click", () => showMore()));
      parts.push(more);
    }
    panel.replaceChildren(...parts);
  };
  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  return () => {
    unsubscribe();
    offScroll();
    for (const u of unlisten) u();
  };
}
