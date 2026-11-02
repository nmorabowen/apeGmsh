// Banner panel: loud reader and mesh warnings, and the zones a reader
// refused ("written by an older apeGmsh"), never a blank view. V4b adds the
// results notices: why results are not drawn on this model (the pairing), and
// why a results stage draws every element (stage activation `ambiguous` or
// `absent`); selectors.noticesOf words them. V4d adds the deformed shape:
// how many drawn nodes have no displacement (`partial`), or why it is refused.
// V4c2a adds the element contour's refusal (an unverified file, disagreeing
// groups, rows that do not join the mesh) and element components refused alone.

import { BANNER_BELOW_SCRUBBER_PX, noticesOf, refusalsOf, scrubberRow, warningsOf } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import { byId, el, listen } from "../ui/dom.ts";

/** The banner's closed line: the message when there is one, else the counts of each kind. */
export function bannerSummary(refused: number, notices: number, warnings: number, only: string | null): string {
  if (refused + notices + warnings === 1 && only !== null) return only;
  const part = (n: number, one: string, many: string) => (n ? [`${n} ${n === 1 ? one : many}`] : []);
  return [...part(refused, "refused", "refused"), ...part(notices, "notice", "notices"), ...part(warnings, "warning", "warnings")].join(" · ");
}

export function mountBanner(store: Store): () => void {
  const banner = byId("banner");
  // Closed, one line: the rail and the scale sit below it and stay readable. A click reads the whole list.
  let open = false;
  let unlisten: (() => void) | null = null;
  const render = (s: State, prev: State | null) => {
    if (
      prev &&
      s.artifacts === prev.artifacts &&
      s.mesh === prev.mesh &&
      s.geometry === prev.geometry &&
      s.results === prev.results &&
      s.lineage === prev.lineage &&
      s.phase === prev.phase &&
      // V4d: the deformed shape's partial or refused line follows its state, the hidden set and the model's ndm.
      s.deform === prev.deform &&
      s.visibility === prev.visibility &&
      s.ndm === prev.ndm &&
      // V4c2a: an element contour's refusal follows the pick, its draw mode and the join verdict.
      s.contour.source === prev.contour.source &&
      s.contour.field === prev.contour.field &&
      s.contour.component === prev.contour.component &&
      s.contour.gauss === prev.contour.gauss &&
      s.contour.join === prev.contour.join &&
      s.contour.notice === prev.contour.notice
    ) return;
    const refusals = refusalsOf(s);
    const notices = noticesOf(s);
    const warnings = warningsOf(s);
    banner.hidden = refusals.length === 0 && notices.length === 0 && warnings.length === 0;
    // Below the results scrubber row when there is one (panels/phase.ts).
    banner.style.top = scrubberRow(s) ? `${BANNER_BELOW_SCRUBBER_PX}px` : "";
    unlisten?.();
    const only = [...refusals, ...notices, ...warnings][0] ?? null;
    const summary = el("button", "banner-summary" + (refusals.length ? " refused" : ""), bannerSummary(refusals.length, notices.length, warnings.length, only));
    summary.type = "button";
    summary.title = open ? "close the list" : "read the whole list";
    unlisten = listen(summary, "click", () => {
      open = !open;
      render(store.get(), null);
    });
    if (open) banner.dataset["open"] = "true";
    else delete banner.dataset["open"];
    banner.replaceChildren(
      summary,
      ...refusals.map((r) => el("div", "refused", r)),
      ...notices.map((n) => el("div", "notice", n)),
      ...warnings.map((w) => el("div", undefined, w)),
    );
  };
  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  return () => {
    unsubscribe();
    unlisten?.();
  };
}
