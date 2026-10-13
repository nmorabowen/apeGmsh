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
import { byId, el } from "../ui/dom.ts";

export function mountBanner(store: Store): () => void {
  const banner = byId("banner");
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
    banner.replaceChildren(
      ...refusals.map((r) => el("div", "refused", r)),
      ...notices.map((n) => el("div", "notice", n)),
      ...warnings.map((w) => el("div", undefined, w)),
    );
  };
  render(store.get(), null);
  return store.subscribe(render);
}
