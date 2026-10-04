// Banner panel: loud reader and mesh warnings, and the zones a reader
// refused ("written by an older apeGmsh"), never a blank view.

import { noticesOf, refusalsOf, warningsOf } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import { byId, el } from "../ui/dom.ts";

export function mountBanner(store: Store): () => void {
  const banner = byId("banner");
  const render = (s: State, prev: State | null) => {
    if (prev && s.artifacts === prev.artifacts && s.mesh === prev.mesh && s.geometry === prev.geometry) return;
    const refusals = refusalsOf(s);
    const notices = noticesOf(s);
    const warnings = warningsOf(s);
    banner.hidden = refusals.length === 0 && notices.length === 0 && warnings.length === 0;
    banner.replaceChildren(
      ...refusals.map((r) => el("div", "refused", r)),
      ...notices.map((n) => el("div", "notice", n)),
      ...warnings.map((w) => el("div", undefined, w)),
    );
  };
  render(store.get(), null);
  return store.subscribe(render);
}
