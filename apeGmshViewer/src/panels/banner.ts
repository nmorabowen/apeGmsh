// Banner panel: loud reader and mesh warnings, and the zones a reader
// refused ("written by an older apeGmsh"), never a blank view. V4b adds the
// results notices: why results are not drawn on this model (the pairing), and
// why a results stage draws every element (stage activation `ambiguous` or
// `absent`); selectors.noticesOf words them. V4d adds the deformed shape:
// how many drawn nodes have no displacement (`partial`), or why it is refused.
// V4c2a adds the element contour's refusal (an unverified file, disagreeing
// groups, rows that do not join the mesh) and element components refused alone.
// VU9a adds the workspace (ADR 0132): a newer file shown read-only, and the
// file an unreadable one was moved aside to, with a control that closes it.
// VF2 (#2924 item 6) adds a derived contour's out-of-plane line, the words of
// the manifest's `contour.derived.warning`, after the warnings.

import { BANNER_BELOW_SCRUBBER_PX, contourView, noticesOf, refusalsOf, SCRUBBER_TOP_PX, scrubberRow, warningsOf, workspaceAsideNotice } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import { byId, el, listen } from "../ui/dom.ts";

/** The banner's closed line: the message when there is one, else the counts of each kind. */
export function bannerSummary(refused: number, notices: number, warnings: number, only: string | null): string {
  if (refused + notices + warnings === 1 && only !== null) return only;
  const part = (n: number, one: string, many: string) => (n ? [`${n} ${n === 1 ? one : many}`] : []);
  return [...part(refused, "refused", "refused"), ...part(notices, "notice", "notices"), ...part(warnings, "warning", "warnings")].join(" · ");
}

/**
 * The banner's top under the scrubber row: 106 px under a one-line bar, else
 * 10 px below the bar's drawn height (`--phase-bar-h`, which panels/phase.ts
 * publishes on the root; a panel may not import another, so the name is
 * spelled here as in style.css). style.css moves `#left-rail` by the same rule.
 */
export function bannerTopBelowScrubber(): string {
  return `max(${BANNER_BELOW_SCRUBBER_PX}px, calc(${SCRUBBER_TOP_PX}px + var(--phase-bar-h, 0px) + 10px))`;
}

/**
 * VF2: a derived contour's out-of-plane line (selectors `derivedPlaneOf`), or
 * null. Its element ids are blobs, which a panel never holds, so
 * renderer/app.ts hands this, closing over the blob store.
 */
export type PlaneLineOf = (s: State) => string | null;

/** A banner mounted without `planeLine` cannot say the line: on a derived contour that throws, never a silent drop. */
function unwiredPlaneLine(s: State): null {
  const v = contourView(s);
  if (v.state === "element" && v.component.derived) {
    throw new Error(`banner: ${v.component.key} is drawn, and the banner was mounted without its out-of-plane line (mountBanner's planeLine)`);
  }
  return null;
}

export function mountBanner(store: Store, planeLine?: PlaneLineOf): () => void {
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
      s.contour.notice === prev.contour.notice &&
      s.workspace.read === prev.workspace.read &&
      s.workspace.aside === prev.workspace.aside &&
      // VF2: the derived contour's plane line follows the model's plane map and what the contour view is.
      s.decls === prev.decls &&
      s.variants === prev.variants &&
      // ADR 0134 D5: the particle lines (selectors `noticesOf`) follow the particle slice.
      s.particles === prev.particles
    ) return;
    const refusals = refusalsOf(s);
    const notices = noticesOf(s);
    const plane = planeLine ? planeLine(s) : unwiredPlaneLine(s);
    const warnings = [...warningsOf(s), ...(plane === null ? [] : [plane])];
    banner.hidden = refusals.length === 0 && notices.length === 0 && warnings.length === 0;
    // Below the results scrubber row when there is one (panels/phase.ts), however many lines it wrapped to:
    // phase.ts publishes the bar's drawn height as --phase-bar-h, and a one-line bar keeps the banner at 106 px (VU3b, #2737).
    banner.style.top = scrubberRow(s) ? bannerTopBelowScrubber() : "";
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
    const aside = s.workspace.aside === null ? null : workspaceAsideNotice(s.workspace.aside);
    const closers: (() => void)[] = [];
    const noticeRow = (n: string) => {
      const row = el("div", "notice", n);
      if (n !== aside) return row;
      // VU9a: the aside line is said once per rename; the user closes it.
      row.dataset["workspaceAside"] = "true";
      const close = el("button", "banner-summary", "Close");
      close.type = "button";
      close.title = "close this notice (the aside file stays on disk)";
      closers.push(listen(close, "click", () => store.dispatch({ type: "dismissWorkspaceAside" })));
      row.append(" ", close);
      return row;
    };
    banner.replaceChildren(
      summary,
      ...refusals.map((r) => el("div", "refused", r)),
      ...notices.map(noticeRow),
      ...warnings.map((w) => el("div", undefined, w)),
    );
    const summaryOff = unlisten;
    unlisten = () => {
      summaryOff?.();
      for (const c of closers) c();
    };
    publish();
  };
  /**
   * Where the banner is, for style.css: `#left-rail` sits below it (VU3b,
   * #2737), and so does `#scale-stack`, the colour scales (VF, #2924). Its top is the expression above (or the CSS 58 px off the scrubber
   * row), and its drawn height is measured: a still draws a closed banner's
   * whole list, up to the open banner's height limit (style.css), and a wrapped
   * bar moves it down. Only the closed height is
   * published: the open list overlays the rail (z-index 20), as it always has,
   * so while it is open the last closed height stands.
   */
  const rootStyle = () => {
    const root = typeof document === "undefined" ? undefined : document.documentElement?.style;
    return typeof root?.setProperty === "function" ? root : null;
  };
  const withdraw = () => {
    const root = rootStyle();
    root?.removeProperty("--banner-top");
    root?.removeProperty("--banner-h");
  };
  const publish = () => {
    const root = rootStyle();
    if (!root) return;
    if (banner.hidden) return withdraw();
    root.setProperty("--banner-top", scrubberRow(store.get()) ? bannerTopBelowScrubber() : "58px");
    if (banner.dataset["open"] === undefined) root.setProperty("--banner-h", `${Math.ceil(banner.getBoundingClientRect().height)}px`);
  };
  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  const sized = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(publish);
  sized?.observe(banner);
  return () => {
    unsubscribe();
    unlisten?.();
    sized?.disconnect();
    withdraw();
  };
}
