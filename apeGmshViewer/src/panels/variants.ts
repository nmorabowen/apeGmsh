// Variants panel (VU10, ADR 0132 D8, D9): the study the user opened, as main
// holds it. One line per variant of `<study>.variants.json`: its label, its
// params, whether it is the window set or the compare set, and two controls,
// Switch (main opens that row's set as the window set) and Compare (main
// opens it read-only in its compare slot). A refused row says why. With a
// compare set open, Difference shows the window variant minus the compare
// variant as the contour. The panel names a variant by its row index and
// label only, never a path (owner ruling option A, #2753); it dispatches the
// clicks and reads the store, and the variant runner (renderer/contour.ts
// `VariantRunner`) carries them to main. It takes no room with no study open.

import type { State, Store } from "../state/store.ts";
import type { VariantRowIn, VariantsState } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const baseName = (path: string): string => path.split(/[\\/]/).pop() ?? path;

/** A row's params as one line, in the manifest's key order (`Ie/Ig = 0.35 · state = cracked`). */
export function paramsLine(params: VariantRowIn["params"]): string {
  const keys = Object.keys(params);
  return keys.length ? keys.map((k) => `${k} = ${String(params[k])}`).join(" · ") : "no params";
}

/** What the panel states for a study, as plain lines (tested in Node; the DOM below shows them). */
export function variantLines(v: VariantsState): {
  title: string;
  newer: string | null;
  rows: { index: number; label: string; params: string; path: string; role: "window" | "compare" | null; canSwitch: boolean; canCompare: boolean }[];
  refused: string[];
  compare: string | null;
  difference: "off" | "on" | "unavailable";
  notice: string | null;
} | null {
  const study = v.study;
  if (!study) return null;
  const c = v.compare;
  const busy = v.switching !== null || c?.status === "opening";
  return {
    title: `${baseName(study.manifest)}${study.version ? ` · ${study.version}` : ""}`,
    newer: study.newer,
    rows: study.rows.map((r) => {
      const isCompare = c !== null && c.status !== "failed" && c.index === r.index;
      return {
        index: r.index,
        label: r.label,
        params: paramsLine(r.params),
        path: r.path,
        role: study.active === r.index ? "window" : isCompare ? "compare" : null,
        canSwitch: !busy && study.active !== r.index,
        canCompare: !busy && study.active !== r.index,
      };
    }),
    refused: study.refused.map((r) => (r.index < 0 ? `${r.code}: ${r.reason}` : `${r.label ?? `row ${r.index}`} (refused, ${r.code}): ${r.reason}`)),
    compare:
      c === null
        ? null
        : c.status === "opening"
          ? `Opening ${c.label} as the compare set…`
          : c.status === "failed"
            ? null
            : `Comparing with ${c.label} (read-only; never the window set)`,
    difference: c?.status !== "open" ? "unavailable" : v.difference ? "on" : "off",
    notice: v.notice,
  };
}

/** The panel's own box, styled from the theme tokens (it sits in #rail with the offers). */
const BOX = "flex: 0 0 auto; min-height: 0; max-height: 40%; overflow: auto; padding: 10px 12px; border-radius: 10px; background: var(--panel); border: 1px solid var(--line);";
const HEAD = "display: flex; align-items: center; gap: 8px; margin: 0 0 6px; font-size: 12px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);";
const BUTTON = "padding: 1px 8px; border-radius: 5px; cursor: pointer; font: 600 12px var(--font); color: var(--text); background: var(--faint); border: 1px solid var(--line);";

function button(text: string, title: string, disabled: boolean, on: () => void, unlisten: (() => void)[]): HTMLButtonElement {
  const b = el("button", "variant-button", text);
  b.type = "button";
  b.title = title;
  b.disabled = disabled;
  b.style.cssText = BUTTON + (disabled ? " cursor: default; color: var(--muted); opacity: 0.6;" : "");
  unlisten.push(listen(b, "click", on));
  return b;
}

export function mountVariants(store: Store): () => void {
  const panel = el("aside");
  panel.id = "variants";
  panel.hidden = true;
  panel.style.cssText = BOX;
  const rail = byId("rail");
  const offers = document.getElementById("offers");
  if (offers && offers.parentElement === rail) offers.after(panel);
  else rail.prepend(panel);
  let unlisten: (() => void)[] = [];
  const render = (s: State, prev: State | null) => {
    if (prev && s.variants === prev.variants) return;
    for (const u of unlisten) u();
    unlisten = [];
    const lines = variantLines(s.variants);
    panel.hidden = lines === null;
    if (lines === null) {
      panel.replaceChildren();
      return;
    }
    const head = el("h3", undefined, "Variants");
    head.style.cssText = HEAD;
    head.append(el("span", "legend-count", String(lines.rows.length)));
    const parts: HTMLElement[] = [head, el("div", "offer-note", lines.title)];
    if (lines.newer) parts.push(el("div", "offer-note", lines.newer));
    for (const r of lines.rows) {
      const row = el("div", "offer-row");
      row.dataset["variant"] = String(r.index);
      const top = el("div", "offer-top");
      top.append(el("span", "offer-name", r.label));
      top.append(el("span", `offer-status${r.role ? " applied" : ""}`, r.role ?? "variant"));
      top.append(
        button("Switch", r.role === "window" ? "This variant is the window set" : `Open ${r.label} as the window set`, !r.canSwitch, () => store.dispatch({ type: "variantSwitch", index: r.index, label: r.label }), unlisten),
      );
      top.append(
        r.role === "compare"
          ? button("Close", `Close the compare set ${r.label}`, s.variants.switching !== null, () => store.dispatch({ type: "variantCompare", row: null }), unlisten)
          : button("Compare", r.role === "window" ? "The window set cannot also be the compare set" : `Open ${r.label} read-only to compare with`, !r.canCompare, () => store.dispatch({ type: "variantCompare", row: { index: r.index, label: r.label } }), unlisten),
      );
      row.append(top);
      const params = el("div", "offer-caption", r.params);
      params.title = r.path;
      row.append(params);
      parts.push(row);
    }
    for (const t of lines.refused) parts.push(el("div", "offer-reason", t));
    if (lines.compare) parts.push(el("div", "offer-note", lines.compare));
    if (lines.difference !== "unavailable") {
      const on = lines.difference === "on";
      parts.push(
        button(on ? "Hide difference" : "Show difference", on ? "Colour by the window variant's own results again" : "Colour by the window variant minus the compare variant, at the step shown", false, () => {
          if (!on && store.get().visibility.colourBy !== "contour") store.dispatch({ type: "setColourBy", by: "contour" });
          store.dispatch({ type: "setDifference", on: !on });
        }, unlisten),
      );
    }
    if (lines.notice) parts.push(el("div", "offer-reason", lines.notice));
    panel.replaceChildren(...parts);
  };
  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  return () => {
    unsubscribe();
    for (const u of unlisten) u();
    panel.remove();
  };
}
