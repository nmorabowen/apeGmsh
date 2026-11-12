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
// VU19: with a compare set open, the panel also lists the compare set's own
// `.user.h5` (read through main's compare doors, never the window set's file),
// each user result with A6's verdict against the compare set's own model and
// results, and Draw for one that can be placed on the window's mesh (one mesh
// by fem_hash; never interpolated).

import { compareMeshOf } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { VariantRowIn, VariantsState } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const baseName = (path: string): string => path.split(/[\\/]/).pop() ?? path;

/** A row's params as one line, in the manifest's key order (`Ie/Ig = 0.35 · state = cracked`). */
export function paramsLine(params: VariantRowIn["params"]): string {
  const keys = Object.keys(params);
  return keys.length ? keys.map((k) => `${k} = ${String(params[k])}`).join(" · ") : "no params";
}

/** One user result of the compare set's file, as the panel lists it: its verdict against the compare set's own model, and what can be drawn. */
export interface CompareCaseLine {
  name: string;
  /** `footfall · on nodes · basis model` */
  what: string;
  /** A6 against the compare set's own model and results */
  verdict: "current" | "stale";
  /** why it is stale; "" when current */
  reason: string;
  /** one per component, when it can be drawn on this mesh; `shown` is the one drawn */
  draws: { component: string; shown: boolean }[];
  /** why it cannot be drawn, or null */
  why: string | null;
}

/** VU19: the compare set's `.user.h5` as the panel states it, or null with no compare set open. */
export interface CompareUserLines {
  /** what the file is, or why it is not read */
  title: string;
  /** whether the file was read (then `cases` and `refused` list it) */
  read: boolean;
  /** its User Combinations zone: none, empty, or how many rows (drawn only as the window set's own) */
  combinations: string | null;
  cases: CompareCaseLine[];
  refused: string[];
  /** why no compare user result is placed on the window's mesh, or null when it can be */
  mesh: string | null;
}

/** The compare set's user results, its title, verdicts and draw controls (`mesh`: selectors `compareMeshOf`). */
function compareUserLines(v: VariantsState, mesh: { ok: true } | { ok: false; reading: boolean; reason: string }): CompareUserLines | null {
  const c = v.compare;
  if (c?.status !== "open") return null;
  const u = v.compareUser;
  const of = `${c.label}'s`;
  const meshWhy = mesh.ok ? null : mesh.reason;
  if (u === null || u.status === "reading") return { title: `Reading ${of} user file…`, read: false, combinations: null, cases: [], refused: [], mesh: meshWhy };
  if (u.status === "none") return { title: `${c.label} has no user file (${baseName(u.path)})`, read: false, combinations: null, cases: [], refused: [], mesh: meshWhy };
  if (u.status === "failed") return { title: `${of} user file is refused: ${u.error}`, read: false, combinations: null, cases: [], refused: [], mesh: meshWhy };
  const info = u.info;
  const cmb = info.combinations;
  const combinations =
    cmb === null ? (info.generations ? "User Combinations: an empty zone (no row)" : null) : !cmb.ok ? `User Combinations refused: ${cmb.reason}` : `${cmb.items.length} User Combination${cmb.items.length === 1 ? "" : "s"} (of the compare variant; never combined across variants)`;
  const r = info.results;
  const cases: CompareCaseLine[] = [];
  const refused: string[] = [];
  if (r !== null && !r.ok) refused.push(`/results refused: ${r.reason}`);
  if (r !== null && r.ok) {
    for (const e of r.items) {
      const stale = e.stale.stale;
      let why: string | null = null;
      if (e.target !== "nodes") why = `on ${e.target}; this app draws a user result on nodes only`;
      else if (e.perState) why = "per state; this app draws a state-free one only";
      else if (stale) why = "stale (A6), so it is not drawn";
      else if (meshWhy !== null) why = meshWhy;
      const shown = v.shown?.name === e.name ? v.shown.component : null;
      cases.push({
        name: e.name,
        what: `${e.name} · on ${e.target} · basis ${e.basis}${e.perState ? " · per state" : ""}`,
        verdict: stale ? "stale" : "current",
        reason: stale ? e.stale.reason || "no reason given" : "",
        draws: why === null ? e.components.map((component) => ({ component, shown: shown === component })) : [],
        why,
      });
    }
    for (const x of r.refused) refused.push(`${x.name} (refused): ${x.reason}${x.detail ? ` (${x.detail})` : ""}`);
  }
  return { title: `${of} user file: ${baseName(u.path)}`, read: true, combinations, cases, refused, mesh: meshWhy };
}

/** What the panel states for a study, as plain lines (tested in Node; the DOM below shows them). */
export function variantLines(
  v: VariantsState,
  mesh: { ok: true } | { ok: false; reading: boolean; reason: string } = { ok: false, reading: false, reason: "the two variants' meshes are not known" },
): {
  title: string;
  newer: string | null;
  rows: { index: number; label: string; params: string; path: string; role: "window" | "compare" | null; canSwitch: boolean; canCompare: boolean }[];
  refused: string[];
  compare: string | null;
  /** VU19: the compare set's user file, or null with no compare set open */
  compareUser: CompareUserLines | null;
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
    compareUser: compareUserLines(v, mesh),
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
    // VU19: whether a compare user result can be drawn also follows the window's model (its fem_hash).
    if (prev && s.variants === prev.variants && s.lineage === prev.lineage && s.visibility.colourBy === prev.visibility.colourBy) return;
    for (const u of unlisten) u();
    unlisten = [];
    const lines = variantLines(s.variants, compareMeshOf(s));
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
    const cu = lines.compareUser;
    if (cu) {
      // VU19: the compare set's own user file, apart from the window set's: each case with its verdict against the compare set's own model.
      const box = el("div", "compare-user");
      box.style.cssText = "margin: 6px 0 2px; padding: 6px 8px; border-radius: 8px; border: 1px dashed var(--line);";
      const head = el("div", "offer-note", cu.title);
      head.style.fontWeight = "600";
      box.append(head);
      if (cu.combinations) box.append(el("div", "offer-caption", cu.combinations));
      for (const k of cu.cases) {
        const row = el("div", "offer-row");
        row.dataset["compareUser"] = k.name;
        const top = el("div", "offer-top");
        top.append(el("span", "offer-name", k.what));
        top.append(el("span", `offer-status${k.verdict === "current" ? " applied" : ""}`, k.verdict));
        const colour = s.visibility.colourBy === "contour";
        for (const d of k.draws) {
          top.append(
            d.shown
              ? button(`Hide ${d.component}`, `Stop drawing ${k.name} ${d.component} of the compare variant`, false, () => store.dispatch({ type: "showCompareUser", pick: null }), unlisten)
              : button(`Draw ${d.component}`, `Colour the window's mesh by ${k.name} ${d.component} of the compare variant (by node id: the two variants are one mesh)`, false, () => {
                  if (!colour) store.dispatch({ type: "setColourBy", by: "contour" });
                  store.dispatch({ type: "showCompareUser", pick: { name: k.name, component: d.component } });
                }, unlisten),
          );
        }
        row.append(top);
        if (k.verdict === "stale") row.append(el("div", "offer-reason", `stale (A6, against its own model and results): ${k.reason}`));
        else if (k.why) row.append(el("div", "offer-caption", `not drawn: ${k.why}`));
        box.append(row);
      }
      for (const t of cu.refused) box.append(el("div", "offer-reason", t));
      if (cu.read && cu.cases.length === 0 && cu.refused.length === 0) box.append(el("div", "offer-caption", "no user result"));
      parts.push(box);
    }
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
