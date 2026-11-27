// The contour colormap control of the Display card: a picker that shows each
// map as its ramp, the number of levels (bands of equal width, each one
// colour), and an editor for the user's own maps.
//
// - A custom map is its id (state/contour.ts `customCmapId`): the name and the
//   stops travel in the state, so the drawn model never depends on the
//   library below. The library is this profile's list of saved maps, kept in
//   localStorage as the scale's place is (ui/scaleChrome.ts): a convenience
//   of this machine, never part of a file.
// - The control is mounted once and kept: the Display card re-appends the same
//   element when it rebuilds, so an editor being typed in is not thrown away.
//   Inputs dispatch on `change` (a committed value), never on every keystroke.

import type { Store } from "../../state/store.ts";
import {
  cmapLabel,
  CMAP_NAMES,
  customCmapId,
  customCmapProblem,
  isBuiltinCmap,
  MAX_CMAP_NAME,
  MAX_LEVELS,
  MAX_STOPS,
  parseCustomCmap,
  rampCss,
  stopsOf,
  withLevels,
  type Stops,
} from "../../state/selectors.ts";
import type { ContourCmap } from "../../state/types.ts";
import { el, listen } from "../../ui/dom.ts";

/** The localStorage key of the saved maps. */
export const LIBRARY_KEY = "apegmsh.viewer.colormaps";

export interface SavedCmap {
  readonly name: string;
  readonly stops: Stops;
}

/** Where the library is kept: localStorage in the app, a stand-in in the tests. */
export interface LibraryStorage {
  read(): string | null;
  write(raw: string): void;
}

const browserStorage: LibraryStorage = {
  read: () => {
    try {
      return localStorage.getItem(LIBRARY_KEY);
    } catch {
      return null;
    }
  },
  write: (raw) => {
    try {
      localStorage.setItem(LIBRARY_KEY, raw);
    } catch {
      // A profile that cannot store keeps the maps for this session only.
    }
  },
};

/** The saved maps in `raw`: every entry that is a valid custom map, the first of a repeated name. */
export function parseLibrary(raw: string | null): SavedCmap[] {
  if (raw === null) return [];
  let v: unknown;
  try {
    v = JSON.parse(raw);
  } catch {
    return [];
  }
  if (!Array.isArray(v)) return [];
  const out: SavedCmap[] = [];
  for (const e of v) {
    const { name, stops } = (e ?? {}) as { name?: unknown; stops?: unknown };
    if (customCmapProblem(name, stops) !== null || out.some((x) => x.name === name)) continue;
    out.push({ name: name as string, stops: stops as Stops });
  }
  return out;
}

/** The library with `map` saved under its name (replacing one of that name, else appended). */
export function librarySave(lib: readonly SavedCmap[], map: SavedCmap): SavedCmap[] {
  const i = lib.findIndex((x) => x.name === map.name);
  return i < 0 ? [...lib, map] : lib.map((x, j) => (j === i ? map : x));
}

const hex2 = (c: number): string => Math.round(Math.min(1, Math.max(0, c)) * 255).toString(16).padStart(2, "0");
/** An sRGB colour (0..1) as an `<input type=color>` value. */
export const toHex = (r: number, g: number, b: number): string => `#${hex2(r)}${hex2(g)}${hex2(b)}`;
/** An `<input type=color>` value as sRGB (0..1); null when it is not `#rrggbb`. */
export function fromHex(v: string): [number, number, number] | null {
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(v);
  return m ? [parseInt(m[1]!, 16) / 255, parseInt(m[2]!, 16) / 255, parseInt(m[3]!, 16) / 255] : null;
}

/** The stops of `stops` read from the other end: position 1 - t, in order. */
export const reversedStops = (stops: Stops): Stops => [...stops].reverse().map(([t, r, g, b]) => [1 - t, r, g, b] as const);

/** A stop added in the middle of the widest gap, in the colour the map has there; the stops unchanged at MAX_STOPS. */
export function withStopAdded(stops: Stops): Stops {
  if (stops.length >= MAX_STOPS) return stops;
  let at = 1;
  for (let i = 1; i < stops.length; i++) if (stops[i]![0] - stops[i - 1]![0] > stops[at]![0] - stops[at - 1]![0]) at = i;
  const t = (stops[at - 1]![0] + stops[at]![0]) / 2;
  const [r, g, b] = rgbAt(stops, t);
  return [...stops.slice(0, at), [t, r, g, b] as const, ...stops.slice(at)];
}

/** The colour of `stops` at `t`, interpolated linearly as the ramp is. */
function rgbAt(stops: Stops, t: number): [number, number, number] {
  for (let i = 1; i < stops.length; i++) {
    const b = stops[i]!;
    if (t <= b[0]) {
      const a = stops[i - 1]!;
      const u = b[0] === a[0] ? 0 : (t - a[0]) / (b[0] - a[0]);
      return [a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2]), a[3] + u * (b[3] - a[3])];
    }
  }
  const l = stops[stops.length - 1]!;
  return [l[1], l[2], l[3]];
}

/** A name for a copy of `cmap` that the library does not hold yet. */
function freshName(base: string, lib: readonly SavedCmap[]): string {
  const root = base.slice(0, MAX_CMAP_NAME - 4);
  for (let i = 1; ; i++) {
    const name = i === 1 ? `${root} (edit)` : `${root} ${i}`;
    if (!lib.some((x) => x.name === name) && name.length <= MAX_CMAP_NAME) return name;
  }
}

interface Draft {
  name: string;
  stops: Stops;
}

export interface ColormapControl {
  readonly root: HTMLElement;
  dispose(): void;
}

export function mountColormaps(store: Store, storage: LibraryStorage = browserStorage): ColormapControl {
  let lib = parseLibrary(storage.read());
  let menuOpen = false;
  let draft: Draft | null = null;
  const unlisten: (() => void)[] = [];
  let menuUnlisten: (() => void)[] = [];
  let editorUnlisten: (() => void)[] = [];

  const root = el("div", "cmap-control");
  const row = el("div", "cmap-row");
  const current = el("button", "cmap-current");
  current.type = "button";
  current.title = "contour colormap: pick one, or make your own";
  const swatch = el("span", "cmap-swatch");
  const name = el("span", "cmap-name");
  current.append(swatch, name);
  const levels = el("input", "contour-levels");
  levels.type = "number";
  levels.min = "0";
  levels.max = String(MAX_LEVELS);
  levels.step = "1";
  levels.title = `levels: bands of equal width over the range, each one colour (2 to ${MAX_LEVELS}); 0 is a continuous ramp`;
  const levelsLabel = el("span", "cmap-levels-label", "levels");
  row.append(current, levelsLabel, levels);
  const menu = el("div", "cmap-menu");
  menu.hidden = true;
  const editor = el("div", "cmap-editor");
  editor.hidden = true;
  root.append(row, menu, editor);

  const dispatchCmap = (cmap: ContourCmap) => {
    if (store.get().contour.cmap !== cmap) store.dispatch({ type: "setContourCmap", cmap });
  };
  const saveLib = (next: SavedCmap[]) => {
    lib = next;
    storage.write(JSON.stringify(lib));
  };

  const option = (cmap: ContourCmap, label: string, on: boolean, saved: SavedCmap | null) => {
    const line = el("div", "cmap-option" + (on ? " on" : ""));
    const pick = el("button", "cmap-pick");
    pick.type = "button";
    pick.title = `colour the contour with ${label}`;
    const sw = el("span", "cmap-swatch");
    sw.style.backgroundImage = rampCss(withLevels(cmap, store.get().contour.levels));
    pick.append(sw, el("span", "cmap-name", label));
    menuUnlisten.push(
      listen(pick, "click", () => {
        dispatchCmap(cmap);
        menuOpen = false;
        renderMenu();
      }),
    );
    line.append(pick);
    if (saved) {
      const edit = el("button", "cmap-edit", "edit");
      edit.type = "button";
      edit.title = `edit ${saved.name}`;
      menuUnlisten.push(listen(edit, "click", () => openEditor({ name: saved.name, stops: saved.stops })));
      line.append(edit);
    }
    menu.append(line);
  };

  const renderMenu = () => {
    for (const u of menuUnlisten) u();
    menuUnlisten = [];
    menu.hidden = !menuOpen;
    current.className = "cmap-current" + (menuOpen ? " open" : "");
    if (!menuOpen) return menu.replaceChildren();
    menu.replaceChildren(el("div", "cmap-group", "Built-in"));
    const shown = store.get().contour.cmap;
    for (const n of CMAP_NAMES) option(n, n, shown === n, null);
    menu.append(el("div", "cmap-group", "Yours"));
    if (!lib.length) menu.append(el("div", "cmap-empty", "none saved yet"));
    for (const m of lib) {
      const id = customCmapId(m.name, m.stops);
      option(id, m.name, shown === id, m);
    }
    const shownCustom = isBuiltinCmap(shown) ? null : parseCustomCmap(shown);
    if (shownCustom && !lib.some((m) => customCmapId(m.name, m.stops) === shown)) option(shown, `${shownCustom.name} (unsaved)`, true, null);
    const fresh = el("button", "cmap-new", "new from current");
    fresh.type = "button";
    fresh.title = "start a map of your own from the one shown";
    menuUnlisten.push(
      listen(fresh, "click", () => {
        const c = store.get().contour.cmap;
        openEditor({ name: freshName(cmapLabel(c), lib), stops: stopsOf(c) });
      }),
    );
    menu.append(fresh);
  };

  const openEditor = (d: Draft) => {
    draft = { name: d.name, stops: d.stops.map((s) => [...s] as const) };
    menuOpen = false;
    renderMenu();
    renderEditor();
    applyDraft();
  };

  /** The draft drawn on the model as it is edited, while it is a valid map. */
  const applyDraft = () => {
    if (draft && customCmapProblem(draft.name, draft.stops) === null) dispatchCmap(customCmapId(draft.name, draft.stops));
  };

  const setDraft = (next: Draft) => {
    draft = next;
    renderEditor();
    applyDraft();
  };

  const renderEditor = () => {
    for (const u of editorUnlisten) u();
    editorUnlisten = [];
    editor.hidden = draft === null;
    if (!draft) return editor.replaceChildren();
    const d = draft;
    const problem = customCmapProblem(d.name, d.stops);
    const head = el("div", "cmap-editor-head");
    const nameIn = el("input", "cmap-name-input");
    nameIn.type = "text";
    nameIn.value = d.name;
    nameIn.maxLength = MAX_CMAP_NAME;
    nameIn.title = "the map's name";
    editorUnlisten.push(listen(nameIn, "change", () => setDraft({ ...d, name: nameIn.value.trim() })));
    head.append(nameIn);
    const preview = el("div", "cmap-preview");
    if (!problem) preview.style.backgroundImage = rampCss(withLevels(customCmapId(d.name, d.stops), store.get().contour.levels));
    const stops = el("div", "cmap-stops");
    d.stops.forEach(([t, r, g, b], i) => {
      const end = i === 0 || i === d.stops.length - 1;
      const line = el("div", "cmap-stop");
      const pos = el("input", "cmap-stop-at");
      pos.type = "number";
      pos.min = "0";
      pos.max = "1";
      pos.step = "0.01";
      pos.value = String(t);
      pos.disabled = end;
      pos.title = end ? "the ends sit at 0 and 1" : "where the stop sits, 0 (min) to 1 (max)";
      const colour = el("input", "cmap-stop-colour");
      colour.type = "color";
      colour.value = toHex(r, g, b);
      colour.title = "the stop's colour";
      const drop = el("button", "cmap-stop-drop", "remove");
      drop.type = "button";
      drop.disabled = end;
      drop.title = end ? "the end stops stay" : "remove this stop";
      editorUnlisten.push(
        listen(pos, "change", () => {
          const v = Number(pos.value);
          if (!Number.isFinite(v)) return;
          const next = d.stops.map((s, j) => (j === i ? ([Math.min(1, Math.max(0, v)), s[1], s[2], s[3]] as const) : s));
          setDraft({ ...d, stops: [...next].sort((x, y) => x[0] - y[0]) });
        }),
        listen(colour, "change", () => {
          const c = fromHex(colour.value);
          if (c) setDraft({ ...d, stops: d.stops.map((s, j) => (j === i ? ([s[0], ...c] as const) : s)) });
        }),
        listen(drop, "click", () => setDraft({ ...d, stops: d.stops.filter((_, j) => j !== i) })),
      );
      line.append(pos, colour, drop);
      stops.append(line);
    });
    const actions = el("div", "cmap-actions");
    const button = (cls: string, label: string, title: string, fn: () => void, disabled = false) => {
      const b = el("button", cls, label);
      b.type = "button";
      b.title = title;
      b.disabled = disabled;
      editorUnlisten.push(listen(b, "click", fn));
      actions.append(b);
    };
    button("cmap-add", "add stop", "add a stop in the widest gap", () => setDraft({ ...d, stops: withStopAdded(d.stops) }), d.stops.length >= MAX_STOPS);
    button("cmap-reverse", "reverse", "read the map from the other end", () => setDraft({ ...d, stops: reversedStops(d.stops) }));
    const saved = lib.some((m) => m.name === d.name);
    button("cmap-save", saved ? "save" : "save as new", `keep ${d.name} in this profile's maps`, () => {
      saveLib(librarySave(lib, { name: d.name, stops: d.stops }));
      renderEditor();
    }, problem !== null);
    button("cmap-delete", "delete", `remove ${d.name} from this profile's maps (the contour keeps it until you pick another)`, () => {
      saveLib(lib.filter((m) => m.name !== d.name));
      draft = null;
      renderEditor();
    }, !saved);
    button("cmap-close", "close", "close the editor (the contour keeps the map)", () => {
      draft = null;
      renderEditor();
    });
    editor.replaceChildren(head, preview, stops, actions);
    if (problem) editor.append(el("div", "legend-note refused-field cmap-problem", `Not drawn: ${problem}.`));
  };

  const sync = () => {
    const c = store.get().contour;
    swatch.style.backgroundImage = rampCss(withLevels(c.cmap, c.levels));
    name.textContent = cmapLabel(c.cmap);
    if (typeof document === "undefined" || document.activeElement !== levels) levels.value = String(c.levels);
    levelsLabel.className = "cmap-levels-label" + (c.levels > 0 ? " on" : "");
  };

  unlisten.push(
    listen(current, "click", () => {
      menuOpen = !menuOpen;
      renderMenu();
    }),
    listen(levels, "change", () => {
      const n = levels.value.trim() === "" ? 0 : Number(levels.value);
      // 1 band is no contour: it is read as continuous, as 0 is.
      const v = n === 1 ? 0 : n;
      if (Number.isInteger(v) && (v === 0 || (v >= 2 && v <= MAX_LEVELS))) {
        levels.setCustomValidity("");
        if (store.get().contour.levels !== v) store.dispatch({ type: "setContourLevels", levels: v });
      } else levels.setCustomValidity(`levels is 0 (continuous) or a whole number from 2 to ${MAX_LEVELS}`);
    }),
  );
  let last = { cmap: "", levels: -1 };
  const unsubscribe = store.subscribe((s) => {
    if (s.contour.cmap === last.cmap && s.contour.levels === last.levels) return;
    const levelsChanged = s.contour.levels !== last.levels;
    last = { cmap: s.contour.cmap, levels: s.contour.levels };
    sync();
    if (menuOpen) renderMenu();
    // The editor's preview shows the levels too.
    if (draft && levelsChanged) renderEditor();
  });
  sync();
  last = { cmap: store.get().contour.cmap, levels: store.get().contour.levels };

  return {
    root,
    dispose: () => {
      unsubscribe();
      for (const u of [...unlisten, ...menuUnlisten, ...editorUnlisten]) u();
    },
  };
}
