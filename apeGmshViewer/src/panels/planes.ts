// Section planes panel (V4e1, #1834; ADR 0083 Part 4, the Qt
// viewers/ui/_clip_planes_panel.py layout): "Add plane" along X, Y, Z or the
// current view; one row per plane with its active box, orientation chips
// (X / Y / Z / Custom, Custom showing the three normal fields), flip and
// delete; an offset slider over the reference mesh's extent along the normal
// beside an unclamped numeric field; and the footer's "Apply cuts" toggle
// with the slice-2 and slice-3 placeholders, disabled. It dispatches events
// and reads selectors; what it needs of the viewport (the slider's box, the
// view direction) app.ts hands it as functions.

import { activeCount, MAX_ACTIVE_PLANES, offsetRange, orientationOf, type Box, type Vec3 } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { ClipPlane } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

/** What the panel needs of the viewport. */
export interface PlanesHost {
  /** the box the offset slider maps: the reference (undeformed) mesh; null with no mesh */
  box: () => Box | null;
  /** the direction the camera looks along, the "View" plane's normal */
  viewNormal: () => readonly [number, number, number];
}

/** One plane's row, as plain data (tested in Node; the DOM below shows it). */
export interface PlaneRow {
  id: string;
  label: string;
  active: boolean;
  flipped: boolean;
  orientation: "x" | "y" | "z" | "custom";
  normal: Vec3;
  /** the offset itself (the numeric field; never clamped) */
  offset: number;
  /** the slider's range and position (clamped to the range); null without a box */
  slider: { min: number; max: number; step: number; value: number } | null;
  /** the offset lies outside the slider's range (the deformation or a typed value put it there) */
  outside: boolean;
}

/** A field number: six significant digits, no trailing noise. */
export const fmt = (v: number): string => String(Number(v.toPrecision(6)));

/** The rows the panel draws: one per plane, in creation order. */
export function planeRows(planes: readonly ClipPlane[], box: Box | null): PlaneRow[] {
  return planes.map((p) => {
    const r = box ? offsetRange(p.normal, box) : null;
    const slider = r ? { min: r.min, max: r.max, step: (r.max - r.min) / 1000, value: Math.min(r.max, Math.max(r.min, p.offset)) } : null;
    return {
      id: p.id,
      label: `Plane ${p.id.replace(/^plane-/, "")}`,
      active: p.active,
      flipped: p.flipped,
      orientation: orientationOf(p.normal),
      normal: p.normal,
      offset: p.offset,
      slider,
      outside: r !== null && (p.offset < r.min || p.offset > r.max),
    };
  });
}

const AXES: readonly { key: "x" | "y" | "z"; label: string; normal: Vec3 }[] = [
  { key: "x", label: "X", normal: [1, 0, 0] },
  { key: "y", label: "Y", normal: [0, 1, 0] },
  { key: "z", label: "Z", normal: [0, 0, 1] },
];

export function mountPlanes(store: Store, host: PlanesHost): () => void {
  const panel = byId("planes");
  let unlisten: (() => void)[] = [];
  /** Planes whose Custom fields the user opened on an axis normal (view state, not a model fact). */
  const expanded = new Set<string>();
  let lastShape = "";
  let lastBox: Box | null = null;
  /** The offset inputs of each drawn row, updated in place while a slider drags (a rebuild would drop the drag). */
  let inputs = new Map<string, { slider: HTMLInputElement | null; field: HTMLInputElement }>();

  const shapeOf = (s: State, box: Box | null): string =>
    JSON.stringify([
      s.mesh === null,
      s.clip.applyCuts,
      s.clip.notice,
      s.clip.planes.map((p) => [p.id, p.active, p.flipped, p.normal, expanded.has(p.id)]),
      box,
    ]);

  const render = (s: State) => {
    const box = s.mesh ? host.box() : null;
    const shape = shapeOf(s, box);
    if (shape === lastShape && box === lastBox) {
      // Only offsets changed: move the inputs, never the one the user is editing.
      for (const r of planeRows(s.clip.planes, box)) {
        const i = inputs.get(r.id);
        if (!i) continue;
        if (i.slider && r.slider && document.activeElement !== i.slider) i.slider.value = String(r.slider.value);
        if (document.activeElement !== i.field) i.field.value = fmt(r.offset);
        i.field.classList.toggle("outside", r.outside);
      }
      return;
    }
    lastShape = shape;
    lastBox = box;
    for (const u of unlisten) u();
    unlisten = [];
    inputs = new Map();
    panel.hidden = s.mesh === null;
    if (panel.hidden) return panel.replaceChildren();

    const n = activeCount(s.clip);
    const head = el("h3", undefined, "Section planes");
    head.append(el("span", "planes-count", s.clip.applyCuts ? `${n} / ${MAX_ACTIVE_PLANES} cutting` : "cuts off"));
    const add = el("div", "legend-mode planes-add");
    add.append(el("span", "planes-label", "Add plane"));
    for (const a of AXES) {
      const b = el("button", undefined, a.label);
      b.title = `a plane normal to ${a.label}, through the model's centre`;
      on(b, "click", () => store.dispatch({ type: "addClipPlane", normal: a.normal }));
      add.append(b);
    }
    const view = el("button", undefined, "View");
    view.title = "a plane normal to the current view, through the model's centre: it cuts away the half nearer the camera";
    on(view, "click", () => store.dispatch({ type: "addClipPlane", normal: [...host.viewNormal()] }));
    add.append(view);
    panel.replaceChildren(head, add);
    // The notice first: with six rows the panel scrolls, and a refusal must be seen where it was asked for.
    if (s.clip.notice) panel.append(el("div", "planes-notice", s.clip.notice));

    for (const r of planeRows(s.clip.planes, box)) panel.append(row(r));

    if (!s.clip.planes.length) panel.append(el("div", "legend-note", "No section planes. A plane keeps the side its normal points to; flip keeps the other."));

    const foot = el("div", "legend-mode planes-foot");
    const cuts = el("button", s.clip.applyCuts ? "on" : "", "Apply cuts");
    cuts.title = "cut with every active plane, or show the whole model and keep the planes";
    on(cuts, "click", () => store.dispatch({ type: "setApplyCuts", on: !store.get().clip.applyCuts }));
    const gizmos = el("button", undefined, "Show plane gizmos");
    gizmos.disabled = true;
    gizmos.title = "not yet available: the plane gizmos arrive with V4e2";
    const face = el("button", undefined, "Contour on cut face");
    face.disabled = true;
    face.title = "not yet available: cut-face caps arrive with V4e3; a solid is cut hollow until then";
    foot.append(cuts, gizmos, face);
    panel.append(foot);
  };

  const row = (r: PlaneRow): HTMLElement => {
    const box = el("div", "plane-row" + (r.active ? "" : " inactive"));
    const head = el("div", "plane-head");
    const label = el("label", "plane-label");
    const active = el("input");
    active.type = "checkbox";
    active.checked = r.active;
    active.title = "cutting";
    const id = r.id;
    on(active, "change", () => store.dispatch({ type: "setClipActive", id, active: active.checked }));
    label.append(active, el("span", undefined, r.label));
    const chips = el("div", "legend-mode plane-chips");
    const custom = r.orientation === "custom" || expanded.has(id);
    for (const a of AXES) {
      const b = el("button", r.orientation === a.key && !custom ? "on" : "", a.label);
      on(b, "click", () => {
        expanded.delete(id);
        store.dispatch({ type: "setClipNormal", id, normal: a.normal });
      });
      chips.append(b);
    }
    const c = el("button", custom ? "on" : "", "Custom");
    on(c, "click", () => {
      if (expanded.has(id)) expanded.delete(id);
      else expanded.add(id);
      render(store.get());
    });
    chips.append(c);
    const flip = el("button", "plane-flip" + (r.flipped ? " on" : ""), "flip");
    flip.title = "keep the other side";
    on(flip, "click", () => store.dispatch({ type: "setClipFlipped", id, flipped: !r.flipped }));
    const del = el("button", "plane-delete", "×");
    del.title = "delete the plane";
    on(del, "click", () => {
      expanded.delete(id);
      store.dispatch({ type: "removeClipPlane", id });
    });
    head.append(label, chips, flip, del);
    box.append(head);

    if (custom) {
      const normal = el("div", "plane-normal");
      const fields = (["x", "y", "z"] as const).map((k, i) => {
        const f = el("input", "plane-number");
        f.value = fmt(r.normal[i]!);
        f.title = `normal ${k} (normalised on entry)`;
        normal.append(el("span", "planes-label", `n${k}`), f);
        return f;
      });
      for (const f of fields) on(f, "change", () => store.dispatch({ type: "setClipNormal", id, normal: fields.map((x) => Number(x.value.trim() || NaN)) }));
      box.append(normal);
    }

    const off = el("div", "plane-offset");
    let slider: HTMLInputElement | null = null;
    if (r.slider) {
      const sl = el("input", "plane-slider");
      sl.type = "range";
      sl.min = String(r.slider.min);
      sl.max = String(r.slider.max);
      sl.step = String(r.slider.step);
      sl.value = String(r.slider.value);
      sl.title = "the offset across the undeformed model";
      on(sl, "input", () => store.dispatch({ type: "setClipOffset", id, offset: Number(sl.value) }));
      off.append(sl);
      slider = sl;
    }
    const field = el("input", "plane-number" + (r.outside ? " outside" : ""));
    field.value = fmt(r.offset);
    field.title = "the offset along the normal (not limited to the model)";
    on(field, "change", () => store.dispatch({ type: "setClipOffset", id, offset: Number(field.value.trim() || NaN) }));
    off.append(field);
    box.append(off);
    inputs.set(id, { slider, field });
    return box;
  };

  const on = (target: HTMLElement, type: "click" | "input" | "change", fn: () => void) => unlisten.push(listen(target, type, fn));

  render(store.get());
  const off = store.subscribe((s) => render(s));
  return () => {
    off();
    for (const u of unlisten) u();
  };
}
