// Phase panel: the phase axis (ADR 0112 D7: geometry -> mesh -> results) as
// a row of buttons, shown when there is more than one phase to choose; on a
// results phase (V4b) it is also the time scrubber: the results stage
// picker, the step slider, the time readout, play/pause and step -1/+1. A
// click dispatches `setPhase` or `setResultStep`; the viewport draws the
// phase the state names, and the effects fetch the step (latest wins). The
// panel only dispatches: it keeps no results data, and play is a timer that
// dispatches the next step. (style.css is not this panel's file, so the
// scrubber's layout is inline, with the theme's variables only.)
//
// ADR 0130 D1 (VU3b): the results are listed by state family. Stages and
// Modes are the results file's stages by kind, and a click opens the first
// one; the stage picker then lists that family only. The Modes button counts
// the stale rows of the `.user.h5` modal table (A6, per row; VU7a), which the
// modal table panel shows beside the mode list. Spectra and User Combinations
// come from the `.user.h5`; their tooltip names every member, a spectrum case
// with its direction, scale, modes and results. A stale member or zone says so
// on its family's button (A6), and a refused zone or row too. Spectra are
// listed; a case is drawn through a User Combination over it. VU7b-2: a click
// on User Combinations shows the first combination as a contour (VU4's
// evaluator, in the app), and a picker in the bar chooses another; a
// combination is one state, so the bar shows no step scrubber for it.

import { combinationShownOf, inactiveElements, SCRUBBER_TOP_PX, scrubberRow, timelineOf, type Timeline } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { PhaseKey, StaleMark } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const label = (k: PhaseKey): string => (k.kind === "stage" ? `stage ${k.index + 1}` : k.kind === "combination" ? k.name : k.kind);

const same = (a: PhaseKey | null, b: PhaseKey): boolean =>
  a !== null &&
  a.kind === b.kind &&
  (a.kind !== "stage" || (b.kind === "stage" && a.index === b.index)) &&
  (a.kind !== "combination" || (b.kind === "combination" && a.name === b.name));

const baseName = (path: string): string => path.split(/[\\/]/).pop() ?? path;

/** ADR 0130 D1's state families, in the phase bar's order. */
export const FAMILIES = ["stages", "modes", "spectra", "combinations"] as const;
export type FamilyKey = (typeof FAMILIES)[number];
export const FAMILY_LABEL: Readonly<Record<FamilyKey, string>> = { stages: "Stages", modes: "Modes", spectra: "Spectra", combinations: "User Combinations" };

/** One family of the phase bar: its members, what of it is stale or refused, and whether a click draws it. */
export interface PhaseFamily {
  key: FamilyKey;
  /** the results stages of the family that record a step (Stages, Modes), in reader order; [] for the `.user.h5` families */
  stages: number[];
  /** how many members it lists */
  count: number;
  /** stale members: combinations, cases, or rows of the modal table under Modes */
  stale: number;
  /** every member is stale (a stale zone, or every row): the button says "stale", not a count */
  staleAll: boolean;
  /** refused rows, or 1 for a refused zone */
  refused: number;
  /** a click opens its first stage (Stages, Modes) or shows its first combination (User Combinations) */
  drawable: boolean;
  /** User Combinations: the combinations a click and the picker can show, in the family's order; [] otherwise */
  members: string[];
  /** the button's tooltip: what it lists, what is stale or refused and why, and why it is not drawn */
  title: string;
}

const staleLine = (what: string, m: StaleMark): string => `${what} is stale: ${m.reason || "the reader gave no reason"}`;

/** At most this many stale rows are named in a tooltip; the rest are counted. */
export const TITLE_ROWS_MAX = 8;

const fmt = (x: number): string => (Number.isNaN(x) ? "none" : String(Number(x.toPrecision(6))));

/** The family a results stage belongs to: a mode shape is a Mode, any other stage a Stage. */
export const familyOfStage = (s: State, stage: number): "stages" | "modes" => (s.results?.stages[stage]?.kind === "mode" ? "modes" : "stages");

/**
 * The state families on offer now: Stages and Modes while the results pair
 * with the model (they are phases of the axis), Spectra and User Combinations
 * while the set's `.user.h5` holds them. A family with no member is left out.
 */
export function familiesOf(s: State): PhaseFamily[] {
  const out: PhaseFamily[] = [];
  const info = s.user?.status === "ready" ? s.user.info : null;
  const file = s.user ? baseName(s.user.path) : "";
  const recorded = (key: "stages" | "modes", notes: string[], stale: number, staleAll: boolean, refused: number) => {
    const stages = s.phase.axis.flatMap((k) => (k.kind === "results" && familyOfStage(s, k.stage) === key ? [k.stage] : []));
    if (!stages.length || !s.results) return;
    const results = s.results;
    const name = (i: number) => results.stages[i]!.name || `${results.stages[i]!.id} (unnamed)`;
    // A stage that records no step cannot be shown: it is listed, and a click opens the first one that can.
    const empty = stages.filter((i) => results.stages[i]!.time.length === 0);
    const shown = stages.filter((i) => results.stages[i]!.time.length > 0);
    const lines = [`${FAMILY_LABEL[key]} of ${baseName(results.path)}: ${stages.map(name).join(", ")}`];
    if (empty.length) lines.push(`${empty.map(name).join(", ")} ${empty.length === 1 ? "records" : "record"} no step`);
    out.push({ key, stages: shown, count: stages.length, stale, staleAll, refused, drawable: shown.length > 0, members: [], title: [...lines, ...notes].join("\n") });
  };
  recorded("stages", [], 0, false, 0);
  const modeNotes: string[] = [];
  let modeStale = 0;
  let modeStaleAll = false;
  let modeRefused = 0;
  const m = info?.modes;
  if (m && !m.ok) {
    modeRefused = 1;
    modeNotes.push(`the modal table of ${file} is refused: ${m.reason}`);
    if (m.stale.stale) modeNotes.push(staleLine(`the modal table of ${file}`, m.stale));
  } else if (m) {
    // A6 per row: each row's lambda against the results file's LAMBDA of that mode.
    const bad = m.rows.filter((r) => r.verdict.stale);
    modeStale = bad.length;
    modeStaleAll = bad.length === m.rows.length;
    const count = `${m.rows.length} ${m.rows.length === 1 ? "row" : "rows"}`;
    modeNotes.push(`the modal table of ${file}: ${count}, ${bad.length ? `${bad.length} stale` : "every row current"}; the modal table panel shows it beside a mode`);
    if (m.stale.stale) modeNotes.push(staleLine(`the modal table of ${file}`, m.stale));
    else {
      for (const r of bad.slice(0, TITLE_ROWS_MAX)) modeNotes.push(`mode ${r.mode} of ${r.host} is stale: ${r.verdict.reason || "the reader gave no reason"}`);
      if (bad.length > TITLE_ROWS_MAX) modeNotes.push(`and ${bad.length - TITLE_ROWS_MAX} more stale rows`);
    }
  }
  recorded("modes", modeNotes, modeStale, modeStaleAll, modeRefused);
  const sp = info?.spectra;
  if (sp && !sp.ok) {
    const lines = [`Spectra: /spectra of ${file} is refused: ${sp.reason}`];
    if (sp.stale.stale) lines.push(staleLine("/spectra", sp.stale));
    out.push({ key: "spectra", stages: [], count: 0, stale: 0, staleAll: false, refused: 1, drawable: false, members: [], title: lines.join("\n") });
  } else if (sp) {
    const lines = ["Spectra, listed; a case is drawn through a User Combination over it:"];
    for (const x of sp.items) {
      const modes = `${x.modes.length} ${x.modes.length === 1 ? "mode" : "modes"}`;
      const damping = x.damping === null ? "damping per mode" : `damping ${fmt(x.damping)}`;
      lines.push(`${x.name}: direction ${x.direction}, scale ${fmt(x.scale)}, ${modes}, ${damping}; ${x.results.join(", ")}`);
    }
    for (const x of sp.refused) lines.push(`${x.name || "(no name)"}: refused, ${x.reason}: ${x.detail}`);
    if (sp.stale.stale) lines.push(staleLine("/spectra", sp.stale));
    const n = sp.items.length;
    out.push({ key: "spectra", stages: [], count: n, stale: sp.stale.stale ? n : 0, staleAll: sp.stale.stale, refused: sp.refused.length, drawable: false, members: [], title: lines.join("\n") });
  }
  const c = info?.combinations;
  if (c && !c.ok) {
    out.push({ key: "combinations", stages: [], count: 0, stale: 0, staleAll: false, refused: 1, drawable: false, members: [], title: `User Combinations: /combinations of ${file} is refused: ${c.reason}` });
  } else if (c) {
    // VU7b-2: a standing combination is shown as a contour over the model, so it needs the mesh.
    const members = s.mesh ? c.items.map((x) => x.name) : [];
    const lines = [
      members.length
        ? "User Combinations; a click shows the first as a contour, and the picker in the bar chooses another:"
        : `User Combinations, listed; none is shown (${s.mesh ? "none of them stands" : "no mesh is loaded"}):`,
    ];
    for (const x of c.items) {
      lines.push(`${x.name}: ${x.rule}, ${x.terms} ${x.terms === 1 ? "term" : "terms"}${x.origin === "model" ? ", from model.h5" : ""}${x.stale ? " (stale: a source of it is stale)" : ""}`);
    }
    for (const x of c.refused) lines.push(`${x.name || "(no name)"}: refused, ${x.reason}: ${x.detail}`);
    const stale = c.items.filter((x) => x.stale).length;
    out.push({ key: "combinations", stages: [], count: c.items.length, stale, staleAll: false, refused: c.refused.length, drawable: members.length > 0, members, title: lines.join("\n") });
  }
  return out;
}

/** What the phase bar says of the `.user.h5` itself: being read, refused, or read with warnings; null when there is nothing to say. */
export function userBadgeOf(s: State): { text: string; title: string; refused: boolean } | null {
  const u = s.user;
  if (!u) return null;
  // Its families are left out while it is read: nothing from before a change is shown as current.
  if (u.status === "reading") return { text: "reading user file", title: `${baseName(u.path)} is being read`, refused: false };
  if (u.status === "failed") return { text: "user file refused", title: `${baseName(u.path)}: ${u.error}`, refused: true };
  const n = u.status === "ready" ? u.info.warnings.length : 0;
  if (u.status === "ready" && n) return { text: `user file: ${n} ${n === 1 ? "warning" : "warnings"}`, title: u.info.warnings.join("\n"), refused: false };
  return null;
}

/** A family button's text after its label: the count, then what is stale and refused. */
export function familyMarks(f: PhaseFamily): { count: string; stale: string; refused: string } {
  return {
    count: ` ${f.count}`,
    stale: f.staleAll ? " · stale" : f.stale ? ` · ${f.stale} stale` : "",
    // Under Modes a refusal is the modal table's, a zone: never a count of modes.
    refused: f.refused ? (f.key === "modes" || f.count === 0 ? " · refused" : ` · ${f.refused} refused`) : "",
  };
}

/**
 * The bar's cap on the scrubber row is `#header`'s (style.css): `100vw - 684px`,
 * short of the ViewCube and the right rail; `100vw - 216px` at 1000 px wide or
 * less, or with the rail collapsed, where the ViewCube moves to the right edge.
 * The capped bar wraps, so a control is never under the ViewCube (#2737 review).
 */
export const BAR_RIGHT_CLEAR_PX = 684;
export const BAR_RIGHT_CLEAR_NARROW_PX = 216;
export const NARROW_MAX_PX = 1000;

/** The pixels the capped bar leaves clear on the right: style.css `#header`'s max-width rule, as a function. */
export function barClearPx(viewportWidth: number, railCollapsed: boolean): number {
  return viewportWidth <= NARROW_MAX_PX || railCollapsed ? BAR_RIGHT_CLEAR_NARROW_PX : BAR_RIGHT_CLEAR_PX;
}

/**
 * The custom property on the document root that holds the scrubber bar's drawn
 * height (px), absent off the scrubber row. style.css moves `#left-rail` below
 * a bar that wrapped, and keeps its 102 px top under a one-line bar.
 */
export const BAR_HEIGHT_VAR = "--phase-bar-h";

/** The scrubber's least width on a line: the picker, the three step buttons, the slider at its least and a little readout. It takes a line of its own below that. */
export const SCRUBBER_MIN_PX = 340;

/** The readout's widest extent; longer text ends in an ellipsis, with the full text as its tooltip. */
export const READOUT_MAX_PX = 240;

/** Milliseconds between steps while playing (10 steps/s; the D5 trigger is 30 steps/s, so playback never outruns it). */
export const PLAY_MS = 100;

const fmtTime = (t: number): string => (Number.isInteger(t) ? String(t) : Math.abs(t) >= 1e-3 && Math.abs(t) < 1e4 ? t.toPrecision(4).replace(/\.?0+$/, "") : t.toExponential(3));

/** The scrubber's readout: `step 3/5 · t = 0.6`, the elements the stage does not have, and a fetch that is not done. */
export function readoutOf(t: Timeline, inactive: number | null): string {
  const time = t.kind === "mode" ? "mode shape" : `t = ${fmtTime(t.time)}`;
  const act = inactive === null ? "" : ` · ${inactive} inactive`;
  const fetch = t.fetch === "ready" ? "" : ` · step ${t.fetch}`;
  return `step ${t.step + 1}/${t.steps} · ${time}${act}${fetch}`;
}

/** A button or picker of the capped bar keeps its width and its one line; only the slider and the readout give way. */
function rigid(e: HTMLElement): void {
  e.style.flex = "0 0 auto";
  e.style.whiteSpace = "nowrap";
}

export function mountPhase(store: Store): () => void {
  const bar = byId("phase");
  let unlisten: (() => void)[] = [];
  let timer: ReturnType<typeof setInterval> | null = null;
  const stop = () => {
    if (timer !== null) clearInterval(timer);
    timer = null;
  };
  const step = (delta: number): boolean => {
    const t = timelineOf(store.get());
    if (!t) return false;
    const next = t.step + delta;
    if (next < 0 || next >= t.steps) return false;
    store.dispatch({ type: "setResultStep", step: next });
    return true;
  };
  const render = (s: State, prev: State | null) => {
    if (prev && s.phase === prev.phase && s.resultStep === prev.resultStep && s.results === prev.results && s.user === prev.user && s.mesh === prev.mesh) return;
    for (const u of unlisten) u();
    unlisten = [];
    const t = timelineOf(s);
    if (!t) stop();
    const families = familiesOf(s);
    const badge = userBadgeOf(s);
    // The `.user.h5` families are worth the bar even beside a single phase.
    const userSays = badge !== null || families.some((f) => !f.drawable);
    bar.hidden = s.phase.axis.length < 2 && !(s.phase.axis.length === 1 && userSays);
    // With the scrubber, the bar takes its own row under the header (style.css centres it on the header row).
    // Capped as #header is, and wrapping: the scrubber takes a line of its own rather than run under the ViewCube.
    const wide = typeof window === "undefined" ? Number.POSITIVE_INFINITY : window.innerWidth;
    const collapsed = typeof document !== "undefined" && document.body?.dataset?.["railCollapsed"] === "true";
    bar.style.cssText = scrubberRow(s)
      ? `top: ${SCRUBBER_TOP_PX}px; left: 16px; transform: none; max-width: calc(100vw - ${barClearPx(wide, collapsed)}px); flex-wrap: wrap;`
      : "";
    // One button per non-results phase, then one per state family (ADR 0130 D1); the stage picker chooses within the family shown.
    const buttons: HTMLElement[] = s.phase.axis
      .filter((k) => k.kind !== "results")
      .map((k) => {
        const b = el("button", same(s.phase.at, k) ? "phase active" : "phase", label(k));
        unlisten.push(listen(b, "click", () => store.dispatch({ type: "setPhase", at: k })));
        return b;
      });
    const at = s.phase.at;
    const shown = at?.kind === "results" ? familyOfStage(s, at.stage) : at?.kind === "combination" ? "combinations" : null;
    for (const f of families) buttons.push(familyButton(f, shown === f.key));
    if (badge) {
      const b = el("span", "phase user-badge", badge.text);
      b.title = badge.title;
      b.style.cssText = `cursor: default; color: var(${badge.refused ? "--error" : "--interp-text"});`;
      buttons.push(b);
    }
    for (const b of buttons) rigid(b);
    const combos = families.find((f) => f.key === "combinations");
    bar.replaceChildren(...buttons, ...(t ? scrubber(t, s) : at?.kind === "combination" && combos ? combinationPicker(at.name, combos, s) : []));
    publishHeight();
  };
  /** The bar's drawn height for style.css (`#left-rail` clears a wrapped bar), or none off the scrubber row. */
  const publishHeight = () => {
    const root = typeof document === "undefined" ? undefined : document.documentElement?.style;
    if (typeof root?.setProperty !== "function") return;
    if (bar.hidden || !scrubberRow(store.get())) root.removeProperty(BAR_HEIGHT_VAR);
    else root.setProperty(BAR_HEIGHT_VAR, `${Math.ceil(bar.getBoundingClientRect().height)}px`);
  };
  const familyButton = (f: PhaseFamily, on: boolean): HTMLElement => {
    const b = el("button", on ? "phase family active" : "phase family", FAMILY_LABEL[f.key]);
    b.dataset["family"] = f.key;
    b.title = f.title;
    const m = familyMarks(f);
    if (m.count) b.append(el("span", "family-count", m.count));
    const mark = (cls: string, text: string, colour: string) => {
      if (!text) return;
      const x = el("span", cls, text);
      // On the active (accent) button the accent's ink keeps the mark legible.
      x.style.cssText = `color: var(${on ? "--accent-ink" : colour}); font-weight: 700;`;
      b.append(x);
    };
    mark("family-stale", m.stale, "--interp-text");
    mark("family-refused", m.refused, "--error");
    if (!f.drawable) {
      b.disabled = true;
      b.style.cssText = "cursor: default;";
      return b;
    }
    unlisten.push(
      listen(b, "click", () => {
        if (on) return;
        stop();
        if (f.key === "combinations") return show(f.members[0]!);
        store.dispatch({ type: "setPhase", at: { kind: "results", stage: f.stages[0]!, step: 0 } });
      }),
    );
    return b;
  };
  /** VU7b-2: show User Combination `name`, as a contour (it is drawn as nothing else). */
  const show = (name: string) => {
    store.dispatch({ type: "setPhase", at: { kind: "combination", name } });
    if (store.get().visibility.colourBy !== "contour") store.dispatch({ type: "setColourBy", by: "contour" });
  };
  /** VU7b-2: the combination picker, and what the combination shown is: its rule, and whether it can be a deformed shape. */
  const combinationPicker = (name: string, f: PhaseFamily, s: State): HTMLElement[] => {
    const box = el("span", "combination");
    box.style.cssText = "display: flex; flex: 0 1 auto; align-items: center; gap: 6px; padding: 0 6px; font: 600 12px var(--font); color: var(--text);";
    const pick = el("select", "combination-pick");
    pick.style.cssText = "font: 600 12px var(--font); color: var(--text); background: var(--bg-1); border: 1px solid var(--line); border-radius: 5px;";
    pick.title = "User Combination shown";
    for (const m of f.members) {
      const o = el("option", undefined, m);
      o.value = m;
      o.selected = m === name;
      pick.append(o);
    }
    unlisten.push(listen(pick, "change", () => show(pick.value)));
    const c = combinationShownOf(s);
    const readout = el("span", "combination-readout", c ? `${c.rule} · ${c.signed === null ? "sign unknown" : c.signed ? "signed" : "unsigned: contour only"}${c.stale ? " · stale" : ""}` : "");
    readout.title = c?.why ?? readout.textContent ?? "";
    readout.style.cssText = `display: inline-block; flex: 0 1 auto; min-width: 0; max-width: ${READOUT_MAX_PX}px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); font: 600 12px var(--mono);`;
    rigid(pick);
    box.append(pick, readout);
    return [box];
  };
  const scrubber = (t: Timeline, s: State): HTMLElement[] => {
    const box = el("span", "scrubber");
    // It shrinks before the bar overflows its cap: the slider to its least width, then the readout to an ellipsis.
    box.style.cssText = `display: flex; flex: 1 1 0; min-width: ${SCRUBBER_MIN_PX}px; align-items: center; gap: 6px; padding: 0 6px; font: 600 12px var(--font); color: var(--text);`;
    const pick = el("select", "results-stage");
    pick.style.cssText = "font: 600 12px var(--font); color: var(--text); background: var(--bg-1); border: 1px solid var(--line); border-radius: 5px;";
    // The family shown: Stages list the static and transient stages, Modes the mode shapes.
    const family = familyOfStage(s, t.stage);
    pick.title = `results stage (${FAMILY_LABEL[family]})`;
    for (const st of t.stages) {
      if (familyOfStage(s, st.index) !== family) continue;
      const o = el("option", undefined, st.label);
      o.value = String(st.index);
      o.selected = st.index === t.stage;
      pick.append(o);
    }
    unlisten.push(listen(pick, "change", () => {
      stop();
      store.dispatch({ type: "setPhase", at: { kind: "results", stage: Number(pick.value), step: 0 } });
    }));
    const back = el("button", "phase step-back", "◀");
    back.title = "step -1";
    back.disabled = t.step === 0;
    const play = el("button", "phase play", timer !== null ? "❚❚" : "▶");
    play.title = timer !== null ? "pause" : "play";
    play.disabled = t.steps < 2;
    const fwd = el("button", "phase step-forward", "▶ |");
    fwd.title = "step +1";
    fwd.disabled = t.step === t.steps - 1;
    const slider = el("input", "results-step");
    slider.type = "range";
    slider.min = "0";
    slider.max = String(t.steps - 1);
    slider.value = String(t.step);
    slider.disabled = t.steps < 2;
    slider.style.cssText = "width: 130px; flex: 0 1 130px; min-width: 48px; accent-color: var(--accent);";
    const inactive = inactiveElements(s);
    const readout = el("span", "results-readout", readoutOf(t, inactive ? inactive.inactive : null));
    // Bounded: a long fetch error ends in an ellipsis; the full text is the tooltip.
    readout.title = readout.textContent ?? "";
    readout.style.cssText = `display: inline-block; flex: 0 1 auto; min-width: 0; max-width: ${READOUT_MAX_PX}px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); font: 600 12px var(--mono);`;
    unlisten.push(
      listen(back, "click", () => {
        stop();
        step(-1);
      }),
      listen(fwd, "click", () => {
        stop();
        step(1);
      }),
      listen(slider, "input", () => {
        stop();
        store.dispatch({ type: "setResultStep", step: Number(slider.value) });
      }),
      listen(play, "click", () => {
        if (timer !== null) {
          stop();
          render(store.get(), null);
          return;
        }
        // From the last step, play starts over.
        const now = timelineOf(store.get());
        if (now && now.step === now.steps - 1) store.dispatch({ type: "setResultStep", step: 0 });
        timer = setInterval(() => {
          if (!step(1)) {
            stop();
            render(store.get(), null);
          }
        }, PLAY_MS);
        render(store.get(), null);
      }),
    );
    for (const b of [pick, back, play, fwd]) rigid(b);
    box.append(pick, back, play, fwd, slider, readout);
    return [box];
  };
  render(store.get(), null);
  const unsubscribe = store.subscribe(render);
  // The cap follows the window's width and the rail, as #header's does (only a real window has them; the Node tests mount without).
  const recap = () => render(store.get(), null);
  const offWindow: (() => void)[] = [];
  if (typeof window !== "undefined") {
    window.addEventListener("resize", recap);
    offWindow.push(() => window.removeEventListener("resize", recap));
    if (typeof MutationObserver !== "undefined") {
      const rail = new MutationObserver(recap);
      rail.observe(document.body, { attributes: true, attributeFilter: ["data-rail-collapsed"] });
      offWindow.push(() => rail.disconnect());
    }
    // The height also moves without a render of this panel (a font arriving): it is published whenever the bar resizes.
    if (typeof ResizeObserver !== "undefined") {
      const sized = new ResizeObserver(publishHeight);
      sized.observe(bar);
      offWindow.push(() => sized.disconnect());
    }
    offWindow.push(() => document.documentElement.style.removeProperty(BAR_HEIGHT_VAR));
  }
  return () => {
    stop();
    unsubscribe();
    for (const u of unlisten) u();
    for (const u of offWindow) u();
  };
}
