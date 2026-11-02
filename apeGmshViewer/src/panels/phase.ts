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
// one; the stage picker then lists that family only. Spectra and User
// Combinations come from the set's `.user.h5` and are listed, not drawn (the
// Spectra family is read by VU7, combinations are evaluated by VU4): their
// buttons are disabled and their tooltip names every member. A stale member
// or zone says so on its family's button (A6), and a refused zone or row too.

import { inactiveElements, SCRUBBER_TOP_PX, scrubberRow, timelineOf, type Timeline } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { PhaseKey, StaleMark } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const label = (k: PhaseKey): string => (k.kind === "stage" ? `stage ${k.index + 1}` : k.kind);

const same = (a: PhaseKey | null, b: PhaseKey): boolean =>
  a !== null && a.kind === b.kind && (a.kind !== "stage" || (b.kind === "stage" && a.index === b.index));

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
  /** how many members it lists; null when its zone is listed by name only (Spectra) */
  count: number | null;
  /** stale members (User Combinations), or 1 for a stale zone (Spectra, the modal table under Modes) */
  stale: number;
  /** refused rows, or 1 for a refused zone */
  refused: number;
  /**
   * Modes only: the `.user.h5` modal table reads, but its rows are not judged
   * against the `.ladruno`'s eigenvalues (A6's per-row rule is VU7's), so it is
   * never shown as current. Python stales every row whose lambda moved.
   */
  unjudged: boolean;
  /** a click opens its first stage */
  drawable: boolean;
  /** the button's tooltip: what it lists, what is stale or refused and why, and why it is not drawn */
  title: string;
}

const staleLine = (what: string, m: StaleMark): string => `${what} is stale: ${m.reason || "the reader gave no reason"}`;

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
  const recorded = (key: "stages" | "modes", notes: string[], stale: number, refused: number, unjudged = false) => {
    const stages = s.phase.axis.flatMap((k) => (k.kind === "results" && familyOfStage(s, k.stage) === key ? [k.stage] : []));
    if (!stages.length || !s.results) return;
    const results = s.results;
    const name = (i: number) => results.stages[i]!.name || `${results.stages[i]!.id} (unnamed)`;
    // A stage that records no step cannot be shown: it is listed, and a click opens the first one that can.
    const empty = stages.filter((i) => results.stages[i]!.time.length === 0);
    const shown = stages.filter((i) => results.stages[i]!.time.length > 0);
    const lines = [`${FAMILY_LABEL[key]} of ${baseName(results.path)}: ${stages.map(name).join(", ")}`];
    if (empty.length) lines.push(`${empty.map(name).join(", ")} ${empty.length === 1 ? "records" : "record"} no step`);
    out.push({ key, stages: shown, count: stages.length, stale, refused, unjudged, drawable: shown.length > 0, title: [...lines, ...notes].join("\n") });
  };
  recorded("stages", [], 0, 0);
  const modeNotes: string[] = [];
  let modeStale = 0;
  let modeRefused = 0;
  let modeUnjudged = false;
  if (info?.modes) {
    if (info.modes.ok) modeNotes.push(`the modal table of ${file} has ${info.modes.rows} rows; this app does not show it yet`);
    else {
      modeRefused = 1;
      modeNotes.push(`the modal table of ${file} is refused: ${info.modes.reason}`);
    }
    if (info.modes.stale.stale) {
      modeStale = 1;
      modeNotes.push(staleLine(`the modal table of ${file}`, info.modes.stale));
    } else if (info.modes.ok) {
      modeUnjudged = true;
      const run = s.results ? baseName(s.results.path) : "the results file";
      modeNotes.push(`its rows are not judged against the eigenvalues of ${run} yet, so the table is not shown as current`);
    }
  }
  recorded("modes", modeNotes, modeStale, modeRefused, modeUnjudged);
  if (info?.spectra) {
    const lines = [`Spectra: ${file} holds /spectra; this app lists it and does not read or draw it yet`];
    if (info.spectra.stale.stale) lines.push(staleLine("/spectra", info.spectra.stale));
    out.push({ key: "spectra", stages: [], count: null, stale: info.spectra.stale.stale ? 1 : 0, refused: 0, unjudged: false, drawable: false, title: lines.join("\n") });
  }
  const c = info?.combinations;
  if (c && !c.ok) {
    out.push({ key: "combinations", stages: [], count: 0, stale: 0, refused: 1, unjudged: false, drawable: false, title: `User Combinations: /combinations of ${file} is refused: ${c.reason}` });
  } else if (c) {
    const lines = ["User Combinations, listed; this app does not evaluate them yet:"];
    for (const x of c.items) {
      lines.push(`${x.name}: ${x.rule}, ${x.terms} ${x.terms === 1 ? "term" : "terms"}${x.origin === "model" ? ", from model.h5" : ""}${x.stale ? " (stale: a source of it is stale)" : ""}`);
    }
    for (const x of c.refused) lines.push(`${x.name || "(no name)"}: refused, ${x.reason}: ${x.detail}`);
    const stale = c.items.filter((x) => x.stale).length;
    out.push({ key: "combinations", stages: [], count: c.items.length, stale, refused: c.refused.length, unjudged: false, drawable: false, title: lines.join("\n") });
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

/** A family button's text after its label: the count, then what is stale (or not judged) and refused. */
export function familyMarks(f: PhaseFamily): { count: string; stale: string; refused: string } {
  const whole = f.count === null || f.key === "modes";
  return {
    count: f.count === null ? "" : ` ${f.count}`,
    stale: f.stale ? (whole ? " · stale" : ` · ${f.stale} stale`) : f.unjudged ? " · rows not judged" : "",
    refused: f.refused ? (whole || f.count === 0 ? " · refused" : ` · ${f.refused} refused`) : "",
  };
}

/**
 * The bar's cap on the scrubber row: `#header`'s (style.css, `100vw - 684px`),
 * which stops short of the ViewCube and the right rail, so the readout is never
 * under the ViewCube (#2737 review).
 */
export const BAR_RIGHT_CLEAR_PX = 684;

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
    if (prev && s.phase === prev.phase && s.resultStep === prev.resultStep && s.results === prev.results && s.user === prev.user) return;
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
    bar.style.cssText = scrubberRow(s) ? `top: ${SCRUBBER_TOP_PX}px; left: 16px; transform: none; max-width: calc(100vw - ${BAR_RIGHT_CLEAR_PX}px);` : "";
    // One button per non-results phase, then one per state family (ADR 0130 D1); the stage picker chooses within the family shown.
    const buttons: HTMLElement[] = s.phase.axis
      .filter((k) => k.kind !== "results")
      .map((k) => {
        const b = el("button", same(s.phase.at, k) ? "phase active" : "phase", label(k));
        unlisten.push(listen(b, "click", () => store.dispatch({ type: "setPhase", at: k })));
        return b;
      });
    const shown = s.phase.at?.kind === "results" ? familyOfStage(s, s.phase.at.stage) : null;
    for (const f of families) buttons.push(familyButton(f, shown === f.key));
    if (badge) {
      const b = el("span", "phase user-badge", badge.text);
      b.title = badge.title;
      b.style.cssText = `cursor: default; color: var(${badge.refused ? "--error" : "--interp-text"});`;
      buttons.push(b);
    }
    for (const b of buttons) rigid(b);
    bar.replaceChildren(...buttons, ...(t ? scrubber(t, s) : []));
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
        store.dispatch({ type: "setPhase", at: { kind: "results", stage: f.stages[0]!, step: 0 } });
      }),
    );
    return b;
  };
  const scrubber = (t: Timeline, s: State): HTMLElement[] => {
    const box = el("span", "scrubber");
    // It shrinks before the bar overflows its cap: the slider to its least width, then the readout to an ellipsis.
    box.style.cssText = "display: flex; flex: 1 1 auto; min-width: 0; align-items: center; gap: 6px; padding: 0 6px; font: 600 12px var(--font); color: var(--text);";
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
  return () => {
    stop();
    unsubscribe();
    for (const u of unlisten) u();
  };
}
