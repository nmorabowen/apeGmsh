// Phase panel: the phase axis (ADR 0112 D7: geometry -> mesh -> results) as
// a row of buttons, shown when there is more than one phase to choose; on a
// results phase (V4b) it is also the time scrubber: the results stage
// picker, the step slider, the time readout, play/pause and step -1/+1. A
// click dispatches `setPhase` or `setResultStep`; the viewport draws the
// phase the state names, and the effects fetch the step (latest wins). The
// panel only dispatches: it keeps no results data, and play is a timer that
// dispatches the next step. (style.css is not this panel's file, so the
// scrubber's layout is inline, with the theme's variables only.)

import { inactiveElements, timelineOf, type Timeline } from "../state/selectors.ts";
import type { State, Store } from "../state/store.ts";
import type { PhaseKey } from "../state/types.ts";
import { byId, el, listen } from "../ui/dom.ts";

const label = (k: PhaseKey): string => (k.kind === "stage" ? `stage ${k.index + 1}` : k.kind);

const same = (a: PhaseKey | null, b: PhaseKey): boolean =>
  a !== null && a.kind === b.kind && (a.kind !== "stage" || (b.kind === "stage" && a.index === b.index));

/** Milliseconds between steps while playing (10 steps/s; the D5 trigger is 30 steps/s, so playback never outruns it). */
export const PLAY_MS = 100;

const fmtTime = (t: number): string => (Number.isInteger(t) ? String(t) : Math.abs(t) >= 1e-3 && Math.abs(t) < 1e4 ? t.toPrecision(4).replace(/\.?0+$/, "") : t.toExponential(3));

/** The scrubber's readout: `step 3 / 5 · t = 0.6` and the fetch and activation state. */
export function readoutOf(t: Timeline, hidden: number | null): string {
  const time = t.kind === "mode" ? "mode shape" : `t = ${fmtTime(t.time)}`;
  const act = hidden === null ? "" : ` · ${hidden} inactive hidden`;
  const fetch = t.fetch === "ready" ? "" : ` · step ${t.fetch}`;
  return `step ${t.step + 1} / ${t.steps} · ${time}${act}${fetch}`;
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
    if (prev && s.phase === prev.phase && s.resultStep === prev.resultStep && s.results === prev.results) return;
    for (const u of unlisten) u();
    unlisten = [];
    const t = timelineOf(s);
    if (!t) stop();
    bar.hidden = s.phase.axis.length < 2;
    // One button per non-results phase, and one "results" button for the results stages (the picker chooses among them).
    const firstResults = s.phase.axis.find((k) => k.kind === "results");
    const keys: PhaseKey[] = s.phase.axis.filter((k) => k.kind !== "results");
    if (firstResults) keys.push(firstResults);
    const buttons = keys.map((k) => {
        const on = k.kind === "results" ? s.phase.at?.kind === "results" : same(s.phase.at, k);
        const b = el("button", on ? "phase active" : "phase", label(k));
        unlisten.push(listen(b, "click", () => {
          if (k.kind !== "results" || s.phase.at?.kind !== "results") store.dispatch({ type: "setPhase", at: k });
        }));
        return b;
      });
    bar.replaceChildren(...buttons, ...(t ? scrubber(t, s) : []));
  };
  const scrubber = (t: Timeline, s: State): HTMLElement[] => {
    const box = el("span", "scrubber");
    box.style.cssText = "display: flex; align-items: center; gap: 6px; padding: 0 6px; font: 600 12px var(--font); color: var(--text);";
    const pick = el("select", "results-stage");
    pick.title = "results stage";
    pick.style.cssText = "font: 600 12px var(--font); color: var(--text); background: var(--hover); border: 1px solid var(--line); border-radius: 5px;";
    for (const st of t.stages) {
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
    const fwd = el("button", "phase step-forward", "▶ |");
    fwd.title = "step +1";
    fwd.disabled = t.step === t.steps - 1;
    const slider = el("input", "results-step");
    slider.type = "range";
    slider.min = "0";
    slider.max = String(t.steps - 1);
    slider.value = String(t.step);
    slider.disabled = t.steps < 2;
    slider.style.cssText = "width: 220px; accent-color: var(--accent);";
    const inactive = inactiveElements(s);
    const readout = el("span", "results-readout", readoutOf(t, inactive ? inactive.hidden : null));
    readout.style.cssText = "min-width: 180px; color: var(--muted); font: 600 12px var(--mono);";
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
