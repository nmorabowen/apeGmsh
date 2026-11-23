// The material tester's interim surface (ADR 0135 Amendment 1): under a uniaxial
// row of the materials window, a form for the strain protocol, a Run button, and
// the material's runs: the response curves overlaid (the newest in the accent, older
// ones muted, each with its own dash), a second plot against the step (the imposed
// history, the tangent, or the work done), and per run a CSV save and a remove (D7).
// Below the protocol, the material's numeric argument slots can be overridden for
// the test only (D6); the changed values show as text to copy into the script by
// hand (D8), since the model is never changed.
// The fork computes; this only dispatches events and draws the state (ADR 0112 D6),
// so it moves to ADR 0137's tool host as it is.
//
// The window is rebuilt on every state change, so what is typed lives in the
// store (`materialTestFormSet`), and each field carries a focus key the frame
// puts the caret back on.

import type { Event, State } from "../../state/store.ts";
import {
  materialTestAxesOf,
  materialTestFormFor,
  materialTestOverridesOf,
  materialTestOverrideText,
  materialTestProtocolOf,
  materialTestProtocolText,
  materialTestSlotsOf,
  materialTestStepsOf,
} from "../../state/selectors.ts";
import type { DeclPath, MaterialTestForm, MaterialTestOverride, MaterialTestRun, MaterialTestSecond, MaterialTestSlot } from "../../state/types.ts";
import { el } from "../../ui/dom.ts";
import { fmt, xyPlot, xySwatch } from "../../ui/xyPlot.ts";

const KINDS: readonly (readonly [MaterialTestForm["kind"], string])[] = [
  ["monotonic", "monotonic"],
  ["cyclic", "cyclic"],
  ["values", "values"],
  ["stress", "stress control"],
];

type Field = "peak" | "steps" | "amplitudes" | "stepsPerRamp" | "values" | "targets";

/** Keep a click inside the tester from selecting the row under it. */
const contained = <T extends HTMLElement>(e: T): T => {
  e.addEventListener("click", (ev) => ev.stopPropagation());
  return e;
};

function select<T extends string>(cls: string, title: string, options: readonly (readonly [T, string])[], value: T, on: (v: T) => void): HTMLElement {
  const s = contained(el("select", cls) as HTMLSelectElement);
  s.title = title;
  for (const [v, label] of options) {
    const o = el("option", undefined, label) as HTMLOptionElement;
    o.value = v;
    o.selected = v === value;
    s.append(o);
  }
  s.addEventListener("change", () => on(s.value as T));
  return s;
}

function input(f: MaterialTestForm, field: Field, label: string, title: string, act: (e: Event) => void, area = false): HTMLElement {
  const wrap = el("label", "mt-field");
  const box = contained(el(area ? "textarea" : "input", "mt-input") as HTMLInputElement);
  if (!area) box.type = "text";
  box.value = f[field];
  box.title = title;
  box.spellcheck = false;
  box.dataset["focusKey"] = `mt-${field}`;
  box.addEventListener("input", () => act({ type: "materialTestFormSet", form: { ...f, [field]: box.value } }));
  wrap.append(el("span", "mt-label", label), box);
  return wrap;
}

function button(text: string, title: string, on: () => void, cls = "window-test"): HTMLElement {
  const b = contained(el("button", cls, text));
  b.title = title;
  b.addEventListener("click", on);
  return b;
}

/** The material's numeric slots, each an input holding its declared value or what was typed; locked ones disabled. */
function slotsEl(f: MaterialTestForm, slots: readonly MaterialTestSlot[], act: (e: Event) => void): HTMLElement {
  const grid = el("div", "mt-slots");
  for (const sl of slots) {
    const typed = f.overrides[String(sl.slot)];
    const text = typed ?? String(sl.value);
    const changed = text.trim() !== "" && Number(text) !== sl.value;
    const wrap = el("label", "mt-field mt-slot" + (changed ? " mt-changed" : ""));
    const box = contained(el("input", "mt-input") as HTMLInputElement);
    box.type = "text";
    box.value = text;
    box.spellcheck = false;
    box.dataset["focusKey"] = `mt-o-${sl.slot}`;
    box.disabled = sl.locked !== null;
    box.title = sl.locked ? `locked: ${sl.locked}` : `declared ${sl.value}; this test only, the model is not changed`;
    box.addEventListener("input", () => act({ type: "materialTestFormSet", form: { ...f, overrides: { ...f.overrides, [String(sl.slot)]: box.value } } }));
    wrap.append(el("span", "mt-label", sl.label), box);
    grid.append(wrap);
  }
  return grid;
}

/** The changed values as text to copy (D8), with a Copy button where the page has a clipboard. */
function copyEl(path: DeclPath, type: string, overrides: readonly MaterialTestOverride[], f: MaterialTestForm, act: (e: Event) => void): HTMLElement {
  const box = el("div", "mt-copy");
  const text = materialTestOverrideText(path, type, overrides);
  box.append(el("div", "mt-copy-text", text));
  const bar = el("div", "mt-bar");
  const clip = (globalThis as { navigator?: { clipboard?: { writeText(t: string): Promise<void> } } }).navigator?.clipboard;
  if (clip) bar.append(button("Copy", "copy the changed values, to edit the script by hand", () => void clip.writeText(text)));
  bar.append(button("Reset", "go back to the declared values", () => act({ type: "materialTestFormSet", form: { ...f, overrides: {} } })));
  box.append(bar);
  return box;
}

/** The second plot's curve of one run, against the step: its imposed history is the stress under stress control. */
function second(r: MaterialTestRun, what: MaterialTestSecond): readonly number[] {
  const p = r.points!;
  if (what === "history") return r.protocol.kind === "stress" ? p.stress : p.strain;
  return what === "tangent" ? p.tangent : p.energy;
}

/** The dash of a run in the overlay: the newest ready run is the main curve (null), older ones count back from 0. */
const dashOf = (ready: readonly MaterialTestRun[], r: MaterialTestRun): number | null => {
  const i = ready.indexOf(r);
  return i === ready.length - 1 ? null : ready.length - 2 - i;
};

function plotsEl(ready: readonly MaterialTestRun[], f: MaterialTestForm, xName: string, yName: string, act: (e: Event) => void): HTMLElement {
  const box = el("div", "mt-plots");
  const main = ready[ready.length - 1]!;
  const older = ready.slice(0, -1);
  const p = main.points!;
  box.append(
    xyPlot({
      xs: p.strain,
      ys: p.stress,
      xName,
      yName,
      title: `${yName} against ${xName}, ${main.type}, ${ready.length} run${ready.length === 1 ? "" : "s"}`,
      others: older.map((r) => ({ xs: r.points!.strain, ys: r.points!.stress, dash: dashOf(ready, r)! })),
    }),
  );
  const imposed = main.protocol.kind === "stress" ? yName : xName;
  const names: Record<MaterialTestSecond, string> = { history: `imposed ${imposed}`, tangent: "tangent", energy: "work done (app, trapezoid)" };
  const titles: Record<MaterialTestSecond, string> = {
    history: `the imposed ${imposed} at each step`,
    tangent: "the material's tangent at each step, as the fork returns it",
    energy: `the running integral of ${yName} over ${xName}, computed by the app (trapezoid); where the ${yName} returns to zero it is the dissipated energy`,
  };
  const pick = el("div", "mt-bar");
  pick.append(
    select<MaterialTestSecond>("mt-second", "what the second plot shows against the step", [["history", names.history], ["tangent", names.tangent], ["energy", names.energy]], f.second, (v) =>
      act({ type: "materialTestFormSet", form: { ...f, second: v } }),
    ),
  );
  const note = el("span", "window-row-decl", titles[f.second]);
  pick.append(note);
  box.append(
    pick,
    xyPlot({
      xs: p.step,
      ys: second(main, f.second),
      xName: "step",
      yName: names[f.second],
      title: `${names[f.second]} against the step, ${main.type}`,
      others: older.map((r) => ({ xs: r.points!.step, ys: second(r, f.second), dash: dashOf(ready, r)! })),
    }),
  );
  return box;
}

/** One line of the legend: the run's dash, what it ran, its build and peak, and its buttons. */
function runLine(s: State, r: MaterialTestRun, dash: number | null | undefined, yName: string, act: (e: Event) => void): HTMLElement {
  const line = el("div", "mt-legend-line");
  if (dash !== undefined) line.append(xySwatch(dash));
  let text = `run ${r.id} · ${materialTestProtocolText(r.protocol)} · ${r.overrides.length ? r.overrides.map((o) => `${o.label} ${o.declared}→${o.value}`).join(", ") : "declared"}`;
  if (r.status === "running") text += " · running on the fork…";
  if (r.status === "ready") {
    let peak = 0;
    for (const y of r.points!.stress) if (Math.abs(y) > Math.abs(peak)) peak = y;
    text += ` · fork build ${(r.build ?? "?").slice(0, 8)} · peak ${yName} ${fmt(peak)}`;
  }
  line.append(el("span", "mt-legend-text", text));
  if (r.status === "ready") {
    const saving = s.materialTest.saving !== null;
    const csv = button("CSV", "save this run's full arrays as CSV, where you choose", () => act({ type: "materialTestSaveRequested", id: r.id })) as HTMLButtonElement;
    csv.disabled = saving;
    line.append(csv);
  }
  line.append(button("×", r.status === "running" ? "stop this run" : "remove this run from the plot", () => act({ type: "materialTestRemoved", id: r.id }), "window-test mt-remove"));
  const box = el("div", "mt-legend-run");
  box.append(line);
  if (r.status === "ready" && r.stopped) box.append(el("div", "mt-error", `stopped at ${r.stopped}`));
  if (r.status === "failed") {
    box.append(el("div", "mt-error", r.error ?? "the run failed"));
    if (r.build) box.append(el("div", "window-row-decl", `fork build ${r.build.slice(0, 8)}`));
  }
  return box;
}

/** The Test button of a uniaxial row's head: opens its form, or closes the one open there. */
export function testButton(s: State, path: DeclPath, act: (e: Event) => void): HTMLElement {
  const open = s.materialTest.form?.decl === path;
  const b = contained(el("button", "window-test" + (open ? " on" : ""), "Test"));
  b.title = open ? "close the material tester" : "run this material on the installed Ladruno fork and plot its response";
  b.addEventListener("click", () => act({ type: "materialTestFormSet", form: open ? null : materialTestFormFor(path) }));
  return b;
}

/** The tester under the row of `path`, or null when its form is not open. */
export function testerEl(s: State, path: DeclPath, type: string, act: (e: Event) => void): HTMLElement | null {
  const f = s.materialTest.form;
  if (!f || f.decl !== path) return null;
  const [xName, yName] = materialTestAxesOf(type);
  const box = contained(el("div", "mt"));
  box.dataset["materialTest"] = path;

  const fields = el("div", "mt-fields");
  fields.append(select("mt-kind", "the imposed history, from zero", KINDS, f.kind, (kind) => act({ type: "materialTestFormSet", form: { ...f, kind } })));
  if (f.kind === "monotonic") {
    fields.append(input(f, "peak", `peak ${xName}`, `the ${xName} the ramp ends at`, act), input(f, "steps", "steps", "steps from zero to the peak", act));
  } else if (f.kind === "cyclic") {
    fields.append(
      input(f, "amplitudes", `amplitudes (${xName})`, "each amplitude once: 0 to +a to -a to 0; separate them with commas or spaces", act),
      input(f, "stepsPerRamp", "steps per ramp", "steps per quarter cycle", act),
    );
  } else if (f.kind === "stress") {
    fields.append(
      input(f, "targets", `targets (${yName})`, `the ${yName} each leg ends at, from zero; the fork finds the ${xName}, and a target beyond the material's peak stops the run there`, act),
      input(f, "stepsPerRamp", "steps per leg", "steps from one target to the next", act),
    );
  } else {
    fields.append(input(f, "values", `${xName} values`, `the ${xName} of each step after zero; separate them with commas, spaces or new lines`, act, true));
  }
  box.append(fields);
  const slots = materialTestSlotsOf(s, path);
  const changed = materialTestOverridesOf(f, slots);
  if (slots.length) {
    box.append(el("div", "mt-label mt-slots-head", "parameters (this test only; the model is not changed)"), slotsEl(f, slots, act));
    if ("overrides" in changed && changed.overrides.length) box.append(copyEl(path, type, changed.overrides, f, act));
  }

  const protocol = materialTestProtocolOf(f, xName);
  const asked = "error" in protocol ? protocol : "error" in changed ? changed : { protocol: protocol.protocol, overrides: changed.overrides };
  const bar = el("div", "mt-bar");
  const run = contained(el("button", "window-test mt-run-button", "Run") as HTMLButtonElement);
  run.disabled = "error" in asked;
  run.title = "error" in asked ? asked.error : "run the fork's uniaxial tester; a run still going is replaced, earlier runs stay for comparison";
  if ("protocol" in asked) run.addEventListener("click", () => act({ type: "materialTestRequested", decl: path, protocol: asked.protocol, overrides: asked.overrides }));
  bar.append(run, el("span", "error" in asked ? "mt-error" : "window-row-decl", "error" in asked ? asked.error : `${materialTestStepsOf(asked.protocol).toLocaleString()} steps · values in the model's units`));
  box.append(bar);

  const mine = s.materialTest.runs.filter((r) => r.decl === path);
  const ready = mine.filter((r) => r.status === "ready");
  if (ready.length) box.append(plotsEl(ready, f, xName, yName, act));
  if (mine.length) {
    const legend = el("div", "mt-legend");
    for (const r of [...mine].reverse()) legend.append(runLine(s, r, r.status === "ready" ? dashOf(ready, r) : undefined, yName, act));
    if (mine.length > 1) legend.append(button("Clear runs", "remove every run of every material", () => act({ type: "materialTestCleared" })));
    box.append(legend);
  }
  if (s.materialTest.saving !== null) box.append(el("div", "window-note", "saving: choose where in the dialog…"));
  else if (s.materialTest.notice) box.append(el("div", "window-note", s.materialTest.notice));
  return box;
}
